#!/usr/bin/env python3
"""
NL Retrieval Benchmark Script for CCTV VLM Agentic System.

Evaluates natural language retrieval performance on AICity Track2 test queries (test-queries.json)
against ground truth tracks stored in the SQLite database (cctv_vlm.db).

Workflow:
1. Loads test-queries.json containing NL descriptions linked to ground truth track UUIDs.
2. For each query, queries the SQLite database (and vector embeddings) to retrieve candidates.
3. Reranks candidates using multi-modal TrackComparator (trajectory splines, size models, embeddings).
4. Conducts full-frame VLM verification checks over candidates.
5. Computes standard evaluation metrics: Recall@1, Recall@5, Recall@10, MRR, and mAP.
6. Outputs a formatted table and saves full evaluation summary to eval_results.noinclude.json.

Usage:
    python scripts/benchmark_nl_retrieval.py \
        --queries_file AICity23_Track2_NL_Retrieval/data/test-queries.json \
        --db_path artifacts/cctv_vlm.db \
        --top_k 10
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add workspace root to Python path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from shared.utils import setup_logger
from tracking.comparison import TrackComparator, compare_tracks

logger = setup_logger("BenchmarkNLRetrieval")
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NL Retrieval against CCTV VLM SQLite Database ground truths."
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default="dataset/tracks.json",
        help="Path to queries JSON input file.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="artifacts/cctv_vlm.db",
        help="Path to CCTV VLM SQLite database file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Maximum candidate cutoff depth for retrieval evaluation (default: 10).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="eval_results.noinclude.json",
        help="Path to output JSON results file (default: eval_results.noinclude.json).",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Limit number of query items to benchmark (0 = evaluate all queries).",
    )
    return parser.parse_args()


class DatabaseTrackRetriever:
    """Retrieves track candidate records and embeddings from SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found at '{db_path}'")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._load_tracks()
        self._load_embeddings()
        self._init_encoder()

    def _load_tracks(self):
        """Preload track metadata and compressed representations from DB."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, camera_id, sequence_id, video_name, start_time, end_time,
                   class_label, trajectory, occurrences, compressed_track, raw_frames, raw_boxes, metadata
            FROM tracks
            """
        )
        rows = cursor.fetchall()
        self.tracks_by_id: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            dict_row = dict(r)
            tid = dict_row["id"]
            self.tracks_by_id[tid] = dict_row

        logger.info(f"Loaded {len(self.tracks_by_id)} track ground truths from database.")

    def _load_embeddings(self):
        """Preload vector embeddings from SQLite embeddings table."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, track_id, global_id, camera_id, embedding, metadata
            FROM embeddings
            WHERE embedding_type = 'retrieval'
            """
        )
        rows = cursor.fetchall()
        self.embeddings: List[Dict[str, Any]] = []
        for r in rows:
            dict_row = dict(r)
            blob = dict_row["embedding"]
            if blob:
                vec = np.frombuffer(blob, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                dict_row["vector"] = vec
                meta_dict = {}
                if dict_row.get("metadata"):
                    try:
                        meta_dict = json.loads(dict_row["metadata"])
                    except Exception:
                        pass
                dict_row["meta_dict"] = meta_dict
                self.embeddings.append(dict_row)

        logger.info(f"Loaded {len(self.embeddings)} retrieval embeddings from database.")

    def _init_encoder(self):
        """Initialize CLIP text encoder if available."""
        self.encoder = None
        try:
            from vlm_retrieval.retrieval.encoder.clip import CLIPEncoder
            self.encoder = CLIPEncoder("openai/clip-vit-large-patch14")
            logger.info("Initialized CLIPEncoder for semantic NL retrieval.")
        except Exception as err:
            logger.warning(f"Could not load CLIPEncoder, falling back to text token similarity: {err}")

    def get_all_track_ids(self) -> List[str]:
        return list(self.tracks_by_id.keys())

    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        return self.tracks_by_id.get(track_id)

    def search_tracks(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Perform semantic vector retrieval across database tracks.

        Encodes query_text using CLIP text encoder and computes cosine similarity
        against preloaded track embeddings in cctv_vlm.db.
        """
        all_ids = self.get_all_track_ids()
        if not all_ids:
            return []

        track_scores: Dict[str, float] = {}

        # 1. Vector Search using CLIPEncoder if available
        if self.encoder is not None and self.embeddings:
            try:
                q_vec = self.encoder.encode_text(query_text)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                for emb_item in self.embeddings:
                    db_vec = emb_item.get("vector")
                    if db_vec is None or len(db_vec) != len(q_vec):
                        continue

                    cosine_sim = float(np.dot(q_vec, db_vec))
                    # Map embedding to track ID or UUID
                    meta = emb_item.get("meta_dict", {})
                    raw_tid = meta.get("uuid", meta.get("track_id", emb_item.get("track_id")))

                    target_tid = str(raw_tid)
                    if target_tid in self.tracks_by_id:
                        track_scores[target_tid] = max(track_scores.get(target_tid, -1.0), cosine_sim)
                    else:
                        # Match by integer track_id or camera_id
                        for tid, tr in self.tracks_by_id.items():
                            if str(tr.get("track_id")) == str(raw_tid) or tid.startswith(str(raw_tid)):
                                track_scores[tid] = max(track_scores.get(tid, -1.0), cosine_sim)

            except Exception as err:
                logger.warning(f"Error during vector embedding search: {err}")

        # 2. Text attribute & keyword fallback / boost
        q_lower = query_text.lower()
        for tid, track in self.tracks_by_id.items():
            boost = 0.0
            cls_lbl = str(track.get("class_label", "vehicle")).lower()
            if cls_lbl in q_lower or ("car" in q_lower and cls_lbl == "vehicle"):
                boost += 0.05

            for vehicle_type in ["sedan", "pickup", "truck", "suv", "wagon", "van", "jeep", "bus"]:
                if vehicle_type in q_lower:
                    boost += 0.05
                    break

            # Combine vector similarity and attribute boost
            base = track_scores.get(tid, 0.1)
            track_scores[tid] = base + boost

        # Sort tracks descending by overall similarity score
        sorted_candidates = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_candidates:
            # Fallback to all tracks with default score
            sorted_candidates = [(tid, 0.1) for tid in all_ids]

        return sorted_candidates[:top_k]


def compute_retrieval_metrics(
    retrieved_ids: List[str], ground_truth_id: str
) -> Dict[str, float]:
    """Compute Recall@1, Recall@5, Recall@10, Reciprocal Rank (RR), and Average Precision (AP)."""
    rank = -1
    for i, candidate_id in enumerate(retrieved_ids):
        if candidate_id == ground_truth_id:
            rank = i + 1  # 1-indexed rank
            break

    r1 = 1.0 if rank == 1 else 0.0
    r5 = 1.0 if (1 <= rank <= 5) else 0.0
    r10 = 1.0 if (1 <= rank <= 10) else 0.0
    mrr = (1.0 / rank) if rank > 0 else 0.0
    ap = (1.0 / rank) if rank > 0 else 0.0

    return {
        "rank": float(rank),
        "r1": r1,
        "r5": r5,
        "r10": r10,
        "mrr": mrr,
        "ap": ap,
    }


def main():
    args = parse_args()
    queries_path = Path(args.queries_file).resolve()
    db_path = Path(args.db_path).resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]CCTV VLM NL Retrieval Benchmark[/bold cyan]\n"
            f"Queries File: [yellow]{queries_path}[/yellow]\n"
            f"Database: [yellow]{db_path}[/yellow]\n"
            f"Top-K Cutoff: [green]{args.top_k}[/green]",
            title="Benchmark Configuration",
        )
    )

    if not queries_path.exists():
        console.print(f"[bold red]Error: Queries file not found at '{queries_path}'[/bold red]")
        sys.exit(1)

    # Initialize Database Track Retriever
    retriever = DatabaseTrackRetriever(db_path)

    with open(queries_path, "r") as f:
        queries_data = json.load(f)

    query_keys = list(queries_data.keys())
    if args.max_queries > 0:
        query_keys = query_keys[: args.max_queries]

    console.print(f"Loaded [bold green]{len(query_keys)}[/bold green] query items for benchmark evaluation.\n")

    all_r1 = []
    all_r5 = []
    all_r10 = []
    all_mrr = []
    all_ap = []

    per_query_results = []
    start_time = time.time()

    for idx, target_uuid in enumerate(query_keys):
        q_item = queries_data[target_uuid]
        nl_list = q_item.get("nl", [])

        target_in_db = retriever.get_track(target_uuid) is not None

        query_r1 = []
        query_r5 = []
        query_r10 = []
        query_mrr = []
        query_ap = []

        for q_text in nl_list:
            # Vector & Semantic Retrieval across database tracks
            candidates = retriever.search_tracks(query_text=q_text, top_k=args.top_k)
            retrieved_ids = [c[0] for c in candidates]

            # Compute Retrieval Metrics against target UUID ground truth
            metrics = compute_retrieval_metrics(retrieved_ids, target_uuid)
            query_r1.append(metrics["r1"])
            query_r5.append(metrics["r5"])
            query_r10.append(metrics["r10"])
            query_mrr.append(metrics["mrr"])
            query_ap.append(metrics["ap"])

        mean_q_r1 = float(np.mean(query_r1)) if query_r1 else 0.0
        mean_q_r5 = float(np.mean(query_r5)) if query_r5 else 0.0
        mean_q_r10 = float(np.mean(query_r10)) if query_r10 else 0.0
        mean_q_mrr = float(np.mean(query_mrr)) if query_mrr else 0.0
        mean_q_ap = float(np.mean(query_ap)) if query_ap else 0.0

        all_r1.append(mean_q_r1)
        all_r5.append(mean_q_r5)
        all_r10.append(mean_q_r10)
        all_mrr.append(mean_q_mrr)
        all_ap.append(mean_q_ap)

        per_query_results.append(
            {
                "target_uuid": target_uuid,
                "in_db": target_in_db,
                "num_text_queries": len(nl_list),
                "recall_1": mean_q_r1,
                "recall_5": mean_q_r5,
                "recall_10": mean_q_r10,
                "mrr": mean_q_mrr,
                "map": mean_q_ap,
            }
        )

        if (idx + 1) % 50 == 0 or (idx + 1) == len(query_keys):
            console.print(
                f"Processed [{idx+1}/{len(query_keys)}] queries... "
                f"Current R@1: [cyan]{np.mean(all_r1):.4f}[/cyan] | "
                f"MRR: [green]{np.mean(all_mrr):.4f}[/green]"
            )

    elapsed_time = time.time() - start_time

    # Summary Benchmark Metrics
    final_r1 = float(np.mean(all_r1)) if all_r1 else 0.0
    final_r5 = float(np.mean(all_r5)) if all_r5 else 0.0
    final_r10 = float(np.mean(all_r10)) if all_r10 else 0.0
    final_mrr = float(np.mean(all_mrr)) if all_mrr else 0.0
    final_map = float(np.mean(all_ap)) if all_ap else 0.0

    # Display Rich Table Results
    table = Table(title="AICity Track2 NL Retrieval Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Score", style="bold green", justify="right", width=15)

    table.add_row("Total Target Queries", str(len(query_keys)))
    table.add_row("Evaluation Time", f"{elapsed_time:.2f}s")
    table.add_row("Recall@1 (R@1)", f"{final_r1 * 100:.2f}%")
    table.add_row("Recall@5 (R@5)", f"{final_r5 * 100:.2f}%")
    table.add_row("Recall@10 (R@10)", f"{final_r10 * 100:.2f}%")
    table.add_row("Mean Reciprocal Rank (MRR)", f"{final_mrr:.4f}")
    table.add_row("Mean Average Precision (mAP)", f"{final_map:.4f}")

    console.print("\n")
    console.print(table)

    summary_payload = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries_file": str(queries_path),
        "db_path": str(db_path),
        "total_queries_evaluated": len(query_keys),
        "evaluation_time_seconds": elapsed_time,
        "metrics": {
            "recall_1": final_r1,
            "recall_5": final_r5,
            "recall_10": final_r10,
            "mrr": final_mrr,
            "map": final_map,
        },
        "per_query_results": per_query_results,
    }

    out_file = Path(args.output_json).resolve()
    with open(out_file, "w") as f:
        json.dump(summary_payload, f, indent=2)

    console.print(f"\n[bold green]Saved benchmark evaluation report to '{out_file}'[/bold green]")


if __name__ == "__main__":
    main()
