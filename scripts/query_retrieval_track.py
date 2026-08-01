#!/usr/bin/env python3
"""
Track Retrieval Script (Retrieval Encoder: SigLIP / CLIP)

Takes a natural language text query and a retrieval embeddings NPZ file,
computes text-to-visual embedding similarities across all tracks, and returns
the track ID with highest cosine similarity along with its retrieval distance.

Usage Examples:
    # Query default registry embeddings with default SigLIP encoder
    poetry run python scripts/query_retrieval_track.py --query "white pickup truck"

    # Query with CLIP model
    poetry run python scripts/query_retrieval_track.py --query "red sedan near camera 1" --retrieval_model openai/clip-vit-base-patch32

    # Specify custom embeddings NPZ file and top-K results
    poetry run python scripts/query_retrieval_track.py --query "red sedan near camera 1" --embeddings artifacts/registry.retrieval.embeddings.npz --top_k 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.utils import compute_cosine_similarity, setup_logger
from vlm_retrieval.retrieval.encoder.factory import get_retrieval_encoder
from vlm_retrieval.retrieval.vector_store import VectorStore

console = Console()
logger = setup_logger("QueryRetrievalTrack")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve the track ID with highest similarity for a text query using Retrieval Encoder (SigLIP / CLIP embeddings).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Natural language text query (e.g. 'white pickup truck', 'red vehicle')",
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=str,
        default="artifacts/registry.retrieval.embeddings.npz",
        help="Path to the retrieval embeddings NPZ file (default: 'artifacts/registry.retrieval.embeddings.npz')",
    )
    parser.add_argument(
        "--json_registry",
        "-j",
        type=str,
        default=None,
        help="Optional path to matching JSON track metadata registry file",
    )
    parser.add_argument(
        "--retrieval_model",
        "--model_name",
        "-m",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Retrieval encoder model name (default: 'google/siglip2-so400m-patch14-384'; supports CLIP e.g. 'openai/clip-vit-base-patch32')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computing device ('auto', 'cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidate matches to report (default: 5)",
    )
    return parser.parse_args()


def extract_track_id(record_id: str, metadata: Dict[str, Any]) -> str:
    """Extract a clean human-readable track/identity ID from record metadata or record key."""
    if metadata:
        if metadata.get("global_id"):
            return str(metadata["global_id"])
        if metadata.get("track_id"):
            return str(metadata["track_id"])
    
    # Extract track ID pattern from record_id string
    match = re.search(r"(global_veh_\d+|clip\d+\.mp4_\d+|cam_\d+_\d+|track_\d+|\d+)", str(record_id))
    if match:
        return match.group(1)
    return str(record_id)


def load_embeddings_from_npz(npz_path: str, json_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load embeddings and associated metadata records from NPZ file."""
    npz_file = Path(npz_path)
    if not npz_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {npz_path}")

    # Use VectorStore if available for robust workspace NPZ parsing
    try:
        vs = VectorStore(npz_path=str(npz_file), json_path=json_path)
        if vs.npz_records:
            records = []
            for rec in vs.npz_records:
                rec_id = rec.get("id", "unknown")
                meta = rec.get("metadata", {})
                vec = rec.get("embedding")
                if vec is None and rec.get("retrieval_embedding") is not None:
                    vec = np.array(rec["retrieval_embedding"], dtype=np.float32)
                
                if vec is not None:
                    track_id = extract_track_id(rec_id, meta)
                    records.append({
                        "id": rec_id,
                        "track_id": track_id,
                        "embedding": np.array(vec, dtype=np.float32),
                        "metadata": meta,
                    })
            if records:
                return records
    except Exception as e:
        logger.warning(f"VectorStore parser fallback triggered for '{npz_path}': {e}")

    # Manual fallback loader for raw key-based NPZ structures
    npz_data = np.load(npz_file, allow_pickle=True)
    records = []

    if "retrieval_embeddings" in npz_data or "embeddings" in npz_data:
        embs = npz_data.get("retrieval_embeddings", npz_data.get("embeddings"))
        metas = npz_data.get("metadatas", npz_data.get("metadata"))
        ids = npz_data.get("ids")

        for i in range(len(embs)):
            vec = np.array(embs[i], dtype=np.float32)
            rec_id = str(ids[i]) if ids is not None and i < len(ids) else f"rec_{i}"
            meta = {}
            if metas is not None and i < len(metas):
                m_raw = metas[i]
                if isinstance(m_raw, str):
                    try:
                        meta = json.loads(m_raw)
                    except Exception:
                        meta = {"raw": m_raw}
                elif isinstance(m_raw, dict):
                    meta = m_raw

            track_id = extract_track_id(rec_id, meta)
            records.append({
                "id": rec_id,
                "track_id": track_id,
                "embedding": vec,
                "metadata": meta,
            })
    else:
        # Dict mapping keys (e.g. clip1.mp4_app_1 or track_4) to embedding vectors
        for key in npz_data.keys():
            raw_arr = npz_data[key]
            if raw_arr.size == 0:
                continue
            if raw_arr.ndim == 2:
                vec = np.mean(raw_arr, axis=0)
            else:
                vec = raw_arr
            vec = np.array(vec, dtype=np.float32)
            track_id = extract_track_id(key, {})
            records.append({
                "id": key,
                "track_id": track_id,
                "embedding": vec,
                "metadata": {},
            })

    return records


def search_track_by_text(
    query: str,
    embeddings_path: str,
    json_path: Optional[str] = None,
    model_name: str = "google/siglip2-so400m-patch14-384",
    device: str = "auto",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Encode query with Retrieval Encoder (SigLIP / CLIP), score candidates, and return top matching track details.

    Returns:
        Dict containing:
            - best_track_id: str
            - highest_similarity: float
            - retrieval_distance: float
            - best_record: dict
            - top_matches: list of dicts
    """
    # Step 1: Resolve and load Retrieval Encoder and encode text query
    console.print(f"[bold cyan]Initializing retrieval encoder:[/bold cyan] {model_name}")
    encoder = get_retrieval_encoder(model_name=model_name, device=device)
    query_embedding = encoder.encode_text(query)

    # Step 2: Load track embeddings from NPZ
    console.print(f"[bold cyan]Loading embeddings from:[/bold cyan] {embeddings_path}")
    records = load_embeddings_from_npz(embeddings_path, json_path=json_path)
    if not records:
        raise ValueError(f"No valid embedding records found in '{embeddings_path}'")

    console.print(f"[bold green]Loaded {len(records)} candidate embeddings.[/bold green]")

    # Step 3: Compute Cosine Similarity and Retrieval Distance for each candidate
    scored_candidates = []
    dim_mismatch_warned = False

    for rec in records:
        rec_emb = rec["embedding"]
        if query_embedding.shape[-1] != rec_emb.shape[-1]:
            if not dim_mismatch_warned:
                logger.warning(
                    f"Embedding dimension mismatch: query dim is {query_embedding.shape[-1]} "
                    f"({model_name}), but record '{rec['id']}' dim is {rec_emb.shape[-1]}. "
                    f"Ensure you are passing embeddings generated with the same retrieval model."
                )
                dim_mismatch_warned = True

        similarity = compute_cosine_similarity(query_embedding, rec_emb)
        distance = 1.0 - similarity

        scored_candidates.append({
            "record_id": rec["id"],
            "track_id": rec["track_id"],
            "cosine_similarity": float(similarity),
            "retrieval_distance": float(distance),
            "metadata": rec["metadata"],
        })

    # Sort descending by cosine similarity (ascending by retrieval distance)
    scored_candidates.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    top_matches = scored_candidates[:top_k]

    best_candidate = top_matches[0]
    return {
        "query": query,
        "best_track_id": best_candidate["track_id"],
        "highest_similarity": best_candidate["cosine_similarity"],
        "retrieval_distance": best_candidate["retrieval_distance"],
        "best_record": best_candidate,
        "top_matches": top_matches,
    }


def main():
    args = parse_args()
    results = search_track_by_text(
        query=args.query,
        embeddings_path=args.embeddings,
        json_path=args.json_registry,
        model_name=args.retrieval_model,
        device=args.device,
        top_k=args.top_k,
    )

    # Render summary panel for winner
    winner_panel = Panel(
        f"[bold gold1]Query:[/bold gold1] {results['query']}\n"
        f"[bold green]Highest Similarity Track ID:[/bold green] [bold yellow]{results['best_track_id']}[/bold yellow]\n"
        f"[bold cyan]Cosine Similarity:[/bold cyan] {results['highest_similarity']:.4f}\n"
        f"[bold magenta]Retrieval Distance:[/bold magenta] {results['retrieval_distance']:.4f}",
        title="[bold blue]Track Retrieval Result[/bold blue]",
        border_style="cyan",
        expand=False,
    )
    console.print()
    console.print(winner_panel)
    console.print()

    # Render Top-K Table
    table = Table(
        title=f"Top-{len(results['top_matches'])} Candidate Matches",
        box=None,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Track ID", style="bold yellow")
    table.add_column("Cosine Sim", justify="right", style="green")
    table.add_column("Retrieval Dist", justify="right", style="magenta")
    table.add_column("Camera ID", style="blue")
    table.add_column("Record ID", style="dim")

    for idx, item in enumerate(results["top_matches"], 1):
        meta = item.get("metadata", {})
        cam_id = str(meta.get("camera_id", meta.get("camera_ids", "N/A")))
        table.add_row(
            str(idx),
            str(item["track_id"]),
            f"{item['cosine_similarity']:.4f}",
            f"{item['retrieval_distance']:.4f}",
            cam_id,
            str(item["record_id"]),
        )

    console.print(table)


if __name__ == "__main__":
    main()
