#!/usr/bin/env python3
"""
Standalone CCTV Inference Script (Serverless)

Runs natural language query inference directly against .npz embedding files / directories
and registry JSON metadata without requiring PostgreSQL or a running HTTP server node.

Supports:
  - Direct fast vector similarity search (SigLIP2 / cosine retrieval)
  - Agentic VLM reasoning planning pipeline (multistage perception tools)

Usage Examples:
    # Direct vector retrieval for "a blue public transport bus" using temp.noinclude.npz
    python scripts/run_standalone_inference.py "a blue public transport bus"

    # Search against a specific .npz directory
    python scripts/run_standalone_inference.py "red sedan car" --npz_dir ./embeddings_dir --top_k 10

    # Run full Agentic VLM reasoning pipeline
    python scripts/run_standalone_inference.py "person wearing black jacket" --mode agentic --reasoning_model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace root to python path to import app modules
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inference_node.config import InferenceConfig
from inference_node.retrieval.encoder import get_retrieval_encoder
from inference_node.retrieval.search import RetrievalEngine
from inference_node.retrieval.vector_store import VectorStore

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone serverless CCTV inference script using .npz embeddings and registry metadata."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="a blue public transport bus",
        help="Natural language query describing target object or person (default: 'a blue public transport bus').",
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default=str(workspace_root / "temp.noinclude.npz")
        if (workspace_root / "temp.noinclude.npz").exists()
        else None,
        help="Path to single .npz embeddings file.",
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        default=None,
        help="Directory containing .npz embedding files.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=str(workspace_root / "temp.noinclude.json")
        if (workspace_root / "temp.noinclude.json").exists()
        else None,
        help="Path to registry JSON metadata file.",
    )
    parser.add_argument(
        "--postgres_url",
        type=str,
        default=None,
        help="Optional PostgreSQL database URL if testing database mode.",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-base-patch16-224",
        help="Retrieval encoder model name.",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="VLM reasoning model for agentic mode (e.g. 'Qwen/Qwen3-VL-8B-Instruct', 'gemini-2.5-flash', 'openai-5.6').",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidates to retrieve.",
    )
    parser.add_argument(
        "--camera_id",
        type=str,
        default=None,
        help="Optional camera identifier filter (e.g. 'cam_1').",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "agentic"],
        default="direct",
        help="Inference mode: 'direct' = fast vector search; 'agentic' = multistage VLM reasoning pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for models (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save search results summary JSON.",
    )
    return parser.parse_args()


def run_direct_inference(args, vector_store: VectorStore) -> List[Dict[str, Any]]:
    """Execute direct semantic vector similarity search using SigLIP2 and VectorStore."""
    console.print(f"[bold cyan]Initializing SigLIP2 retrieval encoder:[/bold cyan] {args.retrieval_model}")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    retrieval_engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=True,
    )

    console.print(f"[bold green]Searching for query:[/bold green] '{args.query}' (top_k={args.top_k}, camera_id={args.camera_id})")
    parsed, results = retrieval_engine.search(
        query=args.query,
        top_k=args.top_k,
        camera_id=args.camera_id,
    )

    console.print(f"\n[bold yellow]Parsed Semantic Query:[/bold yellow] '{parsed.semantic_text}'")
    if parsed.metadata_filters:
        console.print(f"[bold yellow]Metadata Filters Applied:[/bold yellow] {parsed.metadata_filters}")

    # Build detailed formatted result list
    formatted_results = []
    table = Table(title=f"Search Results for '{args.query}'", box=box.ROUNDED)
    table.add_column("Rank", style="bold cyan", justify="right")
    table.add_column("Candidate ID", style="bold white", justify="left")
    table.add_column("Camera ID", style="green", justify="center")
    table.add_column("Track ID", style="magenta", justify="right")
    table.add_column("Time (sec / ms)", style="yellow", justify="left")
    table.add_column("Distance (1-sim)", style="red", justify="right")
    table.add_column("Class Label", style="blue", justify="center")

    for rank, res in enumerate(results, start=1):
        # Look up extra details from vector store record
        rec_info = None
        if vector_store.conn_type == "npz":
            for rec in vector_store.npz_records:
                if rec["id"] == res.id:
                    rec_info = rec
                    break

        meta = rec_info["metadata"] if rec_info else {}
        class_lbl = meta.get("class_label", "object")
        st_time = meta.get("start_time", res.camera_timestamp)
        end_time = meta.get("end_time", st_time)

        table.add_row(
            str(rank),
            res.id,
            res.camera_id,
            str(res.track_id),
            f"{st_time:.2f}s - {end_time:.2f}s ({res.video_pos_ms:.0f}ms)",
            f"{res.distance:.4f}",
            class_lbl,
        )

        formatted_results.append({
            "rank": rank,
            "id": res.id,
            "camera_id": res.camera_id,
            "track_id": res.track_id,
            "global_id": meta.get("global_id", res.track_id),
            "camera_timestamp": res.camera_timestamp,
            "video_pos_ms": res.video_pos_ms,
            "start_time": st_time,
            "end_time": end_time,
            "distance": res.distance,
            "class_label": class_lbl,
            "retrieval_embedding": rec_info.get("retrieval_embedding") if rec_info else None,
            "appearance_embedding": rec_info.get("appearance_embedding") if rec_info else None,
            "track_details": meta.get("track_details") or {
                "compressed_track": meta.get("compressed_track"),
                "trajectory": meta.get("trajectory"),
                "occurrences": meta.get("occurrences"),
            },
        })

    console.print("\n")
    console.print(Panel(table, border_style="cyan", expand=False))
    return formatted_results


def run_agentic_inference(args, vector_store: VectorStore) -> List[Dict[str, Any]]:
    """Execute full multistage Agentic Planning VLM pipeline."""
    from inference_node.agentic_pipeline import AgenticPlannerPipeline
    from inference_node.frame_extractor import FrameExtractor
    from inference_node.vqa import get_vqa_reasoner

    console.print(f"[bold cyan]Initializing SigLIP2 encoder:[/bold cyan] {args.retrieval_model}")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    retrieval_engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=True,
    )

    console.print("[bold cyan]Initializing Frame Extractor...[/bold cyan]")
    frame_extractor = FrameExtractor(video_sources={})

    console.print(f"[bold cyan]Loading Agentic VLM Reasoner:[/bold cyan] {args.reasoning_model}")
    reasoner = get_vqa_reasoner(model_name=args.reasoning_model, device=args.device)

    pipeline = AgenticPlannerPipeline(
        retrieval_engine=retrieval_engine,
        vector_store=vector_store,
        frame_extractor=frame_extractor,
        reasoner=reasoner,
        max_planning_steps=5,
    )

    console.print(f"[bold green]Executing Agentic VLM query:[/bold green] '{args.query}'")
    results = pipeline.query(query_text=args.query, top_k=args.top_k, camera_id=args.camera_id)

    formatted_results = []
    table = Table(title=f"Agentic VLM Pipeline Results for '{args.query}'", box=box.ROUNDED)
    table.add_column("Rank", style="bold cyan", justify="right")
    table.add_column("Camera ID", style="green", justify="center")
    table.add_column("Global Track ID", style="magenta", justify="right")
    table.add_column("Timestamp (Human)", style="yellow", justify="left")
    table.add_column("VLM Score", style="bold green", justify="right")
    table.add_column("VLM Explanation", style="white", justify="left")

    for item in results:
        table.add_row(
            str(item.rank),
            item.camera_id,
            str(item.global_id),
            item.timestamp_human,
            f"{item.vlm_score:.2f}",
            item.vlm_explanation or "Verified target match",
        )
        formatted_results.append(item.model_dump())

    console.print("\n")
    console.print(Panel(table, border_style="cyan", expand=False))
    return formatted_results


def main():
    args = parse_args()

    console.print("\n┌────────────────────────────────────────────────────────────┐")
    console.print("│        Standalone CCTV Inference Node (Serverless)         │")
    console.print("└────────────────────────────────────────────────────────────┘\n")

    # Connect VectorStore
    if args.npz_dir or args.npz_path:
        console.print(f"[bold cyan]Connecting to NPZ Vector Store:[/bold cyan] npz_dir={args.npz_dir}, npz_path={args.npz_path}, json_path={args.json_path}")
        vector_store = VectorStore(
            npz_dir=args.npz_dir,
            npz_path=args.npz_path,
            json_path=args.json_path,
            postgres_url=args.postgres_url,
        )
    else:
        console.print("[bold cyan]Connecting to default Vector Store...[/bold cyan]")
        vector_store = VectorStore(postgres_url=args.postgres_url)

    console.print(f"[bold green]Store Connection Type:[/bold green] '{vector_store.conn_type}'")
    console.print(f"[bold green]Total Events Indexed:[/bold green] {vector_store.get_event_count()}\n")

    if vector_store.get_event_count() == 0:
        console.print("[bold red]Warning: Vector store contains 0 events. Ensure .npz and .json files exist or specify --npz_path / --npz_dir.[/bold red]")

    # Run chosen inference mode
    if args.mode == "direct":
        results = run_direct_inference(args, vector_store)
    else:
        results = run_agentic_inference(args, vector_store)

    # Export output JSON if requested
    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "query": args.query,
            "mode": args.mode,
            "store_type": vector_store.conn_type,
            "total_store_events": vector_store.get_event_count(),
            "results_count": len(results),
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        console.print(f"\n[bold green]Saved results to:[/bold green] {out_path}\n")


if __name__ == "__main__":
    main()
