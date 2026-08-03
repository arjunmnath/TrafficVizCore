#!/usr/bin/env python3
"""
Track Retrieval Script (Retrieval Encoder: SigLIP / CLIP)

Queries the CCTV SQL database (`cctv_vlm.db`), computes text-to-visual embedding
similarities across all tracks, and returns the top track candidate with highest similarity.

Usage Examples:
    poetry run python scripts/query_retrieval_track.py --query "white pickup truck"
    poetry run python scripts/query_retrieval_track.py --query "red sedan" --db_path artifacts/cctv_vlm.db --top_k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.utils import setup_logger
from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore

console = Console()
logger = setup_logger("QueryRetrievalTrack")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve top track ID matches for a text query using Retrieval Encoder against cctv_vlm.db.",
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
        "--db_path",
        "-d",
        type=str,
        default="artifacts/cctv_vlm.db",
        help="Path to the SQL database file (default: 'artifacts/cctv_vlm.db')",
    )
    parser.add_argument(
        "--retrieval_model",
        "--model_name",
        "-m",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Retrieval encoder model name (default: 'openai/clip-vit-large-patch14')",
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


def main():
    args = parse_args()

    logger.info(f"Connecting to SQL VectorStore database at '{args.db_path}'...")
    vector_store = VectorStore(db_path=args.db_path)

    logger.info(f"Initializing retrieval encoder '{args.retrieval_model}'...")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=True,
    )

    parsed, results = engine.search(query=args.query, top_k=args.top_k)

    if not results:
        console.print(f"[bold red]No matching candidates found for query: '{args.query}'[/bold red]")
        sys.exit(0)

    best_candidate = results[0]
    best_similarity = max(0.0, 1.0 - best_candidate.distance)

    # Render summary panel for winner
    winner_panel = Panel(
        f"[bold gold1]Query:[/bold gold1] {args.query}\n"
        f"[bold green]Highest Similarity Track ID:[/bold green] [bold yellow]{best_candidate.track_id}[/bold yellow]\n"
        f"[bold cyan]Cosine Similarity:[/bold cyan] {best_similarity:.4f}\n"
        f"[bold magenta]Retrieval Distance:[/bold magenta] {best_candidate.distance:.4f}",
        title="[bold blue]Track Retrieval Result[/bold blue]",
        border_style="cyan",
        expand=False,
    )
    console.print()
    console.print(winner_panel)
    console.print()

    # Render Top-K Table
    table = Table(
        title=f"Top-{len(results)} Candidate Matches",
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

    for idx, item in enumerate(results, start=1):
        sim = max(0.0, 1.0 - item.distance)
        table.add_row(
            str(idx),
            str(item.track_id),
            f"{sim:.4f}",
            f"{item.distance:.4f}",
            item.camera_id,
            str(item.id),
        )

    console.print(table)


if __name__ == "__main__":
    main()
