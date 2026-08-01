#!/usr/bin/env python3
"""
Interactive CCTV Semantic Search & Agentic VLM Reranking Demo Application

Operates directly against .npz embedding files, registry JSON metadata, and VectorStore.
Supports both fast direct vector search (Retrieval Encoder / cosine retrieval) and multistage
Agentic VLM visual reasoning (Qwen3-VL, Gemini, OpenAI) with an interactive terminal shell loop.

Usage Examples:
    # Launch interactive terminal shell (defaults to direct mode, type /mode agentic to switch)
    python scripts/demo_interactive_retrieval.py

    # Launch interactive shell in agentic VLM mode with Qwen3-VL
    python scripts/demo_interactive_retrieval.py --mode agentic --reasoning_model Qwen/Qwen3-VL-8B-Instruct

    # Run a single query directly and exit
    python scripts/demo_interactive_retrieval.py --query "a red vehicle or motorcycle" --top_k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from vlm_retrieval.agentic_pipeline import AgenticPlannerPipeline
from vlm_retrieval.config import VLMRetrievalConfig
from vlm_retrieval.frame_extractor import FrameExtractor

from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore
from vlm_retrieval.vqa import get_vqa_reasoner

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive terminal demo for SigLIP2 semantic search and Agentic VLM reasoning."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit. If omitted, launches interactive shell session.",
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to .npz embeddings file.",
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
        required=True,
        help="Path to registry JSON metadata file.",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Retrieval encoder model name.",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="VLM reasoning model for agentic mode (e.g. 'Qwen/Qwen3-VL-8B-Instruct', 'gemini-2.5-flash', 'openai-5.6').",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "agentic"],
        default="direct",
        help="Initial inference mode: 'direct' = fast vector search; 'agentic' = VLM reasoning pipeline.",
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
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, cpu.",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="inference_report.md",
        help="Path to save markdown summary report file.",
    )
    return parser.parse_args()


def execute_direct_query(
    query_str: str,
    vector_store: VectorStore,
    retrieval_engine: RetrievalEngine,
    top_k: int = 5,
    camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Executes fast vector similarity search using RetrievalEngine."""
    parsed, results = retrieval_engine.search(
        query=query_str,
        top_k=top_k,
        camera_id=camera_id,
    )

    console.print(f"\n[bold yellow]Parsed Semantic Query:[/bold yellow] '{parsed.semantic_text}'")
    if parsed.metadata_filters:
        console.print(f"[bold yellow]Metadata Filters Applied:[/bold yellow] {parsed.metadata_filters}")

    formatted_results = []
    table = Table(title=f"Search Results for '{query_str}'", box=box.ROUNDED)
    table.add_column("Rank", style="bold cyan", justify="right")
    table.add_column("Candidate ID", style="bold white", justify="left")
    table.add_column("Camera ID", style="green", justify="center")
    table.add_column("Track ID", style="magenta", justify="right")
    table.add_column("Time Range", style="yellow", justify="left")
    table.add_column("Distance (1-sim)", style="red", justify="right")
    table.add_column("Class Label", style="blue", justify="center")

    for rank, res in enumerate(results, start=1):
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
            f"{st_time:.2f}s - {end_time:.2f}s",
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
        })

    console.print("\n")
    console.print(Panel(table, border_style="cyan", expand=False))
    return formatted_results


def execute_agentic_query(
    query_str: str,
    pipeline: AgenticPlannerPipeline,
    top_k: int = 5,
    camera_id: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], List[Any]]:
    """Executes full Agentic VLM reasoning pipeline with perception tools."""
    console.print(f"\n[bold green]Executing Agentic VLM query:[/bold green] '{query_str}'")
    results, trajectory = pipeline.query_with_trajectory(
        query_text=query_str,
        top_k=top_k,
        camera_id=camera_id,
    )

    formatted_results = []
    table = Table(title=f"Agentic VLM Pipeline Results for '{query_str}'", box=box.ROUNDED)
    table.add_column("Rank", style="bold cyan", justify="right")
    table.add_column("Camera ID", style="green", justify="center")
    table.add_column("Global Track ID", style="magenta", justify="right")
    table.add_column("Timestamp", style="yellow", justify="left")
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

    if trajectory:
        console.print("\n[bold yellow]Agentic Reasoning Traces & Execution Steps:[/bold yellow]")
        for step in trajectory:
            console.print(f"\n[bold underline]Step {step.step_number}[/bold underline]: {step.thought.strip()}")
            for call in step.tool_calls:
                console.print(f"  [cyan]🔧 Tool Call [{call.call_id}]:[/cyan] {call.name}({call.arguments})")
            for res in step.tool_results:
                status_str = "[red]ERROR[/red]" if res.is_error else "[green]SUCCESS[/green]"
                console.print(f"  [magenta]📥 Tool Result [{res.call_id}]:[/magenta] Status={status_str} | Content={res.content}")

    return formatted_results, trajectory


def print_help():
    """Prints interactive shell commands help table."""
    table = Table(title="Interactive Shell Commands", box=box.SIMPLE_HEAD)
    table.add_column("Command / Syntax", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_row("query_text", "Execute semantic search for target (e.g. 'blue bus on cam_1')")
    table.add_row("mode direct / mode agentic", "Switch between fast vector search and VLM visual reasoning")
    table.add_row("model <name>", "Switch VLM model (e.g. 'Qwen/Qwen3-VL-8B-Instruct', 'gemini-2.5-flash')")
    table.add_row("top_k <int>", "Set candidate retrieval count (e.g. 'top_k 10')")
    table.add_row("help / ?", "Show this command guide")
    table.add_row("exit / quit / q", "Exit interactive shell")
    console.print(table)


def main():
    args = parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]CCTV Semantic Search & Agentic VLM Interactive Shell[/bold cyan]\n"
            f"[dim]Store Path: {args.npz_path or args.npz_dir or 'Default'}\n"
            f"Retrieval Model: {args.retrieval_model} | Reasoning Model: {args.reasoning_model}\n"
            f"Mode: {args.mode} | Top K: {args.top_k} | Device: {args.device}[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )

    # 1. Load VectorStore
    console.print(f"[bold cyan]Connecting to VectorStore...[/bold cyan]")
    vector_store = VectorStore(
        npz_dir=args.npz_dir,
        npz_path=args.npz_path,
        json_path=args.json_path,
    )
    console.print(f"[bold green]Store Connection Type:[/bold green] '{vector_store.conn_type}'")
    console.print(f"[bold green]Total Events Indexed:[/bold green] {vector_store.get_event_count()}\n")

    if vector_store.get_event_count() == 0:
        console.print("[bold red]Warning: VectorStore contains 0 events. Ensure .npz and .json files exist.[/bold red]")

    # 2. Initialize Retrieval Encoder & RetrievalEngine
    console.print(f"[bold cyan]Loading Retrieval Encoder ({args.retrieval_model})...[/bold cyan]")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)
    retrieval_engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=True,
    )

    # Lazy-loaded Agentic Pipeline components
    pipeline: Optional[AgenticPlannerPipeline] = None
    reasoner: Optional[Any] = None

    def get_agentic_pipeline() -> AgenticPlannerPipeline:
        nonlocal pipeline, reasoner
        if pipeline is None:
            console.print(f"[bold cyan]Loading Agentic VLM Reasoner ({args.reasoning_model})...[/bold cyan]")
            reasoner = get_vqa_reasoner(model_name=args.reasoning_model, device=args.device)
            frame_extractor = FrameExtractor(video_sources={})
            pipeline = AgenticPlannerPipeline(
                retrieval_engine=retrieval_engine,
                vector_store=vector_store,
                frame_extractor=frame_extractor,
                reasoner=reasoner,
                max_planning_steps=5,
            )
        return pipeline

    # Single Query Execution Mode
    if args.query:
        console.print(f"[bold green]Executing single query in '{args.mode}' mode:[/bold green] '{args.query}'")
        if args.mode == "direct":
            execute_direct_query(
                query_str=args.query,
                vector_store=vector_store,
                retrieval_engine=retrieval_engine,
                top_k=args.top_k,
                camera_id=args.camera_id,
            )
        else:
            agentic_pipe = get_agentic_pipeline()
            execute_agentic_query(
                query_str=args.query,
                pipeline=agentic_pipe,
                top_k=args.top_k,
                camera_id=args.camera_id,
            )
        return

    # Interactive Shell Loop
    console.print("[bold green]Interactive Shell Ready![/bold green] Type your search query or 'help' for options.")
    current_mode = args.mode
    current_top_k = args.top_k

    while True:
        try:
            prompt_str = f"[bold cyan]CCTV-Agent[/bold cyan] [[yellow]{current_mode}[/yellow]] > "
            user_input = Prompt.ask(prompt_str).strip()

            if not user_input:
                continue

            cmd_lower = user_input.lower()

            if cmd_lower in ("exit", "quit", "q"):
                console.print("[bold yellow]Exiting interactive search session. Goodbye![/bold yellow]")
                break

            if cmd_lower in ("help", "?"):
                print_help()
                continue

            if cmd_lower.startswith("mode ") or cmd_lower.startswith("/mode "):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].lower() in ("direct", "agentic"):
                    current_mode = parts[1].lower()
                    console.print(f"[bold green]Switched mode to:[/bold green] '{current_mode}'")
                else:
                    current_mode = "agentic" if current_mode == "direct" else "direct"
                    console.print(f"[bold green]Toggled mode to:[/bold green] '{current_mode}'")
                continue

            if cmd_lower.startswith("top_k ") or cmd_lower.startswith("topk "):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    current_top_k = int(parts[1])
                    console.print(f"[bold green]Updated top_k to:[/bold green] {current_top_k}")
                continue

            if cmd_lower.startswith("model ") or cmd_lower.startswith("/model "):
                parts = user_input.split(maxsplit=1)
                if len(parts) >= 2:
                    args.reasoning_model = parts[1]
                    pipeline = None  # Reset pipeline to reload new reasoner model
                    console.print(f"[bold green]Updated reasoning model to:[/bold green] '{args.reasoning_model}'")
                continue

            # Execute Query
            if current_mode == "direct":
                execute_direct_query(
                    query_str=user_input,
                    vector_store=vector_store,
                    retrieval_engine=retrieval_engine,
                    top_k=current_top_k,
                    camera_id=args.camera_id,
                )
            else:
                agentic_pipe = get_agentic_pipeline()
                execute_agentic_query(
                    query_str=user_input,
                    pipeline=agentic_pipe,
                    top_k=current_top_k,
                    camera_id=args.camera_id,
                )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session terminated. Goodbye![/bold yellow]")
            break
        except Exception as err:
            console.print(f"[bold red]Error executing query:[/bold red] {err}")


if __name__ == "__main__":
    main()
