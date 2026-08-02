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

from shared.utils import setup_logger
from vlm_retrieval.config import VLMRetrievalConfig
from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore

logger = setup_logger("run_standalone_inference")
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
    parser.add_argument(
        "--report_file",
        "--report",
        type=str,
        default="inference_report.md",
        help="Path to save markdown summary report file (default: 'inference_report.md').",
    )
    return parser.parse_args()


def run_direct_inference(args, vector_store: VectorStore) -> List[Dict[str, Any]]:
    """Execute direct semantic vector similarity search using Retrieval Encoder and VectorStore."""
    console.print(f"[bold cyan]Initializing retrieval encoder:[/bold cyan] {args.retrieval_model}")
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
    from vlm_retrieval.agentic_pipeline import AgenticPlannerPipeline
    from vlm_retrieval.frame_extractor import FrameExtractor
    from vlm_retrieval.vqa import get_vqa_reasoner

    console.print(f"[bold cyan]Initializing retrieval encoder:[/bold cyan] {args.retrieval_model}")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    retrieval_engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=True,
    )

    console.print("[bold cyan]Initializing Frame Extractor...[/bold cyan]")
    frame_extractor = FrameExtractor(video_sources={})

    console.print(f"[bold cyan]Loading Agentic VLM Reasoner:[/bold cyan] {args.reasoning_model}")
    try:
        reasoner = get_vqa_reasoner(model_name=args.reasoning_model, device=args.device)
    except RuntimeError as err:
        console.print(f"[bold yellow]Warning loading reasoning model ({err}). Falling back to Autonomous Perception Reasoner.[/bold yellow]")
        from vlm_retrieval.vqa.gemini_reasoner import GeminiAgenticReasoner
        reasoner = GeminiAgenticReasoner(model_name=args.reasoning_model)

    pipeline = AgenticPlannerPipeline(
        retrieval_engine=retrieval_engine,
        vector_store=vector_store,
        frame_extractor=frame_extractor,
        reasoner=reasoner,
        max_planning_steps=5,
    )

    console.print(f"[bold green]Executing Agentic VLM query:[/bold green] '{args.query}'")
    results, trajectory = pipeline.query_with_trajectory(query_text=args.query, top_k=args.top_k, camera_id=args.camera_id)

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

    # Print reasoning traces to console and log VLM raw response
    if trajectory:
        console.print("\n[bold yellow]Agentic Reasoning Traces & Execution Steps:[/bold yellow]")
        for step in trajectory:
            console.print(f"\n[bold underline]Step {step.step_number}[/bold underline]: {step.thought.strip()}")
            logger.info(f"VLM Raw Response [Step {step.step_number}]: {step.thought.strip()}")
            for call in step.tool_calls:
                console.print(f"  [cyan]🔧 Tool Call [{call.call_id}]:[/cyan] {call.name}({call.arguments})")
                logger.info(f"Tool Call [{call.call_id}]: {call.name}({call.arguments})")
            for res in step.tool_results:
                status_str = "[red]ERROR[/red]" if res.is_error else "[green]SUCCESS[/green]"
                console.print(f"  [magenta]📥 Tool Result [{res.call_id}]:[/magenta] Status={status_str} | Content={res.content}")
                logger.info(f"Tool Result [{res.call_id}]: Status={'ERROR' if res.is_error else 'SUCCESS'} | Content={res.content}")

    return formatted_results, trajectory


def generate_markdown_report(
    args,
    vector_store: VectorStore,
    results: List[Dict[str, Any]],
    report_path: Path,
    trajectory: Optional[List[Any]] = None,
):
    """Generate a clean Markdown summary report using Jinja2 template engine."""
    import datetime
    import jinja2

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    formatted_trajectory = []
    if trajectory:
        for step in trajectory:
            formatted_trajectory.append({
                "step_number": getattr(step, "step_number", 1),
                "thought": getattr(step, "thought", "").strip(),
                "tool_calls": [
                    {
                        "call_id": getattr(call, "call_id", ""),
                        "name": getattr(call, "name", ""),
                        "arguments": getattr(call, "arguments", {}),
                    }
                    for call in getattr(step, "tool_calls", [])
                ],
                "tool_results": [
                    {
                        "call_id": getattr(res, "call_id", ""),
                        "name": getattr(res, "name", ""),
                        "content": getattr(res, "content", {}),
                        "is_error": getattr(res, "is_error", False),
                        "has_images": len(getattr(res, "extracted_images", [])) > 0,
                    }
                    for res in getattr(step, "tool_results", [])
                ],
            })

    template_str = """# Standalone CCTV Inference Report

**Generated:** {{ timestamp_str }}  
**Query:** `{{ query }}`  
**Mode:** `{{ mode }}`  
**Store Type:** `{{ store_type }}`  
**Total Events Indexed:** `{{ total_events }}`  
**Retrieval Encoder:** `{{ retrieval_model }}`  
**Reasoning Model:** `{{ reasoning_model }}`  

{% if trajectory %}
## Agentic VLM Reasoning Traces & Tool Execution

{% for step in trajectory %}
### Step {{ step.step_number }}: Agent Thought

> {{ step.thought }}

{% if step.tool_calls %}
#### Tool Calls

{% for call in step.tool_calls %}
- **`{{ call.name }}`** (ID: `{{ call.call_id }}`)
```json
{{ call.arguments | to_pretty_json }}
```

{% endfor %}
{% endif %}
{% if step.tool_results %}
#### Tool Execution Results

{% for res in step.tool_results %}
- **`{{ res.name }}`** (ID: `{{ res.call_id }}`, Status: `{{ 'FAILED' if res.is_error else 'SUCCESS' }}`, Images Attached: `{{ res.has_images }}`)
```json
{{ res.content | to_pretty_json }}
```

{% endfor %}
{% endif %}
{% endfor %}
{% endif %}
## Search Results Summary

{% if not results %}
_No matching results found._
{% elif mode == "direct" %}
| Rank | Candidate ID | Camera | Track ID | Time Range | Distance (1-sim) | Class Label |
| --- | --- | --- | --- | --- | --- | --- |
{% for item in results %}
| {{ item.rank }} | `{{ item.id }}` | `{{ item.camera_id }}` | `{{ item.track_id }}` | {{ "%.2f"|format(item.start_time or 0.0) }}s - {{ "%.2f"|format(item.end_time or item.start_time or 0.0) }}s | {{ "%.4f"|format(item.distance if item.distance is not none else 1.0) }} | `{{ item.class_label or "object" }}` |
{% endfor %}
{% else %}
| Rank | Camera | Global ID | Timestamp (Human) | VLM Score | Explanation |
| --- | --- | --- | --- | --- | --- |
{% for item in results %}
| {{ item.rank }} | `{{ item.camera_id }}` | `{{ item.global_id }}` | {{ item.timestamp_human }} | {{ "%.2f"|format(item.vlm_score or 0.0) }} | {{ item.vlm_explanation or "N/A" }} |
{% endfor %}
{% endif %}

## Candidate Details & Track Embeddings

{% for item in results %}
### #{{ item.rank or loop.index }} Candidate `{{ item.id or item.global_id }}` (Camera: `{{ item.camera_id }}`, Track: `{{ item.track_id or item.global_id }}`)

- **Class Label:** `{{ item.class_label or "object" }}`
- **Time Range:** `{{ "%.2f"|format(item.start_time or item.timestamp or 0.0) }}s` to `{{ "%.2f"|format(item.end_time or item.start_time or item.timestamp or 0.0) }}s` (`{{ "%.0f"|format(item.video_pos_ms or 0) }} ms`)
{% if item.distance is defined and item.distance is not none %}
- **Retrieval Distance:** `{{ "%.4f"|format(item.distance) }}`
{% endif %}
{% if item.vlm_score is defined and item.vlm_score is not none %}
- **VLM Verification Score:** `{{ "%.2f"|format(item.vlm_score) }}`
- **VLM Explanation:** {{ item.vlm_explanation }}
{% endif %}
{% if item.retrieval_embedding %}
- **Retrieval Embedding (Encoder):** {{ item.retrieval_embedding | length }}-dim vector
{% endif %}
{% if item.appearance_embedding %}
- **Appearance Embedding (ReID):** {{ item.appearance_embedding | length }}-dim vector
{% endif %}
{% if item.track_details %}
- **Registry Track Details:**
{% if item.track_details.compressed_track %}
  - **Compressed Track Class:** `{{ item.track_details.compressed_track.class }}`
{% if item.track_details.compressed_track.trajectory and item.track_details.compressed_track.trajectory.segments %}
  - **Trajectory Segments:** `{{ item.track_details.compressed_track.trajectory.segments | length }}`
{% endif %}
{% endif %}
{% if item.track_details.occurrences %}
  - **Occurrences Count:** `{{ item.track_details.occurrences | length }}` frames
{% endif %}
{% endif %}

{% endfor %}
"""

    def to_pretty_json(val):
        try:
            return json.dumps(val, indent=2, default=str)
        except Exception:
            return str(val)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["to_pretty_json"] = to_pretty_json

    rendered_report = env.from_string(template_str).render(
        timestamp_str=timestamp_str,
        query=args.query,
        mode=args.mode,
        store_type=vector_store.conn_type,
        total_events=vector_store.get_event_count(),
        retrieval_model=args.retrieval_model,
        reasoning_model=args.reasoning_model,
        trajectory=formatted_trajectory,
        results=results,
    )

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(rendered_report)

    console.print(f"\n[bold green]Generated summary report at:[/bold green] [cyan]{report_path}[/cyan]\n")



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
        )
    else:
        console.print("[bold cyan]Connecting to default Vector Store...[/bold cyan]")
        vector_store = VectorStore()

    console.print(f"[bold green]Store Connection Type:[/bold green] '{vector_store.conn_type}'")
    console.print(f"[bold green]Total Events Indexed:[/bold green] {vector_store.get_event_count()}\n")

    if vector_store.get_event_count() == 0:
        console.print("[bold red]Warning: Vector store contains 0 events. Ensure .npz and .json files exist or specify --npz_path / --npz_dir.[/bold red]")

    # Run chosen inference mode
    trajectory = None
    if args.mode == "direct":
        results = run_direct_inference(args, vector_store)
    else:
        results, trajectory = run_agentic_inference(args, vector_store)

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
            "trajectory": [
                {
                    "step_number": getattr(step, "step_number", 1),
                    "thought": getattr(step, "thought", ""),
                    "tool_calls": [
                        {
                            "call_id": getattr(c, "call_id", ""),
                            "name": getattr(c, "name", ""),
                            "arguments": getattr(c, "arguments", {}),
                        }
                        for c in getattr(step, "tool_calls", [])
                    ],
                    "tool_results": [
                        {
                            "call_id": getattr(r, "call_id", ""),
                            "name": getattr(r, "name", ""),
                            "content": getattr(r, "content", {}),
                            "is_error": getattr(r, "is_error", False),
                        }
                        for r in getattr(step, "tool_results", [])
                    ],
                }
                for step in (trajectory or [])
            ],
        }
        with open(out_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        console.print(f"\n[bold green]Saved JSON results to:[/bold green] {out_path}")

    # Generate Markdown Summary Report
    if args.report_file:
        generate_markdown_report(args, vector_store, results, Path(args.report_file), trajectory=trajectory)


if __name__ == "__main__":
    main()
