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

from vlm_retrieval.config import InferenceConfig
from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore

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
        default=str(workspace_root / "registry.embeddings.npz")
        if (workspace_root / "registry.embeddings.npz").exists()
        else (str(workspace_root / "temp.noinclude.npz") if (workspace_root / "temp.noinclude.npz").exists() else None),
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
        default=str(workspace_root / "registry.tracks.json")
        if (workspace_root / "registry.tracks.json").exists()
        else (str(workspace_root / "temp.noinclude.json") if (workspace_root / "temp.noinclude.json").exists() else None),
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
    parser.add_argument(
        "--report_file",
        "--report",
        type=str,
        default="inference_report.md",
        help="Path to save markdown summary report file (default: 'inference_report.md').",
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
    from vlm_retrieval.agentic_pipeline import AgenticPlannerPipeline
    from vlm_retrieval.frame_extractor import FrameExtractor
    from vlm_retrieval.vqa import get_vqa_reasoner

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

    # Print reasoning traces to console
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


def generate_markdown_report(
    args,
    vector_store: VectorStore,
    results: List[Dict[str, Any]],
    report_path: Path,
    trajectory: Optional[List[Any]] = None,
):
    """Generate a clean Markdown summary report including full reasoning traces and tool call results."""
    import datetime

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = []
    md_lines.append("# Standalone CCTV Inference Report\n")
    md_lines.append(f"**Generated:** {timestamp_str}  ")
    md_lines.append(f"**Query:** `{args.query}`  ")
    md_lines.append(f"**Mode:** `{args.mode}`  ")
    md_lines.append(f"**Store Type:** `{vector_store.conn_type}`  ")
    md_lines.append(f"**Total Events Indexed:** `{vector_store.get_event_count()}`  ")
    md_lines.append(f"**Retrieval Encoder:** `{args.retrieval_model}`  ")
    md_lines.append(f"**Reasoning Model:** `{args.reasoning_model}`  \n")

    if trajectory:
        md_lines.append("## Agentic VLM Reasoning Traces & Tool Execution\n")
        for step in trajectory:
            step_num = getattr(step, "step_number", 1)
            thought = getattr(step, "thought", "")
            tool_calls = getattr(step, "tool_calls", [])
            tool_results = getattr(step, "tool_results", [])

            md_lines.append(f"### Step {step_num}: Agent Thought\n")
            md_lines.append(f"> {thought.strip()}\n")

            if tool_calls:
                md_lines.append("#### Tool Calls\n")
                for call in tool_calls:
                    cid = getattr(call, "call_id", "")
                    name = getattr(call, "name", "")
                    args_dict = getattr(call, "arguments", {})
                    md_lines.append(f"- **`{name}`** (ID: `{cid}`)")
                    md_lines.append("  ```json")
                    md_lines.append(json.dumps(args_dict, indent=2))
                    md_lines.append("  ```\n")

            if tool_results:
                md_lines.append("#### Tool Execution Results\n")
                for res in tool_results:
                    cid = getattr(res, "call_id", "")
                    name = getattr(res, "name", "")
                    content = getattr(res, "content", {})
                    is_err = getattr(res, "is_error", False)
                    img_attached = len(getattr(res, "extracted_images", [])) > 0

                    status_tag = "FAILED" if is_err else "SUCCESS"
                    md_lines.append(f"- **`{name}`** (ID: `{cid}`, Status: `{status_tag}`, Images Attached: `{img_attached}`)")
                    md_lines.append("  ```json")
                    try:
                        md_lines.append(json.dumps(content, indent=2, default=str))
                    except Exception:
                        md_lines.append(str(content))
                    md_lines.append("  ```\n")

    md_lines.append("## Search Results Summary\n")
    if not results:
        md_lines.append("_No matching results found._\n")
    else:
        if args.mode == "direct":
            md_lines.append("| Rank | Candidate ID | Camera | Track ID | Time Range | Distance (1-sim) | Class Label |")
            md_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for item in results:
                rank = item.get("rank")
                cid = item.get("id")
                cam = item.get("camera_id")
                tid = item.get("track_id")
                st = item.get("start_time", 0.0)
                et = item.get("end_time", st)
                dist = item.get("distance", 1.0)
                cls = item.get("class_label", "object")
                md_lines.append(f"| {rank} | `{cid}` | `{cam}` | `{tid}` | {st:.2f}s - {et:.2f}s | {dist:.4f} | `{cls}` |")
        else:
            md_lines.append("| Rank | Camera | Global ID | Timestamp (Human) | VLM Score | Explanation |")
            md_lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in results:
                rank = item.get("rank")
                cam = item.get("camera_id")
                gid = item.get("global_id")
                ts_h = item.get("timestamp_human")
                score = item.get("vlm_score", 0.0)
                exp = item.get("vlm_explanation", "N/A")
                md_lines.append(f"| {rank} | `{cam}` | `{gid}` | {ts_h} | {score:.2f} | {exp} |")

        md_lines.append("\n## Candidate Details & Track Embeddings\n")
        for item in results:
            rank = item.get("rank", 1)
            cid = item.get("id", item.get("global_id"))
            cam = item.get("camera_id")
            tid = item.get("track_id", item.get("global_id"))
            cls = item.get("class_label", "object")
            st = item.get("start_time", item.get("timestamp", 0.0))
            et = item.get("end_time", st)

            md_lines.append(f"### #{rank} Candidate `{cid}` (Camera: `{cam}`, Track: `{tid}`)\n")
            md_lines.append(f"- **Class Label:** `{cls}`")
            md_lines.append(f"- **Time Range:** `{st:.2f}s` to `{et:.2f}s` (`{item.get('video_pos_ms', 0):.0f} ms`)")

            if "distance" in item:
                md_lines.append(f"- **Retrieval Distance:** `{item['distance']:.4f}`")
            if "vlm_score" in item:
                md_lines.append(f"- **VLM Verification Score:** `{item['vlm_score']:.2f}`")
                md_lines.append(f"- **VLM Explanation:** {item.get('vlm_explanation')}")

            ret_emb = item.get("retrieval_embedding")
            app_emb = item.get("appearance_embedding")
            if ret_emb:
                md_lines.append(f"- **Retrieval Embedding (Encoder):** {len(ret_emb)}-dim vector")
            if app_emb:
                md_lines.append(f"- **Appearance Embedding (ReID):** {len(app_emb)}-dim vector")

            track_det = item.get("track_details")
            if isinstance(track_det, dict) and track_det:
                md_lines.append("- **Registry Track Details:**")
                comp_tr = track_det.get("compressed_track") or {}
                if comp_tr:
                    md_lines.append(f"  - **Compressed Track Class:** `{comp_tr.get('class')}`")
                    traj = comp_tr.get("trajectory")
                    if isinstance(traj, dict):
                        md_lines.append(f"  - **Trajectory Segments:** `{len(traj.get('segments', []))}`")
                occs = track_det.get("occurrences")
                if isinstance(occs, list):
                    md_lines.append(f"  - **Occurrences Count:** `{len(occs)}` frames")

            md_lines.append("")

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))

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
