from __future__ import annotations

"""
Image crop extractor to retrieve bounding-box crops of person and vehicle identities
from video sources using the compressed-track registry JSON.

Instead of per-frame occurrence records, crops are derived by sampling the compressed
trajectory at a configurable time interval, reconstructing the bounding box via
BBoxReconstructor, and seeking to the nearest video frame.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

MIN_TIME_GAP_SECONDS = 3.0
workspace_root = Path(__file__).resolve().parent.parent

# Add repo root to path so we can import tracking modules
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from tracking.serialization.json_deserializer import JsonDeserializer
from tracking.compression.reconstruction import BBoxReconstructor


def sample_track_detections(
    comp_track_dict: dict,
    time_gap: float,
) -> list[dict]:
    """
    Sample the compressed track at regular time intervals and return a list of
    synthetic detection records compatible with the rest of the pipeline.

    Each record contains:
        feed_name, track_id, timestamp_seconds, frame (nearest), bbox [x1,y1,x2,y2]
    """
    try:
        track = JsonDeserializer.deserialize_from_dict(comp_track_dict)
    except Exception:
        return []

    t_start = track.metadata.start_timestamp
    t_end = track.metadata.end_timestamp
    feed = track.metadata.camera_id
    tid = track.metadata.track_id

    detections = []

    if time_gap <= 0:
        # Sample all frames of the track
        for frame_idx, frame_t in zip(track.time_model.frames, track.time_model.timestamps):
            try:
                x1, y1, x2, y2 = BBoxReconstructor.reconstruct(track, frame_t)
            except Exception:
                continue
            detections.append(
                {
                    "feed_name": feed,
                    "track_id": tid,
                    "timestamp_seconds": frame_t,
                    "frame": int(frame_idx),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections

    # Sample with a time gap
    t = t_start
    sampled_frames = set()
    while t <= t_end + 1e-9:
        try:
            x1, y1, x2, y2 = BBoxReconstructor.reconstruct(track, t)
        except Exception:
            t += time_gap
            continue

        # Get nearest frame and its actual timestamp
        frame_idx = int(track.time_model.timestamp_to_frame(t))

        if frame_idx in sampled_frames:
            t += max(time_gap, 0.1)
            continue

        frame_t = track.time_model.frame_to_timestamp(frame_idx)

        # Reconstruct at actual frame timestamp for maximum coordinate accuracy
        try:
            x1, y1, x2, y2 = BBoxReconstructor.reconstruct(track, frame_t)
        except Exception:
            pass

        detections.append(
            {
                "feed_name": feed,
                "track_id": tid,
                "timestamp_seconds": frame_t,
                "frame": frame_idx,
                "bbox": [x1, y1, x2, y2],
            }
        )
        sampled_frames.add(frame_idx)

        # Next sample target is at least time_gap after the actual frame's timestamp
        t = frame_t + time_gap

    return detections


def extract_reid_crops(
    json_path: str,
    video_dir: str,
    output_dir: str,
    min_time_gap_seconds: float = MIN_TIME_GAP_SECONDS,
    headless: bool = False,
    global_ids: list[int] | None = None,
    matches_path: str | None = None,
) -> None:
    json_path = Path(json_path)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    console = Console()

    stats = {
        "Unique Tracks": 0,
        "Sampled Frames": 0,
        "Extracted Crops": 0,
        "Missing Videos": 0,
        "Frame Errors": 0,
    }

    activity_log: list[str] = []

    def log_event(tag: str, message: str, style: str):
        timestamp = time.strftime("%H:%M:%S")
        log_line = f"[dim][{timestamp}][/dim] [{style}][{tag}][/{style}] {message}"
        if headless:
            console.print(log_line)
        else:
            activity_log.append(log_line)
            if len(activity_log) > 100:
                activity_log.pop(0)

    log_event("SYSTEM", f"Starting extraction pipeline. JSON: [cyan]{json_path}[/cyan]", "cyan")

    if not json_path.exists():
        log_event("ERROR", f"JSON file not found: [bold red]{json_path}[/bold red]", "red")
        if headless:
            sys.exit(1)
        else:
            console.print(f"[bold red]Error: JSON file not found: {json_path}[/bold red]")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(json_path, "r") as f:
            registry = json.load(f)
    except Exception as e:
        log_event("ERROR", f"Failed to load JSON file: {e}", "red")
        if headless:
            sys.exit(1)
        else:
            console.print(f"[bold red]Error: Failed to load JSON: {e}[/bold red]")
            return

    # ── Build (feed_name, track_id) → comp_track_dict map ────────────────────
    # JSON structure: {feed_name: [{track_id, compressed_track}, ...]}
    tracks_map: dict[tuple[str, int], dict] = {}
    if isinstance(registry, dict):
        for feed_name, tracks_list in registry.items():
            for entry in tracks_list:
                tid = entry["track_id"]
                comp_track = entry.get("compressed_track")
                if comp_track is None:
                    continue
                tracks_map[(feed_name, tid)] = comp_track
    else:
        log_event("ERROR", "Unexpected JSON format — expected a dict keyed by feed name.", "red")
        if headless:
            sys.exit(1)
        return

    # ── Load matches and perform greedy 1-to-1 matching ──────────────────────
    resolved_matches_path = None
    if matches_path:
        resolved_matches_path = Path(matches_path)
    else:
        # Default to output.json in the current working directory or workspace root
        for path_opt in [Path("output.json"), workspace_root / "output.json"]:
            if path_opt.exists():
                resolved_matches_path = path_opt
                break

    matched_nodes = set()
    matched_groups = []

    if resolved_matches_path and resolved_matches_path.exists():
        log_event("SYSTEM", f"Loading matches from: [cyan]{resolved_matches_path}[/cyan]", "cyan")
        try:
            with open(resolved_matches_path) as f:
                matches_data = json.load(f)

            # Sort matches by similarity descending to resolve greedily
            matches_data = sorted(
                matches_data, key=lambda x: x.get("similarity", 0.0), reverse=True
            )

            for m in matches_data:
                node_a = (m["feed_a"], int(m["track_a"]))
                node_b = (m["feed_b"], int(m["track_b"]))
                if node_a in tracks_map and node_b in tracks_map:
                    if node_a not in matched_nodes and node_b not in matched_nodes:
                        matched_nodes.add(node_a)
                        matched_nodes.add(node_b)
                        matched_groups.append((node_a, node_b))
        except Exception as e:
            log_event("WARNING", f"Failed to load matches file: {e}", "yellow")
    else:
        log_event(
            "WARNING",
            f"Matches file not found (path: {resolved_matches_path or 'output.json'}). "
            "All tracks will be treated as local tracks.",
            "yellow",
        )

    # Assign global_id string → list of (feed_name, track_id)
    global_id_map: dict[str, list[tuple[str, int]]] = {}

    # 1. Assign global IDs to Resolved Matched Groups first
    idx = 1
    # Sort matched groups by first node name, then track ID to ensure deterministic order
    for node_a, node_b in sorted(matched_groups, key=lambda x: (x[0][0], x[0][1])):
        global_id_map[f"{idx:03d}"] = [node_a, node_b]
        idx += 1

    # 2. Then assign unique global IDs to remaining local tracks
    for node in sorted(tracks_map.keys(), key=lambda x: (x[0], x[1])):
        if node not in matched_nodes:
            global_id_map[f"{idx:03d}"] = [node]
            idx += 1

    if global_ids:
        target_set = {f"{gid:03d}" for gid in global_ids}
        global_id_map = {gid: nodes for gid, nodes in global_id_map.items() if gid in target_set}
        log_event("FILTER", f"Filtered to global IDs: {global_ids}", "green")

    stats["Unique Tracks"] = sum(len(nodes) for nodes in global_id_map.values())

    # ── Sample detections from each compressed track ──────────────────────────
    # by_video: feed_name → list of detection dicts (enriched with global_id)
    by_video: dict[str, list[dict]] = defaultdict(list)

    for global_id, nodes in global_id_map.items():
        for feed_name, track_id in nodes:
            comp_track = tracks_map[(feed_name, track_id)]
            dets = sample_track_detections(comp_track, min_time_gap_seconds)
            stats["Sampled Frames"] += len(dets)
            for det in dets:
                det["global_id"] = global_id
                by_video[feed_name].append(det)

    log_event(
        "SYSTEM",
        f"Sampled [green]{stats['Sampled Frames']}[/green] frames from "
        f"[green]{stats['Unique Tracks']}[/green] tracks across "
        f"[cyan]{len(by_video)}[/cyan] videos.",
        "cyan",
    )

    total_videos = len(by_video)
    if total_videos == 0:
        log_event("WARNING", "No videos to process.", "yellow")
        return

    # ── Progress / layout helpers ─────────────────────────────────────────────
    overall_progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30, style="grey37", complete_style="cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    video_progress = Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=30, style="grey37", complete_style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    videos_task = overall_progress.add_task("Overall Progress", total=total_videos)
    video_task = video_progress.add_task("Active Video Detections", total=0, visible=False)

    config_info = {
        "JSON Path": str(json_path),
        "Video Dir": str(video_dir),
        "Output Dir": str(output_dir),
        "Time Gap": f"{min_time_gap_seconds}s",
    }

    current_video_name = "None"
    current_video_status = "Waiting..."

    def generate_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )

        header_text = Text(
            "ReID Crop Extractor",
            style="bold white on magenta",
            justify="center",
        )
        layout["header"].update(Panel(header_text, border_style="magenta", box=box.ROUNDED))

        left_table = Table.grid(padding=1, expand=True)
        cfg_tbl = Table(title="[bold cyan]Configuration[/bold cyan]", box=box.ROUNDED, expand=True)
        cfg_tbl.add_column("Parameter", style="cyan")
        cfg_tbl.add_column("Value", style="green")
        for k, v in config_info.items():
            cfg_tbl.add_row(k, str(v))

        stats_tbl = Table(
            title="[bold yellow]Statistics[/bold yellow]", box=box.ROUNDED, expand=True
        )
        stats_tbl.add_column("Metric", style="yellow")
        stats_tbl.add_column("Count", style="white", justify="right")
        for k, v in stats.items():
            stats_tbl.add_row(k, f"[bold]{v}[/bold]" if v > 0 else str(v))

        left_table.add_row(cfg_tbl)
        left_table.add_row(stats_tbl)
        layout["left"].update(Panel(left_table, border_style="cyan", box=box.ROUNDED))

        right_layout = Layout()
        right_layout.split_column(
            Layout(name="progress_section", size=9),
            Layout(name="events_section", ratio=1),
        )
        prog_table = Table.grid(padding=1, expand=True)
        prog_table.add_row(
            Text.assemble(("Active Video: ", "bold cyan"), (current_video_name, "bold white"))
        )
        prog_table.add_row(
            Text.assemble(("Status:       ", "bold yellow"), (current_video_status, "white"))
        )
        prog_table.add_row(overall_progress)
        prog_table.add_row(video_progress)

        right_layout["progress_section"].update(
            Panel(
                prog_table,
                title="[bold yellow]Extraction Progress[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

        events_text = Text()
        for ev in activity_log[-12:]:
            events_text.append(Text.from_markup(ev + "\n"))
        right_layout["events_section"].update(
            Panel(
                events_text,
                title="[bold red]Activity Log[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        layout["body"]["right"].update(right_layout)

        footer_text = Text(
            "Extracting crops from compressed tracks. Press Ctrl+C to abort.",
            style="italic dim white",
            justify="center",
        )
        layout["footer"].update(Panel(footer_text, border_style="grey37", box=box.ROUNDED))
        return layout

    start_time = time.time()

    def process_loop(live_instance=None):
        nonlocal current_video_name, current_video_status

        for video_idx, (video_name, detections) in enumerate(by_video.items(), 1):
            current_video_name = video_name
            current_video_status = f"Initializing {video_name}"
            if live_instance:
                live_instance.update(generate_layout())

            video_path = video_dir / video_name
            log_event(
                "VIDEO",
                f"Processing [bold]{video_name}[/bold] ({len(detections)} sampled frames)",
                "magenta",
            )
            if live_instance:
                live_instance.update(generate_layout())

            if not video_path.exists():
                fallback = workspace_root / "reid" / video_dir.name / video_name
                if fallback.exists():
                    video_path = fallback
                else:
                    log_event("WARNING", f"Missing video file: {video_path}", "yellow")
                    stats["Missing Videos"] += 1
                    overall_progress.update(videos_task, advance=1)
                    if live_instance:
                        live_instance.update(generate_layout())
                    continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                log_event("WARNING", f"Could not open video: {video_path}", "yellow")
                stats["Missing Videos"] += 1
                overall_progress.update(videos_task, advance=1)
                if live_instance:
                    live_instance.update(generate_layout())
                continue

            detections.sort(key=lambda x: x["frame"])

            current_video_status = "Extracting crops..."
            video_progress.update(video_task, total=len(detections), completed=0, visible=True)
            if live_instance:
                live_instance.update(generate_layout())

            current_frame_idx = -1
            frame = None

            for det in detections:
                frame_idx = int(det["frame"])

                if frame_idx != current_frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = cap.read()

                    if not success:
                        log_event(
                            "WARNING",
                            f"Failed reading frame {frame_idx} from {video_name}",
                            "yellow",
                        )
                        stats["Frame Errors"] += 1
                        video_progress.update(video_task, advance=1)
                        if live_instance:
                            live_instance.update(generate_layout())
                        continue

                    current_frame_idx = frame_idx

                if frame is None:
                    video_progress.update(video_task, advance=1)
                    if live_instance:
                        live_instance.update(generate_layout())
                    continue

                x1, y1, x2, y2 = map(int, det["bbox"])
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w - 1))
                x2 = max(0, min(x2, frame_w))
                y1 = max(0, min(y1, frame_h - 1))
                y2 = max(0, min(y2, frame_h))

                bbox_w = x2 - x1
                bbox_h = y2 - y1

                if bbox_w <= 0 or bbox_h <= 0:
                    video_progress.update(video_task, advance=1)
                    if live_instance:
                        live_instance.update(generate_layout())
                    continue

                crop = frame[y1 : y1 + bbox_h, x1 : x1 + bbox_w]

                global_id = str(det["global_id"])
                track_id = det["track_id"]
                timestamp = det["timestamp_seconds"]

                person_dir = output_dir / global_id
                person_dir.mkdir(parents=True, exist_ok=True)

                stem = Path(video_name).stem
                out_name = f"{stem}_f{frame_idx:06d}_t{track_id}_s{timestamp:.2f}.jpg"

                cv2.imwrite(str(person_dir / out_name), crop)
                stats["Extracted Crops"] += 1

                if stats["Extracted Crops"] % 10 == 0:
                    log_event(
                        "CROP",
                        f"Saved crop for ID {global_id} (Total: {stats['Extracted Crops']})",
                        "blue",
                    )

                video_progress.update(video_task, advance=1)
                if live_instance:
                    live_instance.update(generate_layout())

            cap.release()
            current_video_status = "Finished video"
            video_progress.update(video_task, visible=False)
            overall_progress.update(videos_task, advance=1)
            if live_instance:
                live_instance.update(generate_layout())

    if headless:
        process_loop()
    else:
        with Live(generate_layout(), console=console, refresh_per_second=10) as live:
            process_loop(live)

    elapsed = time.time() - start_time
    log_event(
        "SUCCESS",
        f"Finished extracting [green]{stats['Extracted Crops']}[/green] crops in [cyan]{elapsed:.2f}s[/cyan].",
        "green",
    )

    summary_table = Table(box=box.HEAVY, expand=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total Processing Time", f"{elapsed:.2f} seconds")
    for k, v in stats.items():
        summary_table.add_row(k, str(v))

    console.print("\n")
    console.print(
        Panel(
            summary_table,
            title="[bold green]CROP EXTRACTION SUMMARY[/bold green]",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract bounding-box crops from compressed-track registry JSON."
    )
    parser.add_argument(
        "--json_path",
        "-j",
        type=str,
        required=True,
        help="Path to the registry JSON file produced by run_reid_pipeline.py.",
    )
    parser.add_argument(
        "--matches_path",
        "-m",
        type=str,
        default=None,
        help="Path to a cross-camera match JSON to group tracks into global identities.",
    )
    parser.add_argument(
        "--video_dir",
        "-v",
        type=str,
        required=True,
        help="Directory containing the input video files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save the extracted crops.",
    )
    parser.add_argument(
        "--min_time_gap",
        "-t",
        type=float,
        default=MIN_TIME_GAP_SECONDS,
        help="Minimum time gap in seconds between sampled frames per track (default: 3 s).",
    )
    parser.add_argument(
        "--global_ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict extraction to these global IDs (default: all).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the fullscreen Live TUI; print plain-text logs instead.",
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        fallback = workspace_root / args.json_path
        if fallback.exists():
            json_path = fallback

    matches_path = None
    if args.matches_path:
        matches_path = Path(args.matches_path)
        if not matches_path.exists():
            fallback = workspace_root / args.matches_path
            if fallback.exists():
                matches_path = fallback

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        fallback = workspace_root / args.video_dir
        if fallback.exists():
            video_dir = fallback

    extract_reid_crops(
        json_path=str(json_path),
        video_dir=str(video_dir),
        output_dir=str(args.output_dir),
        min_time_gap_seconds=args.min_time_gap,
        headless=args.headless,
        global_ids=args.global_ids,
        matches_path=str(matches_path) if matches_path else None,
    )
