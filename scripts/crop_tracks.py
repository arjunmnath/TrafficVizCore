#!/usr/bin/env python3
from __future__ import annotations

"""
Crop Tracks Script

Produces image crops for each track specified in a compressed track registry JSON file.
Reconstructs bounding boxes using trajectory segments along with width and height size models.
Crops for each track are saved in output directories named `feedname_trackid`.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Add project root directory to python path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from tracking.serialization.json_deserializer import JsonDeserializer
from tracking.compression.reconstruction import BBoxReconstructor
from tracking.domain.track import CompressedTrack

console = Console()


def find_video_file(video_dir: Path, feed_name: str) -> Path | None:
    """Find video file matching feed_name within video_dir."""
    candidates = [
        video_dir / feed_name,
        video_dir / Path(feed_name).name,
        video_dir / f"{Path(feed_name).stem}.mp4",
        video_dir / f"{Path(feed_name).stem}.avi",
        video_dir / f"{Path(feed_name).stem}.mkv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Check camera subfolder (e.g. feed_name="c006_vdo.avi" -> video_dir/c006/vdo.avi)
    if "_" in feed_name:
        parts = feed_name.split("_", 1)
        cam_folder, v_name = parts[0], parts[1]
        c1 = video_dir / cam_folder / v_name
        if c1.exists() and c1.is_file():
            return c1

        sub_dir = video_dir / cam_folder
        if sub_dir.exists() and sub_dir.is_dir():
            for ext in ["vdo.avi", "vdo.mp4", "vdo.mkv", "*.avi", "*.mp4", "*.mkv"]:
                matches = list(sub_dir.glob(ext))
                if matches:
                    return matches[0]

    # Check if feed_name itself is a subfolder
    sub_dir = video_dir / feed_name
    if sub_dir.exists() and sub_dir.is_dir():
        for ext in ["vdo.avi", "vdo.mp4", "vdo.mkv", "*.avi", "*.mp4", "*.mkv"]:
            matches = list(sub_dir.glob(ext))
            if matches:
                return matches[0]

    # Recursive search: match parent folder name or target filename
    cam_prefix = feed_name.split("_")[0] if "_" in feed_name else None
    target_filename = Path(feed_name).name

    for root, _, files in os.walk(video_dir):
        root_path = Path(root)
        for f in files:
            f_path = root_path / f
            if f_path.suffix.lower() not in [".mp4", ".avi", ".mkv", ".mov"]:
                continue
            if cam_prefix and cam_prefix == root_path.name:
                return f_path
            if f == target_filename:
                return f_path

    return None


def produce_crops(
    registry_path: Path,
    video_dir: Path,
    output_dir: Path,
    time_gap: float = 0.0,
) -> None:
    """Read registry JSON and crop all tracks using compressed models and video sources."""
    if not registry_path.exists():
        console.print(f"[bold red]Error:[/bold red] Registry file not found: {registry_path}")
        sys.exit(1)

    console.print(f"[bold cyan]Loading registry JSON from:[/bold cyan] {registry_path}")
    with open(registry_path, "r") as f:
        registry = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Deserialize tracks grouped by video feed name
    tracks_by_feed: Dict[str, List[CompressedTrack]] = defaultdict(list)
    track_count = 0

    if not isinstance(registry, dict):
        console.print("[bold red]Error:[/bold red] Expected JSON object with feed names as keys.")
        sys.exit(1)

    for feed_name, track_list in registry.items():
        for item in track_list:
            if not item:
                continue
            comp_track_dict = item.get("compressed_track", item) if isinstance(item, dict) else item
            if not comp_track_dict:
                continue

            try:
                track = JsonDeserializer.deserialize_from_dict(comp_track_dict)
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not deserialize track in {feed_name}: {e}")
                continue

            camera_feed = feed_name
            tracks_by_feed[camera_feed].append(track)
            track_count += 1

    console.print(
        f"[bold green]Parsed {track_count} tracks across {len(tracks_by_feed)} video feeds.[/bold green]"
    )

    crops_saved = 0

    # 2. Process each video feed
    for feed_name, tracks in tracks_by_feed.items():
        video_path = find_video_file(video_dir, feed_name)
        if not video_path:
            console.print(f"[bold red]Warning:[/bold red] Video source for feed '{feed_name}' not found in '{video_dir}'. Skipping.")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            console.print(f"[bold red]Error:[/bold red] Could not open video file: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_feed_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0  # Default fallback if FPS read fails

        # Build frame map for this video feed using correct video frame indices (v_frame = int(round(t * fps)))
        # Map: v_frame -> List[crop_spec]
        frame_crops_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        total_feed_crops = 0

        feed_stem = Path(feed_name).stem

        for track in tracks:
            track_id = track.metadata.track_id
            # Output directory named feedname_trackid (e.g., clip1.mp4_1 or clip1_1)
            # Create subfolder named feedname_trackid
            track_dir_name = f"{feed_name}_{track_id}"
            track_output_dir = output_dir / track_dir_name

            timestamps = track.time_model.timestamps

            # Sample timestamps based on time_gap
            if time_gap <= 0.0:
                sampled_timestamps = timestamps
            else:
                sampled_timestamps = []
                last_t = -float("inf")
                for t in timestamps:
                    if t - last_t >= time_gap - 1e-6:
                        sampled_timestamps.append(t)
                        last_t = t

            for t in sampled_timestamps:
                v_frame = int(round(t * fps))

                try:
                    x1, y1, x2, y2 = BBoxReconstructor.reconstruct(track, t)
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Could not reconstruct bbox for track {track_id} at t={t}: {e}")
                    continue

                crop_name = f"frame_{v_frame:06d}_t{t:.2f}.jpg"
                frame_crops_map[v_frame].append({
                    "track_output_dir": track_output_dir,
                    "crop_filename": crop_name,
                    "bbox": (x1, y1, x2, y2),
                    "timestamp": t,
                    "track_id": track_id,
                })
                total_feed_crops += 1

        target_frames = set(frame_crops_map.keys())
        max_target_frame = max(target_frames) if target_frames else 0

        console.print(f"[bold cyan]Extracting crops for video feed:[/bold cyan] {feed_name} ([yellow]{video_path.name}[/yellow]) @ {fps:.2f} FPS")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Extracting {feed_name}", total=min(total_feed_frames, max_target_frame + 1) if total_feed_frames > 0 else max_target_frame + 1)

            curr_frame_idx = 0
            while cap.isOpened() and curr_frame_idx <= max_target_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if curr_frame_idx in frame_crops_map:
                    img_h, img_w = frame.shape[:2]

                    for crop_spec in frame_crops_map[curr_frame_idx]:
                        x1, y1, x2, y2 = crop_spec["bbox"]

                        # Clamp bounding box coordinates to image dimensions
                        ix1 = max(0, min(img_w - 1, int(round(x1))))
                        iy1 = max(0, min(img_h - 1, int(round(y1))))
                        ix2 = max(0, min(img_w, int(round(x2))))
                        iy2 = max(0, min(img_h, int(round(y2))))

                        if ix2 > ix1 and iy2 > iy1:
                            crop_img = frame[iy1:iy2, ix1:ix2]
                            out_dir: Path = crop_spec["track_output_dir"]
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / crop_spec["crop_filename"]

                            cv2.imwrite(str(out_path), crop_img)
                            crops_saved += 1

                progress.update(task, completed=curr_frame_idx + 1)
                curr_frame_idx += 1

        cap.release()

    console.print(
        f"\n[bold green]Successfully produced {crops_saved} track crops in directory:[/bold green] {output_dir.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract track crops from compressed track registry JSON."
    )
    parser.add_argument(
        "--registry",
        "-r",
        type=Path,
        required=True,
        help="Path to input compressed track registry JSON file",
    )
    parser.add_argument(
        "--video-dir",
        "-v",
        type=Path,
        required=True,
        help="Directory containing video feed files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory to save extracted crops",
    )
    parser.add_argument(
        "--time-gap",
        type=float,
        default=0.0,
        help="Optional minimum time gap in seconds between sampled crops per track (default: 0 = all frames)",
    )

    args = parser.parse_args()
    produce_crops(
        registry_path=args.registry,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        time_gap=args.time_gap,
    )


if __name__ == "__main__":
    main()
