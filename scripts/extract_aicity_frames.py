#!/usr/bin/env python3
"""
Extract Video Frames Script for AICity23 Track 2 NL Retrieval Dataset.

Extracts frame images (img1/%06d.jpg) from all sequence vdo.avi videos under
train/ and validation/ directories so that exact file paths referenced in
train-tracks.json and test-tracks.json exist on disk.

Usage Example:
    python scripts/extract_aicity_frames.py --data_root AICity23_Track2_NL_Retrieval/data
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames (img1/%06d.jpg) from vdo.avi videos across AICity23 sequences."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="AICity23_Track2_NL_Retrieval/data",
        help="Path to AICity23 dataset data directory containing train/ and validation/.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for video frame extraction (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame images if img1/ directory already exists.",
    )
    return parser.parse_args()


def extract_frames_for_video(vdo_path_str: str, overwrite: bool = False) -> Tuple[str, int, bool]:
    """Extract frames from a single vdo.avi file into an accompanying img1 directory."""
    vdo_path = Path(vdo_path_str)
    img1_dir = vdo_path.parent / "img1"
    os.makedirs(img1_dir, exist_ok=True)

    # Check if frames are already extracted
    existing_frames = list(img1_dir.glob("*.jpg"))
    if existing_frames and not overwrite:
        return (str(vdo_path.parent), len(existing_frames), False)

    cap = cv2.VideoCapture(str(vdo_path))
    if not cap.isOpened():
        return (str(vdo_path.parent), 0, False)

    count = 1
    extracted = 0
    success, frame = cap.read()

    while success:
        frame_filename = f"{count:06d}.jpg"
        frame_path = img1_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        extracted += 1
        count += 1
        success, frame = cap.read()

    cap.release()
    return (str(vdo_path.parent), extracted, True)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    console = Console()
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        console.print(f"[bold red]Error: Data root directory not found: {data_root}[/bold red]")
        sys.exit(1)

    # Discover all vdo.avi files under train/ and validation/
    vdo_files = sorted(list(data_root.glob("**/vdo.avi")))

    console.print(
        Panel.fit(
            "[bold green]AICity23 Video Frame Extractor[/bold green]\n"
            f"Data Root: [white]{data_root}[/white]\n"
            f"Discovered Video Feeds: [cyan]{len(vdo_files)}[/cyan] vdo.avi files | Workers: [magenta]{args.num_workers}[/magenta]",
            border_style="cyan",
        )
    )

    if not vdo_files:
        console.print("[bold red]No vdo.avi files found to process.[/bold red]")
        sys.exit(0)

    total_extracted_frames = 0
    skipped_feeds = 0
    processed_feeds = 0

    pbar = tqdm(total=len(vdo_files), desc="Extracting Video Frames", leave=False)

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(extract_frames_for_video, str(vdo_p), args.overwrite)
                for vdo_p in vdo_files
            ]
            for future in as_completed(futures):
                feed_dir, frame_count, newly_extracted = future.result()
                total_extracted_frames += frame_count
                if newly_extracted:
                    processed_feeds += 1
                else:
                    skipped_feeds += 1
                pbar.update(1)
    else:
        for vdo_p in vdo_files:
            feed_dir, frame_count, newly_extracted = extract_frames_for_video(
                str(vdo_p), args.overwrite
            )
            total_extracted_frames += frame_count
            if newly_extracted:
                processed_feeds += 1
            else:
                skipped_feeds += 1
            pbar.update(1)

    pbar.close()

    console.print(f"\n[bold green]✓ Video frame extraction complete![/bold green]")
    console.print(f"  • Processed feeds: [bold white]{processed_feeds}[/bold white]")
    console.print(f"  • Skipped feeds (already extracted): [bold white]{skipped_feeds}[/bold white]")
    console.print(f"  • Total frames present across feeds: [bold white]{total_extracted_frames}[/bold white]\n")


if __name__ == "__main__":
    main()
