#!/usr/bin/env python3
"""
crop_tracks.py — Generate high-quality crops of tracked objects from video using a differential heuristic.

Usage:
------
    python scripts/crop_tracks.py \
        --registry temp.json \
        --video-dir input_vids \
        --output-dir output_crops \
        --lambda 1.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracking.serialization.json_deserializer import JsonDeserializer
from tracking.domain.track import get_cleared_detection_frame, CompressedTrack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop the highest quality detection frame for each track from video using a differential heuristic."
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="temp.json",
        help="Path to the JSON registry file containing tracks.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="input_vids",
        help="Directory containing the source videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_crops",
        help="Directory to save the cropped track images.",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        dest="lambda_param",
        help="Sensitivity parameter for weighting the differential size penalty.",
    )
    return parser.parse_args()


def load_tracks_from_registry(registry_path: str) -> Dict[str, List[CompressedTrack]]:
    """Loads all tracks from the registry, grouping them by video key."""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    video_to_tracks = {}
    for video_key, track_list in registry.items():
        tracks = []
        for entry in track_list:
            compressed = entry.get("compressed_track", entry)
            if compressed is None:
                continue
            try:
                track = JsonDeserializer.deserialize_from_dict(compressed)
                tracks.append(track)
            except Exception as exc:
                track_id = compressed.get("track_id", "?")
                print(f"[WARN] Skipping track {track_id} in {video_key}: {exc}", file=sys.stderr)
        if tracks:
            video_to_tracks[video_key] = tracks

    return video_to_tracks


def process_video_crops(
    video_key: str,
    video_path: str,
    tracks: List[CompressedTrack],
    output_dir: str,
    lambda_param: float,
) -> None:
    """Finds optimal frames for all tracks of a video, seeks to them, and saves the crops."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        return

    # Find optimal frames/boxes for all tracks in this video
    # Group tracks by the target frame number to scan sequentially or perform minimal seeks
    frame_to_tracks = {}
    for track in tracks:
        try:
            best_frame, best_time, best_bbox, best_score = get_cleared_detection_frame(
                track, lambda_param=lambda_param
            )
            frame_to_tracks.setdefault(best_frame, []).append((track.metadata.track_id, best_bbox, best_score))
        except Exception as exc:
            print(f"[ERROR] Could not evaluate cleared frame for track {track.metadata.track_id}: {exc}", file=sys.stderr)

    if not frame_to_tracks:
        print(f"No valid frames to crop for video '{video_key}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}", file=sys.stderr)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing '{video_key}' ({width}x{height}, {total_frames} frames)...")
    os.makedirs(output_dir, exist_ok=True)

    # Sort target frames to seek/read forward efficiently
    target_frames = sorted(frame_to_tracks.keys())
    success_count = 0

    for target_frame in target_frames:
        if target_frame < 0 or target_frame >= total_frames:
            print(f"[WARN] Requested frame {target_frame} is out of video bounds [0, {total_frames})", file=sys.stderr)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Failed to read frame {target_frame} from {video_key}", file=sys.stderr)
            continue

        for track_id, bbox, score in frame_to_tracks[target_frame]:
            x1, y1, x2, y2 = bbox

            # Clip bounding box to frame dimensions
            ix1 = max(0, min(width - 1, int(round(x1))))
            iy1 = max(0, min(height - 1, int(round(y1))))
            ix2 = max(0, min(width, int(round(x2))))
            iy2 = max(0, min(height, int(round(y2))))

            if ix2 <= ix1 or iy2 <= iy1:
                print(f"[WARN] Crop bbox for track {track_id} is empty/out of frame boundaries", file=sys.stderr)
                continue

            crop = frame[iy1:iy2, ix1:ix2]
            
            # Format output file name: {video_base}_track_{track_id}.jpg
            video_base = Path(video_key).stem
            out_filename = f"{video_base}_track_{track_id}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            
            cv2.imwrite(out_path, crop)
            success_count += 1

    cap.release()
    print(f"Finished '{video_key}': successfully saved {success_count} crops.")


def main():
    args = parse_args()

    try:
        video_to_tracks = load_tracks_from_registry(args.registry)
    except Exception as exc:
        print(f"[ERROR] Failed to load registry: {exc}", file=sys.stderr)
        sys.exit(1)

    if not video_to_tracks:
        print("No tracks found in registry.")
        sys.exit(0)

    for video_key, tracks in video_to_tracks.items():
        # Resolve video path
        video_path = os.path.join(args.video_dir, video_key)
        
        # If not found directly, try matching by searching files in video-dir
        if not os.path.exists(video_path):
            # Try checking the current directory
            if os.path.exists(video_key):
                video_path = video_key
            else:
                # Try search under workspace root / input_vids
                workspace_path = os.path.join(str(_REPO_ROOT), args.video_dir, video_key)
                if os.path.exists(workspace_path):
                    video_path = workspace_path

        process_video_crops(
            video_key=video_key,
            video_path=video_path,
            tracks=tracks,
            output_dir=args.output_dir,
            lambda_param=args.lambda_param,
        )


if __name__ == "__main__":
    main()
