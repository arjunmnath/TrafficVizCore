#!/usr/bin/env python3
"""
render_tracks.py — Render compressed tracks from a registry JSON onto a video.

For each frame the script:
  • Looks up the current video timestamp.
  • Finds every track whose start_time ≤ t < end_time.
  • Draws a fading trail of the center-path up to t.
  • Draws the reconstructed bounding box and label at t.

Usage
-----
    python scripts/render_tracks.py \
        --registry temp.json \
        --video    input_vids/clip1.mp4 \
        --output   output_clip1.mp4 \
        [--trail-duration 3.0]   # seconds of trail to show (default 3 s)
        [--fps-override 30]      # override detected FPS (optional)
        [--use-bbox]             # draw bounding box along with the tracks
        [--no-trail]             # skip trail (show bbox only, requires --use-bbox)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Make sure the repo root is on the path when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracking.serialization.json_deserializer import JsonDeserializer
from tracking.compression.reconstruction import BBoxReconstructor
from tracking.domain.track import CompressedTrack


# ---------------------------------------------------------------------------
# Colour palette — one distinct BGR colour per track id
# ---------------------------------------------------------------------------
_PALETTE = [
    (0,   210, 255),  # amber-yellow
    (255, 100,   0),  # blue
    ( 50, 220,  50),  # green
    (  0,  60, 255),  # red
    (200,   0, 200),  # magenta
    (  0, 180, 180),  # olive-yellow
    (255, 180,   0),  # cyan-ish
    (128,   0, 255),  # purple
    (  0, 128, 255),  # orange
    (  0, 255, 128),  # spring green
]


def track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Core drawing helpers
# ---------------------------------------------------------------------------

def draw_trail(
    frame: np.ndarray,
    track: CompressedTrack,
    current_t: float,
    trail_duration: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a fading trail from (current_t - trail_duration) to current_t."""
    t0 = max(track.metadata.start_timestamp, current_t - trail_duration)
    t1 = current_t

    if t1 <= t0:
        return

    n_pts = max(2, int((t1 - t0) * 40))          # ~40 pts/s resolution
    times = np.linspace(t0, t1, n_pts)
    pts = [track.position(t) for t in times]

    for i in range(len(pts) - 1):
        # Older segments are more transparent -> darker
        alpha = (i + 1) / len(pts)               # 0 ... 1 (newest = 1)
        seg_color = tuple(int(c * alpha) for c in color)

        p1 = (int(round(pts[i][0])),     int(round(pts[i][1])))
        p2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        cv2.line(frame, p1, p2, seg_color, thickness, cv2.LINE_AA)

    # Draw a dot at the head
    head = (int(round(pts[-1][0])), int(round(pts[-1][1])))
    cv2.circle(frame, head, thickness + 2, color, -1, cv2.LINE_AA)


def draw_bbox(
    frame: np.ndarray,
    track: CompressedTrack,
    t: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw the reconstructed bounding box and label at timestamp t."""
    try:
        x1, y1, x2, y2 = BBoxReconstructor.reconstruct(track, t)
    except Exception:
        return

    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(frame, p1, p2, color, thickness, cv2.LINE_AA)

    label = f"{track.metadata.class_label} #{track.metadata.track_id}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y = max(th + baseline + 2, p1[1] - 5)
    bg_p1 = (p1[0], label_y - th - baseline)
    bg_p2 = (p1[0] + tw + 4, label_y + baseline)
    cv2.rectangle(frame, bg_p1, bg_p2, color, -1)
    cv2.putText(
        frame, label,
        (p1[0] + 2, label_y),
        font, font_scale,
        (255, 255, 255),
        font_thick, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------

def load_tracks(registry_path: str, camera_key: str) -> List[CompressedTrack]:
    """Load and deserialize all CompressedTrack objects for a given camera."""
    with open(registry_path, "r") as f:
        registry: Dict = json.load(f)

    if camera_key not in registry:
        raise KeyError(
            f"Camera key '{camera_key}' not found in registry. "
            f"Available keys: {list(registry.keys())}"
        )

    tracks = []
    for entry in registry[camera_key]:
        compressed = entry.get("compressed_track", entry)   # handle both shapes
        if compressed is None:
            print(f"  [WARN] Skipping entry with null compressed_track (track_id={entry.get('track_id', '?')})", file=sys.stderr)
            continue
        try:
            track = JsonDeserializer.deserialize_from_dict(compressed)
            tracks.append(track)
        except Exception as exc:
            tid = compressed.get("track_id", "?")
            print(f"  [WARN] Skipping track {tid}: {exc}", file=sys.stderr)

    print(f"Loaded {len(tracks)} tracks for '{camera_key}'.")
    return tracks


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------

def render(
    registry_path: str,
    video_path: str,
    output_path: str,
    trail_duration: float = 3.0,
    fps_override: Optional[float] = None,
    draw_bboxes: bool = True,
    draw_trails: bool = True,
) -> None:
    camera_key = Path(video_path).name          # e.g. "clip1.mp4"

    # Load tracks
    tracks = load_tracks(registry_path, camera_key)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps      = fps_override or cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps  ({n_frames} frames)")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current timestamp for this frame
        current_t = frame_idx / fps

        # Render each active track
        for track in tracks:
            t_start = track.metadata.start_timestamp
            t_end   = track.metadata.end_timestamp

            if current_t < t_start or current_t >= t_end:
                continue                                # not active yet / already done

            color = track_color(track.metadata.track_id)

            if draw_trails:
                draw_trail(frame, track, current_t, trail_duration, color)
            if draw_bboxes:
                draw_bbox(frame, track, current_t, color)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / max(n_frames, 1) * 100
            print(f"  {frame_idx}/{n_frames} frames ({pct:.1f}%)", end="\r", flush=True)

    print(f"\nDone. Output written to: {output_path}")
    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render compressed tracks onto a video."
    )
    parser.add_argument(
        "--registry", "-r", required=True,
        help="Path to the registry JSON file (e.g. temp.json).",
    )
    parser.add_argument(
        "--video", "-v", required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=(
            "Path for the output video. "
            "Defaults to '<input_stem>_tracks.mp4' next to the input."
        ),
    )
    parser.add_argument(
        "--trail-duration", "-t", type=float, default=3.0,
        help="Duration (seconds) of trailing path shown behind the current position. Default: 3.",
    )
    parser.add_argument(
        "--fps-override", type=float, default=None,
        help="Override the FPS detected from the video metadata.",
    )
    parser.add_argument(
        "--use-bbox", action="store_true",
        help="Draw bounding boxes along with the tracks (default: show tracks/trails only).",
    )
    parser.add_argument(
        "--no-trail", action="store_true",
        help="Disable trail rendering (show bounding box only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = args.video
    if args.output:
        output_path = args.output
    else:
        p = Path(video_path)
        output_path = str(p.with_stem(p.stem + "_tracks"))

    render(
        registry_path   = args.registry,
        video_path      = video_path,
        output_path     = output_path,
        trail_duration  = args.trail_duration,
        fps_override    = args.fps_override,
        draw_bboxes     = args.use_bbox,
        draw_trails     = not args.no_trail,
    )


if __name__ == "__main__":
    main()
