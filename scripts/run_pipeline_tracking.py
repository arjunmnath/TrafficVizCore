#!/usr/bin/env python3
"""
Object tracking coordinator running the custom ReID pipeline stages (VideoFeederStage, YoloDetectionStage, TrackingStage).
Outputs an annotated video with bounding boxes and track labels.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add workspace root to python path to import app modules if needed
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from reid.pipeline import ReIDPipeline
from reid.utils import FrameData
from shared.utils import setup_logger

from reid.stages import (
    VideoFeederStage,
    YoloDetectionStage,
    TrackingStage,
    FeatureStage,
    OfflineAddToRegistryStage,
)

logger = setup_logger("RunPipelineTracking")


def get_color(track_id: int) -> tuple[int, int, int]:
    """Generate a deterministic and visually distinct BGR color for a track ID."""
    np.random.seed(track_id)
    color = tuple(map(int, np.random.randint(0, 256, size=3)))
    # Ensure color is not too dark to be visible on dark backgrounds
    if sum(color) < 150:
        # Boost color brightness
        color = tuple(min(255, c + 100) for c in color)
    return color  # type: ignore


def draw_box(
    frame: np.ndarray, bbox: np.ndarray, label: str, color: tuple[int, int, int], thickness: int = 2
) -> None:
    """Draw a beautifully styled bounding box and label on the frame."""
    x1, y1, x2, y2 = map(int, bbox)

    # Draw main bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Font setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1

    # Get size of text
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Position text slightly above the box, or inside if too close to top edge
    text_y = y1 - 5 if y1 - 5 - text_h > 0 else y1 + text_h + 5
    text_x = x1

    # Bound check position coordinates
    text_y = max(0, min(frame.shape[0] - 5, text_y))
    text_x = max(0, min(frame.shape[1] - text_w - 5, text_x))

    # Draw background box for text to make it extremely readable
    cv2.rectangle(
        frame, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), color, -1
    )

    # Determine text color based on background brightness
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    # Draw text label
    cv2.putText(
        frame, label, (text_x, text_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReID pipeline tracking on a video and output annotated results."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the annotated output video will be saved.",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8s.pt",
        help="Name of YOLO model (default: yolov8s.pt).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration filename (e.g. bytetrack.yaml, botsort.yaml) or custom config YAML path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Object detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 2, 3, 5, 7],
        help="COCO class IDs to track. Default: [0, 2, 3, 5, 7] (person, car, motorcycle, bus, truck). Use empty list/None to track all.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum frames to process (-1 for full video). Useful for quick testing.",
    )
    parser.add_argument(
        "--fuse_appearance",
        action="store_true",
        help="Use appearance aware tracking by running ReID feature extraction.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 half-precision inference for ensemble.",
    )
    parser.add_argument(
        "--reid_model_type",
        "--reid-model-type",
        type=str,
        default="resnetibnreid",
        choices=["resnetibnreid", "vitclipreid"],
        help="ReID feature extractor model variant ('resnetibnreid' or 'vitclipreid')",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 half-precision inference for ensemble.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run ReID on (cpu, cuda, mps, auto).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check video path
    if not os.path.exists(args.video_path):
        logger.error(f"Input video file not found: {args.video_path}")
        sys.exit(1)

    # Check model path
    logger.info("Initializing stages...")
    feeder_stage = VideoFeederStage(args.video_path)
    detector_stage = YoloDetectionStage(args.yolo_model)

    feature_stage = None
    if args.fuse_appearance:
        feature_stage = FeatureStage(
            model_type=args.reid_model_type,
            device=args.device,
            fp16=args.fp16,
        )

    tracker_stage = TrackingStage(tracker_config=args.tracker)
    offline_registry_stage = OfflineAddToRegistryStage()

    # Initialize all stages
    feeder_stage.initialize()
    detector_stage.initialize()
    if feature_stage is not None:
        feature_stage.initialize()
    tracker_stage.initialize()
    offline_registry_stage.initialize()

    # Wrap stages in ReIDPipeline container to pass to process() calls and allow looking up sibling stages
    stages = [feeder_stage, detector_stage]
    if feature_stage is not None:
        stages.append(feature_stage)
    stages.append(tracker_stage)
    stages.append(offline_registry_stage)
    pipeline = ReIDPipeline(stages=stages)

    # Get properties from the video feeder stage
    fps = feeder_stage.fps
    total_frames = feeder_stage.total_frames

    cap_temp = cv2.VideoCapture(args.video_path)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()

    # Set up classes list (if empty, set classes=None to track all classes)
    classes = args.classes if args.classes else None

    # Determine frames to process
    frames_to_process = total_frames
    if args.max_frames > 0:
        frames_to_process = min(args.max_frames, total_frames)

    logger.info(f"Input video resolution: {width}x{height} @ {fps:.2f} FPS")
    logger.info(f"Total video frames: {total_frames}, processing: {frames_to_process}")
    logger.info(f"Tracker configuration: {args.tracker}")
    logger.info(f"Tracking classes: {classes}")
    if feature_stage is not None:
        logger.info("Appearance awareness ENABLED (Ensemble: True)")

    # Initialize video writer
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output codec based on extension
    ext = Path(args.output_path).suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    try:
        # Initialize tqdm progress bar
        with tqdm(total=frames_to_process, desc="Pipeline tracking progress") as pbar:
            frame_idx = 0
            while frame_idx < frames_to_process:
                # 1. Instantiate FrameData
                data = FrameData(feed_idx=1, total_videos=1)

                # 2. Ingest next frame
                data = feeder_stage.process(data, pipeline)
                if data.end_of_stream:
                    break
                if data.skip:
                    continue

                # 3. Object Detection
                data = detector_stage.process(data, pipeline)

                # Filter classes/confidence if required
                if classes is not None and data.boxes is not None and len(data.boxes) > 0:
                    mask = np.isin(data.classes, classes) & (data.scores >= args.conf)
                    data.boxes = data.boxes[mask]
                    data.scores = data.scores[mask]
                    data.classes = data.classes[mask]

                # 3.5. Feature Extraction (if enabled)
                if feature_stage is not None:
                    data = feature_stage.process(data, pipeline)

                # 4. Multi-Object Tracking
                data = tracker_stage.process(data, pipeline)

                # 4.5. Offline Add to Registry
                data = offline_registry_stage.process(data, pipeline)

                # 5. Annotation
                frame = data.frame
                if data.tracks is not None and len(data.tracks) > 0:
                    for t in data.tracks:
                        # track layout is: [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
                        bbox = t[0:4]
                        track_id = int(t[4])
                        cls_id = int(t[6])
                        class_name = getattr(pipeline, "coco_classes", {}).get(cls_id, "unknown")
                        label = f"{class_name} - {track_id}"
                        color = get_color(track_id)
                        draw_box(frame, bbox, label, color)

                # Write the annotated frame to output video
                out.write(frame)

                frame_idx += 1
                pbar.update(1)

        logger.info(f"Tracking finished. Output video saved to: {args.output_path}")

    finally:
        feeder_stage.stop()
        if feature_stage is not None:
            feature_stage.finalize(None)
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
