#!/usr/bin/env python3
"""
Pipeline runner to perform cross-video person and vehicle re-identification tracking using YOLOv8 detectors and ResNet features.
Maintains a simple global registry tracking occurrences of unique identities.
Supports both headless mode (for servers) and UI mode (for live monitoring).
"""

import os
import sys
import json
import argparse
import numpy as np

# Add workspace root to python path to import app modules
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from reid import (
    ReIDPipeline,
    SimpleRegistry,
    RichUIListener,
    HeadlessUIListener,
    resolve_path,
)
from reid.postprocessing import (
    PostProcessingPipeline,
    TrajectoryFusionStage,
    TrajectoryCompressionStage,
    IntraCameraTrajectoryFusionStage,
)

from reid.stages import (
    SamplerStage,
    VideoFeederStage,
    YoloDetectionStage,
    FeatureStage,
    TrackingStage,
    OfflineAddToRegistryStage,
)


def export_results(registries: dict, output_path: str) -> None:
    """Export the registry results to JSON and embeddings to NPZ, outside pipeline scope.

    Args:
        registries (dict): Mapping of feed name to SimpleRegistry.
        output_path (str): Output path for JSON summary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    summary = {}
    for feed_name, reg in registries.items():
        summary[feed_name] = reg.get_results_summary()

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    embeddings = {}
    for feed_name, reg in registries.items():
        for global_id, data in reg.get_embeddings_dict().items():
            embeddings[f"{feed_name}_{global_id}"] = data

    if embeddings:
        npz_path = os.path.splitext(output_path)[0] + ".npz"
        np.savez(npz_path, **embeddings)


VALID_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
}


def collect_video_paths(raw_inputs: list) -> list:
    """Collect all valid video file paths from specified files or directories."""
    video_paths = []
    seen = set()

    for item in raw_inputs:
        abs_item = os.path.abspath(item)
        if not os.path.exists(abs_item):
            print(f"Warning: Input path does not exist: {item}", file=sys.stderr)
            continue

        if os.path.isfile(abs_item):
            if abs_item not in seen:
                seen.add(abs_item)
                video_paths.append(abs_item)
        elif os.path.isdir(abs_item):
            dir_videos = []
            for root, _, files in os.walk(abs_item):
                for f in files:
                    if f.startswith("."):
                        continue
                    ext = os.path.splitext(f)[1].lower()
                    if ext in VALID_VIDEO_EXTENSIONS:
                        full_path = os.path.abspath(os.path.join(root, f))
                        dir_videos.append(full_path)
            dir_videos.sort()
            for v in dir_videos:
                if v not in seen:
                    seen.add(v)
                    video_paths.append(v)

    return video_paths


def generate_feed_names(video_paths: list) -> list:
    """Generate unique, descriptive feed names for a list of video paths."""
    basenames = [os.path.basename(p) for p in video_paths]
    if len(set(basenames)) == len(video_paths):
        return basenames

    feed_names = []
    seen = {}
    for p in video_paths:
        parent = os.path.basename(os.path.dirname(p))
        base = os.path.basename(p)
        candidate = f"{parent}_{base}" if parent else base
        if candidate not in seen:
            seen[candidate] = 1
            feed_names.append(candidate)
        else:
            seen[candidate] += 1
            feed_names.append(f"{candidate}_{seen[candidate]}")
    return feed_names


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-video ReID tracking pipeline on video files or directories."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="Path(s) to video file(s) or directory/directories containing video files.",
    )
    parser.add_argument(
        "--videos",
        "--video",
        "--inputs",
        nargs="+",
        dest="video_flags",
        default=[],
        help="Path(s) to video files or directories containing videos.",
    )
    parser.add_argument(
        "--dir",
        "--directory",
        type=str,
        default=None,
        help="Directory path containing video files or subdirectories.",
    )
    parser.add_argument(
        "--video1", type=str, default=None, help="Path to first video file (legacy option)"
    )
    parser.add_argument(
        "--video2", type=str, default=None, help="Path to second video file (legacy option)"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="trained_model/yolo26x.pt",
        help="Path to YOLOv8 model file",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="ReID matching threshold")
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for JSON summary of occurrences"
    )
    parser.add_argument(
        "--max_frames", type=int, default=0, help="Maximum frames to process per video (0 for all)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run ReID on (cpu, cuda, mps, auto)"
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=5,
        help="Sample FPS rate to reduce computational load (0.0 for full video FPS)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no interactive terminal UI)"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 half-precision inference for ensemble",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 half-precision inference for ensemble",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration filename (e.g. bytetrack.yaml, botsort.yaml) or custom config YAML path",
    )

    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="mean",
        choices=["mean", "attention"],
        dest="fusion_mode",
        help="Trajectory fusion mode for the postprocessing pipeline: "
        "'mean' = simple mean pooling, "
        "'attention' = scaled dot-product self-attention, "
        "'none' = disable postprocessing",
    )

    parser.add_argument(
        "--enable-intra-camera-fusion",
        action="store_true",
        default=True,
        help="Enable intra-camera trajectory fusion stage to merge fragmented tracks",
    )
    parser.add_argument(
        "--intra-camera-threshold",
        type=float,
        default=0.75,
        dest="intra_camera_threshold",
        help="Appearance similarity threshold for intra-camera trajectory fusion (default: 0.75)",
    )

    args = parser.parse_args()

    raw_inputs = []
    if args.inputs:
        raw_inputs.extend(args.inputs)
    if args.video_flags:
        raw_inputs.extend(args.video_flags)
    if args.dir:
        raw_inputs.append(args.dir)
    if args.video1:
        raw_inputs.append(args.video1)
    if args.video2:
        raw_inputs.append(args.video2)

    if not raw_inputs:
        parser.error(
            "No video inputs specified. Pass video file(s) or directory as positional arguments, or via --videos / --dir / --video1."
        )

    videos = collect_video_paths(raw_inputs)
    if not videos:
        parser.error("No valid video files found in specified input(s).")

    feed_names = generate_feed_names(videos)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))

    output_path = os.path.abspath(args.output)

    # Select appropriate listener based on mode
    if args.headless:
        listener = HeadlessUIListener(videos, feed_names=feed_names)
    else:
        listener = RichUIListener(videos, feed_names=feed_names)

    # Show configuration
    model_dir = resolve_path("trained_model", workspace_root)
    config_data = {
        "Video Sources": videos,
        "Feed Names": feed_names,
        "YOLO Model": args.yolo_model,
        "ReID Threshold": f"{args.threshold:.2f}",
        "Device": str(args.device),
        "Max Frames": str(args.max_frames) if args.max_frames > 0 else "All",
        "Sample FPS": str(args.sample_fps) if args.sample_fps > 0 else "Full FPS",
        "Output Path": output_path,
        "Pipeline Mode": "Ensemble (Centroid Fusion)",
        "Ensemble Model Dir": model_dir,
        "YOLO Tracker": args.tracker,
        "FP16 Enabled": str(args.fp16),
        "Intra-Camera Fusion": f"{args.enable_intra_camera_fusion} (threshold={args.intra_camera_threshold})",
    }

    listener.show_configuration(config_data)

    feature_stage = FeatureStage(
        device=args.device,
        fp16=args.fp16,
    )

    # Build postprocessing pipeline
    postprocessing_stages = [
        TrajectoryFusionStage(mode=args.fusion_mode),
        TrajectoryCompressionStage(),
    ]
    if args.enable_intra_camera_fusion:
        postprocessing_stages.append(
            IntraCameraTrajectoryFusionStage(
                appearance_threshold=args.intra_camera_threshold,
                fusion_mode=args.fusion_mode,
            )
        )

    postprocessing_pipeline = PostProcessingPipeline(postprocessing_stages)

    stages = [
        VideoFeederStage(),
        SamplerStage(sample_fps=args.sample_fps, time_based=False),
        YoloDetectionStage(yolo_path=args.yolo_model),
        feature_stage,
        TrackingStage(
            tracker_config=args.tracker,
            postprocessing_pipeline=postprocessing_pipeline,
        ),
        OfflineAddToRegistryStage(),
    ]

    # Create feed-specific registries
    registries = {}
    for feed_name in feed_names:
        registries[feed_name] = SimpleRegistry()

    pipeline = ReIDPipeline(
        stages=stages,
        threshold=args.threshold,
        max_frames=args.max_frames,
        registry=None,  # Assigned dynamically during run loop
    )

    feeder_stage = stages[0]  # VideoFeederStage

    pipeline.initialize(listener)

    for idx, (video, feed_name) in enumerate(zip(videos, feed_names)):
        pipeline.registry = registries[feed_name]
        feeder_stage.set_video_path(video)
        if listener:
            listener.current_video_idx = idx + 1
        pipeline.run(listener)

    # Export results outside the pipeline scope
    export_results(registries, output_path)
    if listener:
        listener.on_pipeline_end(registries, output_path)


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    try:
        main()
    except Exception:
        # Safely shut down any active Rich Live display to prevent terminal garbling
        import rich.live
        import gc

        for obj in gc.get_objects():
            if isinstance(obj, rich.live.Live):
                try:
                    obj.stop()
                except Exception:
                    pass
        console.print_exception(show_locals=True)
        raise
