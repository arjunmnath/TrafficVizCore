#!/usr/bin/env python3
"""
System Evaluation Script for CCTV ReID & Object Tracking on Scene 6 (S06).
Evaluates end-to-end performance on dataset/test/S06 against reference ground truth annotations.
Reports key ReID retrieval metrics (Rank-1, Rank-5, mAP, mINP) and multi-object tracking metrics
(IDF1, HOTA, DetA, AssA, MOTA, IDSW), comparing baseline tracking against intra-camera trajectory fusion.
"""

from __future__ import annotations

import os
import sys
import time
import glob
import csv
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add workspace root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from reid import (
    ReIDPipeline,
    SimpleRegistry,
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
from reid.eval_metrics import (
    compute_reid_retrieval_metrics,
    compute_mot_tracking_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="System Evaluation Tool for ReID & Tracking on Scene 6 (S06)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/test/S06",
        help="Path to S06 dataset directory (default: dataset/test/S06)",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional list of specific video file paths to process",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=200,
        help="Number of frames to process per video (-1 for full video)",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="trained_model/yolov8s.pt",
        help="Path to YOLOv8 model file",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration filename",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ReID matching threshold",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=0.0,
        help="Sampling FPS rate (0.0 for full video FPS)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto' (detects GPU), 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon GPU), or 'cpu'",
    )
    return parser.parse_args()


def discover_s06_videos(dataset_dir: str, custom_videos: Optional[List[str]]) -> List[str]:
    """Discover video files in dataset/test/S06 or fallback to input_vids/S06."""
    if custom_videos:
        return [os.path.abspath(v) for v in custom_videos]

    search_dirs = [
        dataset_dir,
        os.path.join(workspace_root, dataset_dir),
        os.path.join(workspace_root, "input_vids/S06"),
    ]

    discovered = []
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            matches = glob.glob(os.path.join(s_dir, "c*", "vdo.avi"))
            if matches:
                discovered = sorted(matches)
                break

    return discovered


def load_ground_truth_records(videos: List[str], max_frames: int) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, List[Dict[str, Any]]]]]:
    """Load reference ground truth frame records and tracks from MTSC files or dataset/eval."""
    gt_frames_flat: List[Dict[str, Any]] = []
    gt_by_feed: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    for video in videos:
        cam_dir = os.path.dirname(video)
        feed_name = os.path.basename(cam_dir)  # e.g. 'c041'
        mtsc_path = os.path.join(cam_dir, "mtsc", "mtsc_tnt_mask_rcnn.txt")

        if not os.path.exists(mtsc_path):
            # Fallback search in input_vids/S06
            fallback_path = os.path.join(workspace_root, "input_vids", "S06", feed_name, "mtsc", "mtsc_tnt_mask_rcnn.txt")
            if os.path.exists(fallback_path):
                mtsc_path = fallback_path

        if os.path.exists(mtsc_path):
            feed_dict: Dict[int, List[Dict[str, Any]]] = {}
            with open(mtsc_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 6:
                        continue
                    frame_num = int(row[0])
                    if max_frames > 0 and frame_num > max_frames:
                        continue
                    track_id = int(row[1])
                    x = float(row[2])
                    y = float(row[3])
                    w = float(row[4])
                    h = float(row[5])
                    bbox = [x, y, x + w, y + h]

                    if frame_num not in feed_dict:
                        feed_dict[frame_num] = []
                    feed_dict[frame_num].append({"track_id": track_id, "bbox": bbox})

            gt_by_feed[feed_name] = feed_dict

            for f_num in sorted(feed_dict.keys()):
                items = feed_dict[f_num]
                gt_frames_flat.append({
                    "feed": feed_name,
                    "frame": f_num,
                    "boxes": [it["bbox"] for it in items],
                    "ids": [it["track_id"] for it in items],
                })

    return gt_frames_flat, gt_by_feed


def run_pipeline_experiment(
    videos: List[str],
    args: argparse.Namespace,
    enable_intra_camera_fusion: bool = False,
) -> Tuple[Dict[str, SimpleRegistry], List[Dict[str, Any]], float]:
    """Runs the ReID pipeline on Scene 6 video feeds, returning registry, frame predictions, and runtime."""
    postprocessing_stages = [
        TrajectoryFusionStage(mode="attention"),
        TrajectoryCompressionStage(),
    ]
    if enable_intra_camera_fusion:
        postprocessing_stages.append(IntraCameraTrajectoryFusionStage(fusion_mode="attention"))

    postprocessing_pipeline = PostProcessingPipeline(postprocessing_stages)

    feature_stage = FeatureStage(device=args.device, fp16=True)
    stages = [
        VideoFeederStage(),
        SamplerStage(sample_fps=args.sample_fps, time_based=False),
        YoloDetectionStage(yolo_path=args.yolo_model, device=args.device),
        feature_stage,
        TrackingStage(
            tracker_config=args.tracker,
            postprocessing_pipeline=postprocessing_pipeline,
        ),
        OfflineAddToRegistryStage(),
    ]

    registries: Dict[str, SimpleRegistry] = {}
    for video in videos:
        cam_dir = os.path.dirname(video)
        feed_name = os.path.basename(cam_dir) or os.path.basename(video)
        registries[feed_name] = SimpleRegistry()

    pipeline = ReIDPipeline(
        stages=stages,
        threshold=args.threshold,
        max_frames=args.num_frames,
        registry=None,
    )
    # Attach prediction accumulator attribute
    setattr(pipeline, "recorded_predictions", [])

    feeder_stage = stages[0]
    listener = HeadlessUIListener(videos)
    pipeline.initialize(listener)

    start_t = time.time()
    for idx, video in enumerate(videos):
        cam_dir = os.path.dirname(video)
        feed_name = os.path.basename(cam_dir) or os.path.basename(video)
        pipeline.registry = registries[feed_name]
        feeder_stage.set_video_path(video)
        listener.current_video_idx = idx + 1
        pipeline.run(listener)

    elapsed = time.time() - start_t
    recorded_preds = getattr(pipeline, "recorded_predictions", [])
    return registries, recorded_preds, elapsed


def evaluate_system_performance(
    registries: Dict[str, SimpleRegistry],
    recorded_preds: List[Dict[str, Any]],
    gt_frames_flat: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute ReID metrics (Rank-1, Rank-5, mAP, mINP) and MOT metrics (IDF1, HOTA, DetA, AssA, MOTA)."""
    # 1. ReID Metrics
    query_embs = []
    query_ids = []
    gallery_embs = []
    gallery_ids = []

    for feed_name, registry in registries.items():
        for track_id, entry in registry.identities.items():
            app_list = entry.get("appearance_embeddings", [])
            if not app_list:
                continue

            arr = np.array(app_list, dtype=np.float32)
            fused = arr.mean(axis=0)

            if len(arr) >= 2:
                q_emb = arr[0]
                g_emb = fused
                query_embs.append(q_emb)
                query_ids.append(track_id)
                gallery_embs.append(g_emb)
                gallery_ids.append(track_id)
            elif len(arr) == 1:
                query_embs.append(arr[0])
                query_ids.append(track_id)
                gallery_embs.append(arr[0])
                gallery_ids.append(track_id)

    reid_metrics = compute_reid_retrieval_metrics(
        np.array(query_embs, dtype=np.float32) if query_embs else np.empty((0, 128)),
        np.array(query_ids) if query_ids else np.empty((0,)),
        np.array(gallery_embs, dtype=np.float32) if gallery_embs else np.empty((0, 128)),
        np.array(gallery_ids) if gallery_ids else np.empty((0,)),
    )

    # 2. MOT Tracking Metrics
    # Map predictions to global master_track_ids
    track_to_master: Dict[Tuple[str, int], int] = {}
    for feed_name, registry in registries.items():
        for track_id, entry in registry.identities.items():
            ct = entry.get("compressed_track")
            master_id = track_id
            if ct and isinstance(ct, dict) and "metadata" in ct:
                master_id = ct["metadata"].get("track_id", track_id)
            track_to_master[(feed_name, track_id)] = master_id

    # Group predictions by (feed, frame)
    preds_map: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for p in recorded_preds:
        key = (p["feed"], p["frame"])
        if key not in preds_map:
            preds_map[key] = []
        
        master_id = track_to_master.get((p["feed"], p["track_id"]), p["track_id"])
        preds_map[key].append({"track_id": master_id, "bbox": p["bbox"]})

    pred_frames_flat = []
    for gt in gt_frames_flat:
        key = (gt["feed"], gt["frame"])
        preds = preds_map.get(key, [])
        boxes = [p["bbox"] for p in preds]
        ids = [p["track_id"] for p in preds]
        pred_frames_flat.append({"boxes": boxes, "ids": ids})

    mot_metrics = compute_mot_tracking_metrics(gt_frames_flat, pred_frames_flat)

    return reid_metrics, mot_metrics


def main():
    console = Console()
    args = parse_args()

    videos = discover_s06_videos(args.dataset_dir, args.videos)
    if not videos:
        console.print(
            f"[bold red]Error: No video files found in dataset path {args.dataset_dir}[/bold red]"
        )
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold green]CCTV ReID & Tracking System Evaluation[/bold green]\n"
            f"Dataset Path: {args.dataset_dir} | Discovered Videos: {len(videos)} | Max Frames: {args.num_frames}",
            border_style="cyan",
        )
    )

    for v in videos:
        console.print(f"  • Found video feed: [bold white]{v}[/bold white]")

    # Load ground truth annotations
    console.print("\n[bold yellow]1. Loading Ground Truth Annotations (Scene 6)...[/bold yellow]")
    gt_frames_flat, gt_by_feed = load_ground_truth_records(videos, args.num_frames)
    console.print(
        f"  Loaded [bold white]{len(gt_frames_flat)}[/bold white] frame ground truth records across {len(gt_by_feed)} video feeds."
    )

    # Run Baseline Experiment (No Intra-Camera Fusion)
    console.print("\n[bold yellow]2. Running Baseline Experiment (Without Intra-Camera Fusion)...[/bold yellow]")
    reg_base, preds_base, time_base = run_pipeline_experiment(videos, args, enable_intra_camera_fusion=False)
    reid_base, mot_base = evaluate_system_performance(reg_base, preds_base, gt_frames_flat)

    # Run Fused Experiment (With Intra-Camera Fusion Enabled)
    console.print("\n[bold yellow]3. Running Fused Experiment (With Intra-Camera Fusion Enabled)...[/bold yellow]")
    reg_fused, preds_fused, time_fused = run_pipeline_experiment(videos, args, enable_intra_camera_fusion=True)
    reid_fused, mot_fused = evaluate_system_performance(reg_fused, preds_fused, gt_frames_flat)

    # Build Comparative Summary Table
    table = Table(title="System Evaluation Summary & Metrics (S06 Benchmark)")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Baseline", justify="right", style="yellow")
    table.add_column("Intra-Camera Fused", justify="right", style="green")
    table.add_column("Delta / Gain", justify="right", style="bold magenta")

    # ReID Metrics
    table.add_row("ReID", "Rank-1 Accuracy (%)", f"{reid_base['rank1']:.2f}%", f"{reid_fused['rank1']:.2f}%", f"+{reid_fused['rank1'] - reid_base['rank1']:.2f}%")
    table.add_row("ReID", "Rank-5 Accuracy (%)", f"{reid_base['rank5']:.2f}%", f"{reid_fused['rank5']:.2f}%", f"+{reid_fused['rank5'] - reid_base['rank5']:.2f}%")
    table.add_row("ReID", "mAP (%)", f"{reid_base['mAP']:.2f}%", f"{reid_fused['mAP']:.2f}%", f"+{reid_fused['mAP'] - reid_base['mAP']:.2f}%")
    table.add_row("ReID", "mINP (%)", f"{reid_base['mINP']:.2f}%", f"{reid_fused['mINP']:.2f}%", f"+{reid_fused['mINP'] - reid_base['mINP']:.2f}%")

    table.add_section()

    # MOT Metrics
    table.add_row("Tracking", "IDF1 Score (%)", f"{mot_base['IDF1']:.2f}%", f"{mot_fused['IDF1']:.2f}%", f"+{mot_fused['IDF1'] - mot_base['IDF1']:.2f}%")
    table.add_row("Tracking", "HOTA Score (%)", f"{mot_base['HOTA']:.2f}%", f"{mot_fused['HOTA']:.2f}%", f"+{mot_fused['HOTA'] - mot_base['HOTA']:.2f}%")
    table.add_row("Tracking", "DetA (Detection Acc %)", f"{mot_base['DetA']:.2f}%", f"{mot_fused['DetA']:.2f}%", f"+{mot_fused['DetA'] - mot_base['DetA']:.2f}%")
    table.add_row("Tracking", "AssA (Association Acc %)", f"{mot_base['AssA']:.2f}%", f"{mot_fused['AssA']:.2f}%", f"+{mot_fused['AssA'] - mot_base['AssA']:.2f}%")
    table.add_row("Tracking", "MOTA (%)", f"{mot_base['MOTA']:.2f}%", f"{mot_fused['MOTA']:.2f}%", f"+{mot_fused['MOTA'] - mot_base['MOTA']:.2f}%")

    table.add_section()

    # Performance
    table.add_row("Performance", "Execution Time (s)", f"{time_base:.2f}s", f"{time_fused:.2f}s", f"{time_fused - time_base:+.2f}s")

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]System evaluation on dataset/test/S06 finished successfully![/bold green]")


if __name__ == "__main__":
    main()
