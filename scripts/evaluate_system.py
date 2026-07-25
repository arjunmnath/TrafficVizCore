#!/usr/bin/env python3
"""
System Evaluation Script for CCTV ReID & Object Tracking on Scene 6 (S06).
Evaluates end-to-end performance on dataset/test/S06 against reference ground truth annotations.
Reports key ReID retrieval metrics (Rank-1, Rank-5, mAP, mINP) and multi-object tracking metrics
(IDF1, HOTA, DetA, AssA, MOTA, IDSW), comparing baseline tracking against intra-camera trajectory fusion.
Saves concise evaluation reports to JSON and text summary files.
"""

from __future__ import annotations

import os
import sys
import time
import glob
import csv
import json
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
    ReIDPipelineListener,
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


class QuietUIListener(ReIDPipelineListener):
    """Quiet pipeline listener that suppresses per-frame console spam during evaluation."""

    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose

    def on_video_start(
        self, video_path: str, video_idx: int, total_videos: int, total_frames: int, fps: float
    ):
        if self.verbose:
            feed_name = os.path.basename(os.path.dirname(video_path))
            self.console.print(f"    • [{video_idx}/{total_videos}] Processing feed {feed_name}...")

    def on_frame_processed(self, *args, **kwargs):
        pass

    def on_video_end(self, video_path: str, total_frames: int):
        pass


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
    parser.add_argument(
        "--output_json",
        type=str,
        default="artifacts/eval_results.json",
        help="Output JSON file path for detailed metrics (default: artifacts/eval_results.json)",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="artifacts/eval_results.txt",
        help="Output text report file path (default: artifacts/eval_results.txt)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed frame progress logging",
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
    console: Console,
    enable_intra_camera_fusion: bool = False,
) -> Tuple[Dict[str, SimpleRegistry], List[Dict[str, Any]], float]:
    """Runs the ReID pipeline on Scene 6 video feeds cleanly and returns registry, frame predictions, and runtime."""
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
    setattr(pipeline, "recorded_predictions", [])

    feeder_stage = stages[0]
    listener = QuietUIListener(console, verbose=args.verbose)
    pipeline.initialize(listener)

    start_t = time.time()
    for idx, video in enumerate(videos):
        cam_dir = os.path.dirname(video)
        feed_name = os.path.basename(cam_dir) or os.path.basename(video)
        pipeline.registry = registries[feed_name]
        feeder_stage.set_video_path(video)
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
    track_to_master: Dict[Tuple[str, int], int] = {}
    for feed_name, registry in registries.items():
        for track_id, entry in registry.identities.items():
            ct = entry.get("compressed_track")
            master_id = track_id
            if ct and isinstance(ct, dict) and "metadata" in ct:
                master_id = ct["metadata"].get("track_id", track_id)
            track_to_master[(feed_name, track_id)] = master_id

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


def save_evaluation_results(
    output_json: str,
    output_txt: str,
    args: argparse.Namespace,
    videos: List[str],
    reid_base: Dict[str, float],
    mot_base: Dict[str, float],
    reid_fused: Dict[str, float],
    mot_fused: Dict[str, float],
    time_base: float,
    time_fused: float,
) -> None:
    """Save metrics evaluation results to JSON and plain text report files."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)

    results_data = {
        "dataset_dir": args.dataset_dir,
        "videos": videos,
        "num_frames": args.num_frames,
        "device": args.device,
        "yolo_model": args.yolo_model,
        "tracker": args.tracker,
        "reid_threshold": args.threshold,
        "metrics": {
            "baseline": {
                "reid": reid_base,
                "tracking": mot_base,
                "execution_time_seconds": round(time_base, 3),
            },
            "intra_camera_fused": {
                "reid": reid_fused,
                "tracking": mot_fused,
                "execution_time_seconds": round(time_fused, 3),
            },
            "delta_gains": {
                "rank1": round(reid_fused["rank1"] - reid_base["rank1"], 2),
                "rank5": round(reid_fused["rank5"] - reid_base["rank5"], 2),
                "mAP": round(reid_fused["mAP"] - reid_base["mAP"], 2),
                "mINP": round(reid_fused["mINP"] - reid_base["mINP"], 2),
                "IDF1": round(mot_fused["IDF1"] - mot_base["IDF1"], 2),
                "HOTA": round(mot_fused["HOTA"] - mot_base["HOTA"], 2),
                "DetA": round(mot_fused["DetA"] - mot_base["DetA"], 2),
                "AssA": round(mot_fused["AssA"] - mot_base["AssA"], 2),
                "MOTA": round(mot_fused["MOTA"] - mot_base["MOTA"], 2),
            },
        },
    }

    with open(output_json, "w") as f:
        json.dump(results_data, f, indent=4)

    # Save text report
    with open(output_txt, "w") as f:
        f.write("========================================================================\n")
        f.write("        CCTV RE-IDENTIFICATION & TRACKING EVALUATION REPORT             \n")
        f.write("========================================================================\n")
        f.write(f"Dataset Directory: {args.dataset_dir}\n")
        f.write(f"Video Feeds ({len(videos)}): {', '.join([os.path.basename(os.path.dirname(v)) for v in videos])}\n")
        f.write(f"Frames per video: {args.num_frames} | Device: {args.device} | Model: {args.yolo_model}\n")
        f.write("------------------------------------------------------------------------\n\n")

        f.write(f"{'Metric':<25} | {'Baseline':<12} | {'Fused':<12} | {'Delta Gain':<12}\n")
        f.write("-" * 68 + "\n")
        f.write(f"{'Rank-1 Accuracy (%)':<25} | {reid_base['rank1']:11.2f}% | {reid_fused['rank1']:11.2f}% | {reid_fused['rank1'] - reid_base['rank1']:+11.2f}%\n")
        f.write(f"{'Rank-5 Accuracy (%)':<25} | {reid_base['rank5']:11.2f}% | {reid_fused['rank5']:11.2f}% | {reid_fused['rank5'] - reid_base['rank5']:+11.2f}%\n")
        f.write(f"{'mAP (%)':<25} | {reid_base['mAP']:11.2f}% | {reid_fused['mAP']:11.2f}% | {reid_fused['mAP'] - reid_base['mAP']:+11.2f}%\n")
        f.write(f"{'mINP (%)':<25} | {reid_base['mINP']:11.2f}% | {reid_fused['mINP']:11.2f}% | {reid_fused['mINP'] - reid_base['mINP']:+11.2f}%\n")
        f.write("-" * 68 + "\n")
        f.write(f"{'IDF1 Score (%)':<25} | {mot_base['IDF1']:11.2f}% | {mot_fused['IDF1']:11.2f}% | {mot_fused['IDF1'] - mot_base['IDF1']:+11.2f}%\n")
        f.write(f"{'HOTA Score (%)':<25} | {mot_base['HOTA']:11.2f}% | {mot_fused['HOTA']:11.2f}% | {mot_fused['HOTA'] - mot_base['HOTA']:+11.2f}%\n")
        f.write(f"{'DetA Score (%)':<25} | {mot_base['DetA']:11.2f}% | {mot_fused['DetA']:11.2f}% | {mot_fused['DetA'] - mot_base['DetA']:+11.2f}%\n")
        f.write(f"{'AssA Score (%)':<25} | {mot_base['AssA']:11.2f}% | {mot_fused['AssA']:11.2f}% | {mot_fused['AssA'] - mot_base['AssA']:+11.2f}%\n")
        f.write(f"{'MOTA (%)':<25} | {mot_base['MOTA']:11.2f}% | {mot_fused['MOTA']:11.2f}% | {mot_fused['MOTA'] - mot_base['MOTA']:+11.2f}%\n")
        f.write("-" * 68 + "\n")
        f.write(f"{'Execution Time (s)':<25} | {time_base:11.2f}s | {time_fused:11.2f}s | {time_fused - time_base:+11.2f}s\n")
        f.write("========================================================================\n")


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
            "[bold green]CCTV System Evaluation (Scene 6)[/bold green]\n"
            f"Dataset: {args.dataset_dir} | Feeds: {len(videos)} | Frames/feed: {args.num_frames} | Device: {args.device}",
            border_style="cyan",
        )
    )

    # Load ground truth annotations
    console.print("[bold yellow]• Loading ground truth annotations...[/bold yellow]")
    gt_frames_flat, gt_by_feed = load_ground_truth_records(videos, args.num_frames)
    console.print(
        f"  Loaded [bold white]{len(gt_frames_flat)}[/bold white] ground truth frame records."
    )

    # Run Baseline Experiment
    console.print("\n[bold yellow]• Evaluating Baseline Tracking (No Intra-Camera Fusion)...[/bold yellow]")
    reg_base, preds_base, time_base = run_pipeline_experiment(videos, args, console, enable_intra_camera_fusion=False)
    reid_base, mot_base = evaluate_system_performance(reg_base, preds_base, gt_frames_flat)
    console.print(f"  Completed Baseline in [bold white]{time_base:.2f}s[/bold white].")

    # Run Fused Experiment
    console.print("\n[bold yellow]• Evaluating Fused Tracking (With Intra-Camera Fusion Enabled)...[/bold yellow]")
    reg_fused, preds_fused, time_fused = run_pipeline_experiment(videos, args, console, enable_intra_camera_fusion=True)
    reid_fused, mot_fused = evaluate_system_performance(reg_fused, preds_fused, gt_frames_flat)
    console.print(f"  Completed Fused in [bold white]{time_fused:.2f}s[/bold white].")

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

    # Save outputs
    save_evaluation_results(
        args.output_json,
        args.output_txt,
        args,
        videos,
        reid_base,
        mot_base,
        reid_fused,
        mot_fused,
        time_base,
        time_fused,
    )

    console.print(f"\n[bold green]Saved evaluation reports to:[/bold green]")
    console.print(f"  • JSON format: [bold white]{args.output_json}[/bold white]")
    console.print(f"  • Text format: [bold white]{args.output_txt}[/bold white]\n")


if __name__ == "__main__":
    main()
