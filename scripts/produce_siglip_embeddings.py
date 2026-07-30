#!/usr/bin/env python3
"""
Produce SigLIP2 Semantic Retrieval Embeddings Script (Post-Camera Processing Pipeline)

Processes crop collections for Global Vehicle Identities:
1. Gathers all crops belonging to each Global Identity across tracks & cameras.
2. Applies 2-stage quality pruning (hard filters: area, aspect ratio, border truncation, blur; soft scoring).
3. Selects representative semantic views via Farthest Point Sampling (FPS) and temporal redundancy suppression.
4. Batch encodes crops into unit-normalized L2 visual vectors using SigLIP2.
5. Aggregates multi-vector identity profiles and exports to `.npz` and `.json` registry for standalone inference.

Usage Example:
    # Process crops in crops.noinclude and output registry.embeddings.npz
    python scripts/produce_siglip_embeddings.py --crop_dir crops.noinclude --output_npz registry.embeddings.npz

    # Specify custom model and target representative views count
    python scripts/produce_siglip_embeddings.py --crop_dir reid_crops_cleaned --target_k 4 --model_name google/siglip2-base-patch16-224
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from post_camera_processing import (
    AggregationConfig,
    CropCollector,
    DiversityConfig,
    DiversitySampler,
    EmbeddingAggregator,
    EmbeddingExporter,
    QualityConfig,
    QualityFilter,
    SigLIP2BatchEncoder,
)

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce SigLIP2 semantic embeddings NPZ for Global Vehicle Identities."
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        default=str(workspace_root / "crops.noinclude")
        if (workspace_root / "crops.noinclude").exists()
        else (str(workspace_root / "reid_crops_cleaned") if (workspace_root / "reid_crops_cleaned").exists() else "crops"),
        help="Path to crops directory (subdirectories per track or flat image folder).",
    )
    parser.add_argument(
        "--global_match_json",
        type=str,
        default=str(workspace_root / "gobal_match.noinclude.json")
        if (workspace_root / "gobal_match.noinclude.json").exists()
        else None,
        help="Path to cross-camera global match association JSON.",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default=str(workspace_root / "registry.embeddings.npz"),
        help="Target output .npz embedding file path.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(workspace_root / "registry.tracks.json"),
        help="Target output .json registry metadata file path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/siglip2-base-patch16-224",
        help="SigLIP2 retrieval model name (e.g. 'google/siglip2-base-patch16-224').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computing device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--target_k",
        type=int,
        default=3,
        help="Number of representative semantic views to retain per Global Identity (default: 3).",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=1600,
        help="Minimum crop area in pixels to pass hard quality filter (default: 1600).",
    )
    parser.add_argument(
        "--min_blur",
        type=float,
        default=35.0,
        help="Minimum Laplacian variance threshold for blur detection (default: 35.0).",
    )
    parser.add_argument(
        "--export_mode",
        type=str,
        choices=["multi_view", "mean"],
        default="multi_view",
        help="Embedding export strategy ('multi_view' exports K rep vectors; 'mean' exports 1 averaged vector).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]Post-Camera Processing: SigLIP2 Semantic Embedding Generation[/bold cyan]\n"
            f"[dim]Crop Directory: {args.crop_dir}\n"
            f"Output NPZ: {args.output_npz}\n"
            f"Model: {args.model_name} | Device: {args.device} | Target Views (K): {args.target_k}[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )

    start_t = time.time()

    # Stage 1: Crop Collection
    console.print("[bold yellow]Stage 1/5: Gathering raw crops for Global Identities...[/bold yellow]")
    collector = CropCollector()
    identity_crops_map = collector.collect_from_crop_directory(
        crop_dir=args.crop_dir,
        global_match_json=args.global_match_json,
    )

    if not identity_crops_map:
        console.print(f"[bold red]No crops found in directory '{args.crop_dir}'![/bold red]")
        sys.exit(1)

    # Stage 2 & 3 Setup: Quality Filter & Diversity Sampler
    q_config = QualityConfig(
        min_area_px=args.min_area,
        min_laplacian_var=args.min_blur,
    )
    quality_filter = QualityFilter(config=q_config)

    d_config = DiversityConfig(
        target_num_views=args.target_k,
    )
    diversity_sampler = DiversitySampler(config=d_config)

    # Stage 4 Setup: SigLIP2 Batch Encoder
    console.print(f"[bold yellow]Stage 2/5: Initializing SigLIP2 encoder ({args.model_name})...[/bold yellow]")
    encoder = SigLIP2BatchEncoder(
        model_name=args.model_name,
        device=args.device,
    )

    aggregator = EmbeddingAggregator(config=AggregationConfig())
    exporter = EmbeddingExporter()

    # Process all Global Identities
    console.print("[bold yellow]Stage 3-4/5: Pruning unfit crops, sampling representative views & encoding...[/bold yellow]")

    all_profiles = []
    total_raw_crops = 0
    total_passed_crops = 0
    total_encoded_crops = 0

    identity_items = list(identity_crops_map.items())

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing identities...", total=len(identity_items))

        for gid, gid_crops in identity_items:
            raw_crops = gid_crops.crops
            total_raw_crops += len(raw_crops)

            # Step 1: Quality filtering
            filtered_crops = quality_filter.filter_and_score_crops(raw_crops)
            total_passed_crops += len(filtered_crops)

            if not filtered_crops:
                # If all crops failed hard filters, fall back to highest-scoring raw crop without hard drop
                fallback_item = raw_crops[0] if raw_crops else None
                if fallback_item:
                    eval_res = quality_filter.evaluate_crop(fallback_item)
                    eval_res.is_passed = True
                    filtered_crops = [eval_res]

            # Step 2: Diversity sampling (FPS + temporal redundancy suppression)
            rep_crops = diversity_sampler.select_representative_crops(
                filtered_crops, target_k=args.target_k
            )

            # Step 3: Encoding with SigLIP2
            crop_embeddings = encoder.encode_crops(rep_crops)
            total_encoded_crops += len(crop_embeddings)

            # Step 4: Profile assembly
            profile = aggregator.build_profile(
                global_id=gid,
                crops=rep_crops,
                embeddings=crop_embeddings,
                extra_metadata=gid_crops.metadata,
            )
            all_profiles.append(profile)

            progress.update(task, advance=1)

    # Stage 5: Exporting to NPZ & JSON
    console.print("[bold yellow]Stage 5/5: Exporting embeddings to NPZ & JSON registry...[/bold yellow]")
    npz_out, json_out = exporter.export_profiles(
        profiles=all_profiles,
        output_npz_path=args.output_npz,
        output_json_path=args.output_json,
        export_mode=args.export_mode,
    )

    elapsed = time.time() - start_t

    # Summary table
    table = Table(title="Post-Camera Pipeline Summary", box=box.SIMPLE_HEAD)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Global Identities Processed", str(len(all_profiles)))
    table.add_row("Total Raw Crops Gathered", str(total_raw_crops))
    table.add_row("Crops Passed Quality Filter", f"{total_passed_crops} ({total_passed_crops/max(1, total_raw_crops):.1%})")
    table.add_row("Representative Crops Encoded", str(total_encoded_crops))
    table.add_row("Output NPZ File", str(npz_out))
    if json_out:
        table.add_row("Output JSON Registry", str(json_out))
    table.add_row("Total Execution Time", f"{elapsed:.2f} seconds")

    console.print(table)
    console.print(
        f"[bold green]Successfully generated SigLIP2 semantic embeddings NPZ in {elapsed:.2f}s![/bold green]"
    )


if __name__ == "__main__":
    main()
