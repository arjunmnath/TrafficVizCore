#!/usr/bin/env python3
"""
Crop Embeddings Clustering & Visualization Script

Extracts visual embeddings for crop images from selected track directories in crops.noinclude
using a specified retrieval encoder model (e.g. SigLIP2, OpenCLIP, EVACLIP).
Applies K-Means clustering and generates 2D projections (UMAP and t-SNE) with both
static PNG plots and interactive Plotly HTML breakdowns.

Usage examples:
    # Cluster 3 specific tracks with default SigLIP2 model and 5 clusters
    python scripts/cluster_crop_embeddings.py --tracks clip1.mp4_1 clip1.mp4_102 clip2.mp4_1 -k 5

    # Process all tracks in crops.noinclude with 6 clusters
    python scripts/cluster_crop_embeddings.py --tracks all -k 6 -o siglip2_all_crops.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure workspace root is on sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from vlm_retrieval.retrieval.encoder.factory import get_retrieval_encoder

console = Console()


@dataclass
class CropItem:
    filepath: Path
    track_id: str
    filename: str
    frame_id: int


def parse_frame_id(filename: str) -> int:
    """Extract frame index from filenames like 'frame_000001_t0.04.jpg'."""
    match = re.search(r"frame_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def discover_crops(crops_dir: Path, requested_tracks: List[str]) -> List[CropItem]:
    """Discover crop image files from the crops directory matching requested tracks.

    Args:
        crops_dir: Path to directory containing track crop folders.
        requested_tracks: List of track folder names or ['all'].

    Returns:
        List of CropItem instances.
    """
    if not crops_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Crops directory not found: {crops_dir}")
        sys.exit(1)

    # Resolve track folders
    all_track_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if not all_track_dirs:
        console.print(f"[bold red]Error:[/bold red] No track directories found in {crops_dir}")
        sys.exit(1)

    selected_dirs = []
    if len(requested_tracks) == 1 and requested_tracks[0].lower() in ("all", "*"):
        selected_dirs = all_track_dirs
    else:
        # Match exact or partial track names
        track_map = {d.name: d for d in all_track_dirs}
        for track_name in requested_tracks:
            # Handle comma-separated lists if passed in a single string
            sub_tracks = [t.strip() for t in track_name.split(",") if t.strip()]
            for target in sub_tracks:
                if target in track_map:
                    if track_map[target] not in selected_dirs:
                        selected_dirs.append(track_map[target])
                else:
                    # Try wildcard matching
                    matched = [d for d in all_track_dirs if target in d.name]
                    if matched:
                        for m in matched:
                            if m not in selected_dirs:
                                selected_dirs.append(m)
                    else:
                        console.print(f"[yellow]Warning:[/yellow] Track directory '{target}' not found.")

    if not selected_dirs:
        console.print("[bold red]Error:[/bold red] No matching track directories found for input selection.")
        sys.exit(1)

    crop_items: List[CropItem] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for track_dir in selected_dirs:
        track_id = track_dir.name
        img_files = sorted([f for f in track_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
        for img_file in img_files:
            frame_id = parse_frame_id(img_file.name)
            crop_items.append(
                CropItem(
                    filepath=img_file,
                    track_id=track_id,
                    filename=img_file.name,
                    frame_id=frame_id,
                )
            )

    return crop_items


def extract_crop_embeddings(
    crop_items: List[CropItem],
    model_name: str,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract visual embeddings for all crop items using the specified retrieval encoder.

    Args:
        crop_items: List of CropItem objects.
        model_name: Name of retrieval encoder model.
        device: Target compute device ('auto', 'cuda', 'mps', 'cpu').
        batch_size: Processing batch size.

    Returns:
        Numpy array of shape (N, D) containing normalized embeddings.
    """
    console.print(f"\n[bold cyan]Initializing retrieval encoder:[/bold cyan] {model_name} (device={device})")
    encoder = get_retrieval_encoder(model_name=model_name, device=device)

    embeddings_list: List[np.ndarray] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Encoding crops...", total=len(crop_items))

        # Check if encoder has direct batch encoding or fallback to item-by-item
        has_batch_support = hasattr(encoder, "model") and hasattr(encoder, "processor")

        if has_batch_support and hasattr(encoder.model, "get_image_features"):
            import torch

            for i in range(0, len(crop_items), batch_size):
                batch_items = crop_items[i : i + batch_size]
                images = [Image.open(item.filepath).convert("RGB") for item in batch_items]

                try:
                    inputs = encoder.processor(images=images, return_tensors="pt").to(encoder.device)
                    with torch.no_grad():
                        feats = encoder.model.get_image_features(**inputs)
                        if hasattr(feats, "pooler_output"):
                            feats = feats.pooler_output
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    batch_embs = feats.cpu().numpy()
                    embeddings_list.append(batch_embs)
                except Exception as e:
                    # Fallback to single item encoding if batching fails
                    for item in batch_items:
                        img = Image.open(item.filepath).convert("RGB")
                        emb = encoder.encode_image(img)
                        embeddings_list.append(emb[np.newaxis, :])

                progress.update(task, advance=len(batch_items))
        else:
            for item in crop_items:
                img = Image.open(item.filepath).convert("RGB")
                emb = encoder.encode_image(img)
                embeddings_list.append(emb[np.newaxis, :])
                progress.update(task, advance=1)

    all_embeddings = np.vstack(embeddings_list).astype(np.float32)

    # Ensure L2 normalization
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    all_embeddings = all_embeddings / norms

    return all_embeddings


def perform_kmeans_clustering(embeddings: np.ndarray, n_clusters: int) -> tuple[np.ndarray, Any, float]:
    """Apply K-Means clustering on embeddings.

    Returns:
        tuple of (cluster_labels, kmeans_model, silhouette_score)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = len(embeddings)
    effective_k = min(n_clusters, n_samples)
    if effective_k < n_clusters:
        console.print(
            f"[yellow]Warning:[/yellow] Number of samples ({n_samples}) is less than requested clusters ({n_clusters}). Adjusting K to {effective_k}."
        )

    console.print(f"\n[bold cyan]Performing K-Means clustering (K={effective_k})...[/bold cyan]")
    kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    sil_score = -1.0
    if n_samples > effective_k > 1:
        try:
            sil_score = float(silhouette_score(embeddings, cluster_labels, metric="cosine"))
        except Exception:
            sil_score = -1.0

    return cluster_labels, kmeans, sil_score


def compute_2d_projections(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute UMAP and t-SNE 2D projections of high-dimensional embeddings."""
    from sklearn.manifold import TSNE

    n_samples = len(embeddings)
    console.print(f"\n[bold cyan]Computing 2D projections for {n_samples} embeddings...[/bold cyan]")

    # 1. UMAP Projection
    try:
        import umap

        console.print("  • Running UMAP projection...")
        n_neighbors = min(15, max(2, n_samples - 1))
        reducer_umap = umap.UMAP(
            n_components=2,
            random_state=42,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=0.1,
        )
        umap_coords = reducer_umap.fit_transform(embeddings)
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] UMAP failed ({e}). Falling back to randomized PCA for UMAP slot.")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        umap_coords = pca.fit_transform(embeddings)

    # 2. t-SNE Projection
    console.print("  • Running t-SNE projection...")
    perplexity = min(30, max(2, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        metric="cosine",
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
    )
    tsne_coords = tsne.fit_transform(embeddings)

    return umap_coords, tsne_coords


def render_matplotlib_visualization(
    umap_coords: np.ndarray,
    tsne_coords: np.ndarray,
    cluster_labels: np.ndarray,
    crop_items: List[CropItem],
    output_path: Path,
    model_name: str,
    sil_score: float,
) -> None:
    """Render high-resolution 4-panel static Matplotlib figure."""
    track_ids = [item.track_id for item in crop_items]
    unique_tracks = sorted(list(set(track_ids)))
    track_to_color_idx = {tid: i for i, tid in enumerate(unique_tracks)}
    track_color_indices = np.array([track_to_color_idx[tid] for tid in track_ids])

    unique_clusters = sorted(list(set(cluster_labels)))
    k_clusters = len(unique_clusters)

    # Create Matplotlib Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)

    # Title Banner
    sil_str = f" | Silhouette: {sil_score:.3f}" if sil_score > -1.0 else ""
    fig.suptitle(
        f"Crop Embeddings Clustering & Dimensionality Breakdown\n"
        f"Model: {model_name} | Crops: {len(crop_items)} | Tracks: {len(unique_tracks)} | Clusters: {k_clusters}{sil_str}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Colors
    cmap_clusters = plt.colormaps.get_cmap("tab10").resampled(k_clusters)
    cmap_tracks = plt.colormaps.get_cmap("turbo").resampled(max(1, len(unique_tracks)))

    # Panel 1: UMAP by Cluster
    sc1 = axes[0, 0].scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=cluster_labels,
        cmap=cmap_clusters,
        alpha=0.85,
        edgecolors="none",
        s=35,
    )
    axes[0, 0].set_title("UMAP — Colored by K-Means Cluster", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("UMAP Dimension 1")
    axes[0, 0].set_ylabel("UMAP Dimension 2")
    cbar1 = fig.colorbar(sc1, ax=axes[0, 0], ticks=range(k_clusters))
    cbar1.set_label("Cluster ID")

    # Panel 2: UMAP by Track
    sc2 = axes[0, 1].scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=track_color_indices,
        cmap=cmap_tracks,
        alpha=0.85,
        edgecolors="none",
        s=35,
    )
    axes[0, 1].set_title("UMAP — Colored by Track ID", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("UMAP Dimension 1")
    axes[0, 1].set_ylabel("UMAP Dimension 2")

    # Panel 3: t-SNE by Cluster
    sc3 = axes[1, 0].scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=cluster_labels,
        cmap=cmap_clusters,
        alpha=0.85,
        edgecolors="none",
        s=35,
    )
    axes[1, 0].set_title("t-SNE — Colored by K-Means Cluster", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("t-SNE Dimension 1")
    axes[1, 0].set_ylabel("t-SNE Dimension 2")
    cbar3 = fig.colorbar(sc3, ax=axes[1, 0], ticks=range(k_clusters))
    cbar3.set_label("Cluster ID")

    # Panel 4: t-SNE by Track
    sc4 = axes[1, 1].scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=track_color_indices,
        cmap=cmap_tracks,
        alpha=0.85,
        edgecolors="none",
        s=35,
    )
    axes[1, 1].set_title("t-SNE — Colored by Track ID", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("t-SNE Dimension 1")
    axes[1, 1].set_ylabel("t-SNE Dimension 2")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[bold green]Saved static plot to:[/bold green] {output_path}")


def render_plotly_html_report(
    umap_coords: np.ndarray,
    tsne_coords: np.ndarray,
    cluster_labels: np.ndarray,
    crop_items: List[CropItem],
    output_path: Path,
    model_name: str,
) -> None:
    """Render interactive HTML report using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        console.print("[yellow]Warning:[/yellow] Plotly is not installed. Skipping interactive HTML generation.")
        return

    track_ids = [item.track_id for item in crop_items]
    hover_texts = [
        f"<b>Track:</b> {item.track_id}<br>"
        f"<b>Cluster:</b> {c_id}<br>"
        f"<b>File:</b> {item.filename}<br>"
        f"<b>Frame:</b> {item.frame_id}"
        for item, c_id in zip(crop_items, cluster_labels)
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "UMAP by K-Means Cluster",
            "UMAP by Track ID",
            "t-SNE by K-Means Cluster",
            "t-SNE by Track ID",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.1,
    )

    # 1. UMAP by Cluster
    fig.add_trace(
        go.Scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=cluster_labels,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cluster", x=0.45, len=0.4, y=0.8),
            ),
            text=hover_texts,
            hoverinfo="text",
            name="UMAP (Cluster)",
        ),
        row=1,
        col=1,
    )

    # 2. UMAP by Track
    fig.add_trace(
        go.Scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=[hash(t) % 100 for t in track_ids],
                colorscale="Turbo",
                showscale=False,
            ),
            text=hover_texts,
            hoverinfo="text",
            name="UMAP (Track)",
        ),
        row=1,
        col=2,
    )

    # 3. t-SNE by Cluster
    fig.add_trace(
        go.Scatter(
            x=tsne_coords[:, 0],
            y=tsne_coords[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=cluster_labels,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cluster", x=0.45, len=0.4, y=0.2),
            ),
            text=hover_texts,
            hoverinfo="text",
            name="t-SNE (Cluster)",
        ),
        row=2,
        col=1,
    )

    # 4. t-SNE by Track
    fig.add_trace(
        go.Scatter(
            x=tsne_coords[:, 0],
            y=tsne_coords[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=[hash(t) % 100 for t in track_ids],
                colorscale="Turbo",
                showscale=False,
            ),
            text=hover_texts,
            hoverinfo="text",
            name="t-SNE (Track)",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Interactive Crop Clustering & Projections ({model_name})",
        height=900,
        width=1400,
        template="plotly_white",
        showlegend=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    console.print(f"[bold green]Saved interactive HTML report to:[/bold green] {output_path}")


def display_cluster_summary_table(crop_items: List[CropItem], cluster_labels: np.ndarray) -> None:
    """Print rich summary table of cluster breakdown."""
    table = Table(title="K-Means Cluster Summary Breakdown", box=None)
    table.add_column("Cluster ID", style="bold cyan", justify="right")
    table.add_column("Total Crops", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Top Represented Track IDs", style="magenta")

    total_crops = len(crop_items)
    unique_clusters = sorted(list(set(cluster_labels)))

    for c_id in unique_clusters:
        mask = cluster_labels == c_id
        count = int(np.sum(mask))
        pct = (count / total_crops) * 100.0

        # Count track distribution in cluster
        cluster_tracks = [crop_items[i].track_id for i in range(total_crops) if mask[i]]
        from collections import Counter

        track_counts = Counter(cluster_tracks).most_common(3)
        top_tracks_str = ", ".join([f"{tid} ({cnt})" for tid, cnt in track_counts])

        table.add_row(str(c_id), str(count), f"{pct:.1f}%", top_tracks_str)

    console.print()
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode track crops, apply K-Means clustering, and generate UMAP & t-SNE visual breakdowns."
    )
    parser.add_argument(
        "-c",
        "--crops-dir",
        type=Path,
        default=Path("crops.noinclude"),
        help="Path to root crops directory (default: crops.noinclude)",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        nargs="+",
        default=["clip1.mp4_1", "clip1.mp4_102", "clip2.mp4_1"],
        help="One or more track folder names (or 'all' for all tracks)",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Encoder model name (default: google/siglip2-so400m-patch14-384)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "-k",
        "--n-clusters",
        type=int,
        default=5,
        help="Number of K-Means clusters (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output-plot",
        type=Path,
        default=Path("siglip2_kmeans_visualization.png"),
        help="Path to save static matplotlib figure PNG (default: siglip2_kmeans_visualization.png)",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("siglip2_kmeans_visualization.html"),
        help="Path to save interactive Plotly HTML report (default: siglip2_kmeans_visualization.html)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path to dump cluster assignments and 2D coordinates to JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for feature extraction (default: 16)",
    )

    args = parser.parse_args()

    console.print(
        Panel(
            f"[bold green]Crop Clustering & Dimensionality Breakdown[/bold green]\n"
            f"• Crops Dir: {args.crops_dir}\n"
            f"• Tracks Selected: {args.tracks}\n"
            f"• Model: {args.model_name}\n"
            f"• K-Means K: {args.n_clusters}",
            title="Configuration",
        )
    )

    # 1. Discover crops
    crop_items = discover_crops(args.crops_dir, args.tracks)
    console.print(f"[bold green]Discovered {len(crop_items)} crop images[/bold green] across {len(set(item.track_id for item in crop_items))} tracks.")

    # 2. Extract embeddings
    embeddings = extract_crop_embeddings(
        crop_items,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    # 3. K-Means Clustering
    cluster_labels, kmeans_model, sil_score = perform_kmeans_clustering(embeddings, args.n_clusters)
    if sil_score > -1.0:
        console.print(f"  • Silhouette Score (Cosine): [bold yellow]{sil_score:.4f}[/bold yellow]")

    # 4. Dimensionality Reduction
    umap_coords, tsne_coords = compute_2d_projections(embeddings)

    # 5. Display Summary Table
    display_cluster_summary_table(crop_items, cluster_labels)

    # 6. Render Visualizations
    render_matplotlib_visualization(
        umap_coords=umap_coords,
        tsne_coords=tsne_coords,
        cluster_labels=cluster_labels,
        crop_items=crop_items,
        output_path=args.output_plot,
        model_name=args.model_name,
        sil_score=sil_score,
    )

    render_plotly_html_report(
        umap_coords=umap_coords,
        tsne_coords=tsne_coords,
        cluster_labels=cluster_labels,
        crop_items=crop_items,
        output_path=args.output_html,
        model_name=args.model_name,
    )

    # 7. Optional JSON export
    if args.output_json:
        export_data = []
        for i, item in enumerate(crop_items):
            export_data.append(
                {
                    "track_id": item.track_id,
                    "filename": item.filename,
                    "filepath": str(item.filepath),
                    "frame_id": item.frame_id,
                    "cluster_id": int(cluster_labels[i]),
                    "umap_x": float(umap_coords[i, 0]),
                    "umap_y": float(umap_coords[i, 1]),
                    "tsne_x": float(tsne_coords[i, 0]),
                    "tsne_y": float(tsne_coords[i, 1]),
                }
            )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(export_data, f, indent=2)
        console.print(f"[bold green]Saved clustering metadata JSON to:[/bold green] {args.output_json}")

    console.print("\n[bold green]Successfully completed crop encoding, clustering, and visualization![/bold green] 🎉\n")


if __name__ == "__main__":
    main()
