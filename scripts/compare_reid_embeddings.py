#!/usr/bin/env python3
"""
Cosine similarity analysis script to compare feature embeddings between tracked
global IDs and output statistics and heatmaps.

Reads the ReID pipeline registry JSON and NPZ embeddings, computes pairwise
cosine similarities, and saves a heatmap plot.

Tracks are identified by "{feed}:{track_id}" keys.
Supports comparing one or more tracks.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_tracks_by_key(json_path: Path) -> dict[str, dict]:
    """Load the registry JSON and return a flat mapping:

        "{feed_name}:{track_id}" -> compressed_track dict

    Tracks with a null compressed_track are silently skipped.
    """
    if not json_path.exists():
        console.print(f"[bold red]Error: JSON file not found at {json_path}[/bold red]")
        sys.exit(1)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error: Failed to read or parse JSON file: {e}[/bold red]")
        sys.exit(1)

    if not isinstance(data, dict):
        console.print(
            "[bold red]Error: Unsupported JSON format. Expected a dict keyed by feed name.[/bold red]"
        )
        sys.exit(1)

    by_key: dict[str, dict] = {}
    for feed_name, tracks_list in data.items():
        for entry in tracks_list:
            tid = entry["track_id"]
            comp_track = entry.get("compressed_track")
            if comp_track is None:
                continue
            key = f"{feed_name}:{tid}"
            by_key[key] = comp_track

    return by_key


def extract_embeddings_for_key(
    key: str,
    npz: "np.lib.npyio.NpzFile",
    embedding_type: str = "smooth",
) -> tuple[np.ndarray, list[str]]:
    """Extract the embedding matrix and human-readable labels for one track key.

    key format: "{feed_name}:{track_id}"
    NPZ key format: "{feed_name}_{embedding_type}_{track_id}"
    """
    feed_name, track_id_str = key.split(":", 1)
    npz_key = f"{feed_name}_{embedding_type}_{track_id_str}"

    if npz_key not in npz:
        return np.empty((0, 0)), []

    embeddings = npz[npz_key].astype(np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]

    n = len(embeddings)
    labels = [f"{key} | frame {i}" for i in range(n)]
    return embeddings, labels


def compute_pairwise_cosine_similarity(all_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the full pairwise cosine similarity matrix of a set of embeddings.
    """
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    norm_embeddings = all_embeddings / norms
    return np.dot(norm_embeddings, norm_embeddings.T)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare embeddings for one or more tracks and plot cosine similarity heatmap."
    )
    parser.add_argument(
        "--json_path",
        "-j",
        type=str,
        required=True,
        help="Path to the registry JSON produced by run_reid_pipeline.py.",
    )
    parser.add_argument(
        "--npz_path",
        "-n",
        type=str,
        default=None,
        help="Path to the NPZ embeddings file. Defaults to <json_path>.npz.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        help=(
            "Track keys to compare, in the format 'feed_name:track_id' "
            "(e.g. --ids clip1.mp4:1 clip1.mp4:3 clip2.mp4:2)."
        ),
    )
    parser.add_argument(
        "--embedding-type",
        "-e",
        choices=["occ", "smooth"],
        default="smooth",
        help="Which embeddings to use: 'occ' = raw per-frame; 'smooth' = moving-average.",
    )
    parser.add_argument(
        "--output_plot",
        "-o",
        type=str,
        default="similarity_matrix.png",
        help="Output filepath for the similarity heatmap.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap (e.g. coolwarm, viridis, plasma).",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show the interactive plot window before saving.",
    )
    args = parser.parse_args()

    # Determine target track keys
    target_ids = [str(x) for x in args.ids] if args.ids else []

    if not target_ids:
        console.print(
            "[bold red]Error: No track keys specified. Use --ids feed:track_id ...[/bold red]"
        )
        sys.exit(1)


    # Resolve JSON path
    json_path = Path(args.json_path)
    if not json_path.exists():
        script_dir = Path(__file__).resolve().parent
        workspace_root = script_dir.parent
        fallback = workspace_root / args.json_path
        if fallback.exists():
            json_path = fallback

    console.print(f"[bold cyan]Loading registry from:[/bold cyan] {json_path}")
    by_key = load_tracks_by_key(json_path)

    # Resolve and load NPZ
    npz_path = Path(args.npz_path) if args.npz_path else json_path.with_suffix(".npz")
    if not npz_path.exists():
        console.print(f"[bold red]Error: NPZ file not found: {npz_path}[/bold red]")
        sys.exit(1)
    try:
        npz = np.load(npz_path)
    except Exception as e:
        console.print(f"[bold red]Error: Failed to load NPZ: {e}[/bold red]")
        sys.exit(1)

    embedding_type = getattr(args, "embedding_type", "smooth")

    # Validate all keys exist
    available_keys = sorted(by_key.keys())
    for key in target_ids:
        if key not in by_key:
            console.print(f"[bold red]Error: Track key '{key}' not found in registry.[/bold red]")
            console.print(f"Available keys: {available_keys}")
            sys.exit(1)

    # Load and process embeddings per track key
    all_embeddings_list = []
    all_labels = []
    id_indices: dict[str, tuple[int, int]] = {}

    current_idx = 0
    for key in target_ids:
        embs, labels = extract_embeddings_for_key(key, npz, embedding_type)
        n = len(embs)
        comp_track = by_key[key]
        cls = comp_track.get("class", "?")
        start_t = comp_track.get("start_time", 0.0)
        end_t = comp_track.get("end_time", 0.0)
        console.print(
            f"[bold green]{key}[/bold green] ({cls}, {start_t:.2f}s–{end_t:.2f}s): "
            f"Found {n} embedding frames."
        )

        if n == 0:
            console.print(
                f"[bold red]Error: Track '{key}' has zero embeddings in NPZ (key: "
                f"{key.split(':')[0]}_{embedding_type}_{key.split(':')[1]}).[/bold red]"
            )
            sys.exit(1)

        all_embeddings_list.append(embs)
        all_labels.extend(labels)
        id_indices[key] = (current_idx, current_idx + n)
        current_idx += n

    total_occs = len(all_labels)
    all_embeddings = np.concatenate(all_embeddings_list, axis=0)

    # Compute similarity matrix
    similarity_matrix = compute_pairwise_cosine_similarity(all_embeddings)

    # Helper to calculate stats excluding diagonal for self-similarities
    def get_stats(
        matrix: np.ndarray, exclude_diag: bool = False
    ) -> tuple[float, float, float, float]:
        if matrix.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        if exclude_diag and matrix.shape[0] == matrix.shape[1]:
            n = matrix.shape[0]
            if n <= 1:
                return 1.0, 0.0, 1.0, 1.0
            vals = matrix[~np.eye(n, dtype=bool)]
        else:
            vals = matrix.flatten()
        return float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))

    # Print results to the console in a nice Table
    stats_table = Table(title="Embedding Similarity Statistics", box=box.ROUNDED)
    stats_table.add_column("Comparison Group", style="cyan", justify="left")
    stats_table.add_column("Count", style="white", justify="right")
    stats_table.add_column("Mean Similarity", style="green", justify="right")
    stats_table.add_column("Std Dev", style="yellow", justify="right")
    stats_table.add_column("Min Similarity", style="red", justify="right")
    stats_table.add_column("Max Similarity", style="green", justify="right")

    # Add Intra stats for each ID
    for gid in target_ids:
        start, end = id_indices[gid]
        n = end - start
        sub_matrix = similarity_matrix[start:end, start:end]
        mean_val, std_val, min_val, max_val = get_stats(sub_matrix, exclude_diag=True)
        stats_table.add_row(
            f"ID {gid} vs ID {gid} (Intra)",
            f"{n}x{n} ({n * (n - 1) if n > 1 else 1} pairs)",
            f"{mean_val:.4f}",
            f"{std_val:.4f}",
            f"{min_val:.4f}",
            f"{max_val:.4f}",
        )

    # Add Inter stats if comparing 2 or more IDs
    if len(target_ids) >= 2:
        stats_table.add_section()
        for i in range(len(target_ids)):
            for j in range(i + 1, len(target_ids)):
                gid1 = target_ids[i]
                gid2 = target_ids[j]
                start1, end1 = id_indices[gid1]
                start2, end2 = id_indices[gid2]
                n1 = end1 - start1
                n2 = end2 - start2

                sub_matrix = similarity_matrix[start1:end1, start2:end2]
                mean_val, std_val, min_val, max_val = get_stats(sub_matrix, exclude_diag=False)
                stats_table.add_row(
                    f"ID {gid1} vs ID {gid2} (Inter)",
                    f"{n1}x{n2} ({n1 * n2} pairs)",
                    f"{mean_val:.4f}",
                    f"{std_val:.4f}",
                    f"{min_val:.4f}",
                    f"{max_val:.4f}",
                )

    console.print("\n")
    console.print(Panel(stats_table, border_style="cyan", expand=False))

    # Visualization Setup
    # Dynamically scale plot size based on number of occurrences
    fig_size = max(10, total_occs * 0.35)
    # Cap size at 25 for extremely large sets, and skip tick labels if too large
    fig_size = min(fig_size, 25)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)

    # Plot similarity heatmap
    # Using vmin/vmax to focus on positive similarity variance if desired, or auto-scale based on values.
    # We will use the minimum value in the similarity matrix as vmin (capped at 0.0 minimum to prevent massive stretch)
    vmin = max(0.0, float(np.min(similarity_matrix)) - 0.05)
    im = ax.imshow(similarity_matrix, cmap=args.cmap, vmin=vmin, vmax=1.0)

    # Draw grid divider lines to segment the IDs
    for gid in target_ids[:-1]:
        _, end = id_indices[gid]
        ax.axvline(x=end - 0.5, color="black", linestyle="--", linewidth=2.0, alpha=0.8)
        ax.axhline(y=end - 0.5, color="black", linestyle="--", linewidth=2.0, alpha=0.8)

    # Set up ticks
    ax.set_xticks(np.arange(total_occs))
    ax.set_yticks(np.arange(total_occs))

    # Only show labels if there aren't too many occurrences
    if total_occs <= 60:
        ax.set_xticklabels(
            all_labels, rotation=90, ha="right", fontsize=max(4, 12 - total_occs // 5)
        )
        ax.set_yticklabels(all_labels, fontsize=max(4, 12 - total_occs // 5))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        console.print(
            "[yellow]Warning: Too many occurrences (>60). Skipping individual tick labels on heatmap.[/yellow]"
        )

    # Annotate cell values if the total occurrences are very small
    if total_occs <= 20:
        for i in range(total_occs):
            for j in range(total_occs):
                val = similarity_matrix[i, j]
                # Choose text color dynamically for contrast
                color = "white" if val < (vmin + (1.0 - vmin) / 2.0) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    weight="bold",
                )

    # Add background boxes/text annotations for regions on the heatmap
    for gid in target_ids:
        start, end = id_indices[gid]
        n = end - start
        if n > 2:  # Only draw text overlay if block is large enough to avoid clutter
            center = start + n / 2.0 - 0.5
            ax.text(
                center,
                center,
                f"Intra ID {gid}",
                color="black",
                fontsize=max(8, min(14, int(n * 0.8))),
                ha="center",
                va="center",
                alpha=0.35,
                weight="bold",
            )

    # For K == 2, we can also label the Inter block for clarity
    if len(target_ids) == 2:
        gid1, gid2 = target_ids[0], target_ids[1]
        start1, end1 = id_indices[gid1]
        start2, end2 = id_indices[gid2]
        n1 = end1 - start1
        n2 = end2 - start2
        center1 = start1 + n1 / 2.0 - 0.5
        center2 = start2 + n2 / 2.0 - 0.5
        # Inter label top-right
        ax.text(
            center2,
            center1,
            "Inter Similarity",
            color="black",
            fontsize=12,
            ha="center",
            va="center",
            alpha=0.35,
            weight="bold",
            style="italic",
        )
        # Inter label bottom-left
        ax.text(
            center1,
            center2,
            "Inter Similarity",
            color="black",
            fontsize=12,
            ha="center",
            va="center",
            alpha=0.35,
            weight="bold",
            style="italic",
        )

    # Title & Colorbar
    ids_title_str = ", ".join(target_ids)
    plt.title(
        f"Pairwise Embedding Cosine Similarity Heatmap\nTracks: {ids_title_str} ({total_occs} total frames)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom", fontsize=11, weight="bold")

    # Show the interactive plot if requested
    if args.show:
        console.print("\n[bold yellow]Opening interactive plot viewer...[/bold yellow]")
        plt.show()

    # Save visualization plot
    output_path = Path(args.output_plot)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    console.print(
        f"\n[bold green]Success![/bold green] Saved similarity heatmap to: [cyan]{output_path.resolve()}[/cyan]\n"
    )


if __name__ == "__main__":
    main()
