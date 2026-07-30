#!/usr/bin/env python3
"""
Multi-camera track matching script.

Given a JSON registry and NPZ embeddings produced by the ReID pipeline,
computes cross-camera cosine similarity between tracks from different feeds
and produces a ranked list of candidate re-identification matches.

JSON format (per feed):
  {feed_name: [{track_id, compressed_track}, ...]}

NPZ key format (per feed, per track):
  {feed}_{embedding_type}_{track_id}

Where embedding_type is one of:
  - occ    — per-frame raw detection embeddings (from FrameData.features)
  - smooth — per-frame tracker moving-average embeddings

Usage:
    python scripts/match_multicamera.py --json temp.json --npz temp.npz [options]
"""

import argparse
import json
import sys
from itertools import product
from typing import Dict, List, NamedTuple, Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────


class TrackEntry(NamedTuple):
    feed: str
    track_id: int
    class_label: str
    n_frames: int
    embedding: np.ndarray[Any, Any]  # shape (D,) — aggregated prototype


class MatchResult(NamedTuple):
    feed_a: str
    track_a: int
    class_a: str
    feed_b: str
    track_b: int
    class_b: str
    similarity: float


# ──────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────────────────


def l2_normalize(v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def aggregate_embeddings(occ_embeddings: np.ndarray[Any, Any], mode: str) -> np.ndarray[Any, Any]:
    """Reduce a (N, D) matrix of embeddings to a single prototype vector.

    Args:
        occ_embeddings: Shape (N, D).
        mode: One of 'mean' | 'max_pooling' | 'last'.

    Returns:
        Shape (D,) normalized prototype vector.
    """
    if mode == "mean":
        proto = occ_embeddings.mean(axis=0)
    elif mode == "max_pooling":
        proto = occ_embeddings.max(axis=0)
    elif mode == "last":
        proto = occ_embeddings[-1]
    else:
        raise ValueError(f"Unknown aggregation mode: {mode!r}. Choose mean | max_pooling | last.")
    return l2_normalize(proto.astype(np.float32))


def cosine_similarity(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_tracks(
    json_path: str,
    npz_path: str,
    aggregation: str,
    embedding_type: str,
    class_filter: List[str],
) -> Dict[str, List[TrackEntry]]:
    """Load track entries from JSON + NPZ, grouped by feed name.

    Args:
        json_path: Path to the registry JSON export.
        npz_path: Path to the NPZ embeddings file.
        aggregation: Embedding aggregation mode ('mean', 'max_pooling', 'last').
        embedding_type: Primary embedding key prefix preference ('app', 'occ', or 'smooth').
        class_filter: If non-empty, only keep tracks whose class_label is in this list.

    Returns:
        Dict mapping feed_name -> list of TrackEntry.
    """
    with open(json_path) as f:
        registry: Dict[str, List[Dict[str, Any]]] = json.load(f)

    npz = np.load(npz_path)

    feed_tracks: Dict[str, List[TrackEntry]] = {}

    for feed_name, tracks in registry.items():
        entries: List[TrackEntry] = []
        for track in tracks:
            track_id = track["track_id"]
            comp_track = track.get("compressed_track")
            if not comp_track:
                continue

            class_label = comp_track["class"]
            if class_filter and class_label not in class_filter:
                continue

            # Candidate NPZ key formats to support run_reid_pipeline.py output
            candidate_keys = [
                f"{feed_name}_{embedding_type}_{track_id}",
                f"{feed_name}_app_{track_id}",
                f"{feed_name}_occ_{track_id}",
                f"{feed_name}_smooth_{track_id}",
                f"{feed_name}_{track_id}",
                f"app_{track_id}",
                str(track_id),
            ]

            npz_key = None
            for key in candidate_keys:
                if key in npz:
                    npz_key = key
                    break

            if npz_key is None:
                print(
                    f"  [warn] Missing embedding key for track {track_id} (feed '{feed_name}') in NPZ — skipping.",
                    file=sys.stderr,
                )
                continue

            embeddings = npz[npz_key].astype(np.float32)  # (N, D)
            if embeddings.size == 0:
                print(
                    f"  [warn] Empty embeddings array for key '{npz_key}' — skipping.",
                    file=sys.stderr,
                )
                continue

            if embeddings.ndim == 1:
                embeddings = embeddings[np.newaxis, :]

            prototype = aggregate_embeddings(embeddings, aggregation)
            n_frames = len(comp_track["time_model"]["frames"])

            entries.append(
                TrackEntry(
                    feed=feed_name,
                    track_id=track_id,
                    class_label=class_label,
                    n_frames=n_frames,
                    embedding=prototype,
                )
            )

        if entries:
            feed_tracks[feed_name] = entries

    return feed_tracks


# ──────────────────────────────────────────────────────────────────────────────
# Matching
# ──────────────────────────────────────────────────────────────────────────────


def match_cross_camera(
    feed_tracks: Dict[str, List[TrackEntry]],
    threshold: float,
    same_class_only: bool,
) -> List[MatchResult]:
    """Compute pairwise cosine similarity between all tracks from different feeds.

    Args:
        feed_tracks: Grouped tracks per feed.
        threshold: Minimum similarity to include in results.
        same_class_only: If True, only compare tracks with the same class label.

    Returns:
        List of MatchResult, sorted by similarity descending.
    """
    feeds = list(feed_tracks.keys())
    results: List[MatchResult] = []

    for i in range(len(feeds)):
        for j in range(i + 1, len(feeds)):
            feed_a, feed_b = feeds[i], feeds[j]
            for ta, tb in product(feed_tracks[feed_a], feed_tracks[feed_b]):
                if same_class_only and ta.class_label != tb.class_label:
                    continue
                sim = cosine_similarity(ta.embedding, tb.embedding)
                if sim >= threshold:
                    results.append(
                        MatchResult(
                            feed_a=feed_a,
                            track_a=ta.track_id,
                            class_a=ta.class_label,
                            feed_b=feed_b,
                            track_b=tb.track_id,
                            class_b=tb.class_label,
                            similarity=sim,
                        )
                    )

    results.sort(key=lambda r: r.similarity, reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────


def print_results(results: List[MatchResult], top_k: int) -> None:
    shown = results[:top_k] if top_k > 0 else results
    if not shown:
        print("No matches found above the similarity threshold.")
        return

    col_w = 16
    header = (
        f"{'Feed A':<{col_w}} {'Track A':>8}  {'Class A':<12}  "
        f"{'Feed B':<{col_w}} {'Track B':>8}  {'Class B':<12}  {'Similarity':>10}"
    )
    sep = "─" * len(header)
    print(f"\n{'Cross-Camera ReID Match Results':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for r in shown:
        print(
            f"{r.feed_a:<{col_w}} {r.track_a:>8}  {r.class_a:<12}  "
            f"{r.feed_b:<{col_w}} {r.track_b:>8}  {r.class_b:<12}  {r.similarity:>10.4f}"
        )
    print(sep)
    print(f"  Total matches: {len(results)}  (showing top {len(shown)})\n")


def save_results(results: List[MatchResult], output_path: str) -> None:
    data = [
        {
            "feed_a": r.feed_a,
            "track_a": r.track_a,
            "class_a": r.class_a,
            "feed_b": r.feed_b,
            "track_b": r.track_b,
            "class_b": r.class_b,
            "similarity": round(r.similarity, 6),
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-camera track ReID matching from pipeline output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        required=True,
        metavar="PATH",
        help="Path to registry JSON produced by run_reid_pipeline.py",
    )
    parser.add_argument(
        "--npz",
        required=True,
        metavar="PATH",
        help="Path to NPZ embeddings produced by run_reid_pipeline.py",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum cosine similarity to report a match",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        metavar="INT",
        help="Number of top matches to display (0 = all)",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max_pooling", "last"],
        default="mean",
        help="Method to aggregate per-frame embeddings into a track prototype",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["app", "occ", "smooth"],
        default="app",
        help=(
            "Which embeddings to use for matching: "
            "'app' = appearance embeddings (produced by run_reid_pipeline.py); "
            "'occ' = raw per-frame detection features; "
            "'smooth' = tracker moving-average features"
        ),
    )
    parser.add_argument(
        "--class-filter",
        nargs="*",
        default=[],
        metavar="CLASS",
        help="Only compare tracks of these class labels, e.g. --class-filter person car",
    )
    parser.add_argument(
        "--same-class-only",
        action="store_true",
        help="Only compare tracks with the same class label across cameras",
    )
    parser.add_argument(
        "--output", metavar="PATH", default=None, help="Optional path to save match results as JSON"
    )

    args = parser.parse_args()

    print("\n=== Multi-Camera Track Matching ===")
    print(f"  JSON           : {args.json}")
    print(f"  NPZ            : {args.npz}")
    print(f"  Threshold      : {args.threshold}")
    print(f"  Aggregation    : {args.aggregation}")
    print(f"  Embedding type : {args.embedding_type}")
    print(f"  Class filter   : {args.class_filter or 'all'}")
    print(f"  Same class only: {args.same_class_only}")
    print("===================================\n")

    print("Loading tracks...")
    feed_tracks = load_tracks(
        args.json, args.npz, args.aggregation, args.embedding_type, args.class_filter
    )

    if len(feed_tracks) < 2:
        print(
            f"Error: need at least 2 feeds for cross-camera matching, found {len(feed_tracks)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_tracks = sum(len(v) for v in feed_tracks.values())
    for feed, tracks in feed_tracks.items():
        print(f"  {feed}: {len(tracks)} tracks")
    print(f"  Total: {total_tracks} tracks across {len(feed_tracks)} feeds\n")

    print("Computing cross-camera similarities...")
    results = match_cross_camera(feed_tracks, args.threshold, args.same_class_only)

    print_results(results, args.top_k)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
