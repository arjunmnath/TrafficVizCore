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
from scipy.optimize import linear_sum_assignment


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


class GlobalIdentityCluster(NamedTuple):
    global_id: str
    class_label: str
    tracks: List[str]  # e.g. ["clip1.mp4_5", "clip2.mp4_128"]
    track_details: List[Dict[str, Any]]
    camera_ids: List[str]
    num_tracks: int
    matches: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────────────────


def l2_normalize(v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def aggregate_embeddings(occ_embeddings: np.ndarray[Any, Any], mode: str) -> np.ndarray[Any, Any]:
    """Reduce a (N, D) matrix of embeddings to a single prototype vector."""
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
    """Load track entries from JSON + NPZ, grouped by feed name."""
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

            embeddings = npz[npz_key].astype(np.float32)
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
# Matching & N-Track Aggregation
# ──────────────────────────────────────────────────────────────────────────────


def match_cross_camera(
    feed_tracks: Dict[str, List[TrackEntry]],
    threshold: float,
    same_class_only: bool = True,
) -> List[MatchResult]:
    """Compute 1-to-1 bipartite optimal cosine similarity matching between tracks across feeds."""
    feeds = list(feed_tracks.keys())
    results: List[MatchResult] = []

    for i in range(len(feeds)):
        for j in range(i + 1, len(feeds)):
            feed_a, feed_b = feeds[i], feeds[j]
            tracks_a = feed_tracks[feed_a]
            tracks_b = feed_tracks[feed_b]

            classes_a = set(t.class_label for t in tracks_a)
            classes_b = set(t.class_label for t in tracks_b)
            common_classes = classes_a & classes_b if same_class_only else (classes_a | classes_b)

            for cls_lbl in sorted(list(common_classes)):
                cls_tracks_a = [t for t in tracks_a if not same_class_only or t.class_label == cls_lbl]
                cls_tracks_b = [t for t in tracks_b if not same_class_only or t.class_label == cls_lbl]

                if not cls_tracks_a or not cls_tracks_b:
                    continue

                sim_matrix = np.zeros((len(cls_tracks_a), len(cls_tracks_b)), dtype=np.float32)
                for r_idx, ta in enumerate(cls_tracks_a):
                    for c_idx, tb in enumerate(cls_tracks_b):
                        sim_matrix[r_idx, c_idx] = cosine_similarity(ta.embedding, tb.embedding)

                cost_matrix = 1.0 - sim_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r_idx, c_idx in zip(row_ind, col_ind):
                    sim = float(sim_matrix[r_idx, c_idx])
                    if sim >= threshold:
                        ta = cls_tracks_a[r_idx]
                        tb = cls_tracks_b[c_idx]
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


def cluster_tracks_into_identities(
    feed_tracks: Dict[str, List[TrackEntry]],
    match_results: List[MatchResult],
) -> List[GlobalIdentityCluster]:
    """Aggregate N matching tracks into single global identity clusters via Union-Find (DSU)."""
    parent: Dict[str, str] = {}
    track_info: Dict[str, Dict[str, Any]] = {}

    for feed, tracks in feed_tracks.items():
        for t in tracks:
            key = f"{feed}_{t.track_id}"
            parent[key] = key
            track_info[key] = {
                "feed": feed,
                "track_id": t.track_id,
                "track_key": key,
                "class_label": t.class_label,
            }

    def find(i: str) -> str:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: str, j: str) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for r in match_results:
        key_a = f"{r.feed_a}_{r.track_a}"
        key_b = f"{r.feed_b}_{r.track_b}"
        union(key_a, key_b)

    clusters: Dict[str, List[str]] = {}
    for key in parent:
        root = find(key)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(key)

    pairwise_per_cluster: Dict[str, List[MatchResult]] = {}
    for r in match_results:
        key_a = f"{r.feed_a}_{r.track_a}"
        root = find(key_a)
        if root not in pairwise_per_cluster:
            pairwise_per_cluster[root] = []
        pairwise_per_cluster[root].append(r)

    sorted_roots = sorted(clusters.keys(), key=lambda r: (-len(clusters[r]), r))
    identity_clusters: List[GlobalIdentityCluster] = []

    for idx, root in enumerate(sorted_roots):
        members = sorted(clusters[root])
        member_details = [track_info[m] for m in members]
        cams = sorted(list(set(track_info[m]["feed"] for m in members)))

        classes = [track_info[m]["class_label"] for m in members]
        class_label = max(set(classes), key=classes.count) if classes else "object"

        matches_list = [
            {
                "feed_a": r.feed_a,
                "track_a": r.track_a,
                "class_a": r.class_a,
                "feed_b": r.feed_b,
                "track_b": r.track_b,
                "class_b": r.class_b,
                "similarity": round(r.similarity, 6),
            }
            for r in pairwise_per_cluster.get(root, [])
        ]

        gid = f"global_veh_{idx + 1}"
        identity_clusters.append(
            GlobalIdentityCluster(
                global_id=gid,
                class_label=class_label,
                tracks=members,
                track_details=member_details,
                camera_ids=cams,
                num_tracks=len(members),
                matches=matches_list,
            )
        )

    return identity_clusters


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────


def print_results(
    results: List[MatchResult],
    clusters: List[GlobalIdentityCluster],
    top_k: int,
) -> None:
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
    print(f"\n{'Cross-Camera ReID Pairwise Matches':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for r in shown:
        print(
            f"{r.feed_a:<{col_w}} {r.track_a:>8}  {r.class_a:<12}  "
            f"{r.feed_b:<{col_w}} {r.track_b:>8}  {r.class_b:<12}  {r.similarity:>10.4f}"
        )
    print(sep)
    print(f"  Total pairwise matches: {len(results)}  (showing top {len(shown)})\n")

    multi_track_clusters = [c for c in clusters if c.num_tracks > 1]
    print(f"{'Aggregated Multi-Track Global Identities (' + str(len(multi_track_clusters)) + ' identities)':^70}")
    print("─" * 70)
    print(f"{'Global ID':<16} {'Class':<12} {'Tracks (N)':<12} {'Feeds':<16} {'Member Tracks'}")
    print("─" * 70)
    for c in multi_track_clusters[:15]:
        trks_str = ", ".join(c.tracks[:4]) + ("..." if len(c.tracks) > 4 else "")
        feeds_str = ", ".join(c.camera_ids)
        print(f"{c.global_id:<16} {c.class_label:<12} {c.num_tracks:<12} {feeds_str:<16} {trks_str}")
    print("─" * 70 + "\n")


def save_results(clusters: List[GlobalIdentityCluster], output_path: str) -> None:
    data = [
        {
            "global_id": c.global_id,
            "class_label": c.class_label,
            "tracks": c.tracks,
            "track_details": c.track_details,
            "camera_ids": c.camera_ids,
            "num_tracks": c.num_tracks,
            "matches": c.matches,
        }
        for c in clusters
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only compare tracks with the same class label across cameras",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path to save output match results JSON",
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

    print("Computing cross-camera similarities and aggregating identities...")
    results = match_cross_camera(feed_tracks, args.threshold, args.same_class_only)
    clusters = cluster_tracks_into_identities(feed_tracks, results)

    print_results(results, clusters, args.top_k)

    if args.output:
        save_results(clusters, args.output)


if __name__ == "__main__":
    main()
