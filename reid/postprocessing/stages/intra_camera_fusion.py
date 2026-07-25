"""
reid/postprocessing/stages/intra_camera_fusion.py

IntraCameraTrajectoryFusionStage: Postprocessing stage that merges fragmented
trajectories belonging to the same object within a single camera feed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Set
import numpy as np

from reid.postprocessing.base import PostProcessingStage
from reid.postprocessing.pipeline import TerminatedTrack
from reid.postprocessing.stages.trajectory_fusion import attention_fusion, mean_fusion
from tracking.compression.compressor import TrajectoryCompressor


def _cosine_similarity(v1: np.ndarray[Any, Any], v2: np.ndarray[Any, Any]) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _bbox_centroid(bbox: List[float] | Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate (cx, cy) from bbox [x1, y1, x2, y2]."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


class IntraCameraTrajectoryFusionStage(PostProcessingStage):
    """Postprocessing stage to fuse fragmented trajectories of a single object in one camera.

    Tracks of the same object can be fragmented due to occlusions, detection drops,
    or tracker loss. This stage evaluates candidate track pairs based on:
      1. Matching feed_name and class_label
      2. Temporal continuity (Track B starts after Track A ends within max_time_gap,
         or small allowed overlap <= max_overlap_frames)
      3. Spatial proximity (centroid distance <= max_spatial_distance)
      4. High ReID appearance embedding similarity (cosine similarity >= appearance_threshold)

    When a match is found, Track B is assigned the master_track_id of Track A, and
    their histories, per-frame embeddings, fused prototype embedding, and continuous
    CompressedTrack representations are consolidated.

    Args:
        appearance_threshold: Cosine similarity threshold for matching embeddings (default: 0.75).
        max_time_gap: Maximum allowed time gap in seconds between Track A end and Track B start (default: 30.0).
        max_spatial_distance: Maximum allowed pixel distance between Track A end centroid and Track B start centroid (default: 300.0).
        max_overlap_frames: Maximum allowed overlapping frames between Track A and Track B (default: 5).
        retention_seconds: Duration in seconds to retain terminated tracks in memory buffer for matching (default: 300.0).
        fusion_mode: Fusion algorithm used when re-fusing appearance embeddings ('attention' or 'mean').
        temperature: Temperature parameter for attention fusion.
        compressor: Optional TrajectoryCompressor instance used to re-compress merged trajectories.
    """

    def __init__(
        self,
        appearance_threshold: float = 0.75,
        max_time_gap: float = 30.0,
        max_spatial_distance: float = 300.0,
        max_overlap_frames: int = 5,
        retention_seconds: float = 300.0,
        fusion_mode: Literal["mean", "attention"] = "attention",
        temperature: float = 1.0,
        compressor: Optional[TrajectoryCompressor] = None,
    ) -> None:
        self.appearance_threshold = appearance_threshold
        self.max_time_gap = max_time_gap
        self.max_spatial_distance = max_spatial_distance
        self.max_overlap_frames = max_overlap_frames
        self.retention_seconds = retention_seconds
        self.fusion_mode = fusion_mode
        self.temperature = temperature
        self.compressor = compressor or TrajectoryCompressor()

        # Internal buffer: (feed_name, class_label) -> list of TerminatedTrack
        self._buffered_tracks: Dict[Tuple[str, str], List[TerminatedTrack]] = {}

    def _ensure_fused_embedding(self, track: TerminatedTrack) -> None:
        """Ensure track.fused_embedding is computed."""
        if track.fused_embedding is not None:
            return

        if track.appearance_embeddings is not None and len(track.appearance_embeddings) > 0:
            embs = np.asarray(track.appearance_embeddings, dtype=np.float32)
            if embs.ndim == 1:
                embs = embs[np.newaxis, :]
            if self.fusion_mode == "mean":
                track.fused_embedding = mean_fusion(embs)
            else:
                track.fused_embedding = attention_fusion(embs, self.temperature)

    def _get_track_time_range(self, track: TerminatedTrack) -> Tuple[float, float, int, int]:
        """Get (start_timestamp, end_timestamp, start_frame, end_frame) from track history."""
        if track.history is not None:
            timestamps = track.history.get("timestamps", [])
            frames = track.history.get("frames", [])
            if timestamps and frames:
                return timestamps[0], timestamps[-1], frames[0], frames[-1]

        if track.compressed_track is not None:
            tm = track.compressed_track.time_model
            return tm.timestamps[0], tm.timestamps[-1], tm.frames[0], tm.frames[-1]

        return 0.0, 0.0, 0, 0

    def _get_track_centroids(
        self, track: TerminatedTrack
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get (start_centroid, end_centroid) from track history or compressed track."""
        if track.history is not None:
            bboxes = track.history.get("bboxes", [])
            if bboxes:
                start_c = _bbox_centroid(bboxes[0])
                end_c = _bbox_centroid(bboxes[-1])
                return start_c, end_c

        if track.compressed_track is not None:
            ct = track.compressed_track
            t0 = ct.time_model.timestamps[0]
            t1 = ct.time_model.timestamps[-1]
            return ct.position(t0), ct.position(t1)

        return (0.0, 0.0), (0.0, 0.0)

    def _evict_expired_tracks(self, current_timestamp: float, key: Tuple[str, str]) -> None:
        """Evict tracks older than retention_seconds from internal buffer."""
        if key not in self._buffered_tracks:
            return

        retained = []
        for candidate in self._buffered_tracks[key]:
            _, end_t, _, _ = self._get_track_time_range(candidate)
            if (current_timestamp - end_t) <= self.retention_seconds:
                retained.append(candidate)

        self._buffered_tracks[key] = retained

    def process(self, track: TerminatedTrack) -> TerminatedTrack:
        """Process a terminated track and merge it with candidate prior tracks if matched.

        Args:
            track: TerminatedTrack entering stage.

        Returns:
            The processed TerminatedTrack with master_track_id and fused representations updated.
        """
        self._ensure_fused_embedding(track)

        if track.master_track_id is None:
            track.master_track_id = track.track_id

        if track.fused_embedding is None:
            return track

        key = (track.feed_name, track.class_label)

        b_start_t, b_end_t, b_start_f, b_end_f = self._get_track_time_range(track)
        b_start_c, b_end_c = self._get_track_centroids(track)

        self._evict_expired_tracks(b_start_t, key)

        candidates = self._buffered_tracks.get(key, [])
        best_candidate: Optional[TerminatedTrack] = None
        best_score = -1.0

        for candidate in candidates:
            if candidate.track_id == track.track_id:
                continue

            self._ensure_fused_embedding(candidate)
            if candidate.fused_embedding is None:
                continue

            a_start_t, a_end_t, a_start_f, a_end_f = self._get_track_time_range(candidate)
            a_start_c, a_end_c = self._get_track_centroids(candidate)

            # Check temporal ordering
            time_gap = b_start_t - a_end_t

            if time_gap < 0:
                # Overlapping frames check
                overlap_frames = a_end_f - b_start_f + 1
                if overlap_frames > self.max_overlap_frames:
                    continue
            else:
                if time_gap > self.max_time_gap:
                    continue

            # Check spatial proximity between end of A and start of B
            dist = math.sqrt((b_start_c[0] - a_end_c[0]) ** 2 + (b_start_c[1] - a_end_c[1]) ** 2)
            if dist > self.max_spatial_distance:
                continue

            # Check appearance similarity
            sim = _cosine_similarity(candidate.fused_embedding, track.fused_embedding)
            if sim < self.appearance_threshold:
                continue

            # Composite matching score
            spatial_penalty = dist / (2.0 * self.max_spatial_distance)
            score = sim * (1.0 - spatial_penalty)

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            master_id = best_candidate.master_track_id or best_candidate.track_id
            track.master_track_id = master_id
            track.extra["master_track_id"] = master_id
            track.extra["fused_with_track_id"] = best_candidate.track_id

            # Merge track history, appearance embeddings, and re-fuse
            self._merge_tracks(best_candidate, track)

            # Update best_candidate in buffer so subsequent matches build on updated trajectory
            for idx, item in enumerate(self._buffered_tracks[key]):
                if item.track_id == best_candidate.track_id:
                    self._buffered_tracks[key][idx] = best_candidate
                    break
        else:
            # No match found, cache track as candidate for future tracks
            if key not in self._buffered_tracks:
                self._buffered_tracks[key] = []
            self._buffered_tracks[key].append(track)

        return track

    def _merge_tracks(self, primary: TerminatedTrack, secondary: TerminatedTrack) -> None:
        """Merge secondary track data into primary track and synchronize secondary."""
        # 1. Merge histories (frames, timestamps, bboxes)
        if primary.history is not None and secondary.history is not None:
            p_frames = primary.history.get("frames", [])
            p_times = primary.history.get("timestamps", [])
            p_boxes = primary.history.get("bboxes", [])

            s_frames = secondary.history.get("frames", [])
            s_times = secondary.history.get("timestamps", [])
            s_boxes = secondary.history.get("bboxes", [])

            # Combine and sort by frame index
            combined = {}
            for f, t, b in zip(p_frames, p_times, p_boxes):
                combined[f] = (t, b)
            for f, t, b in zip(s_frames, s_times, s_boxes):
                combined[f] = (t, b)  # Secondary overwrites on overlap

            sorted_frames = sorted(combined.keys())
            merged_times = [combined[f][0] for f in sorted_frames]
            merged_boxes = [combined[f][1] for f in sorted_frames]

            merged_history = {
                "start_frame": sorted_frames[0],
                "start_timestamp": merged_times[0],
                "end_frame": sorted_frames[-1],
                "end_timestamp": merged_times[-1],
                "frames": sorted_frames,
                "timestamps": merged_times,
                "bboxes": merged_boxes,
            }
            primary.history = merged_history
            secondary.history = merged_history

        # 2. Merge appearance embeddings
        if (
            primary.appearance_embeddings is not None
            and secondary.appearance_embeddings is not None
        ):
            p_emb = np.asarray(primary.appearance_embeddings, dtype=np.float32)
            s_emb = np.asarray(secondary.appearance_embeddings, dtype=np.float32)
            if p_emb.ndim == 1:
                p_emb = p_emb[np.newaxis, :]
            if s_emb.ndim == 1:
                s_emb = s_emb[np.newaxis, :]

            merged_emb = np.vstack([p_emb, s_emb])
            primary.appearance_embeddings = merged_emb
            secondary.appearance_embeddings = merged_emb

            # Re-fuse combined embedding prototype
            if self.fusion_mode == "mean":
                fused = mean_fusion(merged_emb)
            else:
                fused = attention_fusion(merged_emb, self.temperature)

            primary.fused_embedding = fused
            secondary.fused_embedding = fused

        # 3. Re-compress continuous trajectory if compressor and history available
        if primary.history is not None:
            frames = primary.history.get("frames", [])
            timestamps = primary.history.get("timestamps", [])
            bboxes = primary.history.get("bboxes", [])
            if frames and timestamps and bboxes:
                bbox_tuples = [tuple(b) for b in bboxes]
                master_id = primary.master_track_id or primary.track_id
                compressed = self.compressor.compress(
                    track_id=master_id,
                    camera_id=primary.feed_name,
                    class_label=primary.class_label,
                    frames=frames,
                    timestamps=timestamps,
                    bboxes=bbox_tuples,
                )
                primary.compressed_track = compressed
                secondary.compressed_track = compressed
                primary.extra["compressed_track"] = compressed
                secondary.extra["compressed_track"] = compressed

    def process_tracks(self, tracks: List[TerminatedTrack]) -> List[TerminatedTrack]:
        """Batch process a list of TerminatedTracks for offline camera feed trajectory fusion.

        Args:
            tracks: List of TerminatedTrack objects.

        Returns:
            List of processed TerminatedTrack objects with master_track_ids updated.
        """
        # Sort chronologically by start timestamp
        sorted_tracks = sorted(tracks, key=lambda t: self._get_track_time_range(t)[0])
        processed = []
        for t in sorted_tracks:
            processed.append(self.process(t))
        return processed

    def __repr__(self) -> str:
        return (
            f"IntraCameraTrajectoryFusionStage("
            f"appearance_threshold={self.appearance_threshold}, "
            f"max_time_gap={self.max_time_gap}, "
            f"max_spatial_distance={self.max_spatial_distance}, "
            f"fusion_mode={self.fusion_mode!r})"
        )
