"""Track Comparison Module

Provides methods for comparing two tracks using their spatial trajectories
(polynomial/spline curves), size models, visual embeddings, and spatiotemporal profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tracking.domain.track import CompressedTrack
from tracking.serialization.json_deserializer import JsonDeserializer


@dataclass
class TrackSimilarityResult:
    """Detailed similarity comparison report between two tracks."""

    track_id_a: str
    track_id_b: str
    overall_similarity: float
    trajectory_similarity: float
    size_similarity: float
    embedding_similarity: float
    spatiotemporal_similarity: float
    spatial_distance_px: float
    metrics_breakdown: Dict[str, float] = field(default_factory=dict)


class TrackComparator:
    """Multi-modal track comparison engine evaluating spatial curve similarity,

    size curve similarity, visual feature embedding distance, and temporal overlap.
    """

    def __init__(
        self,
        weight_trajectory: float = 0.35,
        weight_size: float = 0.20,
        weight_embedding: float = 0.30,
        weight_spatiotemporal: float = 0.15,
        spatial_sigma_px: float = 150.0,
    ) -> None:
        self.w_traj = weight_trajectory
        self.w_size = weight_size
        self.w_emb = weight_embedding
        self.w_time = weight_spatiotemporal
        self.spatial_sigma = spatial_sigma_px

        # Normalize weights
        total_w = self.w_traj + self.w_size + self.w_emb + self.w_time
        if total_w > 0:
            self.w_traj /= total_w
            self.w_size /= total_w
            self.w_emb /= total_w
            self.w_time /= total_w

    def compare_tracks(
        self,
        track_a: Union[CompressedTrack, Dict[str, Any], str],
        track_b: Union[CompressedTrack, Dict[str, Any], str],
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
    ) -> TrackSimilarityResult:
        """Compare two tracks and return a comprehensive similarity report.

        Args:
            track_a: First track (CompressedTrack object, dict representation, or track ID string).
            track_b: Second track (CompressedTrack object, dict representation, or track ID string).
            embedding_a: Optional feature vector embedding for track_a.
            embedding_b: Optional feature vector embedding for track_b.

        Returns:
            TrackSimilarityResult with detailed breakdown of similarity scores.
        """
        dict_a = self._to_track_dict(track_a)
        dict_b = self._to_track_dict(track_b)

        id_a = str(dict_a.get("track_id", dict_a.get("id", "track_a")))
        id_b = str(dict_b.get("track_id", dict_b.get("id", "track_b")))

        # 1. Trajectory Spatial Curve Similarity
        traj_sim, mean_dist_px = self._compute_trajectory_similarity(dict_a, dict_b)

        # 2. Size Curve Model Similarity
        size_sim = self._compute_size_similarity(dict_a, dict_b)

        # 3. Visual Embedding Cosine Similarity
        emb_sim = self._compute_embedding_similarity(embedding_a, embedding_b)

        # 4. Spatiotemporal Overlap / Proximity
        time_sim = self._compute_spatiotemporal_similarity(dict_a, dict_b)

        # Composite overall score
        overall = (
            self.w_traj * traj_sim
            + self.w_size * size_sim
            + self.w_emb * emb_sim
            + self.w_time * time_sim
        )

        return TrackSimilarityResult(
            track_id_a=id_a,
            track_id_b=id_b,
            overall_similarity=float(np.clip(overall, 0.0, 1.0)),
            trajectory_similarity=float(np.clip(traj_sim, 0.0, 1.0)),
            size_similarity=float(np.clip(size_sim, 0.0, 1.0)),
            embedding_similarity=float(np.clip(emb_sim, 0.0, 1.0)),
            spatiotemporal_similarity=float(np.clip(time_sim, 0.0, 1.0)),
            spatial_distance_px=float(mean_dist_px),
            metrics_breakdown={
                "trajectory_weight": self.w_traj,
                "size_weight": self.w_size,
                "embedding_weight": self.w_emb,
                "spatiotemporal_weight": self.w_time,
            },
        )

    def _to_track_dict(self, track: Union[CompressedTrack, Dict[str, Any], str]) -> Dict[str, Any]:
        """Normalize input to standard dictionary format."""
        if isinstance(track, CompressedTrack):
            from tracking.serialization.json_serializer import JsonSerializer
            return JsonSerializer.serialize_to_dict(track)
        elif isinstance(track, dict):
            if "compressed_track" in track and isinstance(track["compressed_track"], dict):
                merged = dict(track["compressed_track"])
                merged.update(track)
                return merged
            return track
        elif isinstance(track, str):
            return {"track_id": track}
        return {}

    def _extract_trajectory_points(self, track_dict: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract spatial center trajectory points [(x, y), ...] from track dict."""
        points: List[Tuple[float, float]] = []

        # Check trajectory key
        traj = track_dict.get("trajectory")
        if isinstance(traj, str):
            import json
            try:
                traj = json.loads(traj)
            except Exception:
                traj = None

        if isinstance(traj, dict) and "segments" in traj:
            for seg in traj.get("segments", []):
                pts = seg.get("control_points", seg.get("points", []))
                for pt in pts:
                    if pt and len(pt) >= 2:
                        points.append((float(pt[0]), float(pt[1])))
        elif isinstance(traj, list):
            for pt in traj:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    points.append((float(pt[0]), float(pt[1])))

        # Fallback to occurrences or raw_boxes
        if not points:
            boxes = track_dict.get("raw_boxes", track_dict.get("boxes", []))
            if isinstance(boxes, str):
                import json
                try:
                    boxes = json.loads(boxes)
                except Exception:
                    boxes = []
            if isinstance(boxes, list):
                for box in boxes:
                    if box and isinstance(box, (list, tuple)) and len(box) >= 4:
                        cx = float(box[0]) + float(box[2]) / 2.0
                        cy = float(box[1]) + float(box[3]) / 2.0
                        points.append((cx, cy))

        return points

    def _compute_trajectory_similarity(
        self, dict_a: Dict[str, Any], dict_b: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Compute spatial trajectory similarity score and mean pixel distance."""
        pts_a = self._extract_trajectory_points(dict_a)
        pts_b = self._extract_trajectory_points(dict_b)

        if not pts_a or not pts_b:
            return 0.5, 999.0

        # Resample both trajectories to N normalized points
        n_samples = 20
        arr_a = self._resample_points(pts_a, n_samples)
        arr_b = self._resample_points(pts_b, n_samples)

        dists = np.linalg.norm(arr_a - arr_b, axis=1)
        mean_dist = float(np.mean(dists))

        # Gaussian similarity decay
        sim = math.exp(-0.5 * (mean_dist / self.spatial_sigma) ** 2)
        return sim, mean_dist

    def _resample_points(self, points: List[Tuple[float, float]], n: int) -> np.ndarray:
        """Resample point trajectory into n equidistant points along arc length."""
        arr = np.array(points, dtype=np.float32)
        if len(arr) == 1:
            return np.tile(arr[0], (n, 1))

        # Compute cumulative distance along points
        diffs = np.diff(arr, axis=0)
        dist_step = np.linalg.norm(diffs, axis=1)
        cum_dist = np.insert(np.cumsum(dist_step), 0, 0.0)

        total_dist = cum_dist[-1]
        if total_dist <= 1e-6:
            return np.tile(arr[0], (n, 1))

        target_dists = np.linspace(0, total_dist, n)
        x_interp = np.interp(target_dists, cum_dist, arr[:, 0])
        y_interp = np.interp(target_dists, cum_dist, arr[:, 1])

        return np.column_stack([x_interp, y_interp])

    def _compute_size_similarity(self, dict_a: Dict[str, Any], dict_b: Dict[str, Any]) -> float:
        """Compute similarity between bounding box size models or raw dimensions."""
        w_a, h_a = self._extract_average_dimensions(dict_a)
        w_b, h_b = self._extract_average_dimensions(dict_b)

        if w_a <= 0 or w_b <= 0 or h_a <= 0 or h_b <= 0:
            return 0.5

        dw = abs(w_a - w_b) / (w_a + w_b)
        dh = abs(h_a - h_b) / (h_a + h_b)

        size_diff = (dw + dh) / 2.0
        return float(max(0.0, 1.0 - size_diff))

    def _extract_average_dimensions(self, track_dict: Dict[str, Any]) -> Tuple[float, float]:
        """Extract mean width and height from size_model or raw_boxes."""
        wm = track_dict.get("width_model", {})
        hm = track_dict.get("height_model", {})

        if isinstance(wm, dict) and "parameters" in wm:
            params = wm.get("parameters", {})
            val_w = params.get("val", params.get("coeffs", [50.0])[-1])
            val_h = hm.get("parameters", {}).get("val", 50.0)
            return float(val_w), float(val_h)

        boxes = track_dict.get("raw_boxes", track_dict.get("boxes", []))
        if isinstance(boxes, str):
            import json
            try:
                boxes = json.loads(boxes)
            except Exception:
                boxes = []

        if isinstance(boxes, list) and boxes:
            ws = [float(b[2]) for b in boxes if len(b) >= 4]
            hs = [float(b[3]) for b in boxes if len(b) >= 4]
            if ws and hs:
                return float(np.mean(ws)), float(np.mean(hs))

        return 50.0, 50.0

    def _compute_embedding_similarity(
        self, emb_a: Optional[np.ndarray], emb_b: Optional[np.ndarray]
    ) -> float:
        """Compute normalized cosine similarity between visual feature embeddings."""
        if emb_a is None or emb_b is None:
            return 0.5

        vec_a = np.array(emb_a, dtype=np.float32).flatten()
        vec_b = np.array(emb_b, dtype=np.float32).flatten()

        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.5

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a <= 1e-6 or norm_b <= 1e-6:
            return 0.5

        cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

    def _compute_spatiotemporal_similarity(
        self, dict_a: Dict[str, Any], dict_b: Dict[str, Any]
    ) -> float:
        """Compute spatiotemporal overlap or proximity score."""
        st_a = float(dict_a.get("start_time", 0.0))
        end_a = float(dict_a.get("end_time", st_a))
        st_b = float(dict_b.get("start_time", 0.0))
        end_b = float(dict_b.get("end_time", st_b))

        cam_a = str(dict_a.get("camera", dict_a.get("camera_id", "")))
        cam_b = str(dict_b.get("camera", dict_b.get("camera_id", "")))

        inter_start = max(st_a, st_b)
        inter_end = min(end_a, end_b)
        intersection = max(0.0, inter_end - inter_start)

        union = max(end_a, end_b) - min(st_a, st_b)
        time_iou = (intersection / union) if union > 0 else 0.0

        cam_match = 1.0 if (cam_a and cam_b and cam_a == cam_b) else 0.5

        return float(0.7 * time_iou + 0.3 * cam_match)


def compare_tracks(
    track_a: Union[CompressedTrack, Dict[str, Any], str],
    track_b: Union[CompressedTrack, Dict[str, Any], str],
    embedding_a: Optional[np.ndarray] = None,
    embedding_b: Optional[np.ndarray] = None,
) -> TrackSimilarityResult:
    """Convenience functional interface for track comparison."""
    comparator = TrackComparator()
    return comparator.compare_tracks(track_a, track_b, embedding_a, embedding_b)
