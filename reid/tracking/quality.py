import numpy as np
from typing import Any, Dict, Optional


class TrackQuality:
    """Evaluates the tracking quality of a track based on its lifetime statistics."""

    @staticmethod
    def evaluate(track: Any, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute a normalized quality score in [0.0, 1.0] for a given track.

        Args:
            track (Any): The track object (EnhancedSTrack instance).
            weights (Dict[str, float], optional): Optional custom weights for each metric.

        Returns:
            float: Normalized quality score between 0.0 and 1.0.
        """
        if weights is None:
            weights = {
                "detector_confidence": 0.3,
                "track_duration": 0.2,
                "embedding_stability": 0.3,
                "association_consistency": 0.2,
            }

        # 1. Detector confidence score
        num_det = getattr(track, "num_detections", 1)
        total_conf = getattr(track, "total_conf", getattr(track, "score", 0.0))
        avg_conf = float(total_conf / max(1, num_det))

        # 2. Track duration score (normalized up to 100 frames)
        start_frame = getattr(track, "start_frame", 1)
        frame_id = getattr(track, "frame_id", 1)
        duration = max(1, frame_id - start_frame + 1)
        duration_score = float(min(1.0, duration / 100.0))

        # 3. Embedding stability score
        stability_sum = getattr(track, "embedding_stability_sum", 1.0)
        avg_stability = float(stability_sum / max(1, num_det))
        stability_score = float(np.clip(avg_stability, 0.0, 1.0))

        # 4. Association consistency score
        max_consec = getattr(track, "max_consecutive_associations", 1)
        consec_score = float(min(1.0, max_consec / 50.0))

        recall_count = getattr(track, "recall_count", 0)
        penalty = 0.02 * recall_count

        consistency_score = float(max(0.0, consec_score - penalty))

        # Weighted combination
        score = (
            weights.get("detector_confidence", 0.3) * avg_conf
            + weights.get("track_duration", 0.2) * duration_score
            + weights.get("embedding_stability", 0.3) * stability_score
            + weights.get("association_consistency", 0.2) * consistency_score
        )

        total_weight = sum(weights.values())
        if total_weight > 0:
            score /= total_weight

        return float(np.clip(score, 0.0, 1.0))
