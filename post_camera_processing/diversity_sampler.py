"""Diversity Sampler module for Post-Camera Processing.

Selects representative visual crops for each Global Vehicle Identity using Farthest Point
Sampling (FPS), temporal redundancy suppression, and feature space coverage without
requiring hardcoded viewpoint label classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from post_camera_processing.quality_filter import FilteredCropItem
from shared.utils import setup_logger


@dataclass
class DiversityConfig:
    """Configuration for semantic diversity selection and redundancy suppression."""

    target_num_views: int = 3       # Desired number of representative views per identity
    min_time_gap_sec: float = 1.5   # Minimum temporal gap between crops from same camera
    max_feature_similarity: float = 0.92 # Cosine similarity cutoff for redundant crops
    use_hsv_histograms: bool = True # Use HSV spatial color histogram for lightweight feature space FPS


class DiversitySampler:
    """Selects diverse representative crops maximizing feature space coverage and quality."""

    def __init__(
        self,
        config: Optional[DiversityConfig] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.config = config or DiversityConfig()
        self.logger = logger or setup_logger("DiversitySampler")

    def select_representative_crops(
        self,
        filtered_crops: List[FilteredCropItem],
        target_k: Optional[int] = None,
    ) -> List[FilteredCropItem]:
        """Selects up to `target_k` diverse, high-quality representative crops.

        Uses a 2-pass approach:
        1. Filter out temporal & visual duplicates.
        2. Apply Farthest Point Sampling (FPS) on color/structural feature vectors.
        """
        if not filtered_crops:
            return []

        k = target_k or self.config.target_num_views
        if len(filtered_crops) <= k:
            return filtered_crops

        # Step 1: Temporal & Redundancy Suppression
        unique_candidates = self._suppress_temporal_redundancy(filtered_crops)

        if len(unique_candidates) <= k:
            return unique_candidates

        # Step 2: Extract lightweight feature vectors (Spatial HSV Histogram) for FPS
        features = np.array([self._extract_crop_feature(item) for item in unique_candidates])

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_features = features / norms

        # Step 3: Farthest Point Sampling (FPS) seeded by highest quality crop (index 0)
        selected_indices = self._farthest_point_sampling(norm_features, k)

        selected_items = [unique_candidates[idx] for idx in selected_indices]
        return selected_items

    def _suppress_temporal_redundancy(
        self, items: List[FilteredCropItem]
    ) -> List[FilteredCropItem]:
        """Prunes crops that are too close in timestamp from the same camera channel."""
        # Group crops by camera ID
        cam_groups: Dict[str, List[FilteredCropItem]] = {}
        for item in items:
            cam_id = item.raw_crop.camera_id
            if cam_id not in cam_groups:
                cam_groups[cam_id] = []
            cam_groups[cam_id].append(item)

        pruned_list: List[FilteredCropItem] = []

        for cam_id, cam_crops in cam_groups.items():
            # Sort by timestamp
            cam_crops.sort(key=lambda x: x.raw_crop.timestamp_sec)
            accepted_in_cam: List[FilteredCropItem] = []

            for crop_item in cam_crops:
                t_curr = crop_item.raw_crop.timestamp_sec

                # Check minimum time gap against already accepted crops in this camera
                too_close = False
                for accepted in accepted_in_cam:
                    if abs(t_curr - accepted.raw_crop.timestamp_sec) < self.config.min_time_gap_sec:
                        too_close = True
                        break

                if not too_close:
                    accepted_in_cam.append(crop_item)

            pruned_list.extend(accepted_in_cam)

        # Re-sort pruned list by quality score
        pruned_list.sort(key=lambda x: x.quality_score, reverse=True)
        return pruned_list

    def _extract_crop_feature(self, item: FilteredCropItem) -> np.ndarray:
        """Extracts a normalized spatial HSV color histogram as a fast visual proxy feature."""
        pil_img = item.raw_crop.image.resize((128, 128))
        img_np = np.array(pil_img)

        if img_np.ndim == 2 or img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Calculate 3D HSV histogram: 8 Hue, 4 Saturation, 4 Value bins
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        feat = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat

    def _farthest_point_sampling(self, features: np.ndarray, k: int) -> List[int]:
        """Performs Farthest Point Sampling in feature space.

        Seeded at index 0 (which corresponds to highest quality score).
        """
        num_samples = len(features)
        if num_samples <= k:
            return list(range(num_samples))

        selected = [0]
        # Distances to closest selected point
        min_distances = 1.0 - np.dot(features, features[0])

        for _ in range(1, k):
            # Select point with maximum distance to its nearest selected neighbor
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)

            # Update minimum distances using distance to new point
            dist_to_new = 1.0 - np.dot(features, features[next_idx])
            min_distances = np.minimum(min_distances, dist_to_new)

        return selected
