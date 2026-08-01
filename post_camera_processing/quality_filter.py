"""Quality Filter module for Post-Camera Processing.

Applies hard filtering (area, aspect ratio, border truncation, severe blur) and soft weighted
quality scoring (sharpness, area, contrast, detection confidence, border margin) to retain
only informative, high-resolution, un-blurred crops for semantic retrieval embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from post_camera_processing.crop_collector import RawCropItem
from shared.utils import setup_logger


@dataclass
class QualityConfig:
    """Configuration for hard pruning filters and soft quality scoring weights."""

    # Hard pruning thresholds
    min_area_px: int = 1600  # Minimum crop area (e.g. 40x40 = 1600 px)
    min_width_px: int = 30
    min_height_px: int = 30
    min_aspect_ratio: float = 0.25  # W / H lower bound
    max_aspect_ratio: float = 4.0   # W / H upper bound
    min_laplacian_var: float = 35.0 # Absolute blur threshold
    min_mean_intensity: float = 12.0  # Dark limit
    max_mean_intensity: float = 248.0 # Bright limit

    # Soft scoring weights (must sum to 1.0)
    w_sharpness: float = 0.35
    w_area: float = 0.25
    w_contrast: float = 0.15
    w_confidence: float = 0.15
    w_border: float = 0.10


@dataclass
class FilteredCropItem:
    """Represents a crop item that has passed quality filtering and has been assigned quality metrics."""

    raw_crop: RawCropItem
    quality_score: float
    sharpness_var: float
    area_px: int
    contrast_std: float
    is_passed: bool = True
    prune_reason: Optional[str] = None


class QualityFilter:
    """Filters raw crops using hard thresholds and scores passed crops with weighted metrics."""

    def __init__(
        self,
        config: Optional[QualityConfig] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.config = config or QualityConfig()
        self.logger = logger or setup_logger("QualityFilter")

    def filter_and_score_crops(
        self,
        crops: List[RawCropItem],
        max_crops_per_identity: Optional[int] = None,
    ) -> List[FilteredCropItem]:
        """Filters a list of raw crops, returning sorted high-quality `FilteredCropItem`s."""
        filtered_items: List[FilteredCropItem] = []

        for crop_item in crops:
            res = self.evaluate_crop(crop_item)
            if res.is_passed:
                filtered_items.append(res)

        # Sort descending by composite quality score
        filtered_items.sort(key=lambda x: x.quality_score, reverse=True)

        if max_crops_per_identity and len(filtered_items) > max_crops_per_identity:
            filtered_items = filtered_items[:max_crops_per_identity]

        return filtered_items

    def evaluate_crop(self, raw_crop: RawCropItem) -> FilteredCropItem:
        """Evaluates a single crop against hard filters and calculates soft quality score."""
        pil_img = raw_crop.image
        width, height = pil_img.size
        area = width * height

        # Hard Filter 1: Area & Dimensions
        if area < self.config.min_area_px or width < self.config.min_width_px or height < self.config.min_height_px:
            return FilteredCropItem(
                raw_crop=raw_crop,
                quality_score=0.0,
                sharpness_var=0.0,
                area_px=area,
                contrast_std=0.0,
                is_passed=False,
                prune_reason=f"Area ({area}px) or dimensions ({width}x{height}) below minimum.",
            )

        # Hard Filter 2: Aspect Ratio
        aspect_ratio = width / max(height, 1)
        if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
            return FilteredCropItem(
                raw_crop=raw_crop,
                quality_score=0.0,
                sharpness_var=0.0,
                area_px=area,
                contrast_std=0.0,
                is_passed=False,
                prune_reason=f"Aspect ratio ({aspect_ratio:.2f}) outside range [{self.config.min_aspect_ratio}, {self.config.max_aspect_ratio}].",
            )

        # Convert PIL Image to OpenCV Grayscale for metric calculation
        img_np = np.array(pil_img.convert("L"))

        # Hard Filter 3: Exposure Limits (Mean Pixel Intensity)
        mean_val = float(np.mean(img_np))
        if mean_val < self.config.min_mean_intensity or mean_val > self.config.max_mean_intensity:
            return FilteredCropItem(
                raw_crop=raw_crop,
                quality_score=0.0,
                sharpness_var=0.0,
                area_px=area,
                contrast_std=0.0,
                is_passed=False,
                prune_reason=f"Mean intensity ({mean_val:.1f}) in extreme dark/bright range.",
            )

        # Hard Filter 4: Sharpness / Blur Detection via Laplacian Variance
        laplacian_var = float(cv2.Laplacian(img_np, cv2.CV_64F).var())
        if laplacian_var < self.config.min_laplacian_var:
            return FilteredCropItem(
                raw_crop=raw_crop,
                quality_score=0.0,
                sharpness_var=laplacian_var,
                area_px=area,
                contrast_std=0.0,
                is_passed=False,
                prune_reason=f"Laplacian variance ({laplacian_var:.1f}) below blur threshold {self.config.min_laplacian_var}.",
            )

        # Contrast Standard Deviation
        contrast_std = float(np.std(img_np))

        # --- Soft Quality Score Calculation ---
        # 1. Sharpness score: normalized sigmoid/clip over [min_var, 500]
        s_sharpness = min(1.0, max(0.0, (laplacian_var - self.config.min_laplacian_var) / 400.0))

        # 2. Area score: ratio against target 224x224 = 50176 px
        s_area = min(1.0, area / 50176.0)

        # 3. Contrast score: normalized std dev [0, 80]
        s_contrast = min(1.0, contrast_std / 70.0)

        # 4. Detection confidence
        s_confidence = min(1.0, max(0.0, raw_crop.confidence))

        # 5. Border margin score: 1.0 if not touching borders, lower if near frame edge
        s_border = 1.0
        if raw_crop.bbox:
            x1, y1, x2, y2 = raw_crop.bbox
            if x1 <= 2 or y1 <= 2:
                s_border = 0.5

        composite_score = (
            self.config.w_sharpness * s_sharpness
            + self.config.w_area * s_area
            + self.config.w_contrast * s_contrast
            + self.config.w_confidence * s_confidence
            + self.config.w_border * s_border
        )

        return FilteredCropItem(
            raw_crop=raw_crop,
            quality_score=round(composite_score, 4),
            sharpness_var=round(laplacian_var, 2),
            area_px=area,
            contrast_std=round(contrast_std, 2),
            is_passed=True,
        )
