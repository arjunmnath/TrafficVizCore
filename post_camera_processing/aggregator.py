"""Aggregator module for Post-Camera Processing.

Assembles multi-representative vector profiles and extracts structured spatio-temporal and visual
metadata for each Global Vehicle Identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from post_camera_processing.quality_filter import FilteredCropItem
from shared.utils import setup_logger


@dataclass
class ViewRecord:
    """Represents a single encoded representative visual crop view for a global identity."""

    doc_id: str
    view_idx: int
    embedding: np.ndarray
    quality_score: float
    camera_id: str
    track_id: str
    frame_idx: int
    timestamp_sec: float
    source_path: Optional[str] = None


@dataclass
class SemanticProfile:
    """Complete semantic embedding profile and metadata for a Global Vehicle Identity."""

    global_id: str
    representative_views: List[ViewRecord] = field(default_factory=list)
    aggregated_embedding: Optional[np.ndarray] = None  # Global mean/max pooled vector
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationConfig:
    """Configuration options for identity profile aggregation."""

    include_mean_vector: bool = True
    include_individual_views: bool = True
    class_label: str = "vehicle"


class EmbeddingAggregator:
    """Aggregates crop embeddings and extracts rich identity metadata."""

    def __init__(
        self,
        config: Optional[AggregationConfig] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.config = config or AggregationConfig()
        self.logger = logger or setup_logger("EmbeddingAggregator")

    def build_profile(
        self,
        global_id: str,
        crops: List[FilteredCropItem],
        embeddings: List[np.ndarray],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticProfile:
        """Builds a `SemanticProfile` combining multi-view embeddings and metadata."""
        views: List[ViewRecord] = []
        cams_seen = set()
        tracks_seen = set()
        timestamps = []

        for idx, (crop_item, emb) in enumerate(zip(crops, embeddings)):
            raw = crop_item.raw_crop
            cams_seen.add(raw.camera_id)
            tracks_seen.add(raw.track_id)
            timestamps.append(raw.timestamp_sec)

            view_doc_id = f"{global_id}_v{idx}"
            v_rec = ViewRecord(
                doc_id=view_doc_id,
                view_idx=idx,
                embedding=emb,
                quality_score=crop_item.quality_score,
                camera_id=raw.camera_id,
                track_id=raw.track_id,
                frame_idx=raw.frame_idx,
                timestamp_sec=raw.timestamp_sec,
                source_path=raw.source_path,
            )
            views.append(v_rec)

        # Calculate aggregated mean vector
        agg_emb = None
        if embeddings and self.config.include_mean_vector:
            mean_vec = np.mean(embeddings, axis=0, dtype=np.float32)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec /= norm
            agg_emb = mean_vec

        # Build structured metadata dict
        min_t = min(timestamps) if timestamps else 0.0
        max_t = max(timestamps) if timestamps else 0.0

        meta_dict = {
            "global_id": global_id,
            "camera_ids": sorted(list(cams_seen)),
            "track_ids": sorted(list(tracks_seen)),
            "start_time": min_t,
            "end_time": max_t,
            "duration_sec": max(0.0, max_t - min_t),
            "num_representative_views": len(views),
            "class_label": self.config.class_label,
            "avg_quality_score": round(
                float(np.mean([v.quality_score for v in views])) if views else 0.0, 4
            ),
        }

        if extra_metadata:
            meta_dict.update(extra_metadata)

        profile = SemanticProfile(
            global_id=global_id,
            representative_views=views,
            aggregated_embedding=agg_emb,
            metadata=meta_dict,
        )

        return profile
