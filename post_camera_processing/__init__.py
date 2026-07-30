"""Post Camera Processing package for global vehicle identity crop gathering, quality pruning, semantic diversity sampling, and SigLIP2 semantic embedding production.
"""

from __future__ import annotations

from post_camera_processing.crop_collector import CropCollector, RawCropItem, GlobalIdentityCrops
from post_camera_processing.quality_filter import QualityFilter, QualityConfig, FilteredCropItem
from post_camera_processing.diversity_sampler import DiversitySampler, DiversityConfig
from post_camera_processing.encoder import SigLIP2BatchEncoder
from post_camera_processing.aggregator import EmbeddingAggregator, AggregationConfig, SemanticProfile
from post_camera_processing.exporter import EmbeddingExporter

__all__ = [
    "CropCollector",
    "RawCropItem",
    "GlobalIdentityCrops",
    "QualityFilter",
    "QualityConfig",
    "FilteredCropItem",
    "DiversitySampler",
    "DiversityConfig",
    "SigLIP2BatchEncoder",
    "EmbeddingAggregator",
    "AggregationConfig",
    "SemanticProfile",
    "EmbeddingExporter",
]
