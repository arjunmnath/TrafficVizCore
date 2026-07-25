"""
ReID Package
Exposes ReID pipelines, listeners, registry, utilities, and stages.
"""

from .pipeline import ReIDPipeline
from .registry import SimpleRegistry
from .utils import ReIDPipelineListener, resolve_path
from .ui import RichUIListener, HeadlessUIListener
from .eval_metrics import compute_reid_retrieval_metrics, compute_mot_tracking_metrics

__all__ = [
    "ReIDPipeline",
    "SimpleRegistry",
    "ReIDPipelineListener",
    "resolve_path",
    "RichUIListener",
    "HeadlessUIListener",
    "compute_reid_retrieval_metrics",
    "compute_mot_tracking_metrics",
]
