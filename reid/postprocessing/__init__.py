"""
reid/postprocessing/__init__.py
Exposes the postprocessing pipeline and all built-in stages.
"""

from .pipeline import PostProcessingPipeline, TerminatedTrack
from .base import PostProcessingStage
from .stages.trajectory_fusion import TrajectoryFusionStage
from .stages.trajectory_compression import TrajectoryCompressionStage
from .stages.intra_camera_fusion import IntraCameraTrajectoryFusionStage

__all__ = [
    "PostProcessingPipeline",
    "TerminatedTrack",
    "PostProcessingStage",
    "TrajectoryFusionStage",
    "TrajectoryCompressionStage",
    "IntraCameraTrajectoryFusionStage",
]
