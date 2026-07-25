"""
reid/postprocessing/pipeline.py
TerminatedTrack dataclass and PostProcessingPipeline executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base import PostProcessingStage
from tracking.domain.track import CompressedTrack


@dataclass
class TerminatedTrack:
    """Payload passed through the postprocessing pipeline when a track is terminated.

    Attributes:
        track_id: The tracker-assigned local track ID.
        class_label: Detected object class (e.g. 'person', 'car').
        feed_name: Video feed / camera name that produced this track.

        appearance_embeddings: Per-frame raw detection feature vectors, shape (N, D).
            These are the features extracted directly from the ReID model per frame.

        fused_embedding: Output of the trajectory fusion stage, shape (D,).
            None until TrajectoryFusionStage runs.

        history: Raw track history dict from the Tracker (frames, timestamps, bboxes).
        extra: Arbitrary key-value store for downstream stages to attach data.
    """

    track_id: int
    class_label: str = "unknown"
    feed_name: str = ""

    # Raw per-frame embeddings collected by the registry during the track's lifetime
    appearance_embeddings: Optional[np.ndarray[Any, Any]] = None  # shape (N, D)

    # Set by TrajectoryFusionStage
    fused_embedding: Optional[np.ndarray[Any, Any]] = None  # shape (D,)

    # Tracker history (frames, timestamps, bboxes)
    history: Optional[Dict[str, Any]] = None

    # Set by TrajectoryCompressionStage
    compressed_track: Optional[CompressedTrack] = None

    # Set by IntraCameraTrajectoryFusionStage (points to primary master track_id if fused)
    master_track_id: Optional[int] = None

    # Open-ended store for stage outputs
    extra: Dict[str, Any] = field(default_factory=dict)


class PostProcessingPipeline:
    """Sequential postprocessing pipeline triggered on track termination.

    Stages are executed in order on a TerminatedTrack object. Each stage
    may read from and write to the track in-place.

    Example:
        pipeline = PostProcessingPipeline([
            TrajectoryFusionStage(mode="attention"),
        ])
        # Wire into the tracking hook:
        tracker.on_track_terminated = lambda t: pipeline.run(make_terminated(t))
    """

    def __init__(self, stages: List[PostProcessingStage]) -> None:
        """Initialize the pipeline with an ordered list of stages.

        Args:
            stages: Ordered list of PostProcessingStage instances.
        """
        self.stages = stages

    def run(self, track: TerminatedTrack) -> TerminatedTrack:
        """Execute all stages sequentially on the terminated track.

        Args:
            track: The TerminatedTrack entering the pipeline.

        Returns:
            The TerminatedTrack after all stages have processed it.
        """
        for stage in self.stages:
            track = stage.process(track)
        return track
