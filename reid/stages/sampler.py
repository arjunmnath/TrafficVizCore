import time
from typing import Any
from reid.stages.base import PipelineStage
from reid.utils import FrameData


class SamplerStage(PipelineStage):
    """Stage 0.5: Performs temporal frame downsampling.

    Supports both real-time time-based downsampling and offline count-based downsampling.
    """

    def __init__(self, sample_fps: float = 0.0, time_based: bool = True):
        """Constructor.

        Args:
            sample_fps (float): Target processing framerate (0.0 for full FPS).
            time_based (bool): If True, downsamples based on system clock elapsed time (best for live streams).
                               If False, downsamples based on frame count index interval (best for offline files).
        """
        self.sample_fps = sample_fps
        self.time_based = time_based
        self.last_processed_time = 0.0
        self.interval_seconds = 1.0 / sample_fps if sample_fps > 0 else 0.0

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        """Evaluate sample condition and set data.skip = True if condition fails."""
        if data.skip or data.end_of_stream:
            return data

        fps = data.fps if data.fps > 0 else 30.0
        if self.sample_fps <= 0.0 or self.sample_fps >= fps:
            return data

        if self.time_based:
            current_time = time.time()
            # If elapsed time since last frame is less than sample interval, skip frame
            if current_time - self.last_processed_time < self.interval_seconds:
                data.skip = True
            else:
                self.last_processed_time = current_time
        else:
            # Traditional count-based downsampling (ensure frame 1 is processed)
            frame_interval = max(1, int(round(fps / self.sample_fps)))
            if (data.frame_count - 1) % frame_interval != 0:
                data.skip = True

        return data
