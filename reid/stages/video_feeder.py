import os
from typing import Any
import cv2
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData


class VideoFeederStage(PipelineStage):
    """Stage 0: Synchronous video frame feeder stage with fast grab sampling."""

    def __init__(self, video_path: str = "", sample_fps: float = 0.0):
        """Constructor.

        Args:
            video_path (str): Initial video file path or RTSP stream link.
            sample_fps (float): Target processing framerate (0.0 for full video FPS).
        """
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.video_name = ""
        self.frame_count = 0
        self.frame_interval = 1

    def set_video_path(self, video_path: str) -> None:
        """Update the video path for the next stream ingestion.

        Args:
            video_path (str): Video file or RTSP link.
        """
        self.video_path = video_path

    def _compute_frame_interval(self) -> int:
        if self.sample_fps <= 0.0:
            return 1
        if self.fps > 0.0 and self.sample_fps < self.fps:
            return max(1, int(round(self.fps / self.sample_fps)))
        elif self.sample_fps > 1.0:
            return int(round(self.sample_fps))
        return 1

    def set_sample_fps(self, sample_fps: float) -> None:
        """Update the target sampling FPS rate or step interval."""
        self.sample_fps = sample_fps
        self.frame_interval = self._compute_frame_interval()

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        """Open the video stream and reset the frame counter."""
        if not self.video_path:
            return

        if listener:
            listener.on_init_status(f"Initializing VideoFeeder for {self.video_path}...")

        # If a capture object is already open, release it
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_name = os.path.basename(self.video_path)
        self.frame_count = 0
        self.frame_interval = self._compute_frame_interval()

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        """Fetch the next sampled frame synchronously from cv2.VideoCapture."""
        if self.cap is None or not self.cap.isOpened():
            data.skip = True
            data.end_of_stream = True
            return data

        # Check if SamplerStage is in pipeline and sync sample_fps dynamically
        if hasattr(pipeline, "stages") and pipeline.stages:
            from reid.stages.sampler import SamplerStage

            sampler = next((s for s in pipeline.stages if isinstance(s, SamplerStage)), None)
            if sampler and not sampler.time_based and sampler.sample_fps > 0.0:
                if sampler.sample_fps != self.sample_fps:
                    self.set_sample_fps(sampler.sample_fps)

        if self.frame_count == 0:
            ret, frame = self.cap.read()
            if not ret:
                data.skip = True
                data.end_of_stream = True
                return data
            self.frame_count = 1
            step = 1
        elif self.frame_interval > 1:
            step = self.frame_interval
            # Grab (skip decoding) unneeded frames rapidly
            for _ in range(self.frame_interval - 1):
                if not self.cap.grab():
                    data.skip = True
                    data.end_of_stream = True
                    return data
                self.frame_count += 1

            ret, frame = self.cap.read()
            if not ret:
                data.skip = True
                data.end_of_stream = True
                return data
            self.frame_count += 1
        else:
            ret, frame = self.cap.read()
            if not ret:
                data.skip = True
                data.end_of_stream = True
                return data
            self.frame_count += 1
            step = 1

        timestamp = self.frame_count / self.fps

        # Populate FrameData properties
        data.frame = frame
        data.frame_count = self.frame_count
        data.frame_step = step
        data.feed_name = self.video_name
        data.total_frames = self.total_frames
        data.timestamp = timestamp
        data.fps = self.fps

        data.skip = False
        data.end_of_stream = False
        return data

    def stop(self) -> None:
        """Release the VideoCapture resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
