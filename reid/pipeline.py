import time
from typing import List, Optional

from reid.registry import SimpleRegistry
from reid.utils import ReIDPipelineListener, FrameData
from reid.stages.base import PipelineStage
from reid.stages.video_feeder import VideoFeederStage
from reid.stages.live_feeder import LiveFootageFeedStage


class ReIDPipeline:
    """Execution coordinator running frame-by-frame inference stages sequentially.

    A single pipeline instance processes a single input feed. The feed source is
    determined by the VideoFeederStage. To process multiple feeds, update the
    feeder's video path and call run() again. Result export is handled externally.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        threshold: float = 0.8,
        max_frames: int = 0,
        registry: Optional[SimpleRegistry] = None,
    ):
        """Constructor.

        Args:
            stages (List[PipelineStage]): List of stages representing the pipeline configuration.
            threshold (float): ReID matching threshold.
            max_frames (int): Maximum frames to process (0 for full video).
            registry (SimpleRegistry, optional): Shared registry to reuse across pipeline runs.
        """
        self.stages = stages
        self.threshold = threshold
        self.max_frames = max_frames

        if registry is not None:
            self.registry = registry
        else:
            self.registry = SimpleRegistry()

        # Target COCO classes used for detections: person(0), car(2), motorcycle(3), bus(5), truck(7)
        self.coco_classes = {
            # 0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        """Initialize all stages sequentially."""
        if listener:
            listener.on_init_start()

        for stage in self.stages:
            stage.initialize(listener)

        if listener:
            listener.on_init_end()

    def finalize(self) -> None:
        """Finalize all stages sequentially."""
        for stage in self.stages:
            stage.finalize(self)

    def run(self, listener: ReIDPipelineListener = None) -> None:
        """Execute the pipeline. The input video stream is determined entirely by VideoFeederStage."""
        try:
            feeder_stage = next(
                (s for s in self.stages if isinstance(s, (VideoFeederStage, LiveFootageFeedStage))),
                None,
            )

            if feeder_stage:
                feeder_stage.initialize(listener)
                fps = feeder_stage.fps
                total_frames = feeder_stage.total_frames
            else:
                raise ValueError(
                    "A feeder stage (VideoFeederStage or LiveFootageFeedStage) is required to supply the input stream."
                )

            if listener:
                listener.on_video_start(feeder_stage.video_path, 1, 1, total_frames, fps)

            start_time = time.time()

            try:
                while True:
                    data = FrameData(
                        feed_idx=1,
                        total_videos=1,
                        listener=listener,
                        skip=False,
                        end_of_stream=False,
                    )

                    data.elapsed_time = time.time() - start_time
                    for stage in self.stages:
                        data = stage.process(data, self)


                    if data.end_of_stream:
                        break

                    if self.max_frames > 0 and data.frame_count >= self.max_frames:
                        break
            finally:
                self.finalize()
                feeder_stage.stop()

            if listener:
                listener.on_video_end(feeder_stage.video_path, data.frame_count)

        except KeyboardInterrupt:
            if listener:
                listener.on_error(
                    "Pipeline interrupted by user (Ctrl+C). Saving partial results..."
                )
            for stage in self.stages:
                if isinstance(stage, VideoFeederStage):
                    stage.stop()
