from typing import Any, List
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData
from reid.tracking.detector import Detector


class YoloDetectionStage(PipelineStage):
    """Stage 1: Performs YOLO detection on the input frame. Owns the Detector model instance."""

    def __init__(self, yolo_path: str, device: str = "cpu"):
        """Constructor.

        Args:
            yolo_path (str): Path to YOLO detector weight file (.pt).
            device (str): Compute device.
        """
        self.yolo_path = yolo_path
        self.device = device
        self.detector = None

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if listener:
            listener.on_init_status("Loading YOLOv8 Detector model weights...")
        self.detector = Detector(self.yolo_path, device=self.device)

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        if data.skip or data.end_of_stream:
            return data

        frame = data.frame
        # Run detection. Target COCO classes: person(0), car(2), motorcycle(3), bus(5), truck(7).
        dets = self.detector.detect(frame, conf=0.25, classes=list(pipeline.coco_classes.keys()))

        data.boxes = dets["boxes"]
        data.scores = dets["scores"]
        data.classes = dets["classes"]
        return data
