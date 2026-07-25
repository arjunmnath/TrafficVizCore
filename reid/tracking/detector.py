import numpy as np
from typing import Any, Dict, List, Optional
from ultralytics import YOLO


class Detector:
    """Manual detector wrapper that performs object detection on BGR image frames."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize the detector model.

        Args:
            model_name (str): YOLO model name or path.
            device (str, optional): Computation device ('cpu', 'cuda', 'mps', 'auto').
        """
        import os
        from reid.utils import resolve_model_weights

        # Resolve model path using the new utility
        model_name = resolve_model_weights(model_name)

        if not os.path.exists(model_name):
            raise FileNotFoundError(f"YOLO model weight file not found at: {model_name}")

        if device == "auto":
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model = YOLO(model_name)
        if device is not None and device != "cpu":
            self.model.to(device)

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        classes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Perform object detection on a BGR image frame.

        Args:
            frame (np.ndarray): Input frame image array.
            conf (float): Object detection confidence threshold.
            classes (List[int], optional): Class IDs to track.
            **kwargs: Extra arguments forwarded to YOLO predict.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - "boxes": np.ndarray of shape (N, 4) in xyxy format
                - "scores": np.ndarray of shape (N,)
                - "classes": np.ndarray of shape (N,)
        """
        results = self.model.predict(
            source=frame, conf=conf, classes=classes, verbose=False, **kwargs
        )

        if not results:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "classes": np.empty((0,), dtype=np.int32),
            }

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "classes": np.empty((0,), dtype=np.int32),
            }

        # Convert to numpy arrays
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes_array = result.boxes.cls.int().cpu().numpy()

        return {"boxes": boxes, "scores": scores, "classes": classes_array}
