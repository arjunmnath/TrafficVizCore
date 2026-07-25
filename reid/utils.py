import os
from dataclasses import dataclass
from typing import Any, Optional, Dict
import numpy as np


class SimpleRegistry:
    # Forward declaration type hint helper
    pass


@dataclass
class FrameData:
    """Dataclass holding context payload data passed through the pipeline stages."""

    frame: Optional[np.ndarray] = None
    frame_count: int = 0
    feed_name: str = ""
    total_frames: int = 0
    timestamp: float = 0.0
    fps: float = 0.0

    # Routing execution state
    skip: bool = False
    end_of_stream: bool = False

    # Metadata for listeners/UI logging
    feed_idx: int = 1
    total_videos: int = 1
    elapsed_time: float = 0.0
    listener: Optional[Any] = None

    # Pipeline output fields populated by stages
    boxes: Optional[np.ndarray] = None  # shape (N, 4)
    scores: Optional[np.ndarray] = None  # shape (N,)
    classes: Optional[np.ndarray] = None  # shape (N,)
    features: Optional[np.ndarray] = None  # shape (N, D)
    tracks: Optional[np.ndarray] = None  # shape (M, 8)

    def __repr__(self):
        return f"FrameData(frame_count={self.frame_count}, feed_name={self.feed_name}, total_frames={self.total_frames}, timestamp={self.timestamp}, classes={self.classes}, features={self.features.shape if self.features is not None else None}, tracks={self.tracks.shape if self.tracks is not None else None})"


class ReIDPipelineListener:
    """Interface for listening to ReID pipeline execution events."""

    def on_init_start(self):
        pass

    def on_init_status(self, message: str):
        pass

    def on_init_end(self):
        pass

    def on_video_start(
        self, video_path: str, video_idx: int, total_videos: int, total_frames: int, fps: float
    ):
        pass

    def on_frame_processed(
        self,
        video_name: str,
        video_idx: int,
        total_videos: int,
        frame_count: int,
        total_frames: int,
        elapsed_time: float,
        fps: float,
        registry: "SimpleRegistry",
        log_message: str | None = None,
    ):
        pass

    def on_video_end(self, video_path: str, total_frames: int):
        pass

    def on_pipeline_end(self, registries: Dict[str, "SimpleRegistry"], output_path: str):
        pass

    def on_error(self, message: str):
        pass


def has_minimum_roi_area(bbox: np.ndarray, frame_shape: tuple, threshold: float = 0e-3) -> bool:
    """Validate if the crop bounding box meets size constraints.

    Args:
        bbox (np.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        frame_shape (tuple): Shape of the image frame.

    Returns:
        bool: True if crop is valid.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    W, H, _ = frame_shape
    return (w * h) / (W * H) > threshold


def resolve_path(p: str, base_dir: str) -> str:
    """Resolve relative path to absolute path using base directory.

    Args:
        p (str): Input path.
        base_dir (str): Base directory.

    Returns:
        str: Resolved path.
    """
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def resolve_model_weights(model_path: str) -> str:
    """Resolves a model weight file path to its absolute path.

    If the path is absolute or exists, it is returned.
    Otherwise, we attempt to locate it under the 'trained_model' directory in the workspace.

    Args:
        model_path (str): Path or filename of the model weights.

    Returns:
        str: Resolved absolute path to the weights.
    """
    if not model_path:
        return model_path

    if os.path.isabs(model_path):
        return model_path

    if os.path.exists(model_path):
        return os.path.abspath(model_path)

    # Determine workspace root (where the 'reid' directory and 'trained_model' reside)
    # Since reid/utils.py is at <workspace_root>/reid/utils.py, its parent is the workspace root.
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(utils_dir)

    # Try resolving directly under workspace_root/trained_model
    resolved_path = os.path.join(workspace_root, "trained_model", model_path)
    if os.path.exists(resolved_path):
        return resolved_path

    # Try resolving if model_path already includes trained_model/ or trained_models/ prefix
    resolved_direct = os.path.join(workspace_root, model_path)
    if os.path.exists(resolved_direct):
        return resolved_direct

    # If the user passed "trained_models/yolov8s.pt", and the directory is actually "trained_model"
    normalized_path = model_path
    if normalized_path.startswith("trained_models/"):
        normalized_path = normalized_path.replace("trained_models/", "trained_model/", 1)
        resolved_normalized = os.path.join(workspace_root, normalized_path)
        if os.path.exists(resolved_normalized):
            return resolved_normalized

    # If it is just a filename and is not found under trained_model, try to search it
    # in trained_model by taking the basename
    basename = os.path.basename(model_path)
    resolved_basename = os.path.join(workspace_root, "trained_model", basename)
    if os.path.exists(resolved_basename):
        return resolved_basename

    # Fallback to the join of workspace_root and trained_model/model_path
    return os.path.abspath(os.path.join(workspace_root, "trained_model", model_path))
