import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

from ultralytics.utils import IterableSimpleNamespace
from .tracker_factory import TrackerFactory


class Detections:
    """Mock detections object compatible with Ultralytics tracker requirements."""

    def __init__(
        self, xywh: np.ndarray[Any, Any], conf: np.ndarray[Any, Any], cls: np.ndarray[Any, Any]
    ):
        """Initialize mock detections.

        Args:
            xywh (np.ndarray): Center x, center y, width, height format bounding boxes (N, 4).
            conf (np.ndarray): Confidence scores (N,).
            cls (np.ndarray): Class labels (N,).
        """
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.xywh)

    def __getitem__(self, idx: Any) -> "Detections":
        return Detections(self.xywh[idx], self.conf[idx], self.cls[idx])


class Tracker:
    """Manual tracking API wrapping custom and standard ByteTrack/EnhancedByteTracker algorithms.

    Maintains an internal mapping of track_id -> latest embedding so that the
    on_track_terminated hook can access the appearance vector regardless of
    which tracker backend (BYTETracker vs EnhancedByteTracker) is used.
    """

    def __init__(self, tracker_config: Union[str, Path, dict[Any, Any]]):
        """Initialize the tracker with a YAML configuration file or dict settings.

        Args:
            tracker_config (Union[str, Path, dict]): Tracker configuration settings.
        """
        if isinstance(tracker_config, (str, Path)):
            path = str(tracker_config)
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), "config", os.path.basename(path))

            with open(path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}
        elif isinstance(tracker_config, dict):
            cfg_dict = tracker_config
        else:
            raise TypeError("tracker_config must be a path to a YAML file or a dictionary.")

        # Map to IterableSimpleNamespace as required by Ultralytics trackers
        self.args = IterableSimpleNamespace(**cfg_dict)
        self.args.device = "cpu"  # Keep tracking logic on CPU

        # Create tracker backend via Factory
        self.tracker = TrackerFactory.create(self.args)

        # Internal store: track_id -> latest appearance embedding
        self.track_embeddings: Dict[int, np.ndarray[Any, Any]] = {}

        # Internal store: track_id -> dict of track details/history
        self.track_history: Dict[int, Dict[str, Any]] = {}
        self._hook_wired = False

        # Subscribe to tracker lifecycle events if enabled and supported
        if getattr(self.args, "lifecycle_events_enabled", True) and hasattr(
            self.tracker, "subscribe"
        ):
            setattr(self.tracker, "subscribe", getattr(self.tracker, "subscribe"))
            self.tracker.subscribe(self._handle_tracker_event)

    def _handle_tracker_event(
        self, event_type: str, track_id: int, frame_count: int, timestamp: float, track: Any
    ) -> None:
        """Handle incoming tracker lifecycle events to update internal buffers."""
        if event_type in ("created", "updated", "recalled"):
            bbox = track.xyxy.tolist()  # [x1, y1, x2, y2]
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "start_frame": frame_count,
                    "start_timestamp": timestamp,
                    "end_frame": frame_count,
                    "end_timestamp": timestamp,
                    "bboxes": [bbox],
                    "frames": [frame_count],
                    "timestamps": [timestamp],
                }
            else:
                hist = self.track_history[track_id]
                hist["end_frame"] = frame_count
                hist["end_timestamp"] = timestamp
                hist["bboxes"].append(bbox)
                hist["frames"].append(frame_count)
                hist["timestamps"].append(timestamp)

            # Store the appearance embedding
            smooth_feat = None
            if hasattr(track, "smooth_feat") and track.smooth_feat is not None:
                smooth_feat = track.smooth_feat
            elif hasattr(track, "curr_feat") and track.curr_feat is not None:
                smooth_feat = track.curr_feat

            if smooth_feat is not None:
                self.track_embeddings[track_id] = smooth_feat.copy()

        elif event_type == "terminated":
            # Augment track with last known embedding and history
            setattr(track, "embedding", self.track_embeddings.pop(track.track_id, None))
            setattr(track, "history", self.track_history.pop(track.track_id, None))
            try:
                self.on_track_terminated(track)
            except ValueError:
                pass

    def on_track_terminated(self, track: Any) -> None:
        """Hook called when a track is terminated/removed.

        The track object is augmented with an `embedding` attribute containing
        the last known appearance vector from the internal store.

        Can be overridden by subclasses or set as a callback function on instance.
        """
        raise ValueError("on_track_terminated hook is not attached.")

    def update(
        self,
        boxes: np.ndarray[Any, Any],
        scores: np.ndarray[Any, Any],
        classes: np.ndarray[Any, Any],
        features: Optional[np.ndarray[Any, Any]] = None,
        frame_count: int = 0,
        timestamp: float = 0.0,
    ) -> np.ndarray[Any, Any]:
        """Manually update track states using new frame detections and features.

        Args:
            boxes (np.ndarray): Bounding boxes in xyxy format (N, 4).
            scores (np.ndarray): Confidence scores (N,).
            classes (np.ndarray): Class label indices (N,).
            features (np.ndarray, optional): Appearance embeddings (N, D).
            frame_count (int): Frame number index.
            timestamp (float): Current elapsed timestamp in stream.

        Returns:
            np.ndarray: Tracked targets array of shape (M, 8) where each row is:
                        [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
        """
        prev_removed_ids = {t.track_id for t in getattr(self.tracker, "removed_stracks", [])}
        events_enabled = getattr(self.args, "lifecycle_events_enabled", True) and hasattr(
            self.tracker, "subscribe"
        )

        if len(boxes) == 0:
            # Handle frame with zero detections
            empty_detections = Detections(
                xywh=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.int32),
            )
            if events_enabled:
                tracks = self.tracker.update(
                    empty_detections, feats=np.empty((0, 0)), timestamp=timestamp
                )
            else:
                tracks = self.tracker.update(empty_detections, feats=np.empty((0, 0)))
        else:
            # Convert detections from xyxy to center x, center y, width, height (xywh)
            xywh = np.empty_like(boxes, dtype=np.float32)
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0  # Center x
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0  # Center y
            xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # Width
            xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # Height

            # Wrap into mock detection object
            results = Detections(xywh=xywh, conf=scores, cls=classes)

            # Call tracker update method, forwarding optional pre-extracted features
            if events_enabled:
                tracks = self.tracker.update(results, feats=features, timestamp=timestamp)
            else:
                tracks = self.tracker.update(results, feats=features)

        # Update the internal track_id -> embedding mapping and track history from active tracks (LEGACY FALLBACK)
        if not events_enabled:
            if len(tracks) > 0:
                for t in tracks:
                    track_id = int(t[4])
                    bbox = t[0:4].tolist()  # [x1, y1, x2, y2]

                    if track_id not in self.track_history:
                        self.track_history[track_id] = {
                            "start_frame": frame_count,
                            "start_timestamp": timestamp,
                            "end_frame": frame_count,
                            "end_timestamp": timestamp,
                            "bboxes": [bbox],
                            "frames": [frame_count],
                            "timestamps": [timestamp],
                        }
                    else:
                        hist = self.track_history[track_id]
                        hist["end_frame"] = frame_count
                        hist["end_timestamp"] = timestamp
                        hist["bboxes"].append(bbox)
                        hist["frames"].append(frame_count)
                        hist["timestamps"].append(timestamp)

                    # Try to get the smoothed features from the tracker's internal strack object if it supports appearance ReID
                    smooth_feat = None
                    for strack in getattr(self.tracker, "tracked_stracks", []):
                        if strack.track_id == track_id:
                            if (
                                hasattr(strack, "smooth_feat")
                                and getattr(strack, "smooth_feat") is not None
                            ):
                                smooth_feat = getattr(strack, "smooth_feat")
                            elif (
                                hasattr(strack, "curr_feat")
                                and getattr(strack, "curr_feat") is not None
                            ):
                                smooth_feat = getattr(strack, "curr_feat")
                            break
                    if smooth_feat is not None:
                        self.track_embeddings[track_id] = smooth_feat.copy()
                    elif features is not None:
                        det_idx = int(t[7])
                        if det_idx < len(features):
                            self.track_embeddings[track_id] = features[det_idx].copy()

            # Trigger hook for any newly terminated tracks
            newly_removed = [
                t
                for t in getattr(self.tracker, "removed_stracks", [])
                if t.track_id not in prev_removed_ids
            ]
            for track in newly_removed:
                # Augment the track object with the stored embedding and history
                setattr(track, "embedding", self.track_embeddings.pop(track.track_id, None))
                setattr(track, "history", self.track_history.pop(track.track_id, None))
                self.on_track_terminated(track)

        return tracks

    def reset(self) -> None:
        """Reset the tracker state and IDs."""
        self.tracker.reset()  # type: ignore[no-untyped-call]
        self.track_embeddings.clear()
        self.track_history.clear()

    def terminate_all_tracks(self) -> None:
        """Manually terminate all remaining active and lost tracks at the end of stream."""
        remaining_tracks = []
        if hasattr(self.tracker, "tracked_stracks"):
            remaining_tracks.extend(self.tracker.tracked_stracks)
        if hasattr(self.tracker, "lost_stracks"):
            remaining_tracks.extend(self.tracker.lost_stracks)

        for track in remaining_tracks:
            # Only terminate if we haven't already processed it
            embedding = self.track_embeddings.pop(track.track_id, None)
            if embedding is not None:
                setattr(track, "embedding", embedding)
                setattr(track, "history", self.track_history.pop(track.track_id, None))
                self.on_track_terminated(track)

    @property
    def hook_wired(self) -> bool:
        return self._hook_wired

    def set_hook_wired(self, hook_wired: bool) -> None:
        self._hook_wired = hook_wired
