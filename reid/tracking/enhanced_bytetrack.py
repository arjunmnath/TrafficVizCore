import numpy as np
import torch
from typing import Any, List, Optional, Set, cast

from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils.stracks import parse_bboxes

from .association.base import IoUCost
from .association.appearance import AppearanceCost
from .association.fusion import CostFusion
from .quality import TrackQuality


class EnhancedSTrack(STrack):
    """STrack representation extended with appearance embeddings, recall support, and quality metrics."""

    def __init__(self, xywh: np.ndarray[Any, Any], score: float, cls: Any, feat: Optional[np.ndarray[Any, Any]] = None):
        """Initialize track with appearance features and quality statistics if available.

        Args:
            xywh (np.ndarray): Bounding box in center x, center y, width, height format.
            score (float): Detection confidence score.
            cls (Any): Class label.
            feat (np.ndarray, optional): Appearance feature vector.
        """
        super().__init__(xywh, score, cls)
        self.curr_feat: Optional[np.ndarray[Any, Any]] = None
        self.smooth_feat: Optional[np.ndarray[Any, Any]] = None
        self.alpha: float = 0.9  # Default blending factor for exponential moving average

        # Quality running metrics (O(1) space tracking)
        self.total_conf: float = float(score)
        self.num_detections: int = 1
        self.consecutive_associations: int = 1
        self.max_consecutive_associations: int = 1
        self.recall_count: int = 0
        self.lost_recovered_count: int = 0
        self.embedding_stability_sum: float = 0.0

        if feat is not None and not np.all(feat == 0.0):
            self.update_features(feat)

    def update_features(self, feat: np.ndarray[Any, Any]) -> None:
        """Smooth the appearance feature vector using exponential moving average and track stability.

        Args:
            feat (np.ndarray): The new appearance feature vector.
        """
        from ultralytics.trackers.utils.reid import smooth_feature

        # Accumulate embedding stability
        if feat is not None:
            if self.smooth_feat is not None:
                norm1 = np.linalg.norm(feat)
                norm2 = np.linalg.norm(self.smooth_feat)
                if norm1 > 0.0 and norm2 > 0.0:
                    sim = float(np.dot(feat, self.smooth_feat) / (norm1 * norm2))
                    self.embedding_stability_sum += sim
                else:
                    self.embedding_stability_sum += 1.0
            else:
                self.embedding_stability_sum += 1.0

        curr, smooth = smooth_feature(feat, self.smooth_feat, self.alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update track state, smooth appearance features, and update quality metrics upon match.

        Args:
            new_track (STrack): The matched detection track.
            frame_id (int): ID of the current frame.
        """
        if hasattr(new_track, "curr_feat") and getattr(new_track, "curr_feat") is not None:
            self.update_features(getattr(new_track, "curr_feat"))
        super().update(new_track, frame_id)

        # Update quality metrics
        self.total_conf += float(new_track.score)
        self.num_detections += 1
        self.consecutive_associations += 1
        self.max_consecutive_associations = max(self.max_consecutive_associations, self.consecutive_associations)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """Reactivate track state, smooth appearance features, and update quality metrics.

        Args:
            new_track (STrack): The matching detection.
            frame_id (int): ID of the current frame.
            new_id (bool): Whether to allocate a new track ID.
        """
        if hasattr(new_track, "curr_feat") and getattr(new_track, "curr_feat") is not None:
            self.update_features(getattr(new_track, "curr_feat"))
        super().re_activate(new_track, frame_id, new_id)

        # Update quality metrics
        self.lost_recovered_count += 1
        self.consecutive_associations = 1
        self.max_consecutive_associations = max(self.max_consecutive_associations, self.consecutive_associations)
        self.total_conf += float(new_track.score)
        self.num_detections += 1

    def recall(self, new_track: STrack, frame_id: int) -> None:
        """Recall a previously lost track using new detection data.

        Args:
            new_track (STrack): The matching detection.
            frame_id (int): ID of the current frame.
        """
        # Update Kalman filter state
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.convert_coords(new_track.tlwh)
            )

        # Restore activation state
        self.state = TrackState.Tracked
        self.is_activated = True

        # Restore frame id
        self.frame_id = frame_id

        # Update appearance features
        if hasattr(new_track, "curr_feat") and getattr(new_track, "curr_feat") is not None:
            self.update_features(getattr(new_track, "curr_feat"))

        # Restore metadata
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        self.tracklet_len += 1

        # Update quality metrics
        self.recall_count += 1
        self.lost_recovered_count += 1
        self.consecutive_associations = 1
        self.max_consecutive_associations = max(self.max_consecutive_associations, self.consecutive_associations)
        self.total_conf += float(new_track.score)
        self.num_detections += 1

    def mark_lost(self) -> None:
        """Mark the track as lost."""
        super().mark_lost()
        self.consecutive_associations = 0

    @property
    def quality_score(self) -> float:
        """Compute the normalized track quality score."""
        return TrackQuality.evaluate(self)


class EnhancedByteTracker(BYTETracker):
    """An enhanced ByteTrack tracker incorporating appearance-based association cost and event-dispatching."""

    track_class = EnhancedSTrack

    def __init__(self, args: Any):
        """Initialize custom tracker and its modular cost/fusion components.

        Args:
            args (Any): Track settings configuration namespace.
        """
        super().__init__(args)  # type: ignore[no-untyped-call]

        # Load ReID feature encoder (reusing standard build_encoder helper)
        from ultralytics.trackers.utils.reid import build_encoder

        self.encoder = build_encoder(
            with_reid=False,
            model=getattr(args, "model", "auto"),
            device=getattr(args, "device", None),
        )

        # Initialize modular cost handlers
        self.iou_cost_module = IoUCost()
        self.appearance_cost_module = AppearanceCost()

        # Read cost weights from args
        self.iou_weight = getattr(args, "iou_weight", 0.7)
        self.appearance_weight = getattr(args, "appearance_weight", 0.3)

        # Quality configuration
        self.quality_enabled = getattr(args, "quality_enabled", True)
        self.quality_weights = getattr(args, "quality_weights", {
            "detector_confidence": 0.3,
            "track_duration": 0.2,
            "embedding_stability": 0.3,
            "association_consistency": 0.2,
        })

        # Event configuration
        self.lifecycle_events_enabled = getattr(args, "lifecycle_events_enabled", True)
        self._listeners: List[Any] = []

        # Tracking sets to compute frame transitions
        self._previous_active_ids: Set[int] = set()
        self._previous_removed_ids: Set[int] = set()
        self._created_or_recalled_ids_this_frame: Set[int] = set()
        self.current_timestamp: float = 0.0

    @classmethod
    def setup_predictor(cls, predictor: Any) -> None:
        """Register detection pre-hook to extract features when using model='auto'.

        This classmethod is dynamically called by the Ultralytics tracking pipeline.

        Args:
            predictor (Any): The YOLO predictor instance.
        """
        if not hasattr(predictor, "trackers") or not predictor.trackers:
            return

        tracker = predictor.trackers[0]
        cfg = tracker.args

        # Register hook to extract intermediate feature maps for 'auto' ReID mode
        if getattr(cfg, "with_reid", False) and getattr(cfg, "model", None) == "auto":
            from ultralytics.nn.modules.head import Detect

            model = getattr(predictor, "model", None)
            model_layers = getattr(model, "model", None)
            if isinstance(model_layers, torch.nn.Module):
                model_seq = getattr(model_layers, "model", None)
                if model_seq is not None and hasattr(model_seq, "__getitem__"):
                    last_layer = cast(Any, model_seq)[-1]
                    if isinstance(last_layer, Detect) and not getattr(last_layer, "end2end", False):
                        # Register hook to extract input of Detect layer
                        def pre_hook(module: Any, input_tensor: Any) -> None:
                            predictor._feats = list(input_tensor[0])  # unroll input tensors to prevent mutation

                        predictor._hook = last_layer.register_forward_pre_hook(
                            pre_hook
                        )
                        predictor._feats = None  # Initialize _feats state

    def extract_features(self, frame: Optional[np.ndarray[Any, Any]], bboxes: np.ndarray[Any, Any]) -> Optional[List[np.ndarray[Any, Any]]]:
        """Extract appearance embeddings for detection bounding boxes.

        Args:
            frame (np.ndarray): Image frame or pre-hook/manual extracted features.
            bboxes (np.ndarray): Detected boxes in xywh format.

        Returns:
            Optional[List[np.ndarray]]: List of appearance embeddings, or None.
        """
        # If frame is already pre-extracted features passed manually:
        if frame is not None:
            if isinstance(frame, np.ndarray) and frame.ndim == 2:
                return [f for f in frame]
            elif isinstance(frame, list):
                return frame

        if self.encoder is not None and frame is not None and len(bboxes) > 0:
            return self.encoder(frame, bboxes)  # type: ignore[no-any-return]
        return None

    def init_track(self, results: Any, img: Optional[np.ndarray[Any, Any]] = None) -> List[STrack]:
        """Initialize STrack detections with extracted feature embeddings.

        Args:
            results (Any): YOLO detections list.
            img (Optional[np.ndarray]): Input frame or pre-hook/manual features.

        Returns:
            List[STrack]: Detections represented as STrack instances.
        """
        if len(results) == 0:
            return []

        bboxes = parse_bboxes(results)
        features = self.extract_features(img, bboxes)

        tracks = []
        for i, (xywh, s, c) in enumerate(zip(bboxes, results.conf, results.cls)):
            feat = features[i] if features is not None and i < len(features) else None
            track = self.track_class(xywh, s, c, feat)
            track.alpha = getattr(self.args, "ema_alpha", 0.9)
            tracks.append(track)
        return cast(List[STrack], tracks)

    def get_dists(
        self, tracks: List[STrack], detections: List[STrack]
    ) -> np.ndarray[Any, Any]:
        """Compute the combined cost matrix using custom weights and CostFusion.

        Args:
            tracks (List[STrack]): List of current active tracks.
            detections (List[STrack]): List of new detections.

        Returns:
            np.ndarray: Fused association cost matrix.
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        # 1. Compute individual component cost matrices
        iou_cost = self.iou_cost_module.compute(tracks, detections)
        appearance_cost = self.appearance_cost_module.compute(tracks, detections)

        # 2. Normalize appearance cost (from cosine distance range [0, 2] to [0, 1])
        appearance_cost = appearance_cost / 2.0

        # Check where appearance features are missing and fall back to pure iou_cost
        track_has_no_feat = np.array([getattr(t, "smooth_feat", None) is None for t in tracks])
        det_has_no_feat = np.array([getattr(d, "curr_feat", None) is None for d in detections])

        if track_has_no_feat.any() or det_has_no_feat.any():
            missing_mask = track_has_no_feat[:, None] | det_has_no_feat[None, :]
        else:
            missing_mask = None

        # 3. Combine matrices using CostFusion module
        costs = {"iou": iou_cost, "appearance": appearance_cost}
        weights = {"iou": self.iou_weight, "appearance": self.appearance_weight}

        # Apply official score-appearance-IoU fusion (alpha weight for detection score)
        fused_cost = CostFusion.combine(
            costs=costs,
            weights=weights,
            detections=detections,
            alpha=1.0 - self.appearance_weight,
            missing_mask=missing_mask,
            fuse_score=self.args.fuse_score,
        )

        return fused_cost

    def subscribe(self, callback: Any) -> None:
        """Register a callback to receive tracker events."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unsubscribe(self, callback: Any) -> None:
        """Unregister a callback from tracker events."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def emit_event(self, event_type: str, track_id: int, frame_id: int, timestamp: float, track: Any) -> None:
        """Dispatch an event to all subscribed observers."""
        if not getattr(self.args, "lifecycle_events_enabled", True):
            return
        for listener in self._listeners:
            try:
                listener(event_type, track_id, frame_id, timestamp, track)
            except Exception:
                pass

    def _init_new_tracks(
        self,
        u_detection: List[int],
        detections: List[STrack],
        activated: List[STrack],
        refind: List[STrack] | None = None,
    ) -> None:
        """Activate new tracks from unmatched detections.

        Args:
            u_detection (List[int]): Unmatched detection indices.
            detections (List[STrack]): List of all detections.
            activated (List[STrack]): Active tracks list to append activated tracks.
            refind (List[STrack], optional): Refind tracks list.
        """
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

            self._created_or_recalled_ids_this_frame.add(track.track_id)

            self.emit_event(
                "created",
                track.track_id,
                self.frame_id,
                self.current_timestamp,
                track,
            )

    def update(
        self,
        results: Any,
        img: Optional[np.ndarray[Any, Any]] = None,
        feats: Optional[np.ndarray[Any, Any]] = None,
        **kwargs: Any,
    ) -> np.ndarray[Any, Any]:
        """Update tracker with frame detections, run association, and emit events.

        Args:
            results (Any): YOLO detections list.
            img (Optional[np.ndarray]): Input frame or pre-hook/manual features.
            feats (Optional[np.ndarray]): Appearance embeddings.

        Returns:
            np.ndarray: Tracked targets array.
        """
        self.current_timestamp = kwargs.get("timestamp", 0.0)
        self._created_or_recalled_ids_this_frame = set()

        output = super().update(results, img, feats, **kwargs)

        # Detect and emit transitions
        current_active_ids = {t.track_id for t in self.tracked_stracks}
        current_removed_ids = {t.track_id for t in self.removed_stracks}

        # 1. TrackUpdated: Active tracks updated
        for track in self.tracked_stracks:
            if track.track_id in self._created_or_recalled_ids_this_frame:
                continue
            if track.frame_id == self.frame_id:
                self.emit_event(
                    "updated",
                    track.track_id,
                    self.frame_id,
                    self.current_timestamp,
                    track,
                )

        # 2. TrackLost: Active -> Lost
        for track in self.lost_stracks:
            if track.track_id in self._previous_active_ids:
                self.emit_event(
                    "lost",
                    track.track_id,
                    self.frame_id,
                    self.current_timestamp,
                    track,
                )

        # 3. TrackTerminated: -> Removed
        for track in self.removed_stracks:
            if track.track_id not in self._previous_removed_ids:
                self.emit_event(
                    "terminated",
                    track.track_id,
                    self.frame_id,
                    self.current_timestamp,
                    track,
                )

        self._previous_active_ids = current_active_ids
        self._previous_removed_ids = current_removed_ids

        return output

    def reset(self) -> None:
        """Reset tracker persistent states."""
        super().reset()  # type: ignore[no-untyped-call]
        if hasattr(self, "_previous_active_ids"):
            self._previous_active_ids.clear()
        if hasattr(self, "_previous_removed_ids"):
            self._previous_removed_ids.clear()
        if hasattr(self, "_created_or_recalled_ids_this_frame"):
            self._created_or_recalled_ids_this_frame.clear()
        self.current_timestamp = 0.0
