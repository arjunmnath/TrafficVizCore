import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from tracking.domain.metadata import TrackMetadata
from tracking.domain.interfaces import SizeModel
from tracking.domain.trajectory import PiecewiseTrajectory


class ConfidenceModel:
    """Handles mapping between timestamps/frames and detector confidence scores."""

    def __init__(
        self,
        timestamps: List[float],
        confidences: List[float],
        frames: Optional[List[int]] = None,
    ):
        if len(timestamps) != len(confidences):
            raise ValueError("Timestamps and confidences lists must have the same length.")
        if not timestamps:
            raise ValueError("ConfidenceModel must have at least one timestamp/confidence entry.")

        if frames is not None and len(frames) == len(timestamps):
            sorted_tuples = sorted(zip(timestamps, confidences, frames), key=lambda x: x[0])
            self.timestamps = [p[0] for p in sorted_tuples]
            self.confidences = [float(p[1]) for p in sorted_tuples]
            self.frames = [int(p[2]) for p in sorted_tuples]
        else:
            sorted_tuples = sorted(zip(timestamps, confidences), key=lambda x: x[0])
            self.timestamps = [p[0] for p in sorted_tuples]
            self.confidences = [float(p[1]) for p in sorted_tuples]
            self.frames = []

    def confidence(self, timestamp: float) -> float:
        """Map timestamp to detector confidence score via linear interpolation, clipped to [0, 1]."""
        if len(self.timestamps) == 1:
            val = self.confidences[0]
        else:
            val = float(np.interp(timestamp, self.timestamps, self.confidences))
        return float(np.clip(val, 0.0, 1.0))

    def __call__(self, timestamp: float) -> float:
        return self.confidence(timestamp)

    def serialize(self) -> Dict[str, Any]:
        res = {
            "timestamps": [float(t) for t in self.timestamps],
            "confidences": [float(c) for c in self.confidences],
        }
        if self.frames:
            res["frames"] = [int(f) for f in self.frames]
        return res


class TimeModel:
    """Handles mapping between frame numbers and timestamps, supporting variable FPS."""

    def __init__(self, frames: List[int], timestamps: List[float]):
        if len(frames) != len(timestamps):
            raise ValueError("Frames and timestamps lists must have the same length.")
        if not frames:
            raise ValueError("TimeModel must have at least one frame/timestamp entry.")

        # Ensure sorted order
        sorted_pairs = sorted(zip(frames, timestamps), key=lambda x: x[0])
        self.frames = [p[0] for p in sorted_pairs]
        self.timestamps = [p[1] for p in sorted_pairs]

    def frame_to_timestamp(self, frame: int) -> float:
        """Map frame index to timestamp via linear interpolation."""
        if len(self.frames) == 1:
            return self.timestamps[0]
        return float(np.interp(frame, self.frames, self.timestamps))

    def timestamp_to_frame(self, timestamp: float) -> int:
        """Map timestamp to frame index via linear interpolation."""
        if len(self.timestamps) == 1:
            return self.frames[0]
        return int(round(np.interp(timestamp, self.timestamps, self.frames)))

    def serialize(self) -> Dict[str, Any]:
        return {
            "frames": [int(f) for f in self.frames],
            "timestamps": [float(t) for t in self.timestamps],
        }


class Statistics:
    """Computes and holds aggregate statistics of a track's trajectory."""

    def __init__(
        self,
        avg_speed: float = 0.0,
        max_speed: float = 0.0,
        total_distance: float = 0.0,
        avg_acceleration: float = 0.0,
    ):
        self.avg_speed = avg_speed
        self.max_speed = max_speed
        self.total_distance = total_distance
        self.avg_acceleration = avg_acceleration

    @classmethod
    def compute(cls, trajectory: PiecewiseTrajectory) -> "Statistics":
        """Compute statistics by sampling the continuous trajectory."""
        t0, t1 = trajectory.t0, trajectory.t1
        if t1 <= t0:
            return cls()

        # Sample 100 points along the trajectory
        times = np.linspace(t0, t1, 100)
        speeds = []
        total_dist = 0.0
        prev_pos = None

        accel_mags = []
        prev_vel = None
        prev_t = None

        for t in times:
            pos = trajectory.position(t)
            vel = trajectory.velocity(t)
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            speeds.append(speed)

            if prev_pos is not None:
                total_dist += math.sqrt((pos[0] - prev_pos[0]) ** 2 + (pos[1] - prev_pos[1]) ** 2)
            prev_pos = pos

            if prev_vel is not None and prev_t is not None and t > prev_t:
                dt = t - prev_t
                ax = (vel[0] - prev_vel[0]) / dt
                ay = (vel[1] - prev_vel[1]) / dt
                accel_mags.append(math.sqrt(ax**2 + ay**2))

            prev_vel = vel
            prev_t = t

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        avg_accel = float(np.mean(accel_mags)) if accel_mags else 0.0

        return cls(
            avg_speed=avg_speed,
            max_speed=max_speed,
            total_distance=total_dist,
            avg_acceleration=avg_accel,
        )

    def serialize(self) -> Dict[str, Any]:
        return {
            "avg_speed": float(self.avg_speed),
            "max_speed": float(self.max_speed),
            "total_distance": float(self.total_distance),
            "avg_acceleration": float(self.avg_acceleration),
        }


class CompressedTrack:
    """Primary domain representation of a compressed trajectory track."""

    def __init__(
        self,
        metadata: TrackMetadata,
        time_model: TimeModel,
        size_model: SizeModel,
        trajectory: PiecewiseTrajectory,
        statistics: Optional[Statistics] = None,
        confidence_model: Optional[ConfidenceModel] = None,
    ):
        self.metadata = metadata
        self.time_model = time_model
        self.size_model = size_model
        self.trajectory = trajectory
        self.statistics = statistics or Statistics.compute(trajectory)
        if confidence_model is not None:
            self.confidence_model = confidence_model
        else:
            self.confidence_model = ConfidenceModel(
                timestamps=self.time_model.timestamps,
                confidences=[1.0] * len(self.time_model.timestamps),
                frames=self.time_model.frames,
            )

    def position(self, t: float) -> Tuple[float, float]:
        """Continuous 2D position (cx, cy) at timestamp t."""
        return self.trajectory.position(t)

    def velocity(self, t: float) -> Tuple[float, float]:
        """Continuous 2D velocity (vx, vy) at timestamp t."""
        return self.trajectory.velocity(t)

    def direction(self, t: float) -> float:
        """Continuous direction/heading (radians) at timestamp t."""
        return self.trajectory.direction(t)

    def width(self, t: float) -> float:
        """Continuous width at timestamp t."""
        return self.size_model.width(t)

    def height(self, t: float) -> float:
        """Continuous height at timestamp t."""
        return self.size_model.height(t)

    def confidence(self, t: float) -> float:
        """Continuous detector confidence at timestamp t."""
        return self.confidence_model.confidence(t)

    def acceleration(self, t: float) -> Tuple[float, float]:
        """Continuous 2D acceleration (ax, ay) at timestamp t (numerical derivative)."""
        dt = 1e-3
        v1 = self.velocity(t - dt)
        v2 = self.velocity(t + dt)
        return (v2[0] - v1[0]) / (2 * dt), (v2[1] - v1[1]) / (2 * dt)

    def heading(self, t: float) -> float:
        """Continuous heading angle (radians) at timestamp t (alias of direction)."""
        return self.direction(t)

    def curvature(self, t: float) -> float:
        """Continuous trajectory curvature at timestamp t."""
        vx, vy = self.velocity(t)
        ax, ay = self.acceleration(t)
        speed = math.sqrt(vx**2 + vy**2)
        if speed < 1e-6:
            return 0.0
        return abs(vx * ay - vy * ax) / (speed**3)


def get_cleared_detection_frame(
    track: Any,
    lambda_param: float = 1.0,
    mu_param: float = 0.5,
    dt: float = 1e-3,
    w_conf: float = 1.0,
) -> Tuple[int, float, Tuple[float, float, float, float], float]:
    """Finds the optimal 'cleared' detection frame from a track.

    Multi-heuristic scoring accounts for:
    1. Detector Confidence (C): Higher confidence detections yield clearer crops.
    2. Bounding Box Area (A = w * h): Resolution/size of detection crop.
    3. Differential Relative Size Change Penalty: Penalizes rapid/unstable size jumps (|dw/dt|/w + |dh/dt|/h).
    4. Aspect Ratio Stability Penalty: Penalizes sudden deviations from track's median aspect ratio (r = w/h).

    Score formula:
        rel_dw = |dw/dt| / max(w, 1.0)
        rel_dh = |dh/dt| / max(h, 1.0)
        size_penalty = 1.0 + lambda_param * (rel_dw + rel_dh)

        aspect_dev = |r - median_r| / max(median_r, 1e-6)
        aspect_penalty = 1.0 + mu_param * aspect_dev

        Score = (Area * (C ** w_conf)) / (size_penalty * aspect_penalty)

    Args:
        track: The track object. Supports CompressedTrack, TerminatedTrack,
               or any object that has a 'compressed_track' or 'history' attribute/dict.
        lambda_param: Sensitivity parameter weighting relative size derivative penalty.
        mu_param: Sensitivity parameter weighting aspect ratio deviation penalty.
        dt: The time delta used for numerical differentiation (in seconds).
        w_conf: Exponent weight for detector confidence term.

    Returns:
        Tuple of (best_frame_id, best_timestamp, best_bbox, best_score)
        where best_bbox is (x1, y1, x2, y2).
    """
    actual_track = track
    if hasattr(track, "compressed_track") and track.compressed_track is not None:
        actual_track = track.compressed_track

    if isinstance(actual_track, CompressedTrack):
        timestamps = actual_track.time_model.timestamps
        frames = actual_track.time_model.frames

        if not frames or not timestamps:
            raise ValueError("CompressedTrack has no frames or timestamps.")

        # Compute median aspect ratio across track
        aspect_ratios = []
        for t in timestamps:
            w = actual_track.width(t)
            h = actual_track.height(t)
            aspect_ratios.append(w / max(h, 1e-6))
        median_r = float(np.median(aspect_ratios)) if aspect_ratios else 1.0

        best_frame = None
        best_time = None
        best_bbox = None
        best_score = -1.0

        t0 = actual_track.metadata.start_timestamp
        t1 = actual_track.metadata.end_timestamp

        for frame, t in zip(frames, timestamps):
            w = actual_track.width(t)
            h = actual_track.height(t)
            area = w * h
            r = w / max(h, 1e-6)
            conf = actual_track.confidence(t)

            t_plus = min(t1, t + dt)
            t_minus = max(t0, t - dt)
            denom = t_plus - t_minus

            if denom > 1e-6:
                dw = (actual_track.width(t_plus) - actual_track.width(t_minus)) / denom
                dh = (actual_track.height(t_plus) - actual_track.height(t_minus)) / denom
            else:
                dw = 0.0
                dh = 0.0

            rel_dw = abs(dw) / max(w, 1.0)
            rel_dh = abs(dh) / max(h, 1.0)
            size_penalty = 1.0 + lambda_param * (rel_dw + rel_dh)

            aspect_dev = abs(r - median_r) / max(median_r, 1e-6)
            aspect_penalty = 1.0 + mu_param * aspect_dev

            score = (area * (conf**w_conf)) / (size_penalty * aspect_penalty)

            if score > best_score:
                best_score = score
                best_frame = frame
                best_time = t
                cx, cy = actual_track.position(t)
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                best_bbox = (x1, y1, x2, y2)

        if best_frame is None:
            raise ValueError("CompressedTrack has no frames or timestamps.")

        return best_frame, best_time, best_bbox, best_score

    # Fallback to history sequence format (e.g. dict or object with frames, timestamps, bboxes, confidences/scores)
    frames = None
    timestamps = None
    bboxes = None
    confidences = None

    history = getattr(actual_track, "history", None)
    if history is None and isinstance(actual_track, dict):
        history = actual_track

    if isinstance(history, dict):
        frames = history.get("frames", [])
        timestamps = history.get("timestamps", [])
        bboxes = history.get("bboxes", [])
        confidences = (
            history.get("confidences") or history.get("scores") or history.get("confidence")
        )
    elif (
        hasattr(actual_track, "frames")
        and hasattr(actual_track, "timestamps")
        and hasattr(actual_track, "bboxes")
    ):
        frames = getattr(actual_track, "frames")
        timestamps = getattr(actual_track, "timestamps")
        bboxes = getattr(actual_track, "bboxes")
        confidences = (
            getattr(actual_track, "confidences", None)
            or getattr(actual_track, "scores", None)
            or getattr(actual_track, "confidence", None)
        )

    if (
        not frames
        or not timestamps
        or not bboxes
        or len(frames) != len(timestamps)
        or len(frames) != len(bboxes)
    ):
        raise ValueError("Track lacks standard frame, timestamp, or bounding box history.")

    n = len(timestamps)
    if confidences is None or len(confidences) != n:
        confidences = [1.0] * n

    aspect_ratios = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect_ratios.append(w / max(h, 1e-6))
    median_r = float(np.median(aspect_ratios)) if aspect_ratios else 1.0

    best_frame = None
    best_time = None
    best_bbox = None
    best_score = -1.0

    for i in range(n):
        frame = frames[i]
        t = timestamps[i]
        bbox = bboxes[i]
        conf = float(confidences[i])
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h
        r = w / max(h, 1e-6)

        if n < 2:
            dw = 0.0
            dh = 0.0
        else:
            if i == 0:
                dt_diff = timestamps[1] - timestamps[0]
                if dt_diff > 1e-6:
                    w_next = bboxes[1][2] - bboxes[1][0]
                    h_next = bboxes[1][3] - bboxes[1][1]
                    dw = (w_next - w) / dt_diff
                    dh = (h_next - h) / dt_diff
                else:
                    dw = 0.0
                    dh = 0.0
            elif i == n - 1:
                dt_diff = timestamps[-1] - timestamps[-2]
                if dt_diff > 1e-6:
                    w_prev = bboxes[-2][2] - bboxes[-2][0]
                    h_prev = bboxes[-2][3] - bboxes[-2][1]
                    dw = (w - w_prev) / dt_diff
                    dh = (h - h_prev) / dt_diff
                else:
                    dw = 0.0
                    dh = 0.0
            else:
                dt_diff = timestamps[i + 1] - timestamps[i - 1]
                if dt_diff > 1e-6:
                    w_next = bboxes[i + 1][2] - bboxes[i + 1][0]
                    w_prev = bboxes[i - 1][2] - bboxes[i - 1][0]
                    h_next = bboxes[i + 1][3] - bboxes[i + 1][1]
                    h_prev = bboxes[i - 1][3] - bboxes[i - 1][1]
                    dw = (w_next - w_prev) / dt_diff
                    dh = (h_next - h_prev) / dt_diff
                else:
                    dw = 0.0
                    dh = 0.0

        rel_dw = abs(dw) / max(w, 1.0)
        rel_dh = abs(dh) / max(h, 1.0)
        size_penalty = 1.0 + lambda_param * (rel_dw + rel_dh)

        aspect_dev = abs(r - median_r) / max(median_r, 1e-6)
        aspect_penalty = 1.0 + mu_param * aspect_dev

        score = (area * (conf**w_conf)) / (size_penalty * aspect_penalty)

        if score > best_score:
            best_score = score
            best_frame = frame
            best_time = t
            best_bbox = tuple(bbox)

    if best_frame is None:
        raise ValueError("No frames found in track history.")

    return best_frame, best_time, best_bbox, best_score

