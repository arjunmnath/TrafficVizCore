import math
import numpy as np
from typing import List, Tuple, Optional

from tracking.domain.metadata import TrackMetadata
from tracking.domain.interfaces import SegmentationStrategy, TrajectoryFitter, SizeModel
from tracking.domain.trajectory import PiecewiseTrajectory
from tracking.domain.size_models import LinearModel, ConstantModel, SplineModel
from tracking.domain.track import TimeModel, CompressedTrack, Statistics
from tracking.compression.segmentation import AdaptiveSegmentation
from tracking.compression.fitting import ConstantFitter, LinearFitter, SplineFitter


class CompressedTrackBuilder:
    """Builder pipeline that constructs a CompressedTrack from raw observations."""

    def __init__(self) -> None:
        self._metadata: Optional[TrackMetadata] = None
        self._frames: List[int] = []
        self._timestamps: List[float] = []
        self._bboxes: List[Tuple[float, float, float, float]] = []  # xyxy format

        # Strategies
        self._segmentation_strategy: SegmentationStrategy = AdaptiveSegmentation()
        self._fitter: Optional[TrajectoryFitter] = None  # None means use adaptive fitting

    def set_metadata(
        self,
        track_id: int,
        camera_id: str,
        class_label: str,
    ) -> "CompressedTrackBuilder":
        # Start and end details will be populated automatically from observations
        self._track_id = track_id
        self._camera_id = camera_id
        self._class_label = class_label
        return self

    def add_observation(
        self, frame: int, timestamp: float, bbox: Tuple[float, float, float, float]
    ) -> "CompressedTrackBuilder":
        self._frames.append(frame)
        self._timestamps.append(timestamp)
        self._bboxes.append(bbox)
        return self

    def add_observations(
        self,
        frames: List[int],
        timestamps: List[float],
        bboxes: List[Tuple[float, float, float, float]],
    ) -> "CompressedTrackBuilder":
        self._frames.extend(frames)
        self._timestamps.extend(timestamps)
        self._bboxes.extend(bboxes)
        return self

    def set_segmentation_strategy(self, strategy: SegmentationStrategy) -> "CompressedTrackBuilder":
        self._segmentation_strategy = strategy
        return self

    def set_fitter(self, fitter: TrajectoryFitter) -> "CompressedTrackBuilder":
        self._fitter = fitter
        return self

    def _estimate_velocities_and_headings(
        self, timestamps: List[float], positions: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        n = len(timestamps)
        velocities: List[Tuple[float, float]] = []
        headings: List[float] = []

        if n < 2:
            return [(0.0, 0.0)], [0.0]

        for i in range(n - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt < 1e-6:
                dt = 1e-6
            vx = (positions[i + 1][0] - positions[i][0]) / dt
            vy = (positions[i + 1][1] - positions[i][1]) / dt
            velocities.append((vx, vy))
            headings.append(math.atan2(vy, vx))

        # Replicate for the last point
        velocities.append(velocities[-1])
        headings.append(headings[-1])

        return velocities, headings

    def build(self) -> CompressedTrack:
        if not self._frames:
            raise ValueError("Cannot build CompressedTrack without observations.")

        # Ensure observation lists are sorted by timestamp
        zipped = sorted(zip(self._frames, self._timestamps, self._bboxes), key=lambda x: x[1])
        frames = [z[0] for z in zipped]
        timestamps = [z[1] for z in zipped]
        bboxes = [z[2] for z in zipped]

        n = len(timestamps)

        # 1. Compute center points, width, height from bounding boxes
        positions: List[Tuple[float, float]] = []
        widths: List[float] = []
        heights: List[float] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            positions.append((cx, cy))
            widths.append(w)
            heights.append(h)

        # 2. Build TrackMetadata
        metadata = TrackMetadata(
            track_id=self._track_id,
            camera_id=self._camera_id,
            class_label=self._class_label,
            start_frame=frames[0],
            end_frame=frames[-1],
            start_timestamp=timestamps[0],
            end_timestamp=timestamps[-1],
        )

        # 3. Build TimeModel
        time_model = TimeModel(frames, timestamps)

        # 4. Estimate raw velocities and headings for segmentation
        velocities, headings = self._estimate_velocities_and_headings(timestamps, positions)

        # 5. Segment trajectory
        segment_boundaries = self._segmentation_strategy.segment(
            timestamps, positions, velocities, headings
        )

        # 6. Fit each trajectory segment
        segments = []
        linear_fitter = LinearFitter()
        spline_fitter = SplineFitter()
        constant_fitter = ConstantFitter()

        for idx_start, idx_end in segment_boundaries:
            seg_times = timestamps[idx_start : idx_end + 1]
            seg_pos = positions[idx_start : idx_end + 1]

            if self._fitter is not None:
                # Use specified fitter
                seg = self._fitter.fit(seg_times, seg_pos)
            else:
                # Adaptive fitter selection
                seg_len = len(seg_times)

                # Compute variance to identify constant/stationary segments
                xs = [p[0] for p in seg_pos]
                ys = [p[1] for p in seg_pos]
                var = np.var(xs) + np.var(ys)

                if var < 1.0:  # Very stationary
                    seg = constant_fitter.fit(seg_times, seg_pos)
                elif seg_len <= 3:
                    seg = linear_fitter.fit(seg_times, seg_pos)
                else:
                    try:
                        seg = spline_fitter.fit(seg_times, seg_pos)
                    except Exception:
                        # Fallback to linear if spline fitting fails
                        seg = linear_fitter.fit(seg_times, seg_pos)

            segments.append(seg)

        trajectory = PiecewiseTrajectory(segments)

        # 7. Fit width and height model (adaptive: spline or linear)
        times_arr = np.array(timestamps)
        w_arr = np.array(widths)
        h_arr = np.array(heights)

        size_model: SizeModel
        if n >= 4:
            # Check for U-turns or turning in the trajectory headings
            is_turning = False
            if len(headings) >= 2:
                max_heading_diff = 0.0
                for i in range(len(headings)):
                    for j in range(i + 1, len(headings)):
                        diff = abs((headings[i] - headings[j] + math.pi) % (2 * math.pi) - math.pi)
                        if diff > max_heading_diff:
                            max_heading_diff = diff
                # Turn threshold: 45 degrees (pi / 4)
                if max_heading_diff > (math.pi / 4):
                    is_turning = True

            # Fit linear models first to calculate errors
            w_coeffs = np.polyfit(times_arr, w_arr, 1)
            h_coeffs = np.polyfit(times_arr, h_arr, 1)

            rec_w = np.maximum(1.0, w_coeffs[0] * times_arr + w_coeffs[1])
            rec_h = np.maximum(1.0, h_coeffs[0] * times_arr + h_coeffs[1])

            mean_w_err = float(np.mean(np.abs(w_arr - rec_w)))
            mean_h_err = float(np.mean(np.abs(h_arr - rec_h)))

            avg_w = float(np.mean(w_arr))
            avg_h = float(np.mean(h_arr))

            # Trigger spline fitting if turning or large fitting error
            large_error = (
                mean_w_err > 5.0 or (mean_w_err / max(1.0, avg_w)) > 0.1 or
                mean_h_err > 5.0 or (mean_h_err / max(1.0, avg_h)) > 0.1
            )

            if is_turning or large_error:
                # Downsample control points (e.g. step of 5, scaling with track length)
                downsample_factor = max(1, min(5, n // 5))
                indices = list(range(0, n, downsample_factor))
                if indices[-1] != n - 1:
                    indices.append(n - 1)
                
                # Keep unique indices and ensure they are sorted
                indices = sorted(list(set(indices)))

                # Deduplicate control points by timestamp just in case
                unique_points = []
                seen_times = set()
                for idx in indices:
                    t_val = float(timestamps[idx])
                    if t_val not in seen_times:
                        unique_points.append([t_val, float(widths[idx]), float(heights[idx])])
                        seen_times.add(t_val)
                
                if len(unique_points) >= 2:
                    try:
                        size_model = SplineModel(unique_points)
                    except Exception:
                        # Fallback to linear on any spline construction failures
                        size_model = LinearModel(
                            a=float(w_coeffs[0]),
                            b=float(w_coeffs[1]),
                            c=float(h_coeffs[0]),
                            d=float(h_coeffs[1]),
                        )
                else:
                    size_model = LinearModel(
                        a=float(w_coeffs[0]),
                        b=float(w_coeffs[1]),
                        c=float(h_coeffs[0]),
                        d=float(h_coeffs[1]),
                    )
            else:
                size_model = LinearModel(
                    a=float(w_coeffs[0]),
                    b=float(w_coeffs[1]),
                    c=float(h_coeffs[0]),
                    d=float(h_coeffs[1]),
                )
        elif n >= 2:
            w_coeffs = np.polyfit(times_arr, w_arr, 1)
            h_coeffs = np.polyfit(times_arr, h_arr, 1)
            size_model = LinearModel(
                a=float(w_coeffs[0]),
                b=float(w_coeffs[1]),
                c=float(h_coeffs[0]),
                d=float(h_coeffs[1]),
            )
        else:
            size_model = ConstantModel(w0=widths[0], h0=heights[0])

        # 8. Create CompressedTrack
        statistics = Statistics.compute(trajectory)
        track = CompressedTrack(
            metadata=metadata,
            time_model=time_model,
            size_model=size_model,
            trajectory=trajectory,
            statistics=statistics,
        )

        return track
