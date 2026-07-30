import json
from typing import Any, Dict

from tracking.domain.metadata import TrackMetadata
from tracking.domain.segments import ConstantSegment, LinearSegment, CubicSplineSegment
from tracking.domain.size_models import ConstantModel, LinearModel, PolynomialModel, SplineModel
from tracking.domain.trajectory import PiecewiseTrajectory
from tracking.domain.track import TimeModel, Statistics, CompressedTrack, ConfidenceModel


class JsonDeserializer:
    """Deserializes JSON representations back to CompressedTrack domain objects."""

    @staticmethod
    def _deserialize_segment(seg_data: Dict[str, Any]) -> Any:
        seg_type = seg_data["type"]
        t0 = seg_data["t0"]
        t1 = seg_data["t1"]
        max_err = seg_data.get("max_error", 0.0)

        if seg_type == "constant":
            params = seg_data["parameters"]
            return ConstantSegment(t0=t0, t1=t1, cx=params["cx"], cy=params["cy"], max_err=max_err)
        elif seg_type == "linear":
            params = seg_data["parameters"]
            return LinearSegment(
                t0=t0,
                t1=t1,
                a=params["a"],
                b=params["b"],
                c=params["c"],
                d=params["d"],
                max_err=max_err,
            )
        elif seg_type == "spline":
            control_points = seg_data["control_points"]
            return CubicSplineSegment(control_points=control_points, max_err=max_err)
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

    @staticmethod
    def _deserialize_size_model(w_model: Dict[str, Any], h_model: Dict[str, Any]) -> Any:
        w_type = w_model["type"]
        h_type = h_model["type"]

        if w_type == "constant" and h_type == "constant":
            return ConstantModel(w0=w_model["parameters"]["val"], h0=h_model["parameters"]["val"])
        elif w_type == "linear" and h_type == "linear":
            return LinearModel(
                a=w_model["parameters"]["a"],
                b=w_model["parameters"]["b"],
                c=h_model["parameters"]["a"],
                d=h_model["parameters"]["b"],
            )
        elif w_type == "polynomial" and h_type == "polynomial":
            return PolynomialModel(
                w_coeffs=w_model["parameters"]["coeffs"],
                h_coeffs=h_model["parameters"]["coeffs"],
            )
        elif w_type == "spline" and h_type == "spline":
            # Reconstruct merged control points [t, w, h] from [t, w] and [t, h]
            w_pts = w_model["control_points"]
            h_pts = h_model["control_points"]
            # Align by timestamp
            w_dict = {pt[0]: pt[1] for pt in w_pts}
            h_dict = {pt[0]: pt[1] for pt in h_pts}
            all_times = sorted(list(set(w_dict.keys()) | set(h_dict.keys())))

            merged_pts = []
            for t in all_times:
                w = w_dict.get(t, 1.0)
                h = h_dict.get(t, 1.0)
                merged_pts.append([t, w, h])
            return SplineModel(merged_pts)
        else:
            # Fallback
            return ConstantModel(w0=1.0, h0=1.0)

    @classmethod
    def deserialize_from_dict(cls, data: Dict[str, Any]) -> CompressedTrack:
        """Construct CompressedTrack from a Python dictionary."""
        # 1. Deserialize metadata
        time_model_data = data["time_model"]
        frames = time_model_data["frames"]

        metadata = TrackMetadata(
            track_id=data["track_id"],
            camera_id=data["camera"],
            class_label=data["class"],
            start_frame=frames[0],
            end_frame=frames[-1],
            start_timestamp=data["start_time"],
            end_timestamp=data["end_time"],
        )

        # 2. Deserialize TimeModel
        time_model = TimeModel(frames, time_model_data["timestamps"])

        # 3. Deserialize Trajectory
        segments_data = data["trajectory"]["segments"]
        segments = [cls._deserialize_segment(seg) for seg in segments_data]
        trajectory = PiecewiseTrajectory(segments)

        # 4. Deserialize SizeModel
        size_model = cls._deserialize_size_model(data["width_model"], data["height_model"])

        # 5. Deserialize ConfidenceModel
        conf_model_data = data.get("confidence_model")
        if conf_model_data and "timestamps" in conf_model_data and "confidences" in conf_model_data:
            confidence_model = ConfidenceModel(
                timestamps=conf_model_data["timestamps"],
                confidences=conf_model_data["confidences"],
                frames=conf_model_data.get("frames"),
            )
        else:
            confidence_model = ConfidenceModel(
                timestamps=time_model.timestamps,
                confidences=[1.0] * len(time_model.timestamps),
                frames=time_model.frames,
            )

        # 6. Deserialize Statistics
        stats_data = data.get("statistics", {})
        statistics = Statistics(
            avg_speed=stats_data.get("avg_speed", 0.0),
            max_speed=stats_data.get("max_speed", 0.0),
            total_distance=stats_data.get("total_distance", 0.0),
            avg_acceleration=stats_data.get("avg_acceleration", 0.0),
        )

        return CompressedTrack(
            metadata=metadata,
            time_model=time_model,
            size_model=size_model,
            trajectory=trajectory,
            statistics=statistics,
            confidence_model=confidence_model,
        )

    @classmethod
    def deserialize_from_json(cls, json_str: str) -> CompressedTrack:
        """Construct CompressedTrack from a JSON string."""
        return cls.deserialize_from_dict(json.loads(json_str))
