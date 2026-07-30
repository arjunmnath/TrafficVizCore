import json
from typing import Any, Dict
from tracking.domain.track import CompressedTrack
from tracking.domain.size_models import ConstantModel, LinearModel, PolynomialModel, SplineModel


class JsonSerializer:
    """Serializes a CompressedTrack domain object to the standard JSON schema."""

    @staticmethod
    def serialize_to_dict(track: CompressedTrack) -> Dict[str, Any]:
        """Convert CompressedTrack to a Python dictionary matching the required JSON format."""
        # 1. Serialize metadata
        meta = track.metadata

        # 2. Serialize trajectory segments
        segments_data = []
        for seg in track.trajectory.segments:
            segments_data.append(seg.serialize())

        # 3. Serialize size models (split into width and height models)
        sz = track.size_model
        width_model_data: Dict[str, Any] = {}
        height_model_data: Dict[str, Any] = {}

        if isinstance(sz, ConstantModel):
            width_model_data = {"type": "constant", "parameters": {"val": float(sz.w0)}}
            height_model_data = {"type": "constant", "parameters": {"val": float(sz.h0)}}
        elif isinstance(sz, LinearModel):
            width_model_data = {
                "type": "linear",
                "parameters": {"a": float(sz.a), "b": float(sz.b)},
            }
            height_model_data = {
                "type": "linear",
                "parameters": {"a": float(sz.c), "b": float(sz.d)},
            }
        elif isinstance(sz, PolynomialModel):
            width_model_data = {
                "type": "polynomial",
                "parameters": {"coeffs": [float(c) for c in sz.w_coeffs]},
            }
            height_model_data = {
                "type": "polynomial",
                "parameters": {"coeffs": [float(c) for c in sz.h_coeffs]},
            }
        elif isinstance(sz, SplineModel):
            width_model_data = {
                "type": "spline",
                "control_points": [[float(pt[0]), float(pt[1])] for pt in sz.control_points],
            }
            height_model_data = {
                "type": "spline",
                "control_points": [[float(pt[0]), float(pt[2])] for pt in sz.control_points],
            }
        else:
            # Fallback using the general serialize() dict
            serialized_sz = sz.serialize()
            width_model_data = {
                "type": serialized_sz.get("type", "unknown"),
                "parameters": serialized_sz.get("parameters", {}),
            }
            height_model_data = {
                "type": serialized_sz.get("type", "unknown"),
                "parameters": serialized_sz.get("parameters", {}),
            }

        # 4. Serialize time model
        time_model_data = track.time_model.serialize()

        # 5. Serialize confidence model
        confidence_model_data = track.confidence_model.serialize()

        # 6. Serialize statistics
        stats_data = track.statistics.serialize()

        return {
            "track_id": int(meta.track_id),
            "class": meta.class_label,
            "camera": meta.camera_id,
            "start_time": float(meta.start_timestamp),
            "end_time": float(meta.end_timestamp),
            "trajectory": {"segments": segments_data},
            "width_model": width_model_data,
            "height_model": height_model_data,
            "time_model": time_model_data,
            "confidence_model": confidence_model_data,
            "statistics": stats_data,
        }

    @classmethod
    def serialize_to_json(cls, track: CompressedTrack, indent: int = 4) -> str:
        """Convert CompressedTrack to a JSON string representation."""
        return json.dumps(cls.serialize_to_dict(track), indent=indent)
