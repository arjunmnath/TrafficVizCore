"""Perception & retrieval tools for the Agentic Planning VLM system."""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Dict, List, Optional
from PIL import Image

from inference_node.frame_extractor import FrameExtractor
from inference_node.retrieval.search import RetrievalEngine
from inference_node.retrieval.vector_store import VectorStore
from inference_node.vqa.types import CandidateImage, ToolResult
from shared.utils import setup_logger


class InferenceToolRegistry:
    """Registry exposing perception, vector search, metadata query, and frame extraction tools to the VLM planner."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        vector_store: VectorStore,
        frame_extractor: FrameExtractor,
    ) -> None:
        self.logger = setup_logger("InferenceToolRegistry")
        self.retrieval = retrieval_engine
        self.vector_store = vector_store
        self.frame_extractor = frame_extractor
        self._cached_candidates: Dict[str, CandidateImage] = {}

    def get_tool_declarations(self) -> List[Dict[str, Any]]:
        """Return standardized OpenAI/JSON-schema tool declarations for agentic VLM planning."""
        return [
            {
                "name": "encode_and_search_vector_store",
                "description": "Performs text/image embedding similarity search against the PostgreSQL pgvector track event database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Natural language query describing target appearance or context.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of candidate results to retrieve (default: 10).",
                        },
                        "camera_id": {
                            "type": "string",
                            "description": "Optional camera identifier filter (e.g., 'cam_1').",
                        },
                    },
                    "required": ["query_text"],
                },
            },
            {
                "name": "query_metadata",
                "description": "Queries track event metadata (camera ID, time window, vehicle/person attributes, track ID) from PostgreSQL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera_id": {"type": "string", "description": "Filter by camera ID."},
                        "start_timestamp": {
                            "type": "number",
                            "description": "Start epoch timestamp.",
                        },
                        "end_timestamp": {"type": "number", "description": "End epoch timestamp."},
                        "class_label": {
                            "type": "string",
                            "description": "Target COCO class label ('person', 'car', 'motorcycle', 'bus', 'truck').",
                        },
                        "color": {"type": "string", "description": "Dominant color attribute."},
                        "top_k": {"type": "integer", "description": "Max results to return."},
                    },
                },
            },
            {
                "name": "extract_frame_crop",
                "description": "Extracts raw full frames or target bounding box crops from CCTV video feeds for visual inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera_id": {
                            "type": "string",
                            "description": "Camera ID matching the feed.",
                        },
                        "video_pos_ms": {
                            "type": "number",
                            "description": "Video playback position in milliseconds.",
                        },
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Bounding box coordinates [x1, y1, x2, y2].",
                        },
                    },
                    "required": ["camera_id", "video_pos_ms"],
                },
            },
            {
                "name": "inspect_visual_candidate",
                "description": "Retrieves and returns the candidate image crop for visual verification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera_id": {"type": "string"},
                        "video_pos_ms": {"type": "number"},
                        "track_id": {"type": "integer"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                        "verification_question": {
                            "type": "string",
                            "description": "Question to evaluate on the visual candidate crop.",
                        },
                    },
                    "required": ["camera_id", "video_pos_ms"],
                },
            },
        ]

    def execute_tool(self, call_id: str, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call by name with keyword arguments."""
        self.logger.info(f"Executing tool '{name}' (call_id='{call_id}') with args: {arguments}")
        try:
            if name == "encode_and_search_vector_store":
                return self._execute_vector_search(call_id, arguments)
            elif name == "query_metadata":
                return self._execute_metadata_query(call_id, arguments)
            elif name == "extract_frame_crop":
                return self._execute_frame_crop(call_id, arguments)
            elif name == "inspect_visual_candidate":
                return self._execute_visual_inspection(call_id, arguments)
            else:
                return ToolResult(
                    call_id=call_id,
                    name=name,
                    content=f"Error: Unknown tool '{name}'",
                    is_error=True,
                )
        except Exception as err:
            self.logger.error(f"Error executing tool '{name}': {err}", exc_info=True)
            return ToolResult(
                call_id=call_id,
                name=name,
                content=f"Tool execution failed: {str(err)}",
                is_error=True,
            )

    def _execute_vector_search(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        query_text = args.get("query_text", "")
        top_k = int(args.get("top_k", 10))
        camera_id = args.get("camera_id")

        parsed, candidates = self.retrieval.search(
            query=query_text, top_k=top_k, camera_id=camera_id
        )

        results_data = []
        for cand in candidates:
            cand_key = f"{cand.camera_id}_{cand.track_id}_{cand.video_pos_ms:.0f}"
            results_data.append(
                {
                    "candidate_key": cand_key,
                    "id": cand.id,
                    "camera_id": cand.camera_id,
                    "camera_timestamp": cand.camera_timestamp,
                    "track_id": cand.track_id,
                    "video_pos_ms": cand.video_pos_ms,
                    "bbox": cand.bbox,
                    "retrieval_distance": cand.distance,
                }
            )

        return ToolResult(
            call_id=call_id,
            name="encode_and_search_vector_store",
            content={
                "parsed_semantic_query": parsed.semantic_text,
                "metadata_filters_applied": parsed.metadata_filters,
                "count": len(results_data),
                "candidates": results_data,
            },
        )

    def _execute_metadata_query(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        where_clauses = []
        if args.get("camera_id"):
            where_clauses.append({"camera_id": args["camera_id"]})
        if args.get("start_timestamp") is not None:
            where_clauses.append({"camera_timestamp": {"$gte": float(args["start_timestamp"])}})
        if args.get("end_timestamp") is not None:
            where_clauses.append({"camera_timestamp": {"$lt": float(args["end_timestamp"])}})
        if args.get("class_label"):
            where_clauses.append({"class_label": args["class_label"]})
        if args.get("color"):
            where_clauses.append({"color": args["color"]})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        top_k = int(args.get("top_k", 20))
        metas = self.vector_store.query_metadata(where=where, limit=top_k)

        return ToolResult(
            call_id=call_id,
            name="query_metadata",
            content={
                "count": len(metas),
                "metadatas": metas,
            },
        )

    def _execute_frame_crop(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        camera_id = args.get("camera_id", "")
        video_pos_ms = float(args.get("video_pos_ms", 0.0))
        bbox = args.get("bbox")
        if isinstance(bbox, str):
            bbox = [float(v) for v in bbox.split(",")]

        full_frame, crop = self.frame_extractor.extract_frame(
            camera_id=camera_id,
            video_pos_ms=video_pos_ms,
            bbox=bbox,
        )

        extracted_img = crop if crop is not None else full_frame
        images = [extracted_img] if extracted_img is not None else []

        return ToolResult(
            call_id=call_id,
            name="extract_frame_crop",
            content={
                "camera_id": camera_id,
                "video_pos_ms": video_pos_ms,
                "bbox": bbox,
                "extracted_successfully": len(images) > 0,
            },
            extracted_images=images,
        )

    def _execute_visual_inspection(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        camera_id = args.get("camera_id", "")
        video_pos_ms = float(args.get("video_pos_ms", 0.0))
        track_id = int(args.get("track_id", 0))
        bbox = args.get("bbox")
        if isinstance(bbox, str):
            bbox = [float(v) for v in bbox.split(",")]
        question = args.get("verification_question", "Inspect visual features")

        full_frame, crop = self.frame_extractor.extract_frame(
            camera_id=camera_id,
            video_pos_ms=video_pos_ms,
            bbox=bbox,
        )

        target_img = crop if crop is not None else full_frame
        images = [target_img] if target_img is not None else []

        return ToolResult(
            call_id=call_id,
            name="inspect_visual_candidate",
            content={
                "camera_id": camera_id,
                "video_pos_ms": video_pos_ms,
                "track_id": track_id,
                "bbox": bbox,
                "verification_question": question,
                "image_attached": len(images) > 0,
            },
            extracted_images=images,
        )
