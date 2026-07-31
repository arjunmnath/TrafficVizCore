"""Perception & retrieval tools for the Agentic Planning VLM system.

Tools are pure executors — they accept only inputs, perform their action,
and return observations. They never receive scores, explanations, or
expected outputs as arguments.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PIL import Image

from vlm_retrieval.frame_extractor import FrameExtractor
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore
from vlm_retrieval.vqa.types import ToolResult
from shared.utils import setup_logger


class InferenceToolRegistry:
    """Registry exposing perception, vector search, metadata query, and
    frame extraction tools to the VLM planner.

    Design principles:
    - Tool inputs contain ONLY the parameters needed to execute the action.
    - Tool outputs are the single source of truth; the planner must never
      inject scores or explanations into tool arguments.
    - Each tool does one thing and returns a structured observation.
    """

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

    # ------------------------------------------------------------------
    # Tool declarations (JSON-schema)
    # ------------------------------------------------------------------

    def get_tool_declarations(self) -> List[Dict[str, Any]]:
        """Return standardized tool declarations for agentic VLM planning."""
        return [
            {
                "name": "encode_and_search_vector_store",
                "description": (
                    "Encode a natural-language query into an embedding and "
                    "retrieve the top-K most semantically similar track events "
                    "from the vector store. Returns candidate metadata only — "
                    "does NOT verify visual correctness."
                ),
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
                "description": (
                    "Query track event structural metadata (camera ID, time window, "
                    "track ID, standard COCO class_label). NOTE: Fine-grained visual "
                    "attributes (color, vehicle make/model, clothing) are NOT indexed "
                    "in metadata. Use 'encode_and_search_vector_store' and "
                    "'inspect_visual_candidate' for visual attributes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera_id": {
                            "type": "string",
                            "description": "Filter by camera ID (e.g. 'cam_1').",
                        },
                        "start_timestamp": {
                            "type": "number",
                            "description": "Start epoch timestamp.",
                        },
                        "end_timestamp": {
                            "type": "number",
                            "description": "End epoch timestamp.",
                        },
                        "class_label": {
                            "type": "string",
                            "description": "Target COCO class label ('person', 'car', 'motorcycle', 'bus', 'truck').",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results to return (default: 20).",
                        },
                    },
                },
            },
            {
                "name": "inspect_visual_candidate",
                "description": (
                    "Extract a frame crop from the video at the given position and "
                    "return it for visual verification. The extracted image is "
                    "attached to the response for the VLM to analyze. This tool "
                    "does NOT score or evaluate the candidate — it only extracts "
                    "and returns the visual evidence."
                ),
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
                            "description": "Bounding box coordinates [x1, y1, x2, y2] for cropping.",
                        },
                        "verification_question": {
                            "type": "string",
                            "description": (
                                "The specific question to answer about this candidate "
                                "(e.g., 'Is this a blue public transport bus?')."
                            ),
                        },
                    },
                    "required": ["camera_id", "video_pos_ms", "verification_question"],
                },
            },
            {
                "name": "get_temporal_context",
                "description": (
                    "Retrieve track events occurring near a reference timestamp on a "
                    "given camera. Use this for relationship queries (e.g., 'a blue bus "
                    "followed by a red MPV') to find what other objects appear before or "
                    "after a confirmed candidate."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera_id": {
                            "type": "string",
                            "description": "Camera ID to query.",
                        },
                        "reference_time_ms": {
                            "type": "number",
                            "description": "Reference video position in milliseconds.",
                        },
                        "time_window_ms": {
                            "type": "number",
                            "description": "Time window radius in milliseconds (default: 5000). Events within [ref - window, ref + window] are returned.",
                        },
                        "class_label": {
                            "type": "string",
                            "description": "Optional COCO class label filter.",
                        },
                    },
                    "required": ["camera_id", "reference_time_ms"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def execute_tool(self, call_id: str, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call by name with keyword arguments."""
        self.logger.info(f"Executing tool '{name}' (call_id='{call_id}') with args: {arguments}")
        try:
            handler = self._TOOL_HANDLERS.get(name)
            if handler is None:
                return ToolResult(
                    call_id=call_id,
                    name=name,
                    content=f"Error: Unknown tool '{name}'",
                    is_error=True,
                )
            return handler(self, call_id, arguments)
        except Exception as err:
            self.logger.error(f"Error executing tool '{name}': {err}", exc_info=True)
            return ToolResult(
                call_id=call_id,
                name=name,
                content=f"Tool execution failed: {str(err)}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _execute_vector_search(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        query_text = args.get("query_text", "")
        top_k = int(args.get("top_k", 10))
        camera_id = args.get("camera_id")

        parsed, candidates = self.retrieval.search(
            query=query_text, top_k=top_k, camera_id=camera_id
        )

        results_data = []
        for cand in candidates:
            results_data.append(
                {
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
                "semantic_query": parsed.semantic_text,
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

    def _execute_visual_inspection(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        """Extract a frame crop and return it as an attached image.

        This tool does NOT score or evaluate — it only provides visual evidence.
        The VLM will analyze the attached image in the next reasoning step.
        """
        camera_id = args.get("camera_id", "")
        video_pos_ms = float(args.get("video_pos_ms", 0.0))
        bbox = args.get("bbox")
        if isinstance(bbox, str):
            bbox = [float(v) for v in bbox.split(",")]
        question = args.get("verification_question", "Describe what you see in this image.")

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
                "bbox": bbox,
                "verification_question": question,
                "image_extracted": len(images) > 0,
            },
            extracted_images=images,
        )

    def _execute_temporal_context(self, call_id: str, args: Dict[str, Any]) -> ToolResult:
        """Retrieve track events near a reference timestamp for relationship reasoning."""
        camera_id = args.get("camera_id", "")
        ref_time_ms = float(args.get("reference_time_ms", 0.0))
        window_ms = float(args.get("time_window_ms", 5000.0))
        class_label = args.get("class_label")

        # Convert ms to seconds for timestamp comparison
        ref_time_s = ref_time_ms / 1000.0
        window_s = window_ms / 1000.0

        where_clauses = [
            {"camera_id": camera_id},
            {"camera_timestamp": {"$gte": ref_time_s - window_s}},
            {"camera_timestamp": {"$lt": ref_time_s + window_s}},
        ]
        if class_label:
            where_clauses.append({"class_label": class_label})

        where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]

        metas = self.vector_store.query_metadata(where=where, limit=50)

        return ToolResult(
            call_id=call_id,
            name="get_temporal_context",
            content={
                "camera_id": camera_id,
                "reference_time_ms": ref_time_ms,
                "time_window_ms": window_ms,
                "count": len(metas),
                "nearby_events": metas,
            },
        )

    # Handler dispatch table
    _TOOL_HANDLERS = {
        "encode_and_search_vector_store": _execute_vector_search,
        "query_metadata": _execute_metadata_query,
        "inspect_visual_candidate": _execute_visual_inspection,
        "get_temporal_context": _execute_temporal_context,
    }
