"""Base class for Agentic Planning Visual Question Answering (VLM) reasoning engines.

Implements a ReAct-style execution loop:
    Thought → Tool Call → Tool Execution → Observation → Updated Thought

The planner produces only the next logical step, never an entire execution trace.
Tool outputs are the single source of truth — no fabricated scores or explanations.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from vlm_retrieval.tools import InferenceToolRegistry

from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec, ToolResult
from shared.utils import setup_logger

logger = setup_logger("BaseAgenticVLMReasoner")


class BaseAgenticVLMReasoner(ABC):
    """Abstract base class for agentic VLM reasoning engines.

    Each concrete implementation (Gemini, OpenAI, Qwen) must implement
    `plan_and_execute` following the ReAct loop pattern:

        messages = [system_prompt, user_query]
        while True:
            response = llm(messages)
            if response.contains_tool_call():
                tool_result = execute(tool_call)
                messages.append(response)
                messages.append(tool_result)
                continue
            return response  # final answer
    """

    @abstractmethod
    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: str | None = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        """Perform multistage agentic planning using perception tools.

        Returns:
            trajectory: List of reasoning steps with tool calls and observations.
            ranked_results: Final verified candidates parsed from the LLM's answer.
        """
        pass

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        tools: InferenceToolRegistry,
        model_display_name: str = "CCTV Agentic VLM",
    ) -> str:
        """Construct the system prompt describing the ReAct loop, tools, and rules."""
        tool_decls = tools.get_tool_declarations()
        tool_decls_json = json.dumps(tool_decls, indent=2)
        return (
            f"You are {model_display_name}, an expert CCTV AI Agent performing multi-camera "
            "target search and visual reasoning.\n\n"
            "## Execution Model\n"
            "You operate in a ReAct loop: Think → Act → Observe → Think → ...\n"
            "- Produce only the NEXT logical step, not an entire plan.\n"
            "- Call ONE tool at a time. Wait for its result before deciding the next action.\n"
            "- Tool outputs are the ONLY source of truth. Never assume or fabricate tool results.\n"
            "- When you have sufficient evidence to answer, produce a final answer WITHOUT any tool calls.\n\n"
            f"## Available Tools\n{tool_decls_json}\n\n"
            "## Tool Usage Rules\n"
            "1. `encode_and_search_vector_store` retrieves semantic candidates. "
            "It does NOT verify correctness. Always follow up with `inspect_visual_candidate`.\n"
            "2. `inspect_visual_candidate` extracts a frame crop and returns it to you. "
            "YOU must analyze the attached image and determine if it matches the query. "
            "Do NOT pass scores or explanations as tool arguments.\n"
            "3. `query_metadata` only filters structural metadata (camera_id, timestamp, class_label). "
            "It cannot filter by color, make/model, or visual attributes.\n"
            "4. `get_temporal_context` retrieves nearby track events around a timestamp. "
            "Use this for relationship queries (e.g., 'bus followed by car').\n\n"
            "## Relationship Queries\n"
            "For queries involving multiple objects in a spatial-temporal relationship "
            "(e.g., 'a blue bus followed by a red MPV'):\n"
            "1. Search and verify the first object.\n"
            "2. Use `get_temporal_context` to find nearby objects on the same camera.\n"
            "3. Inspect those candidates to verify the second object.\n"
            "4. Only confirm the relationship if both objects are verified with appropriate timing.\n"
            "5. If insufficient evidence exists, state this explicitly in your final answer.\n\n"
            "## Final Answer Format\n"
            "When you have enough evidence, respond with your analysis followed by a "
            "structured answer block:\n"
            "```\n"
            "<final_answer>\n"
            '{"candidates": [\n'
            '  {"camera_id": "...", "video_pos_ms": ..., "track_id": ..., '
            '"confidence": 0.0-1.0, "explanation": "..."}\n'
            "]}\n"
            "</final_answer>\n"
            "```\n"
            "- `confidence` must be grounded in your visual analysis of inspected crops.\n"
            "- `explanation` must reference specific visual evidence you observed.\n"
            "- Only include candidates you have actually inspected visually.\n"
            "- If no candidates match, return an empty candidates list with an explanation.\n"
        )

    def _build_user_message(self, query: str, camera_id_filter: Optional[str] = None) -> str:
        """Format user query into standard prompt text."""
        msg = f"Find: {query}"
        if camera_id_filter:
            msg += f"\nCamera filter: {camera_id_filter}"
        return msg

    # ------------------------------------------------------------------
    # Final answer parsing
    # ------------------------------------------------------------------

    def _parse_final_answer(
        self,
        response_text: str,
        tools: InferenceToolRegistry,
    ) -> List[RankedResult]:
        """Parse the LLM's final answer into RankedResult objects.

        Extracts the <final_answer> JSON block and converts each candidate
        into a RankedResult, fetching frame thumbnails from the frame extractor.
        """
        results: List[RankedResult] = []

        # Try to extract <final_answer> block
        match = re.search(
            r"<final_answer>\s*(.*?)\s*</final_answer>",
            response_text,
            re.DOTALL,
        )
        if not match:
            # Try bare JSON with candidates key
            match = re.search(
                r'\{\s*"candidates"\s*:\s*\[.*?\]\s*\}',
                response_text,
                re.DOTALL,
            )

        if not match:
            logger.warning("No <final_answer> block found in LLM response. Returning empty results.")
            return results

        try:
            data = json.loads(match.group(1) if match.lastindex else match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse final answer JSON: {e}")
            return results

        candidates = data.get("candidates", [])
        for cand in candidates:
            camera_id = str(cand.get("camera_id", ""))
            video_pos_ms = float(cand.get("video_pos_ms", 0.0))
            track_id = int(cand.get("track_id", 0))
            confidence = float(cand.get("confidence", 0.0))
            explanation = str(cand.get("explanation", ""))

            # Extract frame for thumbnail
            frame = None
            try:
                full_frame, crop = tools.frame_extractor.extract_frame(
                    camera_id=camera_id,
                    video_pos_ms=video_pos_ms,
                    bbox=cand.get("bbox"),
                )
                frame = crop if crop is not None else full_frame
            except Exception as e:
                logger.warning(f"Failed to extract frame for result: {e}")

            results.append(
                RankedResult(
                    camera_id=camera_id,
                    camera_timestamp=float(cand.get("camera_timestamp", video_pos_ms / 1000.0)),
                    video_pos_ms=video_pos_ms,
                    track_id=track_id,
                    vlm_score=round(max(0.0, min(1.0, confidence)), 4),
                    vlm_explanation=explanation,
                    frame=frame,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.vlm_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Tool call parsing (for text-based backends like Qwen)
    # ------------------------------------------------------------------

    def _parse_tool_calls_from_text(self, output_text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse tool calls from raw model output text.

        Supports:
        1. <tool_call>{"name": "...", "arguments": {...}}</tool_call> tags
        2. ```json {"name": "...", "arguments": {...}} ``` code blocks
        3. Inline {"name": "...", "arguments": {...}} JSON objects

        NO keyword fallback — if the model doesn't explicitly emit a structured
        tool call, we treat the response as a final text answer.
        """
        valid_tools = {
            "encode_and_search_vector_store",
            "query_metadata",
            "inspect_visual_candidate",
            "get_temporal_context",
        }

        parsed: List[Tuple[str, Dict[str, Any]]] = []

        # 1. <tool_call>...</tool_call> tags
        tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", output_text, re.DOTALL)
        for match in tc_matches:
            try:
                data = json.loads(match.strip())
                items = data if isinstance(data, list) else [data]
                for item in items:
                    t_name = item.get("name") or item.get("function")
                    t_args = item.get("arguments") or item.get("parameters") or {}
                    if t_name in valid_tools:
                        parsed.append((t_name, t_args))
            except (json.JSONDecodeError, AttributeError):
                pass

        if parsed:
            return parsed

        # 2. ```json ... ``` code blocks
        code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", output_text, re.DOTALL)
        for block in code_blocks:
            try:
                data = json.loads(block.strip())
                items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
                for item in items:
                    if isinstance(item, dict):
                        t_name = item.get("name") or item.get("function")
                        t_args = item.get("arguments") or item.get("parameters") or {}
                        if t_name in valid_tools:
                            parsed.append((t_name, t_args))
            except (json.JSONDecodeError, AttributeError):
                pass

        if parsed:
            return parsed

        # 3. Inline JSON objects with "name" key
        json_objs = re.findall(r'(\{\s*"name"\s*:\s*"[^"]+"\s*,.*?\})', output_text, re.DOTALL)
        for obj_str in json_objs:
            try:
                data = json.loads(obj_str)
                t_name = data.get("name")
                t_args = data.get("arguments") or data.get("parameters") or {}
                if t_name in valid_tools:
                    parsed.append((t_name, t_args))
            except (json.JSONDecodeError, AttributeError):
                pass

        return parsed

    # ------------------------------------------------------------------
    # Device resolution (for local VLM inference)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolves target device for local VLM reasoning, enforcing CUDA requirement."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VLM inference requires a CUDA-enabled GPU, but CUDA is not available on this system."
            )

        if device == "auto" or device is None or device == "":
            return "cuda"

        dev_str = str(device).lower()
        if not dev_str.startswith("cuda"):
            raise RuntimeError(
                f"VLM inference can only be run on CUDA available devices. "
                f"Requested device '{device}' is not allowed."
            )

        return device

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
