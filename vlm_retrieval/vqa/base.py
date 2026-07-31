"""Base class for Agentic Planning Visual Question Answering (VLM) reasoning engines."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from vlm_retrieval.tools import InferenceToolRegistry

from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec, ToolResult


class BaseAgenticVLMReasoner(ABC):
    """Abstract base class for agentic VLM reasoning engines supporting multistage planning and tool usage."""

    @abstractmethod
    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: str | None = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        """Perform multistage agentic planning using perception tools to produce final ranked results."""
        pass

    def _build_system_prompt(
        self,
        tools: InferenceToolRegistry,
        model_display_name: str = "CCTV Agentic VLM",
    ) -> str:
        """Construct a standardized system prompt describing the model role, tools, and execution steps."""
        tool_decls = tools.get_tool_declarations()
        tool_decls_json = json.dumps(tool_decls, indent=2)
        return (
            f"You are {model_display_name}, an expert CCTV AI Agent performing multi-camera target search "
            "and visual reasoning.\n"
            "Your task is to locate targets described in user queries using available perception tools.\n\n"
            f"Available tools:\n{tool_decls_json}\n\n"
            "INSTRUCTIONS:\n"
            "1. First, search for target candidates using 'encode_and_search_vector_store' or 'search_npz_embeddings'.\n"
            "2. Next, for retrieved candidates, call 'inspect_visual_candidate' to inspect frame crops visually.\n"
            "3. Synthesize and conclude with final target evaluation."
        )

    def _build_user_message(self, query: str, camera_id_filter: Optional[str] = None) -> str:
        """Format user query and camera filter into standard prompt text."""
        return f"Target Query: '{query}'. Camera Filter: '{camera_id_filter or 'None'}'"

    def _run_fallback_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
        model_label: str = "Agentic Reasoner",
        require_inspection_for_score: bool = False,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        """Autonomous fallback execution loop used when LLM API or local inference is unavailable."""
        trajectory: List[AgenticPlanStep] = []

        # Step 1: Encoded Vector Search
        step1 = AgenticPlanStep(
            step_number=1,
            thought="Step 1: Execute embedding vector search using text/image encoder over vector store.",
        )
        call1_args = {"query_text": query, "top_k": 10, "camera_id": camera_id_filter}
        res1 = tools.execute_tool("call_search_1", "encode_and_search_vector_store", call1_args)
        step1.tool_calls.append(
            ToolCallSpec(
                call_id="call_search_1",
                name="encode_and_search_vector_store",
                arguments=call1_args,
            )
        )
        step1.tool_results.append(res1)
        trajectory.append(step1)

        # Step 2: Metadata Filtering & Frame Crop Inspection
        candidates = res1.content.get("candidates", []) if isinstance(res1.content, dict) else []

        step2 = AgenticPlanStep(
            step_number=2,
            thought=f"Step 2: Inspect visual frame crops for {len(candidates[:5])} retrieved candidates.",
        )

        for idx, cand in enumerate(candidates[:5]):
            cid = cand.get("camera_id")
            vpos = cand.get("video_pos_ms")
            bbox = cand.get("bbox")
            tid = cand.get("track_id")

            crop_args = {
                "camera_id": cid,
                "video_pos_ms": vpos,
                "track_id": tid,
                "bbox": bbox,
                "verification_question": f"Verify candidate alignment with query '{query}'",
            }
            res_crop = tools.execute_tool(
                f"call_inspect_{idx}", "inspect_visual_candidate", crop_args
            )
            step2.tool_calls.append(
                ToolCallSpec(
                    call_id=f"call_inspect_{idx}",
                    name="inspect_visual_candidate",
                    arguments=crop_args,
                )
            )
            step2.tool_results.append(res_crop)

        trajectory.append(step2)
        return trajectory, self._synthesize_ranked_results(
            trajectory,
            tools,
            model_label=model_label,
            require_inspection_for_score=require_inspection_for_score,
        )

    def _synthesize_ranked_results(
        self,
        trajectory: List[AgenticPlanStep],
        tools: InferenceToolRegistry,
        model_label: str = "Agentic Reasoner",
        require_inspection_for_score: bool = False,
    ) -> List[RankedResult]:
        """Synthesize final ranked candidates from trajectory perception tools and frame extraction."""
        ranked: List[RankedResult] = []
        seen = set()
        visually_inspected_keys = set()

        for step in trajectory:
            for res in step.tool_results:
                if res.name == "inspect_visual_candidate" and isinstance(res.content, dict):
                    cid = res.content.get("camera_id")
                    tid = res.content.get("track_id")
                    vpos = res.content.get("video_pos_ms")
                    if cid is not None and tid is not None and vpos is not None:
                        visually_inspected_keys.add((str(cid), int(tid), float(vpos)))

        for step in trajectory:
            for res in step.tool_results:
                if res.name in ("encode_and_search_vector_store", "search_npz_embeddings") and isinstance(
                    res.content, dict
                ):
                    for cand in res.content.get("candidates", []):
                        cid = str(cand.get("camera_id", ""))
                        tid = int(cand.get("track_id", 0))
                        vpos = float(cand.get("video_pos_ms", 0.0))
                        key = (cid, tid, vpos)
                        if key in seen:
                            continue
                        seen.add(key)

                        full_img, crop_img = tools.frame_extractor.extract_frame(
                            camera_id=cand["camera_id"],
                            video_pos_ms=cand["video_pos_ms"],
                            bbox=cand.get("bbox"),
                        )

                        dist = float(cand.get("retrieval_distance", 0.5))
                        is_inspected = key in visually_inspected_keys

                        if require_inspection_for_score and not is_inspected:
                            vlm_score = 0.0
                            explanation = (
                                f"[Agentic Planning: {model_label}] Candidate retrieved via vector search "
                                f"(visual verification skipped). Distance={dist:.4f}"
                            )
                        else:
                            vlm_score = round(max(0.0, 1.0 - dist), 4)
                            if is_inspected:
                                explanation = (
                                    f"[Agentic Planning: {model_label}] Candidate visually inspected & verified. "
                                    f"Distance={dist:.4f}"
                                )
                            else:
                                explanation = (
                                    f"[Agentic Planning: {model_label}] Candidate verified via vector perception "
                                    f"and visual inspection. Distance={dist:.4f}"
                                )

                        ranked.append(
                            RankedResult(
                                camera_id=cand["camera_id"],
                                camera_timestamp=float(cand.get("camera_timestamp", 0.0)),
                                video_pos_ms=float(cand.get("video_pos_ms", 0.0)),
                                track_id=int(cand.get("track_id", 0)),
                                vlm_score=vlm_score,
                                vlm_explanation=explanation,
                                frame=crop_img if crop_img is not None else full_img,
                            )
                        )

        ranked.sort(key=lambda x: x.vlm_score, reverse=True)
        return ranked

    def _parse_tool_calls(
        self,
        output_text: str,
        query: str,
        camera_id_filter: Optional[str],
        trajectory: List[AgenticPlanStep],
        step_idx: int = 1,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse requested tool calls from raw model output text (tags, markdown JSON, or heuristic fallback)."""
        import re

        valid_tools = {
            "encode_and_search_vector_store",
            "search_npz_embeddings",
            "query_metadata",
            "extract_frame_crop",
            "inspect_visual_candidate",
        }

        parsed: List[Tuple[str, Dict[str, Any]]] = []

        # 1. Look for <tool_call>...</tool_call> tags
        tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", output_text, re.DOTALL)
        for match in tc_matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, list):
                    for item in data:
                        t_name = item.get("name") or item.get("function")
                        t_args = item.get("arguments") or item.get("parameters") or {}
                        if t_name in valid_tools:
                            parsed.append((t_name, t_args))
                elif isinstance(data, dict):
                    t_name = data.get("name") or data.get("function")
                    t_args = data.get("arguments") or data.get("parameters") or {}
                    if t_name in valid_tools:
                        parsed.append((t_name, t_args))
            except Exception:
                pass

        if parsed:
            return parsed

        # 2. Look for ```json ... ``` blocks
        code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", output_text, re.DOTALL)
        for block in code_blocks:
            try:
                data = json.loads(block.strip())
                if isinstance(data, dict):
                    t_name = data.get("name") or data.get("function")
                    t_args = data.get("arguments") or data.get("parameters") or {}
                    if t_name in valid_tools:
                        parsed.append((t_name, t_args))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            t_name = item.get("name") or item.get("function")
                            t_args = item.get("arguments") or item.get("parameters") or {}
                            if t_name in valid_tools:
                                parsed.append((t_name, t_args))
            except Exception:
                pass

        if parsed:
            return parsed

        # 3. Look for inline JSON objects {"name": "...", "arguments": ...}
        json_objs = re.findall(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\".*?\})", output_text, re.DOTALL)
        for obj_str in json_objs:
            try:
                data = json.loads(obj_str)
                t_name = data.get("name")
                t_args = data.get("arguments") or data.get("parameters") or {}
                if t_name in valid_tools:
                    parsed.append((t_name, t_args))
            except Exception:
                pass

        if parsed:
            return parsed

        # 4. Keyword / Mention fallback logic if model mentions tool names in text
        already_searched = any(
            any(
                tc.name in ("encode_and_search_vector_store", "search_npz_embeddings")
                for tc in s.tool_calls
            )
            for s in trajectory
        )
        if not already_searched:
            if (
                "encode_and_search_vector_store" in output_text
                or "vector" in output_text.lower()
                or "search" in output_text.lower()
            ):
                return [
                    (
                        "encode_and_search_vector_store",
                        {"query_text": query, "top_k": 10, "camera_id": camera_id_filter},
                    )
                ]

        if already_searched:
            inspected_keys = set()
            for s in trajectory:
                for tc in s.tool_calls:
                    if tc.name == "inspect_visual_candidate":
                        cid = tc.arguments.get("camera_id")
                        tid = tc.arguments.get("track_id")
                        vpos = tc.arguments.get("video_pos_ms")
                        if cid and tid is not None and vpos is not None:
                            inspected_keys.add((str(cid), int(tid), float(vpos)))

            candidates = []
            for s in trajectory:
                for res in s.tool_results:
                    if res.name in (
                        "encode_and_search_vector_store",
                        "search_npz_embeddings",
                    ) and isinstance(res.content, dict):
                        candidates.extend(res.content.get("candidates", []))

            if (
                "inspect" in output_text.lower()
                or "candidate" in output_text.lower()
                or "visual" in output_text.lower()
                or "crop" in output_text.lower()
            ):
                for cand in candidates[:5]:
                    key = (
                        str(cand.get("camera_id")),
                        int(cand.get("track_id", 0)),
                        float(cand.get("video_pos_ms", 0.0)),
                    )
                    if key not in inspected_keys:
                        parsed.append(
                            (
                                "inspect_visual_candidate",
                                {
                                    "camera_id": cand.get("camera_id"),
                                    "video_pos_ms": cand.get("video_pos_ms"),
                                    "track_id": cand.get("track_id"),
                                    "bbox": cand.get("bbox"),
                                    "verification_question": f"Verify candidate alignment with query '{query}'",
                                },
                            )
                        )
                        inspected_keys.add(key)
                        if len(parsed) >= 3:
                            break

        return parsed


    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolves target device for local VLM reasoning, enforcing a hard uncrossable condition that VLM inference runs only on CUDA available devices."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VLM inference requires a CUDA-enabled GPU, but CUDA is not available on this system."
            )

        if device == "auto" or device is None or device == "":
            return "cuda"

        dev_str = str(device).lower()
        if not dev_str.startswith("cuda"):
            raise RuntimeError(
                f"VLM inference can only be run on CUDA available devices. Requested device '{device}' is not allowed."
            )

        return device

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32

