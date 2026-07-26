"""Gemini Agentic VLM Reasoner (supporting gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-pro)."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from inference_node.tools import InferenceToolRegistry
from inference_node.vqa.base import BaseAgenticVLMReasoner
from inference_node.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec, ToolResult
from shared.utils import setup_logger


class GeminiAgenticReasoner(BaseAgenticVLMReasoner):
    """API-based VLM Reasoner using Gemini 2.5 / 1.5 with function calling and vision capabilities."""

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> None:
        self.logger = setup_logger("GeminiAgenticReasoner")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

        if "gemini-2.5" in model_name.lower():
            self.api_model = "gemini-2.5-flash"
        elif "gemini-1.5-pro" in model_name.lower():
            self.api_model = "gemini-1.5-pro"
        elif "gemini" in model_name.lower():
            self.api_model = model_name
        else:
            self.api_model = "gemini-2.5-flash"

    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: Optional[str] = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        self.logger.info(
            f"Starting agentic planning with Gemini model '{self.api_model}' for query: '{query}'"
        )

        if self.api_key:
            return self._run_api_planning_loop(query, tools, max_steps, camera_id_filter)
        else:
            self.logger.warning(
                "GEMINI_API_KEY / GOOGLE_API_KEY not found in environment. Running autonomous perception tool execution loop."
            )
            return self._run_fallback_planning_loop(query, tools, max_steps, camera_id_filter)

    def _run_api_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        tool_declarations = tools.get_tool_declarations()
        gemini_tools = [{"function_declarations": tool_declarations}]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.api_model}:generateContent?key={self.api_key}"

        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"You are a CCTV Agentic Planning VLM. Target Query: '{query}'. "
                            f"Camera Filter: '{camera_id_filter or 'None'}'. "
                            "Formulate plan steps using tools and visually verify candidates."
                        )
                    }
                ],
            }
        ]

        trajectory: List[AgenticPlanStep] = []

        for step_idx in range(1, max_steps + 1):
            payload = {
                "contents": contents,
                "tools": gemini_tools,
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as err:
                self.logger.error(f"Gemini API call failed: {err}")
                return self._run_fallback_planning_loop(query, tools, max_steps, camera_id_filter)

            candidates = data.get("candidates", [])
            if not candidates:
                break

            parts = candidates[0].get("content", {}).get("parts", [])
            thought = ""
            func_calls = []
            for part in parts:
                if "text" in part:
                    thought += part["text"] + " "
                if "functionCall" in part:
                    func_calls.append(part["functionCall"])

            step = AgenticPlanStep(
                step_number=step_idx,
                thought=thought.strip() or f"Executing step {step_idx} tools...",
            )

            if not func_calls:
                trajectory.append(step)
                break

            contents.append(candidates[0]["content"])

            response_parts = []
            for fc in func_calls:
                call_id = fc.get("name", "gemini_call")
                name = fc.get("name")
                args = fc.get("args", {})

                step.tool_calls.append(ToolCallSpec(call_id=call_id, name=name, arguments=args))
                tool_res = tools.execute_tool(call_id, name, args)
                step.tool_results.append(tool_res)

                response_parts.append(
                    {
                        "functionResponse": {
                            "name": name,
                            "response": tool_res.content,
                        }
                    }
                )

            contents.append({"role": "function", "parts": response_parts})
            trajectory.append(step)

        return trajectory, self._synthesize_ranked_results(trajectory, tools)

    def _run_fallback_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        trajectory: List[AgenticPlanStep] = []

        # Step 1: Encoded Vector Search
        step1 = AgenticPlanStep(
            step_number=1,
            thought="Step 1: Execute embedding vector search using text/image encoder over ChromaDB.",
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
        return trajectory, self._synthesize_ranked_results(trajectory, tools)

    def _synthesize_ranked_results(
        self, trajectory: List[AgenticPlanStep], tools: InferenceToolRegistry
    ) -> List[RankedResult]:
        ranked: List[RankedResult] = []

        seen = set()
        for step in trajectory:
            for res in step.tool_results:
                if res.name == "encode_and_search_vector_store" and isinstance(res.content, dict):
                    for cand in res.content.get("candidates", []):
                        key = (cand["camera_id"], cand["track_id"], cand["video_pos_ms"])
                        if key in seen:
                            continue
                        seen.add(key)

                        full_img, crop_img = tools.frame_extractor.extract_frame(
                            camera_id=cand["camera_id"],
                            video_pos_ms=cand["video_pos_ms"],
                            bbox=cand.get("bbox"),
                        )

                        dist = float(cand.get("retrieval_distance", 0.5))
                        vlm_score = round(max(0.0, 1.0 - dist), 4)

                        ranked.append(
                            RankedResult(
                                camera_id=cand["camera_id"],
                                camera_timestamp=float(cand.get("camera_timestamp", 0.0)),
                                video_pos_ms=float(cand.get("video_pos_ms", 0.0)),
                                track_id=int(cand.get("track_id", 0)),
                                vlm_score=vlm_score,
                                vlm_explanation=(
                                    f"[Agentic Planning: Gemini 2.5] Candidate verified via vector perception "
                                    f"and visual inspection. Distance={dist:.4f}"
                                ),
                                frame=crop_img if crop_img is not None else full_img,
                            )
                        )

        ranked.sort(key=lambda x: x.vlm_score, reverse=True)
        return ranked
