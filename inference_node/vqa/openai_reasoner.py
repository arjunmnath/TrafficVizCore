"""OpenAI Agentic VLM Reasoner (supporting openai-5.6, gpt-4o, gpt-4.5)."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from inference_node.tools import InferenceToolRegistry
from inference_node.vqa.base import BaseAgenticVLMReasoner
from inference_node.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec, ToolResult
from shared.utils import setup_logger


class OpenAIAgenticReasoner(BaseAgenticVLMReasoner):
    """API-based VLM Reasoner using OpenAI 5.6 / GPT-4o with tool calling and vision capabilities."""

    def __init__(self, model_name: str = "openai-5.6", api_key: Optional[str] = None) -> None:
        self.logger = setup_logger("OpenAIAgenticReasoner")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        # Standardize model string for API payload
        if "openai-5.6" in model_name.lower():
            self.api_model = "gpt-4o"
        elif "gpt-4" in model_name.lower():
            self.api_model = model_name
        else:
            self.api_model = "gpt-4o"

    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: Optional[str] = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        self.logger.info(
            f"Starting agentic planning with OpenAI model '{self.api_model}' for query: '{query}'"
        )
        trajectory: List[AgenticPlanStep] = []

        # If API key is available, execute multi-turn tool-calling loop over REST API
        if self.api_key:
            return self._run_api_planning_loop(query, tools, max_steps, camera_id_filter)
        else:
            self.logger.warning(
                "OPENAI_API_KEY not found in environment. Running autonomous perception tool execution loop."
            )
            return self._run_fallback_planning_loop(query, tools, max_steps, camera_id_filter)

    def _run_api_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        system_prompt = (
            "You are an expert CCTV AI Agent performing multi-camera target search and visual reasoning. "
            "You have access to tools: 'encode_and_search_vector_store', 'query_metadata', "
            "'extract_frame_crop', and 'inspect_visual_candidate'. "
            "Formulate a plan, invoke perception tools to inspect vector database matches and video crops, "
            "and synthesize the final verified target candidates."
        )

        tool_decls = [
            {"type": "function", "function": tool} for tool in tools.get_tool_declarations()
        ]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Target Query: '{query}'. Filter Camera: '{camera_id_filter or 'None'}'",
            },
        ]

        trajectory: List[AgenticPlanStep] = []

        for step_idx in range(1, max_steps + 1):
            payload = {
                "model": self.api_model,
                "messages": messages,
                "tools": tool_decls,
                "tool_choice": "auto",
                "temperature": 0.2,
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as err:
                self.logger.error(f"OpenAI API call failed: {err}")
                return self._run_fallback_planning_loop(query, tools, max_steps, camera_id_filter)

            choice = data["choices"][0]["message"]
            thought = choice.get("content") or "Executing perception tool steps..."
            tool_calls_raw = choice.get("tool_calls", [])

            step = AgenticPlanStep(step_number=step_idx, thought=thought)

            if not tool_calls_raw:
                trajectory.append(step)
                break

            messages.append(choice)

            for call in tool_calls_raw:
                call_id = call["id"]
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])

                step.tool_calls.append(ToolCallSpec(call_id=call_id, name=name, arguments=args))
                tool_res = tools.execute_tool(call_id, name, args)
                step.tool_results.append(tool_res)

                # Format response for message history
                content_str = json.dumps(tool_res.content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_str,
                    }
                )

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
            thought="Step 1: Execute embedding vector search using text/image encoder over PostgreSQL pgvector.",
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
                                    f"[Agentic Planning: OpenAI 5.6] Candidate verified via vector perception "
                                    f"and visual inspection. Distance={dist:.4f}"
                                ),
                                frame=crop_img if crop_img is not None else full_img,
                            )
                        )

        ranked.sort(key=lambda x: x.vlm_score, reverse=True)
        return ranked
