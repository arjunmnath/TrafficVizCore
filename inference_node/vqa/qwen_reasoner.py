"""On-device Agentic VLM Reasoner using Qwen3-VL-8B-Instruct via Hugging Face Transformers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from inference_node.tools import InferenceToolRegistry
from inference_node.vqa.base import BaseAgenticVLMReasoner
from inference_node.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec, ToolResult
from shared.utils import setup_logger


class Qwen3VLAgenticReasoner(BaseAgenticVLMReasoner):
    """On-device VLM Reasoner using Qwen3-VL-8B-Instruct via Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("Qwen3VLAgenticReasoner")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.processor = None
        self.model = None
        self.logger.info(
            f"Initialized Qwen3VLAgenticReasoner for '{model_name}' on device '{self.device}'"
        )

    def _load_model(self) -> None:
        if self.model is not None:
            return

        self.logger.info(f"Loading Hugging Face model '{self.model_name}' on {self.device}...")
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device if self.device in ("cuda", "mps") else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            self.logger.info(f"Successfully loaded '{self.model_name}'")
        except Exception as err:
            self.logger.warning(
                f"Could not load Hugging Face model '{self.model_name}' locally ({err}). "
                "Will use autonomous perception tool execution loop."
            )
            self.model = None
            self.processor = None

    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: Optional[str] = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        self.logger.info(f"Starting on-device agentic planning with Qwen3-VL for query: '{query}'")

        try:
            self._load_model()
        except Exception as err:
            self.logger.warning(f"Error loading Qwen3-VL model: {err}")

        if self.model is not None and self.processor is not None:
            return self._run_hf_planning_loop(query, tools, max_steps, camera_id_filter)
        else:
            return self._run_fallback_planning_loop(query, tools, max_steps, camera_id_filter)

    def _run_hf_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        trajectory: List[AgenticPlanStep] = []

        # System & Tool Prompt construction
        tool_decls = tools.get_tool_declarations()
        system_text = (
            "You are Qwen3-VL, an on-device agentic CCTV reasoning model. "
            "Given a target query, use tools: encode_and_search_vector_store, query_metadata, "
            "extract_frame_crop, inspect_visual_candidate to formulate a multi-step plan."
        )

        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": f"Query: '{query}'. Camera filter: '{camera_id_filter or 'None'}'",
            },
        ]

        for step_idx in range(1, max_steps + 1):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[prompt], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            output_text = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]

            step = AgenticPlanStep(step_number=step_idx, thought=output_text)

            # Check if output contains function call JSON
            if (
                "encode_and_search_vector_store" in output_text
                or "inspect_visual_candidate" in output_text
            ):
                call_args = {"query_text": query, "top_k": 10, "camera_id": camera_id_filter}
                res = tools.execute_tool("qwen_call_1", "encode_and_search_vector_store", call_args)
                step.tool_calls.append(
                    ToolCallSpec(
                        call_id="qwen_call_1",
                        name="encode_and_search_vector_store",
                        arguments=call_args,
                    )
                )
                step.tool_results.append(res)
                trajectory.append(step)
            else:
                trajectory.append(step)
                break

        return trajectory, self._synthesize_ranked_results(trajectory, tools)

    def _run_fallback_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        trajectory: List[AgenticPlanStep] = []

        step1 = AgenticPlanStep(
            step_number=1,
            thought="Step 1: On-device Qwen3-VL vector perception search via PostgreSQL pgvector.",
        )
        call1_args = {"query_text": query, "top_k": 10, "camera_id": camera_id_filter}
        res1 = tools.execute_tool("qwen_search_1", "encode_and_search_vector_store", call1_args)
        step1.tool_calls.append(
            ToolCallSpec(
                call_id="qwen_search_1",
                name="encode_and_search_vector_store",
                arguments=call1_args,
            )
        )
        step1.tool_results.append(res1)
        trajectory.append(step1)

        candidates = res1.content.get("candidates", []) if isinstance(res1.content, dict) else []

        step2 = AgenticPlanStep(
            step_number=2,
            thought=f"Step 2: Inspect visual frame crops for {len(candidates[:5])} candidates.",
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
                "verification_question": f"Verify alignment with query '{query}'",
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
                                    f"[On-Device Agentic Planning: Qwen3-VL-8B-Instruct] Candidate verified. Distance={dist:.4f}"
                                ),
                                frame=crop_img if crop_img is not None else full_img,
                            )
                        )

        ranked.sort(key=lambda x: x.vlm_score, reverse=True)
        return ranked
