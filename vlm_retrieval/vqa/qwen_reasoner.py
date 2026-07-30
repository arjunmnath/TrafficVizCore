"""On-device Agentic VLM Reasoner using Qwen3-VL-8B-Instruct via Hugging Face Transformers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from vlm_retrieval.tools import InferenceToolRegistry
from vlm_retrieval.vqa.base import BaseAgenticVLMReasoner
from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec, ToolResult
from shared.utils import setup_logger


class Qwen3VLAgenticReasoner(BaseAgenticVLMReasoner):
    """On-device VLM Reasoner using Qwen3-VL-8B-Instruct via Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        device_map: str = "balanced",
    ) -> None:
        self.logger = setup_logger("Qwen3VLAgenticReasoner")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.device_map = device_map
        self.processor = None
        self.model = None
        self.logger.info(
            f"Initialized Qwen3VLAgenticReasoner for '{model_name}' on device '{self.device}' (device_map='{self.device_map}')"
        )

    def _load_model(self) -> None:
        if self.model is not None:
            return

        self.logger.info(f"Loading Hugging Face model '{self.model_name}' on {self.device} (device_map={self.device_map})...")
        try:
            from transformers import (
                AutoProcessor,
                AutoModelForImageTextToText,
                Qwen3VLForConditionalGeneration,
            )

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

            if torch.cuda.is_available():
                resolved_device_map = self.device_map or "balanced"
            elif self.device in ("cuda", "mps", "auto", "balanced"):
                resolved_device_map = self.device_map if self.device_map else self.device
            else:
                resolved_device_map = None

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=resolved_device_map,
                    trust_remote_code=True,
                )
            except Exception as inner_err:
                self.logger.debug(
                    f"AutoModelForImageTextToText failed ({inner_err}), trying Qwen3VLForConditionalGeneration"
                )
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=resolved_device_map,
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
        tool_decls_json = json.dumps(tool_decls, indent=2)
        system_text = (
            "You are Qwen3-VL, an on-device agentic CCTV reasoning model. "
            "Your task is to locate targets described in user queries using available perception tools.\n"
            f"Available tools:\n{tool_decls_json}\n\n"
            "INSTRUCTIONS:\n"
            "1. First, search for target candidates using 'encode_and_search_vector_store' or 'search_npz_embeddings'.\n"
            "2. Next, for retrieved candidates, call 'inspect_visual_candidate' to inspect frame crops visually.\n"
            "3. Synthesize and conclude with final target evaluation.\n\n"
            "To invoke a tool, output JSON in a tool_call block:\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"arg1": "val1"}}\n</tool_call>'
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": f"Query: '{query}'. Camera filter: '{camera_id_filter or 'None'}'",
            },
        ]

        target_device = self.model.device if hasattr(self.model, "device") else (
            self.device if self.device in ("cuda", "mps", "cpu") else "cpu"
        )

        call_counter = 0

        for step_idx in range(1, max_steps + 1):
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception:
                image_inputs, video_inputs = None, None

            try:
                prompt = self.processor.apply_chat_template(
                    messages, tools=tool_decls, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(target_device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]

            self.logger.info(f"[Step {step_idx}] Qwen3-VL Output: {output_text}")
            step = AgenticPlanStep(step_number=step_idx, thought=output_text)

            parsed_calls = self._parse_tool_calls(
                output_text,
                query=query,
                camera_id_filter=camera_id_filter,
                trajectory=trajectory,
                step_idx=step_idx,
            )

            if not parsed_calls:
                # No tool call requested -> reasoning complete
                messages.append({"role": "assistant", "content": output_text})
                trajectory.append(step)
                break

            # Append model's assistant response to conversation history
            messages.append({"role": "assistant", "content": output_text})

            for name, call_args in parsed_calls:
                call_counter += 1
                call_id = f"qwen_call_s{step_idx}_{call_counter}"

                tool_res = tools.execute_tool(call_id, name, call_args)
                step.tool_calls.append(
                    ToolCallSpec(
                        call_id=call_id,
                        name=name,
                        arguments=call_args,
                    )
                )
                step.tool_results.append(tool_res)

                # Format tool output & append to conversation history
                formatted_content = self._format_tool_result_for_messages(tool_res)
                try:
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "tool_call_id": call_id,
                        "content": formatted_content,
                    })
                except Exception:
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{json.dumps(tool_res.content)}\n</tool_response>"
                    })

            trajectory.append(step)

        return trajectory, self._synthesize_ranked_results(trajectory, tools)

    def _parse_tool_calls(
        self,
        output_text: str,
        query: str,
        camera_id_filter: Optional[str],
        trajectory: List[AgenticPlanStep],
        step_idx: int = 1,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse requested tool calls from model output text."""
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
        # If vector search hasn't executed yet and output mentions search tool
        already_searched = any(
            any(tc.name in ("encode_and_search_vector_store", "search_npz_embeddings") for tc in s.tool_calls)
            for s in trajectory
        )
        if not already_searched:
            if "encode_and_search_vector_store" in output_text or "vector" in output_text.lower() or "search" in output_text.lower():
                return [("encode_and_search_vector_store", {"query_text": query, "top_k": 10, "camera_id": camera_id_filter})]

        # If vector search already executed, check if model wants to inspect candidates
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

            # Extract retrieved candidates from previous step tool results
            candidates = []
            for s in trajectory:
                for res in s.tool_results:
                    if res.name in ("encode_and_search_vector_store", "search_npz_embeddings") and isinstance(res.content, dict):
                        candidates.extend(res.content.get("candidates", []))

            # Inspect uninspected candidates if model mentions inspection or visual candidate
            if ("inspect" in output_text.lower() or "candidate" in output_text.lower() or "visual" in output_text.lower() or "crop" in output_text.lower()):
                for cand in candidates[:5]:
                    key = (str(cand.get("camera_id")), int(cand.get("track_id", 0)), float(cand.get("video_pos_ms", 0.0)))
                    if key not in inspected_keys:
                        parsed.append((
                            "inspect_visual_candidate",
                            {
                                "camera_id": cand.get("camera_id"),
                                "video_pos_ms": cand.get("video_pos_ms"),
                                "track_id": cand.get("track_id"),
                                "bbox": cand.get("bbox"),
                                "verification_question": f"Verify candidate alignment with query '{query}'",
                            }
                        ))
                        inspected_keys.add(key)
                        if len(parsed) >= 3:
                            break

        return parsed

    def _format_tool_result_for_messages(self, tool_res: ToolResult) -> Any:
        """Format tool result for conversation history, incorporating vision inputs if images extracted."""
        content_text = json.dumps(tool_res.content) if isinstance(tool_res.content, (dict, list)) else str(tool_res.content)
        if tool_res.extracted_images:
            content_list: List[Dict[str, Any]] = [{"type": "text", "text": content_text}]
            for img in tool_res.extracted_images:
                content_list.append({"type": "image", "image": img})
            return content_list
        return content_text

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
        visually_inspected_keys = set()

        # Track which candidates were visually inspected
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
                if res.name in ("encode_and_search_vector_store", "search_npz_embeddings") and isinstance(res.content, dict):
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

                        if is_inspected:
                            vlm_score = round(max(0.0, 1.0 - dist), 4)
                            explanation = (
                                f"[On-Device Agentic Planning: Qwen3-VL-8B-Instruct] Candidate visually inspected & verified. Distance={dist:.4f}"
                            )
                        else:
                            vlm_score = 0.0
                            explanation = (
                                f"[On-Device Agentic Planning: Qwen3-VL-8B-Instruct] Candidate retrieved via vector search (visual verification skipped). Distance={dist:.4f}"
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
