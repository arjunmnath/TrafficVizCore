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
            return self._run_fallback_planning_loop(
                query,
                tools,
                max_steps,
                camera_id_filter,
                model_label="Qwen3-VL-8B-Instruct",
                require_inspection_for_score=True,
            )

    def _run_hf_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        trajectory: List[AgenticPlanStep] = []

        # System & Tool Prompt construction using base helpers
        tool_decls = tools.get_tool_declarations()
        system_text = (
            self._build_system_prompt(tools, model_display_name="Qwen3-VL")
            + "\n\nTo invoke a tool, output JSON in a tool_call block:\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"arg1": "val1"}}\n</tool_call>'
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": self._build_user_message(query, camera_id_filter)},
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

        return trajectory, self._synthesize_ranked_results(
            trajectory,
            tools,
            model_label="Qwen3-VL-8B-Instruct",
            require_inspection_for_score=True,
        )

    def _format_tool_result_for_messages(self, tool_res: ToolResult) -> Any:
        """Format tool result for conversation history, incorporating vision inputs if images extracted."""
        content_text = json.dumps(tool_res.content) if isinstance(tool_res.content, (dict, list)) else str(tool_res.content)
        if tool_res.extracted_images:
            content_list: List[Dict[str, Any]] = [{"type": "text", "text": content_text}]
            for img in tool_res.extracted_images:
                content_list.append({"type": "image", "image": img})
            return content_list
        return content_text


