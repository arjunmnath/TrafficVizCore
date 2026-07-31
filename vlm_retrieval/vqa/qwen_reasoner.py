"""On-device Agentic VLM Reasoner using Qwen3-VL via Hugging Face Transformers.

ReAct-style execution loop using <tool_call> XML tags for tool invocation.
When inspect_visual_candidate returns images, they are injected as PIL image
objects into the next message for Qwen's vision encoder to process.
"""

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
    """On-device VLM Reasoner using Qwen3-VL via Hugging Face Transformers."""

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
            f"Initialized Qwen3VLAgenticReasoner for '{model_name}' on device "
            f"'{self.device}' (device_map='{self.device_map}')"
        )

    def _load_model(self) -> None:
        if self.model is not None:
            return

        self.logger.info(
            f"Loading Hugging Face model '{self.model_name}' on {self.device} "
            f"(device_map={self.device_map})..."
        )
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            Qwen3VLForConditionalGeneration,
        )

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        resolved_device_map = self.device_map or "balanced"

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=resolved_device_map,
                trust_remote_code=True,
            )
        except Exception as inner_err:
            self.logger.debug(
                f"AutoModelForImageTextToText failed ({inner_err}), "
                "trying Qwen3VLForConditionalGeneration"
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=resolved_device_map,
                trust_remote_code=True,
            )
        self.model.eval()
        self.logger.info(f"Successfully loaded '{self.model_name}'")

    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: Optional[str] = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        self.logger.info(
            f"Starting ReAct loop with Qwen3-VL for query: '{query}'"
        )

        self._load_model()

        if self.model is None or self.processor is None:
            raise RuntimeError(
                f"Failed to load Qwen3-VL model '{self.model_name}'. "
                "Cannot run agentic planning without a loaded VLM."
            )

        return self._run_react_loop(query, tools, max_steps, camera_id_filter)

    def _run_react_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        trajectory: List[AgenticPlanStep] = []
        tool_decls = tools.get_tool_declarations()

        # Build system prompt with tool call format instructions
        system_text = (
            self._build_system_prompt(tools, model_display_name="Qwen3-VL")
            + "\n\nTo invoke a tool, output JSON in a tool_call block:\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"arg1": "val1"}}\n</tool_call>'
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": self._build_user_message(query, camera_id_filter)},
        ]

        target_device = (
            self.model.device
            if hasattr(self.model, "device")
            else self.device
        )
        call_counter = 0

        for step_idx in range(1, max_steps + 1):
            # Process vision inputs from message history
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception:
                image_inputs, video_inputs = None, None

            # Tokenize and generate
            try:
                prompt = self.processor.apply_chat_template(
                    messages,
                    tools=tool_decls,
                    tokenize=False,
                    add_generation_prompt=True,
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]

            self.logger.info(f"[Step {step_idx}] Qwen3-VL Output: {output_text}")
            step = AgenticPlanStep(step_number=step_idx, thought=output_text)

            # Parse tool calls — no keyword fallback
            parsed_calls = self._parse_tool_calls_from_text(output_text)

            if not parsed_calls:
                # No tool call → this is the final answer
                messages.append({"role": "assistant", "content": output_text})
                trajectory.append(step)
                self.logger.info(f"[Step {step_idx}] Final answer (no tool calls)")
                break

            # Append assistant response to conversation
            messages.append({"role": "assistant", "content": output_text})

            # Execute each tool call (typically one per step in ReAct)
            for name, call_args in parsed_calls:
                call_counter += 1
                call_id = f"qwen_s{step_idx}_{call_counter}"

                tool_res = tools.execute_tool(call_id, name, call_args)
                step.tool_calls.append(
                    ToolCallSpec(call_id=call_id, name=name, arguments=call_args)
                )
                step.tool_results.append(tool_res)

                # Build tool response message with vision support
                tool_message = self._format_tool_response(name, call_id, tool_res)
                messages.append(tool_message)

            trajectory.append(step)
            self.logger.info(
                f"[Step {step_idx}] Executed {len(parsed_calls)} tool call(s): "
                f"{[n for n, _ in parsed_calls]}"
            )

        # Parse final answer from the last thought
        final_text = trajectory[-1].thought if trajectory else ""
        ranked_results = self._parse_final_answer(final_text, tools)

        return trajectory, ranked_results

    def _format_tool_response(
        self,
        tool_name: str,
        call_id: str,
        tool_res: ToolResult,
    ) -> Dict[str, Any]:
        """Format tool result for conversation history with vision support.

        If the tool returned extracted images, they are included as image
        content parts so Qwen's vision encoder can process them.
        """
        content_text = (
            json.dumps(tool_res.content)
            if isinstance(tool_res.content, (dict, list))
            else str(tool_res.content)
        )

        if tool_res.extracted_images:
            # Multi-modal response: text + images
            content_list: List[Dict[str, Any]] = [
                {"type": "text", "text": content_text}
            ]
            for img in tool_res.extracted_images:
                content_list.append({"type": "image", "image": img})

            # Try native tool role first, fall back to user role
            try:
                return {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": call_id,
                    "content": content_list,
                }
            except Exception:
                return {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<tool_response>\n{content_text}\n</tool_response>"}
                    ] + [{"type": "image", "image": img} for img in tool_res.extracted_images],
                }
        else:
            # Text-only response
            try:
                return {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": call_id,
                    "content": content_text,
                }
            except Exception:
                return {
                    "role": "user",
                    "content": f"<tool_response>\n{content_text}\n</tool_response>",
                }
