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


# Max characters for tool result text before truncation.
# Keeps context window manageable on <=16GB GPUs.
_MAX_TOOL_RESULT_CHARS = 3000


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

        target_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        resolved_device_map = self.device_map or "balanced"

        load_kwargs = {
            "dtype": target_dtype,
            "device_map": resolved_device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        self.logger.info(
            f"Loading with device_map='{resolved_device_map}', "
            f"dtype={target_dtype}, low_cpu_mem_usage=True"
        )

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, **load_kwargs,
            )
        except Exception as inner_err:
            self.logger.debug(
                f"AutoModelForImageTextToText failed ({inner_err}), "
                "trying Qwen3VLForConditionalGeneration"
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs,
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

    def _build_compact_system_prompt(self) -> str:
        """Build a compact system prompt for Qwen that avoids duplicating tool
        schemas (apply_chat_template already injects them via tools=...)."""
        return (
            "You are Qwen3-VL, an expert CCTV AI Agent performing multi-camera "
            "target search and visual reasoning.\n\n"
            "## Execution Model\n"
            "You operate in a ReAct loop: Think → Act → Observe → Think → ...\n"
            "- Produce only the NEXT logical step, not an entire plan.\n"
            "- Call ONE tool at a time. Wait for its result before deciding next.\n"
            "- Tool outputs are the ONLY source of truth. Never fabricate results.\n"
            "- When you have sufficient evidence, produce a final answer with NO tool calls.\n\n"
            "## Tool Rules\n"
            "- `encode_and_search_vector_store`: retrieves candidates. Does NOT verify.\n"
            "- `inspect_visual_candidate`: extracts a frame crop and returns it. "
            "YOU analyze the image and judge if it matches.\n"
            "- `get_temporal_context`: finds nearby events for relationship queries.\n"
            "- `query_metadata`: filters by camera_id/timestamp/class only.\n\n"
            "## Relationship Queries\n"
            "For queries like 'bus followed by car': verify first object, use "
            "`get_temporal_context` for the second, then inspect.\n\n"
            "## Final Answer\n"
            "When done, respond with analysis and:\n"
            "<final_answer>\n"
            '{"candidates": [{"camera_id": "...", "video_pos_ms": ..., '
            '"track_id": ..., "confidence": 0.0-1.0, '
            '"explanation": "evidence from inspection"}]}\n'
            "</final_answer>\n\n"
            "To call a tool, use:\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"key": "value"}}\n</tool_call>'
        )

    def _run_react_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        trajectory: List[AgenticPlanStep] = []
        tool_decls = tools.get_tool_declarations()

        # Use compact system prompt — apply_chat_template(tools=...) adds
        # the full tool schemas separately, so we don't duplicate them.
        system_text = self._build_compact_system_prompt()

        user_msg = self._build_user_message(query, camera_id_filter)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_msg},
        ]

        target_device = (
            self.model.device
            if hasattr(self.model, "device")
            else self.device
        )
        call_counter = 0

        for step_idx in range(1, max_steps + 1):
            # Free cached GPU memory before each generation
            torch.cuda.empty_cache()

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

            seq_len = inputs.input_ids.shape[-1]
            self.logger.info(f"[Step {step_idx}] Input sequence length: {seq_len} tokens")

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            # Free input tensors immediately
            del inputs
            torch.cuda.empty_cache()

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    generated_ids[:, :seq_len] if seq_len else generated_ids,
                    generated_ids,
                )
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            del generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

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

        # If the loop exhausted max_steps and the last step still had tool
        # calls (meaning the model never emitted a voluntary final answer),
        # make one extra generation with a forcing prompt and no tools.
        last_step_had_tools = trajectory and trajectory[-1].tool_calls
        if last_step_had_tools:
            self.logger.info(
                "ReAct loop exhausted max_steps without a final answer. "
                "Making one extra generation to force a summary."
            )
            # Append a forcing prompt as a user message
            messages.append({
                "role": "user",
                "content": self._build_final_answer_prompt(),
            })

            torch.cuda.empty_cache()
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception:
                image_inputs, video_inputs = None, None

            # Generate WITHOUT tool declarations to prevent more tool calls
            try:
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
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

            del inputs
            torch.cuda.empty_cache()

            seq_len_force = generated_ids.shape[-1]
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    generated_ids[:, :seq_len_force] if seq_len_force else generated_ids,
                    generated_ids,
                )
            ]
            forced_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            del generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

            if forced_text.strip():
                forced_step = AgenticPlanStep(
                    step_number=len(trajectory) + 1,
                    thought=forced_text.strip(),
                )
                trajectory.append(forced_step)
                self.logger.info("Received forced final answer from Qwen3-VL.")

        # Parse final answer from the last thought
        final_text = trajectory[-1].thought if trajectory else ""
        ranked_results = self._parse_final_answer(final_text, tools)

        return trajectory, ranked_results

    def _truncate_tool_content(self, content: Any) -> str:
        """Serialize tool content to JSON and truncate if too large.

        Large tool results (e.g., 20 search candidates with full metadata)
        can blow up the context window on memory-constrained GPUs. We keep
        the essential structure but truncate oversized payloads.
        """
        text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        if len(text) <= _MAX_TOOL_RESULT_CHARS:
            return text

        # For search results, keep only the most relevant fields per candidate
        if isinstance(content, dict) and "candidates" in content:
            compact = {
                "count": content.get("count", 0),
                "candidates": [],
            }
            for cand in content.get("candidates", []):
                compact["candidates"].append({
                    "camera_id": cand.get("camera_id"),
                    "track_id": cand.get("track_id"),
                    "video_pos_ms": cand.get("video_pos_ms"),
                    "bbox": cand.get("bbox"),
                    "retrieval_distance": cand.get("retrieval_distance"),
                })
            text = json.dumps(compact)
            if len(text) <= _MAX_TOOL_RESULT_CHARS:
                return text

        # Hard truncation as last resort
        return text[:_MAX_TOOL_RESULT_CHARS] + '... (truncated)'

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
        content_text = self._truncate_tool_content(tool_res.content)

        if tool_res.extracted_images:
            # Resize images to reduce vision encoder memory
            resized_images = []
            for img in tool_res.extracted_images:
                img_copy = img.copy()
                img_copy.thumbnail((384, 384))
                resized_images.append(img_copy)

            # Multi-modal response: text + images
            content_list: List[Dict[str, Any]] = [
                {"type": "text", "text": content_text}
            ]
            for img in resized_images:
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
                    ] + [{"type": "image", "image": img} for img in resized_images],
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
