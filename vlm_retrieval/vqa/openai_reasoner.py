"""OpenAI Agentic VLM Reasoner — ReAct-style execution loop.

Uses OpenAI's native tool_calls protocol. When inspect_visual_candidate
returns images, they are encoded as base64 image content in tool response
messages so the VLM can actually see the crops.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    from vlm_retrieval.tools import InferenceToolRegistry
from vlm_retrieval.vqa.base import BaseAgenticVLMReasoner
from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec
from shared.utils import setup_logger


class OpenAIAgenticReasoner(BaseAgenticVLMReasoner):
    """API-based VLM Reasoner using OpenAI with tool calling and vision."""

    def __init__(self, model_name: str = "openai-5.6", api_key: Optional[str] = None) -> None:
        self.logger = setup_logger("OpenAIAgenticReasoner")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

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
            f"Starting ReAct loop with OpenAI model '{self.api_model}' for query: '{query}'"
        )

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment. "
                "Cannot run agentic planning without an LLM."
            )

        return self._run_react_loop(query, tools, max_steps, camera_id_filter)

    def _run_react_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        system_prompt = self._build_system_prompt(tools, model_display_name="OpenAI CCTV Agent")
        user_message = self._build_user_message(query, camera_id_filter)

        tool_decls = [
            {"type": "function", "function": tool}
            for tool in tools.get_tool_declarations()
        ]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
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
            except urllib.error.HTTPError as http_err:
                error_body = ""
                try:
                    error_body = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if http_err.code == 429:
                    err_msg = f"API Rate Limit / Quota Exceeded (HTTP 429): {error_body[:500]}"
                else:
                    err_msg = f"OpenAI API Error HTTP {http_err.code} ({http_err.reason}): {error_body[:500]}"
                self.logger.error(err_msg)
                raise RuntimeError(f"Terminating execution: {err_msg}") from http_err
            except Exception as err:
                self.logger.error(f"OpenAI API call failed at step {step_idx}: {err}")
                raise RuntimeError(f"Terminating execution due to OpenAI API failure: {err}") from err

            choice = data["choices"][0]["message"]
            thought = choice.get("content") or ""
            tool_calls_raw = choice.get("tool_calls", [])

            step = AgenticPlanStep(
                step_number=step_idx,
                thought=thought or f"Step {step_idx}: processing...",
            )

            # If no tool calls → this is the final answer
            if not tool_calls_raw:
                trajectory.append(step)
                self.logger.info(f"[Step {step_idx}] Final answer (no tool calls)")
                break

            # Append assistant message (with tool_calls) to conversation
            messages.append(choice)

            # Execute each tool call
            for call in tool_calls_raw:
                call_id = call["id"]
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])

                step.tool_calls.append(
                    ToolCallSpec(call_id=call_id, name=name, arguments=args)
                )
                tool_res = tools.execute_tool(call_id, name, args)
                step.tool_results.append(tool_res)

                # Build tool response message
                content_parts: List[Dict[str, Any]] = []

                # Text content with structured tool output
                content_text = json.dumps(tool_res.content) if isinstance(
                    tool_res.content, (dict, list)
                ) else str(tool_res.content)
                content_parts.append({"type": "text", "text": content_text})

                # Encode extracted images as base64 for vision
                for img in tool_res.extracted_images:
                    try:
                        buf = io.BytesIO()
                        img_copy = img.copy()
                        img_copy.thumbnail((512, 512))
                        img_copy.save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        })
                    except Exception as img_err:
                        self.logger.warning(f"Failed to encode image for OpenAI: {img_err}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content_parts if len(content_parts) > 1 else content_text,
                })

            trajectory.append(step)
            self.logger.info(
                f"[Step {step_idx}] Executed {len(tool_calls_raw)} tool call(s): "
                f"{[c['function']['name'] for c in tool_calls_raw]}"
            )

        # If the loop exhausted max_steps and the last step still had tool
        # calls (meaning the model never emitted a voluntary final answer),
        # make one extra API call with tools disabled to force a summary.
        last_step_had_tools = trajectory and trajectory[-1].tool_calls
        if last_step_had_tools:
            self.logger.info(
                "ReAct loop exhausted max_steps without a final answer. "
                "Making one extra call to force a summary."
            )
            # Append a forcing prompt
            messages.append({
                "role": "user",
                "content": self._build_final_answer_prompt(),
            })

            # Call OpenAI WITHOUT tools so it cannot emit more tool calls
            force_payload = {
                "model": self.api_model,
                "messages": messages,
                "temperature": 0.2,
            }
            force_req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(force_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(force_req) as resp:
                    force_data = json.loads(resp.read().decode("utf-8"))
                forced_text = force_data["choices"][0]["message"].get("content", "")
                if forced_text.strip():
                    forced_step = AgenticPlanStep(
                        step_number=len(trajectory) + 1,
                        thought=forced_text.strip(),
                    )
                    trajectory.append(forced_step)
                    self.logger.info("Received forced final answer from OpenAI.")
            except Exception as err:
                self.logger.warning(f"Failed to get forced final answer from OpenAI: {err}")

        # Parse final answer from the last thought
        final_text = trajectory[-1].thought if trajectory else ""
        ranked_results = self._parse_final_answer(final_text, tools)

        return trajectory, ranked_results
