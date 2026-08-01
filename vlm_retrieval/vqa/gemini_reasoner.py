"""Gemini Agentic VLM Reasoner — ReAct-style execution loop.

Uses Gemini's native function calling protocol (functionCall / functionResponse).
When inspect_visual_candidate returns images, they are encoded as inline image
parts in the function response so Gemini can actually see the crop.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    from vlm_retrieval.tools import InferenceToolRegistry
from vlm_retrieval.vqa.base import BaseAgenticVLMReasoner
from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec
from shared.utils import setup_logger


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class GeminiAgenticReasoner(BaseAgenticVLMReasoner):
    """API-based VLM Reasoner using Gemini with function calling and vision."""

    def __init__(self, model_name: str = "gemini-3.5-flash", api_key: Optional[str] = None) -> None:
        self.logger = setup_logger("GeminiAgenticReasoner")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

        m_lower = model_name.lower()
        if "gemini-3.5" in m_lower or "3.5" in m_lower:
            self.api_model = "gemini-3.5-flash"
        elif "gemini-3.1" in m_lower or "3.1" in m_lower:
            self.api_model = "gemini-3.1-pro-preview"
        elif "gemini-flash-latest" in m_lower:
            self.api_model = "gemini-flash-latest"
        elif "gemini-2.5" in m_lower:
            self.api_model = "gemini-2.5-flash"
        elif "gemini-1.5-pro" in m_lower:
            self.api_model = "gemini-1.5-pro"
        elif "gemini" in m_lower:
            self.api_model = model_name
        else:
            self.api_model = "gemini-3.5-flash"

    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: Optional[str] = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        self.logger.info(
            f"Starting ReAct loop with Gemini model '{self.api_model}' for query: '{query}'"
        )

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY / GOOGLE_API_KEY not found in environment. "
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
        tool_declarations = tools.get_tool_declarations()
        gemini_tools = [{"function_declarations": tool_declarations}]

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.api_model}:generateContent?key={self.api_key}"
        )

        system_prompt = self._build_system_prompt(tools, model_display_name="Gemini CCTV Agent")
        user_prompt = self._build_user_message(query, camera_id_filter)

        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
            }
        ]

        trajectory: List[AgenticPlanStep] = []

        for step_idx in range(1, max_steps + 1):
            payload = {
                "contents": contents,
                "tools": gemini_tools,
            }

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as http_err:
                # Read the response body for actionable error details
                error_body = ""
                try:
                    error_body = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                self.logger.error(
                    f"Gemini API HTTP {http_err.code} at step {step_idx}: "
                    f"{http_err.reason}\nResponse: {error_body[:1000]}"
                )
                if http_err.code == 403:
                    raise RuntimeError(
                        f"Gemini API returned 403 Forbidden. Check that:\n"
                        f"  1. Your GEMINI_API_KEY / GOOGLE_API_KEY is valid\n"
                        f"  2. The Generative Language API is enabled in your Google Cloud project\n"
                        f"  3. The model '{self.api_model}' is available for your API key\n"
                        f"API response: {error_body[:500]}"
                    ) from http_err
                raise RuntimeError(
                    f"Gemini API call failed (HTTP {http_err.code}): {error_body[:500]}"
                ) from http_err
            except Exception as err:
                self.logger.error(f"Gemini API call failed at step {step_idx}: {err}")
                raise RuntimeError(f"Gemini API call failed: {err}") from err

            candidates = data.get("candidates", [])
            if not candidates:
                self.logger.warning("Gemini returned no candidates. Ending loop.")
                break

            response_content = candidates[0].get("content", {})
            parts = response_content.get("parts", [])

            # Extract thought text and function calls from response
            thought = ""
            func_calls = []
            for part in parts:
                if "text" in part:
                    thought += part["text"] + " "
                if "functionCall" in part:
                    func_calls.append(part["functionCall"])

            step = AgenticPlanStep(
                step_number=step_idx,
                thought=thought.strip() or f"Step {step_idx}: processing...",
            )

            # If no function calls → this is the final answer
            if not func_calls:
                trajectory.append(step)
                self.logger.info(f"[Step {step_idx}] Final answer (no tool calls)")
                break

            # Append assistant response to conversation
            contents.append(response_content)

            # Execute each tool call and build function response parts
            response_parts = []
            for fc in func_calls:
                name = fc.get("name")
                args = fc.get("args", {})
                call_id = f"gemini_s{step_idx}_{name}"

                step.tool_calls.append(
                    ToolCallSpec(call_id=call_id, name=name, arguments=args)
                )
                tool_res = tools.execute_tool(call_id, name, args)
                step.tool_results.append(tool_res)

                # Build function response with text content
                fn_response_parts: List[Dict[str, Any]] = []

                # Add the structured content as text
                content_text = json.dumps(tool_res.content) if isinstance(
                    tool_res.content, (dict, list)
                ) else str(tool_res.content)
                fn_response_parts.append({"text": content_text})

                # If images were extracted, encode them inline for vision
                for img in tool_res.extracted_images:
                    try:
                        buf = io.BytesIO()
                        img_copy = img.copy()
                        img_copy.thumbnail((512, 512))
                        img_copy.save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        fn_response_parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64,
                            }
                        })
                    except Exception as img_err:
                        self.logger.warning(f"Failed to encode image for Gemini: {img_err}")

                response_parts.append(
                    {
                        "functionResponse": {
                            "name": name,
                            "response": {"parts": fn_response_parts},
                        }
                    }
                )

            contents.append({"role": "function", "parts": response_parts})
            trajectory.append(step)
            self.logger.info(
                f"[Step {step_idx}] Executed {len(func_calls)} tool call(s): "
                f"{[fc.get('name') for fc in func_calls]}"
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
            contents.append({
                "role": "user",
                "parts": [{"text": self._build_final_answer_prompt()}],
            })

            # Call Gemini WITHOUT tools so it cannot emit more function calls
            force_payload = {"contents": contents}
            force_req = urllib.request.Request(
                url,
                data=json.dumps(force_payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(force_req) as resp:
                    force_data = json.loads(resp.read().decode("utf-8"))
                force_candidates = force_data.get("candidates", [])
                if force_candidates:
                    force_parts = force_candidates[0].get("content", {}).get("parts", [])
                    forced_text = " ".join(p.get("text", "") for p in force_parts).strip()
                    if forced_text:
                        # Record as an extra trajectory step
                        forced_step = AgenticPlanStep(
                            step_number=len(trajectory) + 1,
                            thought=forced_text,
                        )
                        trajectory.append(forced_step)
                        self.logger.info("Received forced final answer from Gemini.")
            except Exception as err:
                self.logger.warning(f"Failed to get forced final answer from Gemini: {err}")

        # Parse final answer from the last thought
        final_text = trajectory[-1].thought if trajectory else ""
        ranked_results = self._parse_final_answer(final_text, tools)

        return trajectory, ranked_results
