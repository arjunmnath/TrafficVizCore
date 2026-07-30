"""Gemini Agentic VLM Reasoner (supporting gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-pro)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from vlm_retrieval.tools import InferenceToolRegistry
from vlm_retrieval.vqa.base import BaseAgenticVLMReasoner
from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult, ToolCallSpec
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
            return self._run_fallback_planning_loop(
                query, tools, max_steps, camera_id_filter, model_label="Gemini 2.5"
            )

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

        system_prompt = self._build_system_prompt(tools, model_display_name="Gemini CCTV Agent")
        user_prompt = self._build_user_message(query, camera_id_filter)

        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{system_prompt}\n\n{user_prompt}"
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
                return self._run_fallback_planning_loop(
                    query, tools, max_steps, camera_id_filter, model_label="Gemini 2.5"
                )

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

        return trajectory, self._synthesize_ranked_results(
            trajectory, tools, model_label="Gemini 2.5"
        )

