"""OpenAI Agentic VLM Reasoner (supporting openai-5.6, gpt-4o, gpt-4.5)."""

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

        if self.api_key:
            return self._run_api_planning_loop(query, tools, max_steps, camera_id_filter)
        else:
            self.logger.warning(
                "OPENAI_API_KEY not found in environment. Running autonomous perception tool execution loop."
            )
            return self._run_fallback_planning_loop(
                query, tools, max_steps, camera_id_filter, model_label="OpenAI 5.6"
            )

    def _run_api_planning_loop(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int,
        camera_id_filter: Optional[str],
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        system_prompt = self._build_system_prompt(tools, model_display_name="OpenAI CCTV Agent")
        user_message = self._build_user_message(query, camera_id_filter)

        tool_decls = [
            {"type": "function", "function": tool} for tool in tools.get_tool_declarations()
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
            except Exception as err:
                self.logger.error(f"OpenAI API call failed: {err}")
                return self._run_fallback_planning_loop(
                    query, tools, max_steps, camera_id_filter, model_label="OpenAI 5.6"
                )

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

        return trajectory, self._synthesize_ranked_results(
            trajectory, tools, model_label="OpenAI 5.6"
        )

