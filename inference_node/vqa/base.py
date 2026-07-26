"""Base class for Agentic Planning Visual Question Answering (VLM) reasoning engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from inference_node.tools import InferenceToolRegistry
from inference_node.vqa.types import AgenticPlanStep, RankedResult


class BaseAgenticVLMReasoner(ABC):
    """Abstract base class for agentic VLM reasoning engines supporting multistage planning and tool usage."""

    @abstractmethod
    def plan_and_execute(
        self,
        query: str,
        tools: InferenceToolRegistry,
        max_steps: int = 5,
        camera_id_filter: str | None = None,
    ) -> Tuple[List[AgenticPlanStep], List[RankedResult]]:
        """Perform multistage agentic planning using perception tools to produce final ranked results."""
        pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        if device in ("cuda", "mps"):
            return torch.bfloat16 if device == "cuda" else torch.float16
        return torch.float32
