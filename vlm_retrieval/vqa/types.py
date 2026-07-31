"""Shared data types for the Agentic Planning VLM inference system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from PIL import Image


@dataclass
class ToolCallSpec:
    """Specification of a tool call emitted by an agentic VLM planner."""

    call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a tool call."""

    call_id: str
    name: str
    content: Any
    is_error: bool = False
    extracted_images: List[Image.Image] = field(default_factory=list)


@dataclass
class AgenticPlanStep:
    """A single reasoning and tool execution step in the agentic planning trajectory."""

    step_number: int
    thought: str
    tool_calls: List[ToolCallSpec] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)


@dataclass
class RankedResult:
    """A final scored and verified result returned by the Agentic Planning system."""

    camera_id: str
    camera_timestamp: float
    video_pos_ms: float
    track_id: int
    vlm_score: float
    vlm_explanation: str
    frame: Optional[Image.Image] = None
