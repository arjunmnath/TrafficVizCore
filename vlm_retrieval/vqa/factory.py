"""Factory for instantiating Agentic VLM reasoning adapters."""

from __future__ import annotations

from vlm_retrieval.vqa.base import BaseAgenticVLMReasoner


def get_vqa_reasoner(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "auto",
    device_map: str = "balanced",
) -> BaseAgenticVLMReasoner:
    """Factory to resolve a model name to its respective Agentic VLM Reasoner adapter.

    Args:
        model_name: Name of the model:
          - API models: 'openai-5.6', 'gpt-4o', 'gemini-2.5-flash', 'gemini-1.5-pro'
          - On-device models: 'Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'
        device: Device to load local models on ("auto", "cuda", "mps", "cpu")
        device_map: Multi-GPU allocation strategy ("balanced", "auto", "sequential", etc.)

    Returns:
        An instance of BaseAgenticVLMReasoner
    """
    model_name_lower = model_name.lower()

    if "openai" in model_name_lower or "gpt" in model_name_lower:
        from vlm_retrieval.vqa.openai_reasoner import OpenAIAgenticReasoner

        return OpenAIAgenticReasoner(model_name=model_name)

    elif "gemini" in model_name_lower:
        from vlm_retrieval.vqa.gemini_reasoner import GeminiAgenticReasoner

        return GeminiAgenticReasoner(model_name=model_name)

    elif "qwen" in model_name_lower:
        from vlm_retrieval.vqa.qwen_reasoner import Qwen3VLAgenticReasoner

        return Qwen3VLAgenticReasoner(model_name=model_name, device=device, device_map=device_map)

    else:
        raise ValueError(
            f"Unsupported agentic reasoning model: '{model_name}'. "
            "Supported models include: "
            "API: 'openai-5.6', 'gpt-4o', 'gemini-2.5-flash', 'gemini-1.5-pro'; "
            "On-device: 'Qwen/Qwen3-VL-8B-Instruct'."
        )
