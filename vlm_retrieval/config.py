from pydantic import BaseModel
from typing import Dict, Optional


class VLMRetrievalConfig(BaseModel):
    npz_dir: Optional[str] = None
    npz_path: Optional[str] = None
    json_path: Optional[str] = None

    retrieval_model: str = "google/siglip2-so400m-patch14-384"
    reasoning_model: str = (
        "Qwen/Qwen3-VL-8B-Instruct"  # Options: 'Qwen/Qwen3-VL-8B-Instruct', 'openai-5.6', 'gemini-2.5-flash'
    )

    video_sources: Dict[str, str] = {}  # camera_id -> video file path
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    max_planning_steps: int = 5
    metadata_filter_enabled: bool = True

    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    device_map: str = "balanced"  # "balanced", "auto", "sequential"


# Backwards compatibility alias
InferenceConfig = VLMRetrievalConfig
