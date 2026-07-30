from pydantic import BaseModel, Field
from typing import Dict, Optional


class InferenceConfig(BaseModel):
    postgres_table: str = "track_events"
    postgres_url: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    npz_dir: Optional[str] = None
    npz_path: Optional[str] = None
    json_path: Optional[str] = None

    retrieval_model: str = "google/siglip2-base-patch16-224"
    reasoning_model: str = (
        "openai-5.6"  # Options: 'openai-5.6', 'gemini-2.5-flash', 'Qwen/Qwen3-VL-8B-Instruct'
    )

    video_sources: Dict[str, str] = {}  # camera_id -> video file path
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    max_planning_steps: int = 5
    metadata_filter_enabled: bool = True

    api_port: int = 8100
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
