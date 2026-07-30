"""Multistage Agentic Planning VLM Pipeline replacing single-step RAG retrieval."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import List, Optional

from PIL import Image

from inference_node.frame_extractor import FrameExtractor
from inference_node.retrieval.search import RetrievalEngine
from inference_node.retrieval.vector_store import VectorStore
from inference_node.tools import InferenceToolRegistry
from inference_node.vqa import BaseAgenticVLMReasoner
from shared.schemas import QueryResultItem
from shared.utils import setup_logger


class AgenticPlannerPipeline:
    """Orchestrates multistage tool-assisted perception, vector search, metadata filtering, and visual inspection."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        vector_store: VectorStore,
        frame_extractor: FrameExtractor,
        reasoner: BaseAgenticVLMReasoner,
        max_planning_steps: int = 5,
    ) -> None:
        self.logger = setup_logger("AgenticPlannerPipeline")
        self.retrieval = retrieval_engine
        self.vector_store = vector_store
        self.frame_extractor = frame_extractor
        self.reasoner = reasoner
        self.max_planning_steps = max_planning_steps

        # Instantiate perception tools registry
        self.tools = InferenceToolRegistry(
            retrieval_engine=self.retrieval,
            vector_store=self.vector_store,
            frame_extractor=self.frame_extractor,
        )

    def _image_to_base64(self, img: Image.Image, max_size: int = 320) -> str:
        img_copy = img.copy()
        img_copy.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img_copy.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        camera_id: Optional[str] = None,
    ) -> List[QueryResultItem]:
        """Execute the multistage agentic planning pipeline."""
        self.logger.info(f"Executing Agentic Planning VLM pipeline for query: '{query_text}'")

        trajectory, ranked_results = self.reasoner.plan_and_execute(
            query=query_text,
            tools=self.tools,
            max_steps=self.max_planning_steps,
            camera_id_filter=camera_id,
        )

        limit_k = top_k or 5
        final_ranked = ranked_results[:limit_k]

        results: List[QueryResultItem] = []
        for index, result in enumerate(final_ranked):
            ts_human = datetime.fromtimestamp(result.camera_timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

            thumbnail = None
            if result.frame:
                try:
                    thumbnail = self._image_to_base64(result.frame)
                except Exception as err:
                    self.logger.warning(f"Could not format thumbnail for result: {err}")

            results.append(
                QueryResultItem(
                    rank=index + 1,
                    camera_id=result.camera_id,
                    timestamp=result.camera_timestamp,
                    video_pos_ms=result.video_pos_ms,
                    timestamp_human=ts_human,
                    global_id=result.track_id,
                    class_label="verified_target",
                    color="unknown",
                    type=None,
                    vlm_score=result.vlm_score,
                    vlm_explanation=result.vlm_explanation,
                    thumbnail_b64=thumbnail,
                )
            )

        return results
