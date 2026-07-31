"""Agentic Pipeline — thin orchestration layer over the ReAct-style VLM reasoner.

The pipeline wires together the retrieval engine, vector store, frame extractor,
and VLM reasoner. It delegates all planning and tool execution to the reasoner's
ReAct loop and converts the final results into QueryResultItem objects.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from PIL import Image

from vlm_retrieval.frame_extractor import FrameExtractor
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore
from vlm_retrieval.tools import InferenceToolRegistry
from vlm_retrieval.vqa import BaseAgenticVLMReasoner
from vlm_retrieval.vqa.types import AgenticPlanStep, RankedResult
from shared.schemas import QueryResultItem
from shared.utils import setup_logger


class AgenticPlannerPipeline:
    """Orchestrates the ReAct-style agentic VLM retrieval pipeline.

    External API is unchanged: callers use `query()` or `query_with_trajectory()`.
    Internally, all planning, tool execution, and evidence accumulation is
    handled by the VLM reasoner's ReAct loop.
    """

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

    def query_with_trajectory(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        camera_id: Optional[str] = None,
    ) -> Tuple[List[QueryResultItem], List[AgenticPlanStep]]:
        """Execute the ReAct agentic pipeline and return both final results and execution trajectory."""
        self.logger.info(f"Executing ReAct VLM pipeline for query: '{query_text}'")

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
            ts_human = datetime.fromtimestamp(
                result.camera_timestamp, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")

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

        return results, trajectory

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        camera_id: Optional[str] = None,
    ) -> List[QueryResultItem]:
        """Execute the ReAct agentic pipeline and return final results."""
        results, _ = self.query_with_trajectory(
            query_text=query_text, top_k=top_k, camera_id=camera_id
        )
        return results
