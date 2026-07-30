"""Metadata filtering and semantic similarity search with PostgreSQL pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from inference_node.retrieval.encoder import BaseRetrievalEncoder
from inference_node.retrieval.query_parser import ParsedQuery, parse_query
from inference_node.retrieval.vector_store import VectorStore
from shared.utils import setup_logger


@dataclass
class RetrievalResult:
    """A candidate returned by the retrieval stage."""

    id: str
    camera_id: str
    camera_timestamp: float
    track_id: int
    video_pos_ms: float
    bbox: Optional[str]
    video_path: Optional[str]
    distance: float


class RetrievalEngine:
    """Orchestrates query parsing, metadata filtering, and semantic vector search over PostgreSQL pgvector."""

    def __init__(
        self,
        encoder: BaseRetrievalEncoder,
        vector_store: VectorStore,
        metadata_filter_enabled: bool = True,
    ) -> None:
        self.logger = setup_logger("RetrievalEngine")
        self.encoder = encoder
        self.vector_store = vector_store
        self.metadata_filter_enabled = metadata_filter_enabled

    def search(
        self,
        query: str,
        top_k: int = 20,
        camera_id: Optional[str] = None,
    ) -> tuple[ParsedQuery, List[RetrievalResult]]:
        """Parse query, apply metadata filters, and retrieve top-K candidates."""
        parsed = parse_query(query)

        if camera_id:
            parsed.metadata_filters["camera_id"] = camera_id

        where = None
        if self.metadata_filter_enabled and parsed.metadata_filters:
            where = self._build_postgres_where(parsed.metadata_filters)
            self.logger.info(f"Metadata filters: {parsed.metadata_filters}")

        self.logger.info(f"Retrieval: semantic='{parsed.semantic_text}' top_k={top_k}")
        query_embedding = self.encoder.encode_text(parsed.semantic_text)
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        results = [self._to_retrieval_result(item) for item in raw_results]
        return parsed, results

    @staticmethod
    def _build_postgres_where(filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed filters into a PostgreSQL query filter structure."""
        clauses: List[Dict[str, Any]] = []

        if "camera_id" in filters:
            clauses.append({"camera_id": filters["camera_id"]})

        if "camera_timestamp_gte" in filters:
            clauses.append({"camera_timestamp": {"$gte": float(filters["camera_timestamp_gte"])}})
        if "camera_timestamp_lt" in filters:
            clauses.append({"camera_timestamp": {"$lt": float(filters["camera_timestamp_lt"])}})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _to_retrieval_result(item: dict) -> RetrievalResult:
        meta = item.get("metadata", {})
        return RetrievalResult(
            id=item["id"],
            camera_id=str(meta.get("camera_id", "")),
            camera_timestamp=float(meta.get("camera_timestamp", meta.get("timestamp", 0.0))),
            track_id=int(meta.get("track_id", 0)),
            video_pos_ms=float(meta.get("video_pos_ms", 0.0)),
            bbox=meta.get("bbox"),
            video_path=meta.get("video_path"),
            distance=float(item.get("distance", 1.0)),
        )
