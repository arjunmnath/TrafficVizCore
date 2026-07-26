from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_node.agentic_pipeline import AgenticPlannerPipeline
from inference_node.retrieval.vector_store import VectorStore
from shared.schemas import QueryRequest, QueryResponse
from shared.utils import setup_logger

logger = setup_logger("InferenceAPI")

_pipeline: AgenticPlannerPipeline | None = None
_vector_store: VectorStore | None = None


def create_app(pipeline: AgenticPlannerPipeline, vector_store: VectorStore) -> FastAPI:
    global _pipeline, _vector_store
    _pipeline = pipeline
    _vector_store = vector_store

    app = FastAPI(
        title="CCTV Agentic Inference Node",
        description="Text-to-timestamp CCTV search using an Agentic Planning VLM with perception tools",
        version="3.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": _pipeline is not None}

    @app.get("/stats")
    async def stats():
        count = _vector_store.get_event_count() if _vector_store else 0
        return {
            "event_count": count,
            "status": "ready" if _pipeline else "initializing",
        }

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Search CCTV footage by natural language query via Agentic Planning VLM."""
        if _pipeline is None:
            return QueryResponse(query=request.query, results=[])

        logger.info(f"Agentic Query: '{request.query}' (top_k={request.top_k})")

        results = _pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            camera_id=request.camera_id,
        )

        logger.info(f"Returning {len(results)} agentic results")
        return QueryResponse(query=request.query, results=results)

    return app
