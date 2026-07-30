"""Adapter-based visual retrieval — no VLM dependencies."""

from vlm_retrieval.retrieval.encoder import (
    BaseRetrievalEncoder,
    SigLIP2Encoder,
    OpenCLIPEncoder,
    get_retrieval_encoder,
)
from vlm_retrieval.retrieval.query_parser import ParsedQuery, parse_query
from vlm_retrieval.retrieval.search import RetrievalEngine, RetrievalResult
from vlm_retrieval.retrieval.vector_store import VectorStore

__all__ = [
    "BaseRetrievalEncoder",
    "SigLIP2Encoder",
    "OpenCLIPEncoder",
    "get_retrieval_encoder",
    "ParsedQuery",
    "parse_query",
    "RetrievalEngine",
    "RetrievalResult",
    "VectorStore",
]
