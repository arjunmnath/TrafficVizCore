"""Retrieval encoder implementations and factory."""

from vlm_retrieval.retrieval.encoder.base import BaseRetrievalEncoder
from vlm_retrieval.retrieval.encoder.siglip2 import SigLIP2Encoder
from vlm_retrieval.retrieval.encoder.clip import CLIPEncoder
from vlm_retrieval.retrieval.encoder.openclip import OpenCLIPEncoder
from vlm_retrieval.retrieval.encoder.evaclip import EVACLIPEncoder
from vlm_retrieval.retrieval.encoder.factory import get_retrieval_encoder

__all__ = [
    "BaseRetrievalEncoder",
    "SigLIP2Encoder",
    "CLIPEncoder",
    "OpenCLIPEncoder",
    "EVACLIPEncoder",
    "get_retrieval_encoder",
]
