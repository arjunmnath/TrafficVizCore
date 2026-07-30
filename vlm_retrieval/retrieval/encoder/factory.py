"""Factory for instantiating retrieval encoders."""

from __future__ import annotations

from vlm_retrieval.retrieval.encoder.base import BaseRetrievalEncoder


def get_retrieval_encoder(model_name: str, device: str = "auto") -> BaseRetrievalEncoder:
    """Factory to resolve a model name to its respective Retrieval Encoder adapter.

    Args:
        model_name: Name/path of the model (e.g. google/siglip2-so400m-patch14-384,
          openclip-vit, openclip:ViT-B-32/laion2b_s34b_b79k)
        device: Device to load the model on ("auto", "cuda", "mps", "cpu")

    Returns:
        An instance of BaseRetrievalEncoder
    """
    model_name_lower = model_name.lower()
    if "eva" in model_name_lower:
        from vlm_retrieval.retrieval.encoder.evaclip import EVACLIPEncoder

        return EVACLIPEncoder(model_name=model_name, device=device)
    elif "siglip" in model_name_lower:
        from vlm_retrieval.retrieval.encoder.siglip2 import SigLIP2Encoder

        return SigLIP2Encoder(model_name=model_name, device=device)
    elif "openclip" in model_name_lower or "vit-" in model_name_lower or "clip" in model_name_lower:
        from vlm_retrieval.retrieval.encoder.openclip import OpenCLIPEncoder

        return OpenCLIPEncoder(model_name=model_name, device=device)
    else:
        # Default to SigLIP2 for backwards compatibility with any standard HuggingFace models,
        # but log a warning.
        import logging

        logger = logging.getLogger("RetrievalEncoderFactory")
        logger.warning(
            f"Unrecognized retrieval model '{model_name}'. Defaulting to SigLIP2 encoder."
        )
        from vlm_retrieval.retrieval.encoder.siglip2 import SigLIP2Encoder

        return SigLIP2Encoder(model_name=model_name, device=device)
