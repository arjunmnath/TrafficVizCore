"""Encoder module for Post-Camera Processing.

Batch-encodes selected representative vehicle crops using SigLIP2 into unit-normalized L2 visual
embeddings compatible with the vector store and downstream Agentic VLM retrieval engine.
"""

from __future__ import annotations

from typing import List, Optional, Union
import numpy as np
from PIL import Image

from post_camera_processing.quality_filter import FilteredCropItem
from vlm_retrieval.retrieval.encoder.factory import get_retrieval_encoder
from shared.utils import setup_logger


class SigLIP2BatchEncoder:
    """Handles batched visual feature extraction via SigLIP2 retrieval model."""

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        device: str = "auto",
        batch_size: int = 32,
        logger: Optional[Any] = None,
    ) -> None:
        self.logger = logger or setup_logger("SigLIP2BatchEncoder")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.logger.info(f"Initializing SigLIP2 retrieval encoder: '{model_name}' on device '{device}'")
        self.encoder = get_retrieval_encoder(model_name=model_name, device=device)

    def encode_crops(self, crop_items: List[FilteredCropItem]) -> List[np.ndarray]:
        """Encodes a list of `FilteredCropItem` into unit-normalized numpy vectors."""
        if not crop_items:
            return []

        pil_images = [item.raw_crop.image for item in crop_items]
        embeddings = self.encode_pil_images(pil_images)
        return embeddings

    def encode_pil_images(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Batch encodes PIL Images into normalized embeddings."""
        if not images:
            return []

        embeddings_list: List[np.ndarray] = []
        n_samples = len(images)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_images = images[start_idx:end_idx]

            for img in batch_images:
                try:
                    vec = self.encoder.encode_image(img)
                    # Ensure float32 array
                    vec = np.array(vec, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    embeddings_list.append(vec)
                except Exception as err:
                    self.logger.error(f"Error encoding image batch item: {err}")
                    # Fallback zero vector or dummy if failure occurs
                    dummy_dim = getattr(self.encoder, "embedding_dim", 768)
                    dummy_vec = np.zeros((dummy_dim,), dtype=np.float32)
                    embeddings_list.append(dummy_vec)

        return embeddings_list
