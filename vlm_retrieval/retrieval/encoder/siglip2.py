"""SigLIP2 text and image encoder implementation."""

from __future__ import annotations

from typing import Union
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from vlm_retrieval.retrieval.encoder.base import BaseRetrievalEncoder
from shared.utils import setup_logger


class SigLIP2Encoder(BaseRetrievalEncoder):
    """Encodes text queries and cropped object images using SigLIP2."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("SigLIP2Encoder")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.logger.info(f"Loading SigLIP2 encoder: {model_name} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.logger.info("SigLIP2 encoder loaded")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a semantic query string into a normalized embedding vector."""
        if not text.strip():
            raise ValueError("Semantic query text must not be empty")

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()[0]

    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Encode a cropped object image into a normalized embedding vector."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        pil_image = image.convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()[0]
