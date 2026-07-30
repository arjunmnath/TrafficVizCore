"""EVA-CLIP text and image encoder implementation."""

from __future__ import annotations

from typing import Union
import numpy as np
import torch
from PIL import Image

from vlm_retrieval.retrieval.encoder.base import BaseRetrievalEncoder
from shared.utils import setup_logger


class EVACLIPEncoder(BaseRetrievalEncoder):
    """Encodes text queries and cropped object images using EVA-CLIP models.

    Designed to support:
      1. Lightweight EVA02 CLIP variants (B and L) hosted on the Hugging Face Hub via timm/open_clip.
      2. Large BAAI models (8B/18B) loaded via transformers AutoModel (trust_remote_code=True).
    """

    def __init__(
        self,
        model_name: str = "eva02-b-16",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("EVACLIPEncoder")
        self.device = self._resolve_device(device)
        self.model_name = model_name

        model_name_lower = model_name.lower()

        if "/" in model_name:
            self.hf_repo = model_name
        elif "eva-clip-18b" in model_name_lower:
            self.hf_repo = "BAAI/EVA-CLIP-18B"
        else:
            self.hf_repo = "BAAI/EVA-CLIP-8B"

        self.logger.info(
            f"Loading EVA-CLIP via HF transformers: model={self.hf_repo} on {self.device}"
        )
        try:
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.hf_repo, trust_remote_code=True)
            self.model = (
                AutoModel.from_pretrained(self.hf_repo, trust_remote_code=True)
                .to(self.device)
                .eval()
            )
        except Exception as e:
            self.logger.error(f"Failed to load EVA-CLIP via HF AutoModel: {e}")
            raise e

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a semantic query string into a normalized embedding vector."""
        if not text.strip():
            raise ValueError("Semantic query text must not be empty")

        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
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
