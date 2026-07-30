"""Hugging Face CLIP text and image encoder implementation."""

from __future__ import annotations

from typing import Union
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from vlm_retrieval.retrieval.encoder.base import BaseRetrievalEncoder
from shared.utils import setup_logger


class OpenCLIPEncoder(BaseRetrievalEncoder):
    """Encodes text queries and cropped object images using Hugging Face CLIP."""

    def __init__(
        self,
        model_name: str = "openclip-vit",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("OpenCLIPEncoder")
        self.device = self._resolve_device(device)

        self.model_name = model_name
        if "/" in model_name:
            hf_model_name = model_name
        elif "laion2b" in model_name.lower():
            hf_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        else:
            hf_model_name = "openai/clip-vit-large-patch14"

        self.logger.info(f"Loading Hugging Face CLIP encoder: {hf_model_name} on {self.device}")

        try:
            self.processor = CLIPProcessor.from_pretrained(hf_model_name)
            self.model = CLIPModel.from_pretrained(hf_model_name).to(self.device).eval()
        except Exception as e:
            self.logger.error(f"Failed to load HF CLIP model: {e}")
            raise e

        self.logger.info("Hugging Face CLIP encoder loaded successfully")

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
