"""Abstract base class for retrieval encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch
from PIL import Image


class BaseRetrievalEncoder(ABC):
    """Abstract base class for retrieval encoders."""

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query into a normalized numpy embedding vector."""
        pass

    @abstractmethod
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Encode image crop into a normalized numpy embedding vector."""
        pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
