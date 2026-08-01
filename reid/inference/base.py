import abc
import torch
import numpy as np
from typing import Any, List, Union


class BaseReIDExtractor(abc.ABC):
    """Abstract Base Class (Adapter Interface) for ReID feature extraction models."""

    @abc.abstractmethod
    def extract(self, image: Any, is_bgr: bool = True) -> torch.Tensor:
        """Extract feature embedding for a single image.

        Args:
            image (Any): Image input (numpy array, PIL Image, torch Tensor, or file path).
            is_bgr (bool): Whether numpy image is BGR (OpenCV format) or RGB.

        Returns:
            torch.Tensor: 1D feature tensor of shape (D,).
        """
        pass

    @abc.abstractmethod
    def extract_batch(self, images: List[Any], is_bgr: bool = True) -> torch.Tensor:
        """Extract feature embeddings for a batch of images.

        Args:
            images (List[Any]): List of image inputs.
            is_bgr (bool): Whether numpy images are BGR format.

        Returns:
            torch.Tensor: 2D feature tensor of shape (N, D).
        """
        pass

    @property
    @abc.abstractmethod
    def device(self) -> str:
        """Return the device string (e.g. 'cuda', 'cpu')."""
        pass

    @property
    def models(self) -> List[Any]:
        """Return list of underlying submodels/model representations.
        
        Default returns [self] for single-model adapters.
        """
        return [self]
