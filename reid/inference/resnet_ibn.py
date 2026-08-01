import torch
import numpy as np
from typing import List, Any, Optional

from reid.inference.base import BaseReIDExtractor
from reid.inference.config import EnsembleConfig
from reid.inference.model_factory import build_ensemble_model
from reid.inference.preprocessing import preprocess_images
from reid.inference.utils import get_device


class ResNetIBNReID(BaseReIDExtractor):
    """Production-grade ReID feature extractor using 3 ensembled ResNet-IBN/ResNeXt-IBN models with mean centroid fusion."""

    def __init__(
        self,
        device: str = "cuda",
        fp16: bool = True,
        checkpoint_paths: Optional[List[str]] = None,
    ):
        self._device_str = device if device != "auto" else get_device(device)
        self.fp16 = fp16

        from reid.utils import resolve_model_weights

        if checkpoint_paths is None:
            checkpoint_paths = [
                resolve_model_weights("resnet101_ibn_a_2.pth"),
                resolve_model_weights("resnet101_ibn_a_3.pth"),
                resolve_model_weights("resnext101_ibn_a_2.pth"),
            ]

        # Config containing the 3 ensembled submodels
        self.config = EnsembleConfig(
            checkpoint_paths=checkpoint_paths,
            device=self._device_str,
            fp16=self.fp16,
        )

        # Build unified ensemble model
        self.model = build_ensemble_model(self.config)
        self._submodels = self.model.submodels

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def models(self) -> List[Any]:
        return self._submodels

    def extract(self, image: Any, is_bgr: bool = True) -> torch.Tensor:
        """Extract embeddings for a single image."""
        res = self.extract_batch([image], is_bgr=is_bgr)
        return res[0]

    def extract_batch(self, images: List[Any], is_bgr: bool = True) -> torch.Tensor:
        """Extract embeddings for a batch of images and fuse them using mean centroid fusion."""
        if not images:
            raise ValueError("No images provided for feature extraction.")

        # Preprocess the entire list of images once
        tensor_batch = preprocess_images(
            images=images,
            image_size=self.config.image_size,
            pixel_mean=self.config.pixel_mean,
            pixel_std=self.config.pixel_std,
            is_bgr=is_bgr,
        )

        device_obj = torch.device(self._device_str)
        tensor_batch = tensor_batch.to(device_obj)
        is_cuda = device_obj.type == "cuda"

        num_samples = tensor_batch.shape[0]
        batch_size = self.config.batch_size
        feats_list = []

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                sub_batch = tensor_batch[start_idx:end_idx]

                if self.fp16 and is_cuda:
                    with torch.cuda.amp.autocast(enabled=True):
                        sub_feats = self.model(sub_batch)
                else:
                    sub_feats = self.model(sub_batch)

                feats_list.append(sub_feats)

        fused_features = torch.cat(feats_list, dim=0)
        return fused_features
