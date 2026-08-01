import os
import torch
import numpy as np
from typing import List, Any, Optional

from reid.inference.base import BaseReIDExtractor
from reid.inference.preprocessing import preprocess_images
from reid.inference.utils import get_device


class ViTCLIPReID(BaseReIDExtractor):
    """ONNX Runtime-based ReID feature extractor using ViT-CLIP (vehicle_vit_clip_reid.onnx)."""

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        batch_size: int = 32,
        image_size: tuple = (256, 256),
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        **kwargs,
    ):
        self._device_str = device if device != "auto" else get_device(device)
        self.batch_size = batch_size
        self.image_size = image_size
        self.pixel_mean = pixel_mean or [0.485, 0.456, 0.406]
        self.pixel_std = pixel_std or [0.229, 0.224, 0.225]

        from reid.utils import resolve_model_weights

        if model_path is None:
            model_path = resolve_model_weights("vehicle_vit_clip_reid.onnx")
        else:
            model_path = resolve_model_weights(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ViT-CLIP ONNX model file not found at: {model_path}")

        self.model_path = model_path
        self._init_session()

    def _init_session(self) -> None:
        """Initialize the ONNX Runtime InferenceSession with CUDA/MPS/CPU providers."""
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        requested_providers = []

        if "CUDAExecutionProvider" in available_providers:
            requested_providers.append("CUDAExecutionProvider")
        if "MPSExecutionProvider" in available_providers:
            requested_providers.append("MPSExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            requested_providers.append("CPUExecutionProvider")

        if not requested_providers:
            requested_providers = ["CUDAExecutionProvider", "MPSExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(self.model_path, providers=requested_providers)
        except Exception:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        outputs = self.session.get_outputs()
        self.output_name = outputs[0].name

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def models(self) -> List[Any]:
        return [self.session]

    def extract(self, image: Any, is_bgr: bool = True) -> torch.Tensor:
        """Extract feature embedding for a single image."""
        res = self.extract_batch([image], is_bgr=is_bgr)
        return res[0]

    def extract_batch(self, images: List[Any], is_bgr: bool = True) -> torch.Tensor:
        """Extract feature embeddings for a batch of images using ONNX Runtime."""
        if not images:
            raise ValueError("No images provided for feature extraction.")

        # Preprocess the images into standard CHW float32 tensor batch
        tensor_batch = preprocess_images(
            images=images,
            image_size=self.image_size,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            is_bgr=is_bgr,
        )

        num_samples = tensor_batch.shape[0]
        feats_list = []

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            sub_batch_tensor = tensor_batch[start_idx:end_idx]
            chw_image_batch = sub_batch_tensor.cpu().numpy().astype(np.float32)

            # ONNX inference snippet:
            # embedding = sess.run(None, {"input": chw_image_batch})[0]
            embedding = self.session.run(
                [self.output_name], {self.input_name: chw_image_batch}
            )[0]

            feats_list.append(embedding)

        all_embeddings = np.concatenate(feats_list, axis=0)
        return torch.from_numpy(all_embeddings)
