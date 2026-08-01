import os
import sys
import types
import collections.abc
from typing import Optional, Any

from .utils import get_device

# Mock torch._six to support newer PyTorch versions (PyTorch 2.x+)
torch_six_mock = types.ModuleType("torch._six")
torch_six_mock.container_abcs = collections.abc
sys.modules["torch._six"] = torch_six_mock

import torch

from .model.make_model import make_model
from .config import InferenceConfig, EnsembleConfig


def build_model_from_config(config: InferenceConfig) -> torch.nn.Module:
    """Builds a model instance based on the InferenceConfig and loads the checkpoint weights."""
    # Convert configuration to the mock yacs node expected by make_model
    yacs_cfg = config.to_yacs_mock()

    # Instantiate model.
    # Pass a dummy class number (e.g. 1000), since during test/inference
    # the classifier layer is not used and not loaded from checkpoint.
    model = make_model(yacs_cfg, num_class=1000)

    # Load parameters
    if config.checkpoint_path:
        if not os.path.exists(config.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint weight file not found at: {config.checkpoint_path}"
            )
        model.load_param(config.checkpoint_path)

    # Place on device
    device = torch.device(get_device(config.device))
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    return model


def build_ensemble_model(config: EnsembleConfig) -> torch.nn.Module:
    """Builds an EnsembleModel instance based on the EnsembleConfig."""
    from .model.ensemble_model import EnsembleModel

    model = EnsembleModel(config)
    return model


def build_reid_model(
    variant: str = "resnetibnreid",
    device: str = "cuda",
    fp16: bool = True,
    model_path: Optional[str] = None,
    **kwargs: Any,
):
    """Factory function creating a ReID feature extractor adapter instance.

    Supported variants:
        - 'resnetibnreid' (or 'resnet_ibn', 'ensemble'): 3-model PyTorch ResNet-IBN ensemble.
        - 'vitclipreid' (or 'vit_clip', 'onnx'): ONNX Runtime-based ViT-CLIP model.

    Args:
        variant (str): ReID model variant name.
        device (str): Inference device ('cuda', 'cpu', 'auto').
        fp16 (bool): Half precision enable flag (for PyTorch models).
        model_path (Optional[str]): Custom path to model weights/ONNX file.

    Returns:
        BaseReIDExtractor: Initialized ReID model adapter.
    """
    from reid.inference.base import BaseReIDExtractor
    from reid.inference.resnet_ibn import ResNetIBNReID
    from reid.inference.vit_clip import ViTCLIPReID

    normalized = variant.lower().strip()

    if normalized in ("resnetibnreid", "resnet_ibn", "ensemble"):
        return ResNetIBNReID(device=device, fp16=fp16)
    elif normalized in ("vitclipreid", "vit_clip", "onnx"):
        return ViTCLIPReID(device=device, model_path=model_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown ReID variant: '{variant}'. Supported variants are: 'resnetibnreid', 'vitclipreid'."
        )
