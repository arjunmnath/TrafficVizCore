import os
from typing import Any, List, Optional
import numpy as np
import torch

from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, has_minimum_roi_area, FrameData
from reid.inference import build_reid_model, BaseReIDExtractor


class FeatureStage(PipelineStage):
    """Stage 2: Extracts ReID features using configured adapter model (ResNetIBNReID or ViTCLIPReID)."""

    def __init__(
        self,
        model_type: str = "resnetibnreid",
        device: str = "cpu",
        fp16: bool = True,
        model_path: Optional[str] = None,
        extractor: Optional[BaseReIDExtractor] = None,
    ):
        """Constructor.

        Args:
            model_type (str): ReID model variant name ('resnetibnreid' or 'vitclipreid').
            device (str): Inference device.
            fp16 (bool): Whether to enable half precision.
            model_path (Optional[str]): Optional custom model path.
            extractor (Optional[BaseReIDExtractor]): Pre-instantiated ReID extractor adapter.
        """
        self.model_type = model_type
        self.device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp16 = fp16
        self.model_path = model_path
        self.extractor = extractor

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if self.extractor is None:
            if listener:
                listener.on_init_status(f"Loading ReID model adapter ({self.model_type})...")
            self.extractor = build_reid_model(
                variant=self.model_type,
                device=self.device,
                fp16=self.fp16,
                model_path=self.model_path,
            )


        if listener:
            num_submodels = len(getattr(self.extractor, "models", [self.extractor]))
            listener.on_init_status(
                f"Loaded {self.model_type} ReID model ({num_submodels} submodel(s)) successfully."
            )

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        if data.skip or data.end_of_stream:
            return data

        frame = data.frame
        boxes = data.boxes
        scores = data.scores
        classes = data.classes

        if boxes is None or len(boxes) == 0:
            data.features = np.empty((0, 0), dtype=np.float32)
            return data

        features = []
        valid_crops = []
        valid_idxs = []

        for idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if not has_minimum_roi_area(box, frame.shape):
                features.append(None)
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            valid_crops.append(crop)
            valid_idxs.append(idx)
            features.append(None)  # Placeholder

        if len(valid_crops) > 0:
            embeddings = self.extractor.extract_batch(valid_crops, is_bgr=True)
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
            else:
                embeddings_np = np.asarray(embeddings, dtype=np.float32)

            for embed_idx, orig_idx in enumerate(valid_idxs):
                features[orig_idx] = embeddings_np[embed_idx]

        # Resolve missing feature dimensions
        valid_feat = next((f for f in features if f is not None), None)
        if valid_feat is not None:
            feat_dim = len(valid_feat)
        else:
            # All crops are invalid; run a dummy extraction to determine feature dimension
            dummy_crop = np.zeros((128, 64, 3), dtype=np.uint8)
            dummy_feat = self.extractor.extract(dummy_crop, is_bgr=True)
            if isinstance(dummy_feat, torch.Tensor):
                dummy_feat_np = dummy_feat.cpu().numpy()
            else:
                dummy_feat_np = np.asarray(dummy_feat, dtype=np.float32)
            feat_dim = dummy_feat_np.shape[0]

        zeros = np.zeros(feat_dim, dtype=np.float32)
        features = [f if f is not None else zeros for f in features]

        data.features = np.stack(features, axis=0)

        return data
