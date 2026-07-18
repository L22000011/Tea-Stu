from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .shared_multimodal_encoder import SharedMultimodalEncoder
from .task_heads import ReliabilityClassificationHead, ReliabilityPoseHead


class SuperTeacher(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.dim = int(model_cfg.get("dim", 512))
        self.tokens_per_modality = int(model_cfg.get("tokens_per_modality", 32))
        self.num_classes = int(model_cfg.get("num_classes", 27))
        self.encoder = SharedMultimodalEncoder(config)
        self.hpe_head = ReliabilityPoseHead(self.dim, self.tokens_per_modality)
        self.har_head = ReliabilityClassificationHead(self.dim, self.tokens_per_modality, self.num_classes)

    def forward(self, inputs: Dict[str, torch.Tensor], selected_modalities: Sequence[str], task: str = "hpe") -> dict:
        tokens, _, selected = self.encoder(inputs, selected_modalities)
        if task == "hpe":
            return self.hpe_head(tokens, selected)
        if task == "har":
            return self.har_head(tokens, selected)
        raise ValueError(f"Unknown task: {task}")


def build_super_teacher(config: dict) -> SuperTeacher:
    return SuperTeacher(config)

