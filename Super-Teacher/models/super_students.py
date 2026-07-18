from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .shared_multimodal_encoder import SharedMultimodalEncoder
from .task_heads import ReliabilityClassificationHead, ReliabilityPoseHead


class SuperStudent(nn.Module):
    def __init__(self, config: dict, task: str) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.task = task
        self.dim = int(model_cfg.get("dim", 512))
        self.tokens_per_modality = int(model_cfg.get("tokens_per_modality", 32))
        self.num_classes = int(model_cfg.get("num_classes", 27))
        self.encoder = SharedMultimodalEncoder(config)
        if task == "hpe":
            self.head = ReliabilityPoseHead(self.dim, self.tokens_per_modality)
        elif task == "har":
            self.head = ReliabilityClassificationHead(self.dim, self.tokens_per_modality, self.num_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, inputs: Dict[str, torch.Tensor], selected_modalities: Sequence[str]) -> dict:
        tokens, _, selected = self.encoder(inputs, selected_modalities)
        return self.head(tokens, selected)


def build_hpe_student(config: dict) -> SuperStudent:
    return SuperStudent(config, task="hpe")


def build_har_student(config: dict) -> SuperStudent:
    return SuperStudent(config, task="har")

