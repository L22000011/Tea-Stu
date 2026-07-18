from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn


class ReliabilityPoseHead(nn.Module):
    def __init__(self, dim: int = 512, tokens_per_modality: int = 32, num_joints: int = 17) -> None:
        super().__init__()
        self.tokens_per_modality = tokens_per_modality
        self.num_joints = num_joints
        self.pose_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_joints * 3),
        )
        self.logvar_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, tokens: torch.Tensor, selected: Sequence[str]) -> Dict[str, torch.Tensor | List[str]]:
        modality_tokens = list(tokens.split(self.tokens_per_modality, dim=1))
        pooled = torch.stack([item.mean(dim=1) for item in modality_tokens], dim=1)
        modality_poses = self.pose_head(pooled).view(tokens.size(0), len(selected), self.num_joints, 3)
        logvars = self.logvar_head(pooled).squeeze(-1)
        alphas = torch.softmax(-logvars, dim=1)
        pose = torch.sum(alphas[:, :, None, None] * modality_poses, dim=1)
        fused_token = torch.sum(alphas[:, :, None] * pooled, dim=1)
        return {
            "pose": pose,
            "tokens": tokens,
            "shared_tokens": tokens,
            "fused_token": fused_token,
            "modality_poses": modality_poses,
            "logvars": logvars,
            "alphas": alphas,
            "reliability_weights": alphas,
            "modalities": list(selected),
        }


class ReliabilityClassificationHead(nn.Module):
    def __init__(self, dim: int = 512, tokens_per_modality: int = 32, num_classes: int = 27) -> None:
        super().__init__()
        self.tokens_per_modality = tokens_per_modality
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )
        self.logvar_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, tokens: torch.Tensor, selected: Sequence[str]) -> Dict[str, torch.Tensor | List[str]]:
        modality_tokens = list(tokens.split(self.tokens_per_modality, dim=1))
        pooled = torch.stack([item.mean(dim=1) for item in modality_tokens], dim=1)
        modality_logits = self.classifier(pooled)
        logvars = self.logvar_head(pooled).squeeze(-1)
        alphas = torch.softmax(-logvars, dim=1)
        logits = torch.sum(alphas[:, :, None] * modality_logits, dim=1)
        fused_token = torch.sum(alphas[:, :, None] * pooled, dim=1)
        return {
            "logits": logits,
            "tokens": tokens,
            "shared_tokens": tokens,
            "fused_token": fused_token,
            "modality_logits": modality_logits,
            "logvars": logvars,
            "alphas": alphas,
            "reliability_weights": alphas,
            "modalities": list(selected),
        }

