from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class ModalityTokenEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_modalities: int = 5,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.modality_embedding = nn.Embedding(max_modalities, dim)

    def forward(self, projected: Dict[str, torch.Tensor], modalities: List[str], reference: List[str]) -> torch.Tensor:
        tokens = []
        for name in modalities:
            modality_index = reference.index(name)
            embed = self.modality_embedding.weight[modality_index][None, None, :]
            tokens.append(projected[name] + embed)
        return self.encoder(torch.cat(tokens, dim=1))


class PoseExpertHead(nn.Module):
    def __init__(self, dim: int = 512, num_joints: int = 17) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pose = nn.Linear(dim, num_joints * 3)
        self.logvar = nn.Linear(dim, 1)
        self.num_joints = num_joints

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.norm(tokens.mean(dim=1))
        pose = self.pose(pooled).view(pooled.size(0), self.num_joints, 3)
        logvar = self.logvar(pooled).squeeze(-1).clamp(-8.0, 8.0)
        return pose, logvar


class ReliabilityFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_joints: int = 17,
        tokens_per_modality: int = 32,
        modalities: List[str] | None = None,
        tau: float = 1.0,
        fusion_mode: str = "uncertainty",
    ) -> None:
        super().__init__()
        self.reference_modalities = modalities or ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
        self.tokens_per_modality = tokens_per_modality
        self.tau = tau
        self.fusion_mode = fusion_mode
        self.shared_head = PoseExpertHead(dim=dim, num_joints=num_joints)
        self.score_head = nn.Linear(dim, 1)
        self.final_norm = nn.LayerNorm(dim)
        self.final_head = nn.Linear(dim, num_joints * 3)
        self.num_joints = num_joints

    def forward(self, encoded_tokens: torch.Tensor, selected_modalities: List[str]) -> Dict[str, torch.Tensor | list]:
        chunks = list(encoded_tokens.split(self.tokens_per_modality, dim=1))
        modality_poses = []
        logvars = []
        pooled_tokens = []
        for chunk in chunks:
            pose, logvar = self.shared_head(chunk)
            modality_poses.append(pose)
            logvars.append(logvar)
            pooled_tokens.append(chunk.mean(dim=1))

        logvar_tensor = torch.stack(logvars, dim=1)
        if self.fusion_mode == "uniform":
            alphas = torch.ones_like(logvar_tensor) / max(logvar_tensor.size(1), 1)
        elif self.fusion_mode == "attention":
            scores = torch.cat([self.score_head(token).view(token.size(0), 1) for token in pooled_tokens], dim=1)
            alphas = torch.softmax(scores / float(self.tau), dim=1)
        else:
            alphas = torch.softmax(-logvar_tensor / float(self.tau), dim=1)
        stacked_poses = torch.stack(modality_poses, dim=1)
        fused_pose = torch.sum(alphas[:, :, None, None] * stacked_poses, dim=1)

        pooled = torch.stack(pooled_tokens, dim=1)
        fused_token = torch.sum(alphas[:, :, None] * pooled, dim=1)
        residual_pose = self.final_head(self.final_norm(fused_token)).view(fused_token.size(0), self.num_joints, 3)
        pose = 0.5 * fused_pose + 0.5 * residual_pose

        return {
            "pose": pose,
            "tokens": encoded_tokens.mean(dim=1),
            "modality_poses": modality_poses,
            "logvars": logvar_tensor,
            "alphas": alphas,
            "modalities": selected_modalities,
        }
