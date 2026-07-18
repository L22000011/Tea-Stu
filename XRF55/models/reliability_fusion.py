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
        max_modalities: int = 3,
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


class ClassificationExpertHead(nn.Module):
    def __init__(self, dim: int = 512, num_classes: int = 55, semantic_dim: int = 1024) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.logits = nn.Linear(dim, num_classes)
        self.logvar = nn.Linear(dim, 1)
        self.semantic = nn.Linear(dim, semantic_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.norm(tokens.mean(dim=1))
        logits = self.logits(pooled)
        logvar = self.logvar(pooled).squeeze(-1).clamp(-8.0, 8.0)
        semantic = self.semantic(pooled)
        return logits, logvar, semantic


class ClassificationReliabilityFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_classes: int = 55,
        semantic_dim: int = 1024,
        tokens_per_modality: int = 16,
        modalities: List[str] | None = None,
        tau: float = 1.0,
        fusion_mode: str = "uncertainty",
    ) -> None:
        super().__init__()
        self.reference_modalities = modalities or ["wifi", "rfid", "mmwave"]
        self.tokens_per_modality = tokens_per_modality
        self.tau = tau
        self.fusion_mode = fusion_mode
        self.shared_head = ClassificationExpertHead(dim=dim, num_classes=num_classes, semantic_dim=semantic_dim)
        self.score_head = nn.Linear(dim, 1)
        self.final_norm = nn.LayerNorm(dim)
        self.final_head = nn.Linear(dim, num_classes)
        self.final_semantic = nn.Linear(dim, semantic_dim)

    def forward(self, encoded_tokens: torch.Tensor, selected_modalities: List[str]) -> Dict[str, torch.Tensor | list]:
        chunks = list(encoded_tokens.split(self.tokens_per_modality, dim=1))
        modality_logits = []
        modality_semantics = []
        logvars = []
        pooled_tokens = []
        for chunk in chunks:
            logits, logvar, semantic = self.shared_head(chunk)
            modality_logits.append(logits)
            modality_semantics.append(semantic)
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

        stacked_logits = torch.stack(modality_logits, dim=1)
        stacked_semantics = torch.stack(modality_semantics, dim=1)
        weighted_logits = torch.sum(alphas[:, :, None] * stacked_logits, dim=1)
        pooled = torch.stack(pooled_tokens, dim=1)
        fused_token = torch.sum(alphas[:, :, None] * pooled, dim=1)
        residual_logits = self.final_head(self.final_norm(fused_token))
        logits = 0.5 * weighted_logits + 0.5 * residual_logits
        semantic = 0.5 * torch.sum(alphas[:, :, None] * stacked_semantics, dim=1) + 0.5 * self.final_semantic(fused_token)

        return {
            "logits": logits,
            "tokens": fused_token,
            "semantic": semantic,
            "modality_logits": modality_logits,
            "modality_semantics": modality_semantics,
            "logvars": logvar_tensor,
            "alphas": alphas,
            "modalities": selected_modalities,
        }
