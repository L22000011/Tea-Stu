from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from utils.modality import ALL_MODALITIES, canonicalize_modalities
from .legacy_backbones import CSIFeatureExtractor, DepthFeatureExtractor, LidarFeatureExtractor, MMwaveFeatureExtractor
from .skeleton_prompt_encoder import SkeletonPromptEncoder


class SharedMultimodalEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.dim = int(model_cfg.get("dim", 512))
        self.tokens_per_modality = int(model_cfg.get("tokens_per_modality", 32))
        self.modalities = canonicalize_modalities(model_cfg.get("modalities", ALL_MODALITIES))
        self.modality_to_index = {name: idx for idx, name in enumerate(ALL_MODALITIES)}
        backbone_root = Path(model_cfg.get("backbone_root", "backbones"))
        pretrained = model_cfg.get("pretrained_paths", {})

        self.extractors = nn.ModuleDict()
        if "vk" in self.modalities:
            self.extractors["vk"] = SkeletonPromptEncoder(
                dim=self.dim,
                num_output_tokens=self.tokens_per_modality,
                graph_layers=int(model_cfg.get("graph_layers", 2)),
            )
        if "depth" in self.modalities:
            self.extractors["depth"] = DepthFeatureExtractor(backbone_root, pretrained.get("depth"))
        if "lidar" in self.modalities:
            self.extractors["lidar"] = LidarFeatureExtractor(backbone_root, pretrained.get("lidar"))
        if "mmwave" in self.modalities:
            self.extractors["mmwave"] = MMwaveFeatureExtractor(backbone_root, pretrained.get("mmwave"))
        if "wifi-csi" in self.modalities:
            self.extractors["wifi-csi"] = CSIFeatureExtractor(backbone_root, pretrained.get("wifi-csi"))

        self.token_projectors = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim)) for name in self.modalities}
        )
        self.modality_embedding = nn.Embedding(len(ALL_MODALITIES), self.dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(model_cfg.get("fusion_heads", 8)),
            dim_feedforward=int(model_cfg.get("fusion_ffn_dim", self.dim * 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(model_cfg.get("fusion_layers", 2)))
        self.output_norm = nn.LayerNorm(self.dim)

        if bool(model_cfg.get("freeze_pretrained_backbones", True)):
            self.freeze_pretrained_backbones()

    def freeze_pretrained_backbones(self) -> None:
        for name, module in self.extractors.items():
            if name == "vk":
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _pool_or_sample(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) == self.tokens_per_modality:
            return tokens
        tokens = tokens.transpose(1, 2)
        tokens = torch.nn.functional.adaptive_avg_pool1d(tokens, self.tokens_per_modality)
        return tokens.transpose(1, 2)

    def _extract_one(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name == "vk":
            _, projected = self.extractors[name](tensor)
            raw_tokens = projected
        else:
            raw_tokens = self.extractors[name](tensor)
        raw_tokens = torch.nan_to_num(raw_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
        raw_tokens = self._pool_or_sample(raw_tokens)
        raw_tokens = self.token_projectors[name](raw_tokens)
        index = torch.full((raw_tokens.size(1),), self.modality_to_index[name], dtype=torch.long, device=raw_tokens.device)
        return raw_tokens + self.modality_embedding(index)[None, :, :]

    def forward(self, inputs: Dict[str, torch.Tensor], selected_modalities: Sequence[str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        selected = canonicalize_modalities(selected_modalities)
        selected = [name for name in selected if name in inputs and name in self.extractors]
        if not selected:
            raise ValueError("No selected modality is available in the current batch.")
        projected = {name: self._extract_one(name, inputs[name]) for name in selected}
        tokens = torch.cat([projected[name] for name in selected], dim=1)
        encoded = self.output_norm(self.fusion_encoder(tokens))
        return encoded, projected, selected

