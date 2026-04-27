from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch import nn

from utils.modality import ALL_MODALITIES, canonicalize_modalities
from .legacy_backbones import DepthFeatureExtractor, LidarHARFeatureExtractor, MMwaveFeatureExtractor
from .reliability_fusion import ClassificationReliabilityFusion, ModalityTokenEncoder
from .skeleton_prompt_encoder import SkeletonPromptEncoder


SAFE_KEYS = {
    "vk": "vk",
    "depth": "depth",
    "lidar": "lidar",
    "mmwave": "mmwave",
}


class TokenProjector(nn.Module):
    def __init__(self, dim: int = 512, output_tokens: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_tokens),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.permute(0, 2, 1)).permute(0, 2, 1)


class VKRCDHARModel(nn.Module):
    def __init__(
        self,
        modalities: Sequence[str] = ALL_MODALITIES,
        dim: int = 512,
        tokens_per_modality: int = 32,
        num_classes: int = 27,
        backbone_root: str | Path = "backbones",
        freeze_pretrained_backbones: bool = True,
        graph_layers: int = 2,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        tau: float = 1.0,
        fusion_mode: str = "uncertainty",
        pretrained_paths: Dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.modalities = canonicalize_modalities(modalities)
        self.reference_modalities = list(ALL_MODALITIES)
        self.num_classes = num_classes
        pretrained_paths = pretrained_paths or {}

        self.vk_encoder = SkeletonPromptEncoder(
            dim=dim,
            num_output_tokens=tokens_per_modality,
            graph_layers=graph_layers,
        )
        self.extractors = nn.ModuleDict()
        if "depth" in self.modalities:
            self.extractors[SAFE_KEYS["depth"]] = DepthFeatureExtractor(backbone_root, pretrained_paths.get("depth"))
        if "lidar" in self.modalities:
            self.extractors[SAFE_KEYS["lidar"]] = LidarHARFeatureExtractor(backbone_root, pretrained_paths.get("lidar"))
        if "mmwave" in self.modalities:
            self.extractors[SAFE_KEYS["mmwave"]] = MMwaveFeatureExtractor(backbone_root, pretrained_paths.get("mmwave"))

        self.projectors = nn.ModuleDict({
            SAFE_KEYS[name]: TokenProjector(dim=dim, output_tokens=tokens_per_modality)
            for name in self.modalities
            if name != "vk"
        })
        self.token_encoder = ModalityTokenEncoder(
            dim=dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            max_modalities=len(self.reference_modalities),
        )
        self.fusion = ClassificationReliabilityFusion(
            dim=dim,
            num_classes=num_classes,
            tokens_per_modality=tokens_per_modality,
            modalities=self.reference_modalities,
            tau=tau,
            fusion_mode=fusion_mode,
        )

        if freeze_pretrained_backbones:
            self.freeze_sensor_backbones()

    def freeze_sensor_backbones(self) -> None:
        for extractor in self.extractors.values():
            extractor.eval()
            for param in extractor.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "VKRCDHARModel":
        super().train(mode)
        for extractor in self.extractors.values():
            extractor.eval()
        return self

    def extract_features(self, inputs: Dict[str, torch.Tensor], selected_modalities: list[str]) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        if "vk" in selected_modalities:
            _, projected = self.vk_encoder(inputs["vk"])
            features["vk"] = projected
        for name in selected_modalities:
            if name == "vk":
                continue
            safe_key = SAFE_KEYS[name]
            raw = self.extractors[safe_key](inputs[name])
            features[name] = self.projectors[safe_key](raw)
        return features

    def forward(self, inputs: Dict[str, torch.Tensor], selected_modalities: Sequence[str]) -> Dict[str, Any]:
        selected = canonicalize_modalities(selected_modalities)
        if not selected:
            raise ValueError("At least one modality must be selected.")
        missing = [name for name in selected if name not in inputs]
        if missing:
            raise KeyError(f"Batch does not contain selected modalities: {missing}")
        projected = self.extract_features(inputs, selected)
        encoded = self.token_encoder(projected, selected, self.reference_modalities)
        output = self.fusion(encoded, selected)
        output["projected_tokens"] = projected
        return output


def build_model(config: Dict[str, Any], modalities: Sequence[str] | None = None) -> VKRCDHARModel:
    model_cfg = config.get("model", {})
    selected_modalities = modalities or model_cfg.get("modalities") or config.get("modality") or ALL_MODALITIES
    return VKRCDHARModel(
        modalities=selected_modalities,
        dim=int(model_cfg.get("dim", 512)),
        tokens_per_modality=int(model_cfg.get("tokens_per_modality", 32)),
        num_classes=int(model_cfg.get("num_classes", 27)),
        backbone_root=model_cfg.get("backbone_root", "backbones"),
        freeze_pretrained_backbones=bool(model_cfg.get("freeze_pretrained_backbones", True)),
        graph_layers=int(model_cfg.get("graph_layers", 2)),
        fusion_layers=int(model_cfg.get("fusion_layers", 2)),
        fusion_heads=int(model_cfg.get("fusion_heads", 8)),
        tau=float(config.get("loss", {}).get("tau", model_cfg.get("tau", 1.0))),
        fusion_mode=model_cfg.get("fusion_mode", "uncertainty"),
        pretrained_paths=model_cfg.get("pretrained_paths", {}),
    )
