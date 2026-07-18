from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn

from utils.modality import ALL_MODALITIES, canonicalize_modalities
from .official_backbones import MMWaveFeatureExtractor, RFIDFeatureExtractor, WiFiFeatureExtractor
from .reliability_fusion import ClassificationReliabilityFusion, ModalityTokenEncoder


class TokenProjector(nn.Module):
    def __init__(self, input_dim: int = 1024, dim: int = 512, output_tokens: int = 16) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.dim = dim
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_tokens * dim)
        self.act = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.act(self.fc(self.norm(features)))
        return tokens.view(features.size(0), self.output_tokens, self.dim)


class XRFRCDModel(nn.Module):
    def __init__(
        self,
        modalities: Sequence[str] = ALL_MODALITIES,
        num_classes: int = 55,
        feature_dim: int = 1024,
        dim: int = 512,
        semantic_dim: int = 1024,
        tokens_per_modality: int = 16,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_mode: str = "uncertainty",
        tau: float = 1.0,
        freeze_pretrained_backbones: bool = False,
        pretrained_paths: Dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.modalities = canonicalize_modalities(modalities)
        self.reference_modalities = list(ALL_MODALITIES)
        self.num_classes = num_classes
        pretrained_paths = pretrained_paths or {}

        self.extractors = nn.ModuleDict()
        if "wifi" in self.modalities:
            self.extractors["wifi"] = WiFiFeatureExtractor(pretrained_paths.get("wifi"), num_classes=num_classes)
        if "rfid" in self.modalities:
            self.extractors["rfid"] = RFIDFeatureExtractor(pretrained_paths.get("rfid"), num_classes=num_classes)
        if "mmwave" in self.modalities:
            self.extractors["mmwave"] = MMWaveFeatureExtractor(pretrained_paths.get("mmwave"), num_classes=num_classes)

        self.projectors = nn.ModuleDict(
            {
                name: TokenProjector(input_dim=feature_dim, dim=dim, output_tokens=tokens_per_modality)
                for name in self.modalities
            }
        )
        self.token_encoder = ModalityTokenEncoder(
            dim=dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            max_modalities=len(self.reference_modalities),
        )
        self.fusion = ClassificationReliabilityFusion(
            dim=dim,
            num_classes=num_classes,
            semantic_dim=semantic_dim,
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

    def train(self, mode: bool = True) -> "XRFRCDModel":
        super().train(mode)
        for extractor in self.extractors.values():
            if not any(param.requires_grad for param in extractor.parameters()):
                extractor.eval()
        return self

    def extract_features(self, inputs: Dict[str, torch.Tensor], selected_modalities: list[str]) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        for name in selected_modalities:
            raw = self.extractors[name](inputs[name])
            features[name] = self.projectors[name](raw)
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


def build_model(config: Dict[str, Any], modalities: Sequence[str] | None = None) -> XRFRCDModel:
    model_cfg = config.get("model", {})
    selected_modalities = modalities or model_cfg.get("modalities") or config.get("modality") or ALL_MODALITIES
    tau = float(config.get("loss", {}).get("tau", model_cfg.get("tau", 1.0)))
    return XRFRCDModel(
        modalities=selected_modalities,
        num_classes=int(model_cfg.get("num_classes", 55)),
        feature_dim=int(model_cfg.get("feature_dim", 1024)),
        dim=int(model_cfg.get("dim", 512)),
        semantic_dim=int(model_cfg.get("semantic_dim", 1024)),
        tokens_per_modality=int(model_cfg.get("tokens_per_modality", 16)),
        fusion_layers=int(model_cfg.get("fusion_layers", 2)),
        fusion_heads=int(model_cfg.get("fusion_heads", 8)),
        fusion_mode=model_cfg.get("fusion_mode", "uncertainty"),
        tau=tau,
        freeze_pretrained_backbones=bool(model_cfg.get("freeze_pretrained_backbones", False)),
        pretrained_paths=model_cfg.get("pretrained_paths", {}),
    )
