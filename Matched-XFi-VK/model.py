"""Matched X-Fi-style fusion with the formal VK/non-visual inputs.

This module is an adapted baseline, not a claim that the original X-Fi
implementation is identical to the VK-RMD codebase. It keeps the main
X-Fi-style ingredients (cross-modal pooling, modality-specific KV projection,
and cross-attention injection) while using the same encoders and VK input
protocol as the formal model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch import nn

from utils.modality import ALL_MODALITIES, canonicalize_modalities

from models.skeleton_prompt_encoder import SkeletonPromptEncoder

try:
    from models.vk_rcd import TokenProjector
    from models.legacy_backbones import CSIFeatureExtractor, DepthFeatureExtractor, LidarFeatureExtractor, MMwaveFeatureExtractor
except ImportError:
    from models.vk_rcd_har import TokenProjector
    from models.legacy_backbones import DepthFeatureExtractor, LidarHARFeatureExtractor, MMwaveFeatureExtractor
    CSIFeatureExtractor = None
    LidarFeatureExtractor = LidarHARFeatureExtractor


class KVProjection(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(x)
        return self.to_k(x), self.to_v(x)


class XFiStyleFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        tokens_per_modality: int,
        max_modalities: int,
        num_heads: int,
        depth: int,
        output_kind: str,
        num_joints: int = 17,
        num_classes: int = 27,
    ) -> None:
        super().__init__()
        self.tokens_per_modality = tokens_per_modality
        self.output_kind = output_kind
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.kv_layers = nn.ModuleList([KVProjection(dim) for _ in range(max_modalities)])
        self.cross_modal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        output_dim = num_joints * 3 if output_kind == "hpe" else num_classes
        self.head = nn.Linear(dim, output_dim)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) == self.tokens_per_modality:
            return tokens
        return torch.nn.functional.adaptive_avg_pool1d(
            tokens.transpose(1, 2), self.tokens_per_modality
        ).transpose(1, 2)

    def forward(self, projected: Dict[str, torch.Tensor], modalities: Sequence[str]) -> Dict[str, Any]:
        features = [projected[name] for name in modalities]
        multimodal = torch.cat(features, dim=1)
        cross = self._pool(self.cross_modal(multimodal))

        for _ in range(self.depth):
            injected = []
            for index, feature in enumerate(features):
                key, value = self.kv_layers[index](self._pool(feature))
                update, _ = self.cross_attention(cross, key, value, need_weights=False)
                injected.append(update + cross)
            cross = self._pool(torch.cat([self.cross_ffn(item) + item for item in injected], dim=1))

        pooled = self.norm(cross.mean(dim=1))
        prediction = self.head(pooled)
        if self.output_kind == "hpe":
            prediction = prediction.view(prediction.size(0), self.num_joints, 3)
            return {"pose": prediction, "tokens": pooled, "modalities": list(modalities)}
        return {"logits": prediction, "tokens": pooled, "modalities": list(modalities)}


class MatchedXFiVKModel(nn.Module):
    def __init__(
        self,
        project: str,
        modalities: Sequence[str],
        dim: int = 512,
        tokens_per_modality: int = 32,
        backbone_root: str | Path = "backbones",
        pretrained_paths: Dict[str, str] | None = None,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        num_classes: int = 27,
    ) -> None:
        super().__init__()
        self.project = project.upper()
        self.modalities = canonicalize_modalities(modalities)
        self.reference_modalities = list(ALL_MODALITIES)
        pretrained_paths = pretrained_paths or {}
        self.vk_encoder = SkeletonPromptEncoder(
            dim=dim,
            num_output_tokens=tokens_per_modality,
            graph_layers=2,
        )
        self.extractors = nn.ModuleDict()
        if "depth" in self.modalities:
            self.extractors["depth"] = DepthFeatureExtractor(backbone_root, pretrained_paths.get("depth"))
        if "lidar" in self.modalities:
            self.extractors["lidar"] = LidarFeatureExtractor(backbone_root, pretrained_paths.get("lidar"))
        if "mmwave" in self.modalities:
            self.extractors["mmwave"] = MMwaveFeatureExtractor(backbone_root, pretrained_paths.get("mmwave"))
        if "wifi-csi" in self.modalities:
            self.extractors["wifi_csi"] = CSIFeatureExtractor(backbone_root, pretrained_paths.get("wifi-csi"))

        self.projectors = nn.ModuleDict()
        for name in self.modalities:
            if name == "vk":
                continue
            self.projectors[name.replace("-", "_")] = TokenProjector(dim, tokens_per_modality)

        self.fusion = XFiStyleFusion(
            dim=dim,
            tokens_per_modality=tokens_per_modality,
            max_modalities=len(self.reference_modalities),
            num_heads=fusion_heads,
            depth=fusion_layers,
            output_kind="hpe" if self.project == "HPE" else "har",
            num_classes=num_classes,
        )
        self._freeze_extractors()

    def _freeze_extractors(self) -> None:
        for extractor in self.extractors.values():
            extractor.eval()
            for parameter in extractor.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "MatchedXFiVKModel":
        super().train(mode)
        for extractor in self.extractors.values():
            extractor.eval()
        return self

    def _extract(self, inputs: Dict[str, torch.Tensor], selected: Sequence[str]) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        if "vk" in selected:
            _, projected["vk"] = self.vk_encoder(inputs["vk"])
        for name in selected:
            if name == "vk":
                continue
            key = name.replace("-", "_")
            raw = self.extractors[key](inputs[name])
            projected[name] = self.projectors[key](raw)
        return projected

    def forward(self, inputs: Dict[str, torch.Tensor], selected_modalities: Sequence[str]) -> Dict[str, Any]:
        selected = canonicalize_modalities(selected_modalities)
        if not selected:
            raise ValueError("At least one modality must be selected")
        projected = self._extract(inputs, selected)
        output = self.fusion(projected, selected)
        output["projected_tokens"] = projected
        return output


def build_model(config: Dict[str, Any], project: str) -> MatchedXFiVKModel:
    model_cfg = config.get("model", {})
    modalities = config.get("student_modalities") or model_cfg.get("modalities") or ALL_MODALITIES
    return MatchedXFiVKModel(
        project=project,
        modalities=modalities,
        dim=int(model_cfg.get("dim", 512)),
        tokens_per_modality=int(model_cfg.get("tokens_per_modality", 32)),
        backbone_root=model_cfg.get("backbone_root", "backbones"),
        pretrained_paths=model_cfg.get("pretrained_paths", {}),
        fusion_layers=int(model_cfg.get("fusion_layers", 2)),
        fusion_heads=int(model_cfg.get("fusion_heads", 8)),
        num_classes=int(model_cfg.get("num_classes", 27)),
    )
