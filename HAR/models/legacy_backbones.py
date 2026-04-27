from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn


def _load_torch_file(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_weight_path(backbone_root: str | Path, relative_path: str, weights: Optional[str | Path] = None) -> Path:
    path = Path(weights) if weights else Path(backbone_root) / relative_path
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pretrained backbone file was not found: {path}")
    return path


class DepthFeatureExtractor(nn.Module):
    def __init__(self, backbone_root: str | Path = "backbones", weights: Optional[str | Path] = None) -> None:
        super().__init__()
        from backbones.depth_benchmark.depth_ResNet18 import Depth_ResNet18

        model = Depth_ResNet18()
        weight_path = _resolve_weight_path(backbone_root, "depth_benchmark/depth_Resnet18.pt", weights)
        model.load_state_dict(_load_torch_file(weight_path))
        self.part = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.part(x).view(x.size(0), 512, -1)
        return x.permute(0, 2, 1)


class MMwaveFeatureExtractor(nn.Module):
    def __init__(self, backbone_root: str | Path = "backbones", weights: Optional[str | Path] = None) -> None:
        super().__init__()
        from backbones.mmwave_benchmark.mmwave_point_transformer_TD import mmwave_PointTransformerReg

        model = mmwave_PointTransformerReg()
        weight_path = _resolve_weight_path(backbone_root, "mmwave_benchmark/mmwave_all_random_TD.pt", weights)
        model.load_state_dict(_load_torch_file(weight_path))
        self.part = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.part(x)
        return features


class LidarHARFeatureExtractor(nn.Module):
    def __init__(self, backbone_root: str | Path = "backbones", weights: Optional[str | Path] = None) -> None:
        super().__init__()
        from backbones.lidar_benchmark import lidar_point_transformer

        project_root = Path(backbone_root).expanduser().resolve().parent
        lidar_cls = getattr(lidar_point_transformer, "lidar_PointTransformer_cls", None)
        if lidar_cls is None:
            lidar_cls = getattr(lidar_point_transformer, "lidar_PointTransformerReg")
        model = lidar_cls(root=str(project_root))
        weight_path = _resolve_weight_path(backbone_root, "lidar_benchmark/lidar_all_random.pt", weights)
        model.load_state_dict(_load_torch_file(weight_path), strict=False)
        self.fc1 = model.backbone.fc1
        self.transformer1 = model.backbone.transformer1
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        nblocks = 5
        for idx in range(nblocks - 4):
            self.transition_downs.append(model.backbone.transition_downs[idx])
            self.transformers.append(model.backbone.transformers[idx])
        self.nblocks = nblocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        for idx in range(self.nblocks - 4):
            xyz, points = self.transition_downs[idx](xyz, points)
            points = self.transformers[idx](xyz, points)[0]
        return points.view(points.size(0), -1, 512)

