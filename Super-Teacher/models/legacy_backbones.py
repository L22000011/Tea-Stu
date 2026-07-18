from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import Resize


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


def _register_csi_pickle_modules(csi_root: Path) -> None:
    module_dir = csi_root / "models"
    sys.path.insert(0, str(csi_root))
    sys.path.insert(0, str(module_dir))
    module_names = ["CTrans", "wisppn_resnet", "mynetwork"]
    for name in module_names:
        module_path = module_dir / f"{name}.py"
        full_name = f"models.{name}"
        if full_name in sys.modules:
            continue
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        parent = sys.modules.get("models")
        if parent is not None:
            setattr(parent, name, module)


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


class LidarFeatureExtractor(nn.Module):
    def __init__(self, backbone_root: str | Path = "backbones", weights: Optional[str | Path] = None) -> None:
        super().__init__()
        from backbones.lidar_benchmark.lidar_point_transformer import lidar_PointTransformerReg

        project_root = Path(backbone_root).expanduser().resolve().parent
        model = lidar_PointTransformerReg(root=str(project_root))
        weight_path = _resolve_weight_path(backbone_root, "lidar_benchmark/lidar_all_random.pt", weights)
        model.load_state_dict(_load_torch_file(weight_path))
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


class CSIFeatureExtractor(nn.Module):
    def __init__(self, backbone_root: str | Path = "backbones", weights: Optional[str | Path] = None) -> None:
        super().__init__()
        csi_root = Path(backbone_root).expanduser().resolve() / "CSI_benchmark"
        sys.path.insert(0, str(csi_root))
        _register_csi_pickle_modules(csi_root)
        weight_path = _resolve_weight_path(backbone_root, "CSI_benchmark/protocol3_random_1.pkl", weights)
        model = _load_torch_file(weight_path)
        self.part = nn.Sequential(
            model.encoder_conv1,
            model.encoder_bn1,
            model.encoder_relu,
            model.encoder_layer1,
            model.encoder_layer2,
            model.encoder_layer3,
            model.encoder_layer4,
        )
        self.resize = Resize([136, 32])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.transpose(x, 2, 3)
        x = torch.flatten(x, 3, 4)
        x = self.resize(x)
        x = self.part(x).view(x.size(0), 512, -1)
        return x.permute(0, 2, 1)
