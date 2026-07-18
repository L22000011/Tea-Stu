from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn


def _load_weights(module: nn.Module, path: str | Path | None) -> None:
    if path is None:
        return
    weight_path = Path(path)
    if not weight_path.exists():
        print(f"[XRF55Backbone] Weight file not found, using random initialization: {weight_path}")
        return
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(
        f"[XRF55Backbone] Loaded weights from {weight_path} | "
        f"missing={len(missing)} | unexpected={len(unexpected)}"
    )


def _conv3x3_1d(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


def _conv1x1_1d(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, groups: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3_1d(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3_1d(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNetLargeBert1D(nn.Module):
    def __init__(self, inchannel: int, layers: list[int], activity_num: int = 55) -> None:
        super().__init__()
        self.inplanes = 256
        self.conv1 = nn.Conv1d(inchannel, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(256, layers[0], stride=1)
        self.layer2 = self._make_layer(256, layers[1], stride=2)
        self.layer3 = self._make_layer(512, layers[2], stride=2)
        self.layer4 = self._make_layer(1024, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, activity_num)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1_1d(self.inplanes, planes, stride),
                nn.BatchNorm1d(planes),
            )
        layers = [BasicBlock1D(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        pooled = self.avg_pool(x).view(x.size(0), -1)
        logits = self.fc(pooled)
        return logits, pooled


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_channel: int, out_channel: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetLargeBert2D(nn.Module):
    def __init__(self, num_classes: int = 55) -> None:
        super().__init__()
        self.in_channel = 128
        self.conv1 = nn.Conv2d(17, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(128, 2)
        self.layer2 = self._make_layer(256, 2, stride=2)
        self.layer3 = self._make_layer(512, 2, stride=2)
        self.layer4 = self._make_layer(1024, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, channel: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel),
            )
        layers = [BasicBlock2D(self.in_channel, channel, stride=stride, downsample=downsample)]
        self.in_channel = channel
        for _ in range(1, blocks):
            layers.append(BasicBlock2D(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 5 and x.size(1) == 1:
            x = x[:, 0]
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        pooled = torch.flatten(self.avgpool(x), 1)
        logits = self.fc(pooled)
        return logits, pooled


class WiFiFeatureExtractor(nn.Module):
    def __init__(self, weight_path: str | Path | None = None, num_classes: int = 55) -> None:
        super().__init__()
        self.model = ResNetLargeBert1D(inchannel=270, layers=[2, 2, 2, 2], activity_num=num_classes)
        _load_weights(self.model, weight_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.model(x)
        return features


class RFIDFeatureExtractor(nn.Module):
    def __init__(self, weight_path: str | Path | None = None, num_classes: int = 55) -> None:
        super().__init__()
        self.model = ResNetLargeBert1D(inchannel=23, layers=[2, 2, 2, 2], activity_num=num_classes)
        _load_weights(self.model, weight_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.model(x)
        return features


class MMWaveFeatureExtractor(nn.Module):
    def __init__(self, weight_path: str | Path | None = None, num_classes: int = 55) -> None:
        super().__init__()
        self.model = ResNetLargeBert2D(num_classes=num_classes)
        _load_weights(self.model, weight_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.model(x)
        return features
