from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


COCO17_BONES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def smooth_l1_pose_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def mpjpe_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.linalg.norm(prediction - target, dim=-1))


def bone_lengths(pose: torch.Tensor, bones: Sequence[Tuple[int, int]] = COCO17_BONES) -> torch.Tensor:
    values = []
    for start, end in bones:
        values.append(torch.linalg.norm(pose[:, start] - pose[:, end], dim=-1))
    return torch.stack(values, dim=-1)


def bone_length_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bones: Sequence[Tuple[int, int]] = COCO17_BONES,
) -> torch.Tensor:
    return F.l1_loss(bone_lengths(prediction, bones), bone_lengths(target, bones))


def bone_relation_matrix(pose: torch.Tensor) -> torch.Tensor:
    diff = pose[:, :, None, :] - pose[:, None, :, :]
    return torch.linalg.norm(diff, dim=-1)

