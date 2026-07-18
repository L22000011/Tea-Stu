from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from losses.pose_losses import COCO17_BONES


class BoneGraphMixer(nn.Module):
    def __init__(self, dim: int, num_joints: int = 17, bones: Sequence[Tuple[int, int]] = COCO17_BONES) -> None:
        super().__init__()
        adjacency = torch.eye(num_joints)
        for start, end in bones:
            adjacency[start, end] = 1.0
            adjacency[end, start] = 1.0
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adjacency", adjacency)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        neighbor_tokens = torch.einsum("ij,bjd->bid", self.adjacency, tokens)
        mixed = self.mlp(torch.cat([tokens, neighbor_tokens], dim=-1))
        return self.norm(tokens + mixed)


class SkeletonPromptEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 256,
        num_joints: int = 17,
        num_output_tokens: int = 32,
        graph_layers: int = 2,
        coord_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_output_tokens = num_output_tokens
        # Kept for checkpoint/config compatibility; adaptive normalization is used instead.
        self.coord_scale = coord_scale
        self.image_width = 640.0
        self.image_height = 480.0
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )
        self.joint_embedding = nn.Embedding(num_joints, dim)
        self.graph_mixers = nn.ModuleList([BoneGraphMixer(dim, num_joints) for _ in range(graph_layers)])
        self.token_projector = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(num_joints, num_output_tokens),
            nn.ReLU(),
        )

    def normalize_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        coords = torch.nan_to_num(keypoints[..., :2].float(), nan=0.0, posinf=0.0, neginf=0.0)
        abs_max = coords.abs().amax(dim=1, keepdim=True)
        coord_min = coords.amin(dim=1, keepdim=True)

        normalized_range = (abs_max[..., 0:1] <= 2.0) & (abs_max[..., 1:2] <= 2.0)
        signed_normalized = normalized_range & ((coord_min[..., 0:1] < 0.0) | (coord_min[..., 1:2] < 0.0))
        positive_normalized = normalized_range & ~signed_normalized

        small_pixel_range = (abs_max[..., 0:1] <= 256.0) & (abs_max[..., 1:2] <= 256.0) & ~normalized_range

        signed_xy = (coords + 1.0) * 0.5
        positive_xy = coords
        small_pixel_xy = torch.stack(
            [
                coords[..., 0] / 224.0,
                coords[..., 1] / 224.0,
            ],
            dim=-1,
        )
        pixel_xy = torch.stack(
            [
                coords[..., 0] / self.image_width,
                coords[..., 1] / self.image_height,
            ],
            dim=-1,
        )

        out = torch.where(signed_normalized.expand_as(coords), signed_xy, pixel_xy)
        out = torch.where(positive_normalized.expand_as(coords), positive_xy, out)
        out = torch.where(small_pixel_range.expand_as(coords), small_pixel_xy, out)
        return out.clamp(0.0, 1.0)

    def forward(self, keypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints = torch.nan_to_num(keypoints.float(), nan=0.0, posinf=0.0, neginf=0.0)
        coords = self.normalize_keypoints(keypoints)
        centered_coords = coords * 2.0 - 1.0
        joint_ids = torch.arange(self.num_joints, device=keypoints.device)
        tokens = (
            self.coord_mlp(coords)
            + self.position_mlp(centered_coords)
            + self.joint_embedding(joint_ids)[None, :, :]
        )
        for mixer in self.graph_mixers:
            tokens = mixer(tokens)
        projected = self.token_projector(tokens.permute(0, 2, 1)).permute(0, 2, 1)
        return tokens, projected
