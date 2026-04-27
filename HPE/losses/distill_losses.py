from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .pose_losses import bone_length_loss, smooth_l1_pose_loss


def compute_teacher_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    weights: Dict[str, float],
) -> tuple[torch.Tensor, Dict[str, float]]:
    pose_loss = smooth_l1_pose_loss(output["pose"], target)
    bone_loss = bone_length_loss(output["pose"], target)
    total = pose_loss + float(weights.get("lambda_bone", 0.1)) * bone_loss
    logs = {
        "loss": float(total.detach().cpu()),
        "pose_loss": float(pose_loss.detach().cpu()),
        "bone_loss": float(bone_loss.detach().cpu()),
    }
    return total, logs


def _restricted_teacher_alpha(
    teacher_alpha: torch.Tensor,
    teacher_modalities: list[str],
    student_modalities: list[str],
) -> torch.Tensor:
    indices = [teacher_modalities.index(name) for name in student_modalities if name in teacher_modalities]
    if len(indices) != len(student_modalities):
        raise ValueError("Teacher alpha does not contain every student modality.")
    restricted = teacher_alpha[:, indices]
    denom = restricted.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return restricted / denom


def compute_student_loss(
    student: Dict[str, Any],
    target: torch.Tensor,
    weights: Dict[str, float],
    teacher: Dict[str, Any] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    gt_loss = smooth_l1_pose_loss(student["pose"], target)
    loss = gt_loss
    logs = {"loss_gt": float(gt_loss.detach().cpu())}

    if teacher is not None:
        out_kd = smooth_l1_pose_loss(student["pose"], teacher["pose"].detach())
        loss = loss + float(weights.get("lambda_out", 0.5)) * out_kd
        logs["loss_out_kd"] = float(out_kd.detach().cpu())

        if "tokens" in student and "tokens" in teacher:
            token_kd = F.mse_loss(student["tokens"], teacher["tokens"].detach())
            loss = loss + float(weights.get("lambda_token", 0.2)) * token_kd
            logs["loss_token_kd"] = float(token_kd.detach().cpu())

        bone_kd = bone_length_loss(student["pose"], teacher["pose"].detach())
        loss = loss + float(weights.get("lambda_bone", 0.1)) * bone_kd
        logs["loss_bone_kd"] = float(bone_kd.detach().cpu())

        if "alphas" in student and "alphas" in teacher:
            try:
                teacher_alpha = _restricted_teacher_alpha(
                    teacher["alphas"].detach(),
                    teacher["modalities"],
                    student["modalities"],
                )
                rel_kd = F.kl_div(
                    torch.log(student["alphas"].clamp_min(1e-6)),
                    teacher_alpha.clamp_min(1e-6),
                    reduction="batchmean",
                )
                loss = loss + float(weights.get("lambda_rel", 0.05)) * rel_kd
                logs["loss_rel_kd"] = float(rel_kd.detach().cpu())
            except ValueError:
                logs["loss_rel_kd"] = 0.0

    if "modality_poses" in student and "logvars" in student:
        unc_terms = []
        for idx, pose in enumerate(student["modality_poses"]):
            logvar = student["logvars"][:, idx].view(-1, 1, 1)
            err = F.smooth_l1_loss(pose, target, reduction="none").mean(dim=(1, 2), keepdim=True)
            unc_terms.append(torch.exp(-logvar) * err + logvar)
        unc_loss = torch.cat(unc_terms, dim=1).mean()
        loss = loss + float(weights.get("lambda_unc", 0.1)) * unc_loss
        logs["loss_unc"] = float(unc_loss.detach().cpu())

    logs["loss"] = float(loss.detach().cpu())
    return loss, logs

