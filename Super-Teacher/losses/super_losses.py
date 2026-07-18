from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .pose_losses import bone_length_loss, mpjpe_loss, smooth_l1_pose_loss


def _zero_like_output(output: dict | None, target: torch.Tensor | None) -> torch.Tensor:
    if output:
        tensor = next(value for value in output.values() if torch.is_tensor(value))
        return tensor.sum() * 0.0
    if target is not None:
        return target.sum() * 0.0
    return torch.tensor(0.0)


def _reliability_regularizer(output: dict, weight: float) -> torch.Tensor:
    if weight <= 0 or "alphas" not in output:
        return _zero_like_output(output, None)
    alphas = output["alphas"].clamp_min(1e-6)
    entropy = -(alphas * alphas.log()).sum(dim=1).mean()
    return -weight * entropy


def compute_hpe_teacher_loss(output: dict, target: torch.Tensor, weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    pose_loss = smooth_l1_pose_loss(output["pose"], target)
    geo_loss = mpjpe_loss(output["pose"], target)
    bone_loss = bone_length_loss(output["pose"], target)
    rel_loss = _reliability_regularizer(output, float(weights.get("lambda_rel_entropy", 0.0)))
    total = pose_loss + geo_loss + float(weights.get("lambda_bone", 0.1)) * bone_loss + rel_loss
    return total, {
        "hpe_pose": float(pose_loss.detach().cpu()),
        "hpe_mpjpe": float(geo_loss.detach().cpu()),
        "hpe_bone": float(bone_loss.detach().cpu()),
    }


def compute_har_teacher_loss(output: dict, target: torch.Tensor, weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce_loss = F.cross_entropy(output["logits"], target.long())
    rel_loss = _reliability_regularizer(output, float(weights.get("lambda_rel_entropy", 0.0)))
    return ce_loss + rel_loss, {"har_ce": float(ce_loss.detach().cpu())}


def compute_super_teacher_losses(
    hpe_output: dict | None,
    hpe_target: torch.Tensor | None,
    har_output: dict | None,
    har_target: torch.Tensor | None,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    terms = []
    logs: Dict[str, float] = {}
    if hpe_output is not None and hpe_target is not None:
        hpe_loss, hpe_logs = compute_hpe_teacher_loss(hpe_output, hpe_target, weights)
        terms.append(float(weights.get("lambda_hpe", 1.0)) * hpe_loss)
        logs.update(hpe_logs)
    if har_output is not None and har_target is not None:
        har_loss, har_logs = compute_har_teacher_loss(har_output, har_target, weights)
        terms.append(float(weights.get("lambda_har", 1.0)) * har_loss)
        logs.update(har_logs)
    if not terms:
        target = hpe_target if hpe_target is not None else har_target
        output = hpe_output if hpe_output is not None else har_output
        return _zero_like_output(output, target), logs
    return sum(terms), logs


def _token_kd_loss(student_output: dict, teacher_output: dict, weight: float) -> torch.Tensor:
    if weight <= 0:
        return _zero_like_output(student_output, None)
    if "fused_token" not in student_output or "fused_token" not in teacher_output:
        return _zero_like_output(student_output, None)
    return weight * F.mse_loss(student_output["fused_token"], teacher_output["fused_token"].detach())


def _reliability_kd_loss(student_output: dict, teacher_output: dict, weight: float) -> torch.Tensor:
    if weight <= 0:
        return _zero_like_output(student_output, None)
    if "alphas" not in student_output or "alphas" not in teacher_output:
        return _zero_like_output(student_output, None)
    student_modalities = student_output.get("modalities", [])
    teacher_modalities = teacher_output.get("modalities", [])
    if not student_modalities or not teacher_modalities:
        if student_output["alphas"].shape == teacher_output["alphas"].shape:
            return weight * F.mse_loss(student_output["alphas"], teacher_output["alphas"].detach())
        return _zero_like_output(student_output, None)
    teacher_index = {name: idx for idx, name in enumerate(teacher_modalities)}
    pairs = [(s_idx, teacher_index[name]) for s_idx, name in enumerate(student_modalities) if name in teacher_index]
    if not pairs:
        return _zero_like_output(student_output, None)
    student_idx = torch.tensor([item[0] for item in pairs], device=student_output["alphas"].device, dtype=torch.long)
    teacher_idx = torch.tensor([item[1] for item in pairs], device=teacher_output["alphas"].device, dtype=torch.long)
    student_alpha = student_output["alphas"].index_select(1, student_idx)
    teacher_alpha = teacher_output["alphas"].index_select(1, teacher_idx).detach()
    teacher_alpha = teacher_alpha / teacher_alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    student_alpha = student_alpha / student_alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weight * F.mse_loss(student_alpha, teacher_alpha)


def compute_hpe_student_loss(
    student_output: dict,
    target: torch.Tensor,
    weights: Dict[str, float],
    teacher_output: dict | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    gt_loss = smooth_l1_pose_loss(student_output["pose"], target) + mpjpe_loss(student_output["pose"], target)
    bone_loss = bone_length_loss(student_output["pose"], target)
    total = gt_loss + float(weights.get("lambda_bone", 0.1)) * bone_loss
    logs = {"gt": float(gt_loss.detach().cpu()), "bone": float(bone_loss.detach().cpu())}
    if teacher_output is not None:
        pose_kd = F.smooth_l1_loss(student_output["pose"], teacher_output["pose"].detach())
        total = total + float(weights.get("lambda_pose_kd", 0.5)) * pose_kd
        total = total + _token_kd_loss(student_output, teacher_output, float(weights.get("lambda_token", 0.0)))
        total = total + _reliability_kd_loss(student_output, teacher_output, float(weights.get("lambda_rel", 0.0)))
        logs["pose_kd"] = float(pose_kd.detach().cpu())
    return total, logs


def compute_har_student_loss(
    student_output: dict,
    target: torch.Tensor,
    weights: Dict[str, float],
    teacher_output: dict | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce_loss = F.cross_entropy(student_output["logits"], target.long())
    total = ce_loss
    logs = {"ce": float(ce_loss.detach().cpu())}
    if teacher_output is not None:
        temperature = float(weights.get("temperature", 4.0))
        kd_loss = F.kl_div(
            F.log_softmax(student_output["logits"] / temperature, dim=1),
            F.softmax(teacher_output["logits"].detach() / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        total = total + float(weights.get("lambda_logit", 0.7)) * kd_loss
        total = total + _token_kd_loss(student_output, teacher_output, float(weights.get("lambda_token", 0.0)))
        total = total + _reliability_kd_loss(student_output, teacher_output, float(weights.get("lambda_rel", 0.0)))
        logs["logit_kd"] = float(kd_loss.detach().cpu())
    return total, logs
