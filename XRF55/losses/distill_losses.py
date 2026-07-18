from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def compute_teacher_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    semantic_target: torch.Tensor,
    weights: Dict[str, float],
) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(output["logits"], target.long())
    loss = ce
    semantic = torch.zeros((), device=target.device)
    semantic_mod = torch.zeros((), device=target.device)

    if "semantic" in output and semantic_target is not None:
        semantic = F.l1_loss(output["semantic"], semantic_target)
        loss = loss + float(weights.get("lambda_semantic", 0.0)) * semantic

    if "modality_semantics" in output and semantic_target is not None:
        terms = [F.l1_loss(pred, semantic_target) for pred in output["modality_semantics"]]
        if terms:
            semantic_mod = torch.stack(terms).mean()
            loss = loss + float(weights.get("lambda_semantic_mod", 0.0)) * semantic_mod

    if "modality_logits" in output and "logvars" in output:
        unc_terms = []
        for idx, logits in enumerate(output["modality_logits"]):
            ce_i = F.cross_entropy(logits, target.long(), reduction="none")
            logvar = output["logvars"][:, idx].clamp(-8.0, 8.0)
            unc_terms.append(torch.exp(-logvar) * ce_i + logvar)
        unc = torch.stack(unc_terms, dim=1).mean()
        loss = loss + float(weights.get("lambda_unc", 0.05)) * unc
    else:
        unc = torch.zeros((), device=target.device)

    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_ce": float(ce.detach().cpu()),
        "loss_semantic": float(semantic.detach().cpu()),
        "loss_semantic_mod": float(semantic_mod.detach().cpu()),
        "loss_unc": float(unc.detach().cpu()),
    }


def _restricted_teacher_alpha(
    teacher_alpha: torch.Tensor,
    teacher_modalities: list[str],
    student_modalities: list[str],
) -> torch.Tensor:
    indices = [teacher_modalities.index(name) for name in student_modalities if name in teacher_modalities]
    if len(indices) != len(student_modalities):
        raise ValueError("Teacher alpha does not contain every student modality.")
    restricted = teacher_alpha[:, indices]
    return restricted / restricted.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def compute_student_loss(
    student: Dict[str, Any],
    target: torch.Tensor,
    semantic_target: torch.Tensor,
    weights: Dict[str, float],
    teacher: Dict[str, Any] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(student["logits"], target.long())
    loss = ce
    logs = {"loss_ce": float(ce.detach().cpu())}

    semantic = torch.zeros((), device=target.device)
    semantic_mod = torch.zeros((), device=target.device)
    if "semantic" in student and semantic_target is not None:
        semantic = F.l1_loss(student["semantic"], semantic_target)
        loss = loss + float(weights.get("lambda_semantic", 0.0)) * semantic
        logs["loss_semantic"] = float(semantic.detach().cpu())

    if "modality_semantics" in student and semantic_target is not None:
        terms = [F.l1_loss(pred, semantic_target) for pred in student["modality_semantics"]]
        if terms:
            semantic_mod = torch.stack(terms).mean()
            loss = loss + float(weights.get("lambda_semantic_mod", 0.0)) * semantic_mod
            logs["loss_semantic_mod"] = float(semantic_mod.detach().cpu())

    if teacher is not None:
        temperature = float(weights.get("temperature", 4.0))
        kd = F.kl_div(
            F.log_softmax(student["logits"] / temperature, dim=1),
            F.softmax(teacher["logits"].detach() / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)
        loss = loss + float(weights.get("lambda_kd", 0.7)) * kd
        logs["loss_kd"] = float(kd.detach().cpu())

        if "tokens" in student and "tokens" in teacher:
            token_kd = F.mse_loss(student["tokens"], teacher["tokens"].detach())
            loss = loss + float(weights.get("lambda_token", 0.1)) * token_kd
            logs["loss_token"] = float(token_kd.detach().cpu())

        if "semantic" in student and "semantic" in teacher:
            semantic_kd = F.mse_loss(student["semantic"], teacher["semantic"].detach())
            loss = loss + float(weights.get("lambda_semantic_kd", 0.05)) * semantic_kd
            logs["loss_semantic_kd"] = float(semantic_kd.detach().cpu())

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
                loss = loss + float(weights.get("lambda_rel", 0.03)) * rel_kd
                logs["loss_rel"] = float(rel_kd.detach().cpu())
            except ValueError:
                logs["loss_rel"] = 0.0

    if "modality_logits" in student and "logvars" in student:
        unc_terms = []
        for idx, logits in enumerate(student["modality_logits"]):
            ce_i = F.cross_entropy(logits, target.long(), reduction="none")
            logvar = student["logvars"][:, idx].clamp(-8.0, 8.0)
            unc_terms.append(torch.exp(-logvar) * ce_i + logvar)
        unc = torch.stack(unc_terms, dim=1).mean()
        loss = loss + float(weights.get("lambda_unc", 0.05)) * unc
        logs["loss_unc"] = float(unc.detach().cpu())

    logs["loss"] = float(loss.detach().cpu())
    return loss, logs
