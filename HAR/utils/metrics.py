from __future__ import annotations

from typing import Dict

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


def classification_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = torch.argmax(logits.detach(), dim=1)
    target = target.detach().long()
    correct = (pred == target).sum().item()
    accuracy = correct / max(target.numel(), 1)

    f1_scores = []
    for cls in range(num_classes):
        pred_pos = pred == cls
        true_pos = target == cls
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1_scores.append((2 * tp) / denom)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {"acc": float(accuracy), "macro_f1": float(macro_f1)}

