from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_similarity_transform(
    target: np.ndarray,
    prediction: np.ndarray,
    compute_optimal_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    mu_x = target.mean(0)
    mu_y = prediction.mean(0)
    x0 = target - mu_x
    y0 = prediction - mu_y
    ss_x = np.square(x0).sum()
    ss_y = np.square(y0).sum()
    norm_x = np.sqrt(ss_x)
    norm_y = np.sqrt(ss_y)
    if norm_x < 1e-8 or norm_y < 1e-8:
        return 0.0, prediction, np.eye(target.shape[1]), 1.0, np.zeros(target.shape[1])
    x0 = x0 / norm_x
    y0 = y0 / norm_y
    a = np.dot(x0.T, y0)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    v = vt.T
    t = np.dot(v, u.T)
    det_t = np.linalg.det(t)
    v[:, -1] *= np.sign(det_t)
    s[-1] *= np.sign(det_t)
    t = np.dot(v, u.T)
    trace_ta = s.sum()
    if compute_optimal_scale:
        b = trace_ta * norm_x / norm_y
        d = 1 - trace_ta**2
        z = norm_x * trace_ta * np.dot(y0, t) + mu_x
    else:
        b = 1.0
        d = 1 + ss_y / ss_x - 2 * trace_ta * norm_y / norm_x
        z = norm_y * np.dot(y0, t) + mu_x
    c = mu_x - b * np.dot(mu_y, t)
    return d, z, t, b, c


def mpjpe_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.sum(np.square(prediction - target), axis=2))))


def pa_mpjpe_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    n, num_joints, _ = prediction.shape
    errors = np.zeros((n, num_joints), dtype=np.float64)
    for idx in range(n):
        _, aligned, rotation, scale, offset = compute_similarity_transform(
            target[idx],
            prediction[idx],
            compute_optimal_scale=True,
        )
        frame_prediction = scale * prediction[idx].dot(rotation) + offset
        errors[idx] = np.sqrt(np.sum(np.square(frame_prediction - target[idx]), axis=1))
    return float(np.mean(errors))


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = prediction.detach().float().cpu()
    gt = target.detach().float().cpu()
    mse = torch.mean((pred - gt) ** 2).item()
    pred_np = pred.numpy()
    gt_np = gt.numpy()
    return {
        "mse": float(mse),
        "mpjpe": mpjpe_numpy(pred_np, gt_np),
        "pa_mpjpe": pa_mpjpe_numpy(pred_np, gt_np),
    }


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

