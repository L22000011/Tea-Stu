"""Runtime readout ablations that leave the formal model files untouched."""

from __future__ import annotations

from types import MethodType

import torch


VALID_MODES = ("chunk_only", "global_only", "dual")


def _forward(self, encoded_tokens: torch.Tensor, selected_modalities: list[str]):
    chunks = list(encoded_tokens.split(self.tokens_per_modality, dim=1))
    predictions = []
    logvars = []
    pooled_tokens = []
    for chunk in chunks:
        prediction, logvar = self.shared_head(chunk)
        predictions.append(prediction)
        logvars.append(logvar)
        pooled_tokens.append(chunk.mean(dim=1))

    logvar_tensor = torch.stack(logvars, dim=1)
    if self.fusion_mode == "uniform":
        alphas = torch.ones_like(logvar_tensor) / max(logvar_tensor.size(1), 1)
    elif self.fusion_mode == "attention":
        scores = torch.cat(
            [self.score_head(token).view(token.size(0), 1) for token in pooled_tokens], dim=1
        )
        alphas = torch.softmax(scores / float(self.tau), dim=1)
    else:
        alphas = torch.softmax(-logvar_tensor / float(self.tau), dim=1)

    stacked = torch.stack(predictions, dim=1)
    if stacked.ndim == 4:  # HPE: B x M x J x 3
        chunk_prediction = torch.sum(alphas[:, :, None, None] * stacked, dim=1)
    else:  # HAR: B x M x C
        chunk_prediction = torch.sum(alphas[:, :, None] * stacked, dim=1)

    pooled = torch.stack(pooled_tokens, dim=1)
    fused_token = torch.sum(alphas[:, :, None] * pooled, dim=1)
    global_prediction = self.final_head(self.final_norm(fused_token))
    is_hpe = hasattr(self, "num_joints")
    if is_hpe:
        global_prediction = global_prediction.view(fused_token.size(0), self.num_joints, 3)

    mode = self.readout_mode
    if mode == "chunk_only":
        final_prediction = chunk_prediction
    elif mode == "global_only":
        final_prediction = global_prediction
    else:
        final_prediction = 0.5 * chunk_prediction + 0.5 * global_prediction

    output_key = "pose" if is_hpe else "logits"
    expert_key = "modality_poses" if is_hpe else "modality_logits"
    return {
        output_key: final_prediction,
        "tokens": encoded_tokens.mean(dim=1) if is_hpe else fused_token,
        expert_key: predictions,
        "logvars": logvar_tensor,
        "alphas": alphas,
        "modalities": selected_modalities,
        "chunk_prediction": chunk_prediction,
        "global_prediction": global_prediction,
        "readout_mode": mode,
    }


def apply_readout_mode(model, mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown readout mode {mode!r}; expected one of {VALID_MODES}")
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        raise AttributeError("Model does not expose a fusion module.")
    fusion.readout_mode = mode
    fusion.forward = MethodType(_forward, fusion)

