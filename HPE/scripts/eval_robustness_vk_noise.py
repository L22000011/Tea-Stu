from __future__ import annotations

import copy

import torch
from tqdm import tqdm

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import move_batch_to_device, write_csv
from utils.metrics import AverageMeter, compute_metrics


def corrupt_vk(vk: torch.Tensor, noise_std: float, drop_joint_ratio: float) -> torch.Tensor:
    output = vk.clone()
    if noise_std > 0:
        output = output + torch.randn_like(output) * noise_std
    if drop_joint_ratio > 0:
        mask = torch.rand(output.shape[:2], device=output.device) < drop_joint_ratio
        output[mask] = 0.0
    return output


@torch.no_grad()
def evaluate_noise(model, dataloader, device, modality_set, noise_std, drop_joint_ratio, max_batches=None):
    meters = {name: AverageMeter() for name in ["mse", "mpjpe", "pa_mpjpe"]}
    model.eval()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"vk_noise:{noise_std}:{drop_joint_ratio}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        inputs = copy.copy(batch["inputs"])
        inputs["vk"] = corrupt_vk(inputs["vk"], noise_std, drop_joint_ratio)
        output = model(inputs, modality_set)
        metrics = compute_metrics(output["pose"], batch["target"])
        batch_size = batch["target"].size(0)
        for key, value in metrics.items():
            meters[key].update(value, batch_size)
    return {key: meter.avg for key, meter in meters.items()}


def main() -> None:
    parser = base_parser("Evaluate VK keypoint noise and missing-joint robustness.")
    parser.add_argument("--modality-set", type=str, default="vk,depth,lidar,mmwave,wifi-csi")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/vk_noise_robustness.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    modality_set = [item.strip() for item in args.modality_set.split(",") if item.strip()]
    rows = []
    for noise_std in [0.0, 5.0, 10.0, 20.0]:
        for drop_ratio in [0.0, 0.1, 0.3, 0.5]:
            metrics = evaluate_noise(
                model,
                val_loader,
                device,
                modality_set,
                noise_std,
                drop_ratio,
                max_batches=config.get("max_eval_batches"),
            )
            rows.append({
                "method": config.get("method", "VK-RCD"),
                "modality_set": "+".join(modality_set),
                "noise_std": noise_std,
                "drop_joint_ratio": drop_ratio,
                "mse": metrics["mse"],
                "mpjpe": metrics["mpjpe"],
                "pa_mpjpe": metrics["pa_mpjpe"],
            })
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()

