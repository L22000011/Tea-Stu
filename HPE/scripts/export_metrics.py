from __future__ import annotations

import time

import torch
from tqdm import tqdm

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import move_batch_to_device, write_csv
from utils.metrics import count_parameters


@torch.no_grad()
def estimate_fps(model, dataloader, device, modality_set, max_batches):
    model.eval()
    total_samples = 0
    start = time.time()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="fps")):
        if batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        output = model(batch["inputs"], modality_set)
        total_samples += output["pose"].size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.time() - start, 1e-6)
    return total_samples / elapsed


def main() -> None:
    parser = base_parser("Export model complexity and runtime metrics.")
    parser.add_argument("--modality-set", type=str, default="vk,depth,lidar,mmwave,wifi-csi")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/model_complexity.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    modality_set = [item.strip() for item in args.modality_set.split(",") if item.strip()]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fps = estimate_fps(model, val_loader, device, modality_set, max_batches=int(config.get("max_eval_batches", 20)))
    peak_memory = torch.cuda.max_memory_allocated() if device.type == "cuda" else "NA"
    rows = [{
        "method": config.get("method", "VK-RCD"),
        "modality_set": "+".join(modality_set),
        "params": count_parameters(model),
        "fps": fps,
        "peak_memory": peak_memory,
    }]
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()

