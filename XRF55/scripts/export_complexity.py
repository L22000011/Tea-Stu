from __future__ import annotations

import argparse
import time

import torch

from _shared import build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from utils.metrics import count_parameters
from utils.modality import canonicalize_modalities
from utils.reporting import write_csv, write_markdown_table


def main() -> None:
    parser = argparse.ArgumentParser("Export XRF55 model complexity metrics.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--modalities", type=str, default="wifi+rfid+mmwave")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    config = prepare_config(args)
    device = get_device(args.device)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    _, val_loader = build_dataloaders(args.dataset, config)
    batch = next(iter(val_loader))
    inputs = {name: tensor.to(device) for name, tensor in batch["inputs"].items()}
    selected = canonicalize_modalities(args.modalities.split("+"))

    model.eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(inputs, selected)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.repeats):
            _ = model(inputs, selected)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    batch_size = next(iter(inputs.values())).size(0)
    fps = (batch_size * args.repeats) / max(elapsed, 1e-8)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    row = {
        "method": config.get("method", "XRF-RCD"),
        "scene": config.get("scene", "dml"),
        "modality_set": "+".join(selected),
        "params": count_parameters(model),
        "fps": fps,
        "peak_memory_mb": peak_memory,
        "batch_size": batch_size,
        "repeats": args.repeats,
    }
    write_csv(args.output_csv, [row])
    write_markdown_table(args.output_md, [row], "XRF55 complexity export")


if __name__ == "__main__":
    main()
