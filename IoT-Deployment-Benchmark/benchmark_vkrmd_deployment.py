from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


DEFAULT_COMBOS = {
    "HPE": [
        "vk",
        "depth",
        "mmwave",
        "vk+depth",
        "depth+mmwave",
        "vk+depth+lidar+mmwave+wifi-csi",
    ],
    "HAR": [
        "vk",
        "depth",
        "mmwave",
        "vk+depth",
        "depth+mmwave",
        "vk+depth+lidar+mmwave",
    ],
}


def parse_combo(text: str) -> list[str]:
    aliases = {
        "v": "vk",
        "vk": "vk",
        "d": "depth",
        "depth": "depth",
        "l": "lidar",
        "lidar": "lidar",
        "r": "mmwave",
        "mmwave": "mmwave",
        "w": "wifi-csi",
        "wifi": "wifi-csi",
        "wifi-csi": "wifi-csi",
    }
    parts = [item.strip().lower() for item in text.replace(",", "+").split("+") if item.strip()]
    parsed = []
    for part in parts:
        if part not in aliases:
            raise ValueError(f"Unknown modality alias in combo '{text}': {part}")
        parsed.append(aliases[part])
    return parsed


def combo_label(modalities: list[str]) -> str:
    labels = {"vk": "V", "depth": "D", "lidar": "L", "mmwave": "R", "wifi-csi": "W"}
    return "+".join(labels[name] for name in modalities)


def import_project(project_root: Path):
    scripts_dir = project_root / "scripts"
    for path in [str(scripts_dir), str(project_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    shared = importlib.import_module("_shared")
    engine = importlib.import_module("training.engine")
    return shared, engine


def load_one_batch(shared, engine, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    parser = argparse.Namespace(
        dataset=str(args.dataset),
        config=str(args.config),
        output_dir=None,
        checkpoint=str(args.checkpoint),
        resume=None,
        backbone_root=args.backbone_root,
        max_train_batches=None,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
    )
    config = shared.prepare_config(parser)
    if args.batch_size is not None:
        config.setdefault("loader", {})
        config["loader"]["batch_size"] = int(args.batch_size)
    _, val_loader = shared.build_dataloaders(str(args.dataset), config)
    model = shared.build_model_from_config(config, device)
    shared.load_optional_checkpoint(model, str(args.checkpoint), device)
    model.eval()
    batch = next(iter(val_loader))
    batch = engine.move_batch_to_device(batch, device)
    return {"model": model, "batch": batch, "config": config}


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 ** 2)


def cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)


def run_once(model: torch.nn.Module, batch: dict[str, Any], modalities: list[str]) -> None:
    _ = model(batch["inputs"], modalities)


def benchmark_combo(
    model: torch.nn.Module,
    batch: dict[str, Any],
    modalities: list[str],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            run_once(model, batch, modalities)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in tqdm(range(iters), desc=f"bench:{combo_label(modalities)}"):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_once(model, batch, modalities)
                end.record()
                torch.cuda.synchronize(device)
                timings.append(float(start.elapsed_time(end)))
            peak_memory = cuda_memory_mb(device)
        else:
            timings = []
            for _ in tqdm(range(iters), desc=f"bench:{combo_label(modalities)}"):
                start_time = time.perf_counter()
                run_once(model, batch, modalities)
                timings.append((time.perf_counter() - start_time) * 1000.0)
            peak_memory = 0.0
    mean_ms = statistics.mean(timings)
    std_ms = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    sorted_timings = sorted(timings)
    p50_ms = sorted_timings[len(sorted_timings) // 2]
    p95_ms = sorted_timings[min(len(sorted_timings) - 1, int(math.ceil(0.95 * len(sorted_timings))) - 1)]
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "latency_ms_p50": p50_ms,
        "latency_ms_p95": p95_ms,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "peak_memory_mb": peak_memory,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "model_name",
        "checkpoint",
        "modalities",
        "num_modalities",
        "batch_size",
        "params",
        "model_size_mb",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_per_sample",
        "fps",
        "throughput_samples_per_sec",
        "peak_memory_mb",
        "device",
        "gpu_name",
        "warmup",
        "iters",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark VK-RMD deployment cost under missing-modality inference.")
    parser.add_argument("--project", choices=["HPE", "HAR"], required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-root", default=None)
    parser.add_argument("--model-name", default="VK-RMD Student-VK")
    parser.add_argument("--combos", nargs="*", default=None, help="Examples: vk depth vk+depth depth+mmwave")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader batch size; use 1 for single-stream deployment latency.")
    parser.add_argument("--max-eval-batches", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=Path("IoT-Deployment-Benchmark/results/deployment_efficiency_l40.csv"))
    parser.add_argument("--manifest-json", type=Path, default=Path("IoT-Deployment-Benchmark/results/deployment_efficiency_l40_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    project_root = args.project_root or (repo / args.project)
    device = torch.device(args.device)
    shared, engine = import_project(project_root)
    bundle = load_one_batch(shared, engine, args, device)
    model = bundle["model"]
    batch = bundle["batch"]
    params = count_params(model)
    size_mb = model_size_mb(model)
    batch_size = next(iter(batch["inputs"].values())).shape[0]
    combos = args.combos or DEFAULT_COMBOS[args.project]
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    rows = []
    print(
        f"[Benchmark] project={args.project} | model={args.model_name} | "
        f"params={params:,} | size={size_mb:.2f} MB | batch_size={batch_size} | "
        f"device={device} | gpu={gpu_name}"
    )
    for combo_text in combos:
        modalities = parse_combo(combo_text)
        missing = [name for name in modalities if name not in batch["inputs"]]
        if missing:
            raise KeyError(f"Batch does not contain requested modalities {missing}; available={list(batch['inputs'])}")
        metrics = benchmark_combo(model, batch, modalities, device, args.warmup, args.iters)
        row = {
            "task": args.project,
            "model_name": args.model_name,
            "checkpoint": str(args.checkpoint),
            "modalities": combo_label(modalities),
            "num_modalities": len(modalities),
            "batch_size": batch_size,
            "params": params,
            "model_size_mb": size_mb,
            **metrics,
            "latency_ms_per_sample": metrics["latency_ms_mean"] / max(batch_size, 1),
            "throughput_samples_per_sec": (1000.0 * batch_size / metrics["latency_ms_mean"]) if metrics["latency_ms_mean"] > 0 else 0.0,
            "device": str(device),
            "gpu_name": gpu_name,
            "warmup": args.warmup,
            "iters": args.iters,
        }
        rows.append(row)
        print(
            f"[Benchmark] {row['modalities']}: "
            f"{row['latency_ms_mean']:.3f}+/-{row['latency_ms_std']:.3f} ms | "
            f"{row['fps']:.2f} FPS | {row['throughput_samples_per_sec']:.2f} samples/s | "
            f"{row['peak_memory_mb']:.1f} MB"
        )
    write_rows(args.output_csv, rows)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(
            {
                "project": args.project,
                "project_root": str(project_root),
                "dataset": str(args.dataset),
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "device": args.device,
                "gpu_name": gpu_name,
                "warmup": args.warmup,
                "iters": args.iters,
                "output_csv": str(args.output_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.manifest_json}")


if __name__ == "__main__":
    main()
