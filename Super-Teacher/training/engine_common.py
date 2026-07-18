from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from tqdm import tqdm

from utils.har_metrics import AverageMeter as HARAverageMeter
from utils.har_metrics import classification_metrics
from utils.hpe_metrics import AverageMeter, compute_metrics, count_parameters
from utils.modality import all_nonempty_combinations
from utils.reporting import ensure_output_dir, plot_history, save_dependencies, save_json, save_yaml, write_markdown_table


def move_batch_to_device(batch: Dict[str, Any], device: torch.device, task: str) -> Dict[str, Any]:
    target = batch["target"].to(device, non_blocking=True)
    if task == "har":
        target = target.long()
    else:
        target = target.float()
    return {
        "inputs": {name: tensor.to(device, non_blocking=True) for name, tensor in batch["inputs"].items()},
        "target": target,
        "meta": batch.get("meta", []),
    }


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_model_checkpoint(model: nn.Module, path: str | Path, device: torch.device, strict: bool = False) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint


def load_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    device: torch.device,
    strict: bool = False,
) -> tuple[int, Dict[str, float]]:
    checkpoint = load_model_checkpoint(model, path, device, strict=strict)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    metrics = dict(checkpoint.get("metrics", {}))
    print(f"Resumed from {path} at epoch {start_epoch}; metrics={metrics}")
    return start_epoch, metrics


def prepare_experiment(output_dir: Path, config: Dict[str, Any], model: nn.Module, device: torch.device, stage: str) -> int:
    ensure_output_dir(output_dir)
    params = count_parameters(model)
    save_yaml(output_dir / "config_used.yaml", config)
    save_dependencies(output_dir / "dependencies.txt")
    save_json(
        output_dir / "experiment_info.json",
        {
            "method": config.get("method", stage),
            "stage": stage,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "parameter_count": params,
            "output_dir": str(output_dir),
            "dataset_root": config.get("dataset_root", "NA"),
            "split": config.get("split_to_use", "unknown"),
            "protocol": config.get("protocol", "protocol3"),
        },
    )
    print("=" * 80)
    print(f"Experiment: {config.get('method', stage)}")
    print(f"Stage: {stage}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Parameters: {params:,}")
    print(f"Split: {config.get('split_to_use', 'unknown')} | Protocol: {config.get('protocol', 'protocol3')}")
    print("=" * 80)
    return params


@torch.no_grad()
def evaluate_hpe_model(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    modality_set: Sequence[str],
    max_batches: int | None = None,
    task: str = "hpe",
) -> Dict[str, float]:
    model.eval()
    meters = {name: AverageMeter() for name in ["mse", "mpjpe", "pa_mpjpe"]}
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"eval:hpe:{'+'.join(modality_set)}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device, "hpe")
        try:
            output = model(batch["inputs"], modality_set, task=task)
        except TypeError:
            output = model(batch["inputs"], modality_set)
        metrics = compute_metrics(output["pose"], batch["target"])
        batch_size = batch["target"].size(0)
        for key, value in metrics.items():
            meters[key].update(value, batch_size)
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate_har_model(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    modality_set: Sequence[str],
    max_batches: int | None = None,
    task: str = "har",
) -> Dict[str, float]:
    model.eval()
    loss_meter = HARAverageMeter()
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    num_classes = int(getattr(model, "num_classes", 27))
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"eval:har:{'+'.join(modality_set)}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device, "har")
        try:
            output = model(batch["inputs"], modality_set, task=task)
        except TypeError:
            output = model(batch["inputs"], modality_set)
        logits = output["logits"]
        loss = torch.nn.functional.cross_entropy(logits, batch["target"])
        pred = torch.argmax(logits, dim=1)
        batch_size = batch["target"].size(0)
        correct += (pred == batch["target"]).sum().item()
        total += batch_size
        loss_meter.update(float(loss.detach().cpu()), batch_size)
        all_logits.append(logits.detach().cpu())
        all_targets.append(batch["target"].detach().cpu())
    if total == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}
    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_targets), num_classes)
    return {"loss": loss_meter.avg, "acc": correct / total, "macro_f1": metrics["macro_f1"]}


def evaluate_hpe_combinations(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    combinations: List[List[str]],
    max_batches: int | None = None,
    task: str = "hpe",
) -> List[Dict[str, Any]]:
    rows = []
    params = count_parameters(model)
    for combo in combinations:
        metrics = evaluate_hpe_model(model, dataloader, device, combo, max_batches=max_batches, task=task)
        rows.append({"modality_set": "+".join(combo), **metrics, "params": params})
    return rows


def evaluate_har_combinations(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    combinations: List[List[str]],
    max_batches: int | None = None,
    task: str = "har",
) -> List[Dict[str, Any]]:
    rows = []
    params = count_parameters(model)
    for combo in combinations:
        metrics = evaluate_har_model(model, dataloader, device, combo, max_batches=max_batches, task=task)
        rows.append({"modality_set": "+".join(combo), **metrics, "params": params})
    return rows


def save_final_hpe_eval(model: nn.Module, val_loader: Iterable[Dict[str, Any]], device: torch.device, config: Dict[str, Any], output_dir: Path, modalities: Sequence[str], task: str = "hpe") -> Dict[str, float]:
    metrics = evaluate_hpe_model(model, val_loader, device, modalities, config.get("max_eval_batches"), task=task)
    row = {"method": config.get("method", "Super-HPE"), "split": config.get("split_to_use", "unknown"), "modality_set": "+".join(modalities), **metrics, "params": count_parameters(model)}
    write_csv(output_dir / "final_eval_hpe.csv", [row])
    write_markdown_table(output_dir / "final_eval_hpe.md", [row], "Final HPE evaluation")
    return metrics


def save_final_har_eval(model: nn.Module, val_loader: Iterable[Dict[str, Any]], device: torch.device, config: Dict[str, Any], output_dir: Path, modalities: Sequence[str], task: str = "har") -> Dict[str, float]:
    metrics = evaluate_har_model(model, val_loader, device, modalities, config.get("max_eval_batches"), task=task)
    row = {"method": config.get("method", "Super-HAR"), "split": config.get("split_to_use", "unknown"), "modality_set": "+".join(modalities), **metrics, "params": count_parameters(model)}
    write_csv(output_dir / "final_eval_har.csv", [row])
    write_markdown_table(output_dir / "final_eval_har.md", [row], "Final HAR evaluation")
    return metrics


def combinations_for(modalities: Sequence[str]) -> List[List[str]]:
    return all_nonempty_combinations(list(modalities))
