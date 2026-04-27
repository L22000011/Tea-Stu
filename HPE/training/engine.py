from __future__ import annotations

import csv
import platform
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from tqdm import tqdm

from losses import compute_student_loss, compute_teacher_loss
from utils.metrics import AverageMeter, compute_metrics, count_parameters
from utils.modality import ALL_MODALITIES, all_nonempty_combinations, sample_missing_modalities
from utils.reporting import (
    append_csv_row,
    ensure_output_dir,
    plot_history,
    save_dependencies,
    save_json,
    save_yaml,
    write_markdown_table,
)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "inputs": {name: tensor.to(device, non_blocking=True) for name, tensor in batch["inputs"].items()},
        "target": batch["target"].to(device, non_blocking=True),
        "meta": batch.get("meta", []),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mpjpe": best_metric,
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
) -> tuple[int, float]:
    checkpoint = load_model_checkpoint(model, path, device, strict=strict)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_mpjpe = float(checkpoint.get("best_mpjpe", float("inf")))
    print(f"Resumed from {path} at epoch {start_epoch}; best_mpjpe={best_mpjpe:.6f}")
    return start_epoch, best_mpjpe


def prepare_experiment(
    output_dir: Path,
    config: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
    stage: str,
) -> int:
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


def make_result_row(
    config: Dict[str, Any],
    modality_set: Sequence[str],
    metrics: Dict[str, float],
    params: int,
) -> Dict[str, Any]:
    return {
        "method": config.get("method", "VK-RCD"),
        "split": config.get("split_to_use", "unknown"),
        "protocol": config.get("protocol", "protocol3"),
        "modality_set": "+".join(modality_set),
        "mse": metrics["mse"],
        "mpjpe": metrics["mpjpe"],
        "pa_mpjpe": metrics["pa_mpjpe"],
        "params": params,
        "fps": "NA",
        "peak_memory": "NA",
    }


def save_final_eval(
    model: nn.Module,
    val_loader: Iterable[Dict[str, Any]],
    device: torch.device,
    config: Dict[str, Any],
    output_dir: Path,
    modality_set: Sequence[str],
    params: int,
) -> Dict[str, float]:
    metrics = evaluate_model(model, val_loader, device, modality_set, max_batches=config.get("max_eval_batches"))
    row = make_result_row(config, modality_set, metrics, params)
    write_csv(output_dir / "final_eval.csv", [row])
    write_markdown_table(output_dir / "final_eval.md", [row], f"{config.get('method', 'VK-RCD')} final evaluation")
    print(
        f"Final eval | modality={'+'.join(modality_set)} | "
        f"MSE={metrics['mse']:.6f} | MPJPE={metrics['mpjpe']:.6f} | PA-MPJPE={metrics['pa_mpjpe']:.6f}"
    )
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    modality_set: Sequence[str],
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    meters = {name: AverageMeter() for name in ["mse", "mpjpe", "pa_mpjpe"]}
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"eval:{'+'.join(modality_set)}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        output = model(batch["inputs"], modality_set)
        metrics = compute_metrics(output["pose"], batch["target"])
        batch_size = batch["target"].size(0)
        for key, value in metrics.items():
            meters[key].update(value, batch_size)
    return {key: meter.avg for key, meter in meters.items()}


def evaluate_combinations(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    combinations: List[List[str]] | None = None,
    max_batches: int | None = None,
) -> List[Dict[str, Any]]:
    combos = combinations or all_nonempty_combinations(ALL_MODALITIES)
    rows = []
    params = count_parameters(model)
    for combo in combos:
        metrics = evaluate_model(model, dataloader, device, combo, max_batches=max_batches)
        rows.append({
            "modality_set": "+".join(combo),
            "mse": metrics["mse"],
            "mpjpe": metrics["mpjpe"],
            "pa_mpjpe": metrics["pa_mpjpe"],
            "params": params,
            "fps": "NA",
            "peak_memory": "NA",
        })
    return rows


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if {"modality_set", "mpjpe", "pa_mpjpe"}.issubset(rows[0].keys()):
        try:
            from utils.xfi_report import DEFAULT_XFI_TABLE, generate_xfi_hpe_report

            if Path(DEFAULT_XFI_TABLE).exists():
                report = generate_xfi_hpe_report(path)
                print(f"X-Fi comparison TeX: {report['tex']}")
                print(f"X-Fi comparison PDF: {report['pdf']}")
                print(
                    "X-Fi comparison summary | "
                    f"MPJPE improved {report['mpjpe_improved']}/{report['compared']} | "
                    f"PA-MPJPE improved {report['pa_mpjpe_improved']}/{report['compared']} | "
                    f"avg delta MPJPE {report['avg_delta_mpjpe_mm']} mm | "
                    f"avg delta PA-MPJPE {report['avg_delta_pa_mpjpe_mm']} mm"
                )
        except Exception as exc:
            print(f"Skipped X-Fi comparison report for {path}: {exc}")


def train_teacher(
    model: nn.Module,
    train_loader: Iterable[Dict[str, Any]],
    val_loader: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    output_dir = ensure_output_dir(config.get("output_dir", "outputs/teacher_full"))
    epochs = int(config.get("training_epochs", 100))
    lr = float(config.get("learning_rate", 1e-4))
    max_train_batches = config.get("max_train_batches")
    eval_every = int(config.get("eval_every", 10))
    save_every = int(config.get("save_every", 5))
    modalities = config.get("train_modalities", ALL_MODALITIES)
    loss_weights = config.get("loss", {})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    params = prepare_experiment(output_dir, config, model, device, "teacher")
    start_epoch = 0
    best_mpjpe = float("inf")
    resume_path = config.get("resume")
    if resume_path:
        start_epoch, best_mpjpe = load_training_checkpoint(model, optimizer, resume_path, device, strict=False)

    history_rows: List[Dict[str, Any]] = []
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            model.train()
            total = AverageMeter()
            print(f"[{config.get('method', 'teacher')}] epoch {epoch + 1}/{epochs} | params={params:,} | modalities={'+'.join(modalities)}")
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"teacher:{epoch}")):
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["inputs"], modalities)
                loss, _ = compute_teacher_loss(output, batch["target"], loss_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
                optimizer.step()
                total.update(float(loss.detach().cpu()), batch["target"].size(0))

            val_metrics: Dict[str, float] | None = None
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_metrics = evaluate_model(model, val_loader, device, modalities, max_batches=config.get("max_eval_batches"))
                if val_metrics["mpjpe"] < best_mpjpe:
                    best_mpjpe = val_metrics["mpjpe"]
                    save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, best_mpjpe, config)
                    print(f"Saved new best model: {output_dir / 'best.pth'} | best_mpjpe={best_mpjpe:.6f}")

            save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, best_mpjpe, config)
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", model, optimizer, epoch, best_mpjpe, config)

            row = {
                "epoch": epoch + 1,
                "train_loss": total.avg,
                "val_mse": val_metrics["mse"] if val_metrics else "NA",
                "val_mpjpe": val_metrics["mpjpe"] if val_metrics else "NA",
                "val_pa_mpjpe": val_metrics["pa_mpjpe"] if val_metrics else "NA",
                "best_mpjpe": best_mpjpe,
                "lr": lr,
                "params": params,
                "modalities": "+".join(modalities),
            }
            history_rows.append(row)
            append_csv_row(output_dir / "epoch_history.csv", row)
            print(
                f"Epoch {epoch + 1}/{epochs} done | train_loss={total.avg:.6f} | "
                f"val_mpjpe={row['val_mpjpe']} | best_mpjpe={best_mpjpe:.6f}"
            )
    except KeyboardInterrupt:
        save_checkpoint(output_dir / "interrupted.pth", model, optimizer, current_epoch, best_mpjpe, config)
        print(f"Training interrupted. Saved checkpoint to {output_dir / 'interrupted.pth'}")
        raise

    best_path = output_dir / "best.pth"
    if best_path.exists():
        load_model_checkpoint(model, best_path, device, strict=False)
    final_metrics = save_final_eval(model, val_loader, device, config, output_dir, modalities, params)
    save_json(output_dir / "final_summary.json", {"best_mpjpe": best_mpjpe, "final_metrics": final_metrics})
    plot_history(output_dir / "training_curve.png", history_rows)


def train_student(
    student: nn.Module,
    teacher: nn.Module | None,
    train_loader: Iterable[Dict[str, Any]],
    val_loader: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    output_dir = ensure_output_dir(config.get("output_dir", "outputs/student"))
    epochs = int(config.get("training_epochs", 100))
    lr = float(config.get("learning_rate", 1e-4))
    max_train_batches = config.get("max_train_batches")
    eval_every = int(config.get("eval_every", 10))
    save_every = int(config.get("save_every", 5))
    available_modalities = config.get("student_modalities", ALL_MODALITIES)
    teacher_modalities = config.get("teacher_modalities", ALL_MODALITIES)
    drop_counts = config.get("drop_counts", [1, 2, 3])
    loss_weights = config.get("loss", {})
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(int(config.get("init_rand_seed", 0)))
    params = prepare_experiment(output_dir, config, student, device, "student")
    start_epoch = 0
    best_mpjpe = float("inf")
    resume_path = config.get("resume")
    if resume_path:
        start_epoch, best_mpjpe = load_training_checkpoint(student, optimizer, resume_path, device, strict=False)

    if teacher is not None:
        teacher.eval()

    history_rows: List[Dict[str, Any]] = []
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            student.train()
            total = AverageMeter()
            print(
                f"[{config.get('method', 'student')}] epoch {epoch + 1}/{epochs} | params={params:,} | "
                f"student_modalities={'+'.join(available_modalities)} | drop_counts={drop_counts}"
            )
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"student:{epoch}")):
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break
                batch = move_batch_to_device(batch, device)
                selected = sample_missing_modalities(available_modalities, drop_counts, rng)
                optimizer.zero_grad(set_to_none=True)

                teacher_output = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_output = teacher(batch["inputs"], teacher_modalities)

                student_output = student(batch["inputs"], selected)
                loss, _ = compute_student_loss(student_output, batch["target"], loss_weights, teacher_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.get("grad_clip", 1.0)))
                optimizer.step()
                total.update(float(loss.detach().cpu()), batch["target"].size(0))

            val_metrics: Dict[str, float] | None = None
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                eval_modalities = config.get("eval_modalities", available_modalities)
                val_metrics = evaluate_model(student, val_loader, device, eval_modalities, max_batches=config.get("max_eval_batches"))
                if val_metrics["mpjpe"] < best_mpjpe:
                    best_mpjpe = val_metrics["mpjpe"]
                    save_checkpoint(output_dir / "best.pth", student, optimizer, epoch, best_mpjpe, config)
                    print(f"Saved new best model: {output_dir / 'best.pth'} | best_mpjpe={best_mpjpe:.6f}")

            save_checkpoint(output_dir / "last.pth", student, optimizer, epoch, best_mpjpe, config)
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", student, optimizer, epoch, best_mpjpe, config)

            row = {
                "epoch": epoch + 1,
                "train_loss": total.avg,
                "val_mse": val_metrics["mse"] if val_metrics else "NA",
                "val_mpjpe": val_metrics["mpjpe"] if val_metrics else "NA",
                "val_pa_mpjpe": val_metrics["pa_mpjpe"] if val_metrics else "NA",
                "best_mpjpe": best_mpjpe,
                "lr": lr,
                "params": params,
                "student_modalities": "+".join(available_modalities),
                "drop_counts": "/".join(str(item) for item in drop_counts),
            }
            history_rows.append(row)
            append_csv_row(output_dir / "epoch_history.csv", row)
            print(
                f"Epoch {epoch + 1}/{epochs} done | train_loss={total.avg:.6f} | "
                f"val_mpjpe={row['val_mpjpe']} | best_mpjpe={best_mpjpe:.6f}"
            )
    except KeyboardInterrupt:
        save_checkpoint(output_dir / "interrupted.pth", student, optimizer, current_epoch, best_mpjpe, config)
        print(f"Training interrupted. Saved checkpoint to {output_dir / 'interrupted.pth'}")
        raise

    best_path = output_dir / "best.pth"
    if best_path.exists():
        load_model_checkpoint(student, best_path, device, strict=False)
    final_modalities = config.get("eval_modalities", available_modalities)
    final_metrics = save_final_eval(student, val_loader, device, config, output_dir, final_modalities, params)
    save_json(output_dir / "final_summary.json", {"best_mpjpe": best_mpjpe, "final_metrics": final_metrics})
    plot_history(output_dir / "training_curve.png", history_rows)
