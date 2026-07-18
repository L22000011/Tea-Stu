from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn
from tqdm import tqdm

from losses import compute_har_student_loss, compute_hpe_student_loss
from utils.har_metrics import AverageMeter as HARAverageMeter
from utils.hpe_metrics import AverageMeter
from utils.modality import sample_missing_modalities
from utils.reporting import append_csv_row, ensure_output_dir, plot_history, save_json
from .engine_common import (
    evaluate_har_model,
    evaluate_hpe_model,
    load_model_checkpoint,
    load_training_checkpoint,
    move_batch_to_device,
    prepare_experiment,
    save_checkpoint,
    save_final_har_eval,
    save_final_hpe_eval,
)


def train_hpe_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Iterable[Dict[str, Any]],
    val_loader: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    _train_student("hpe", student, teacher, train_loader, val_loader, config, device)


def train_har_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Iterable[Dict[str, Any]],
    val_loader: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    _train_student("har", student, teacher, train_loader, val_loader, config, device)


def _train_student(
    task: str,
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Iterable[Dict[str, Any]],
    val_loader: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    output_dir = ensure_output_dir(config.get("output_dir", f"outputs/super_{task}_student"))
    epochs = int(config.get("training_epochs", 100 if task == "hpe" else 50))
    lr = float(config.get("learning_rate", 1e-4))
    max_train_batches = config.get("max_train_batches")
    eval_every = int(config.get("eval_every", 5))
    save_every = int(config.get("save_every", 5))
    student_modalities = config.get("student_modalities", config.get("modality", []))
    teacher_modalities = config.get("teacher_modalities", config.get("modality", []))
    eval_modalities = config.get("eval_modalities", student_modalities)
    drop_counts = config.get("drop_counts", [1, 2, 3])
    weights = config.get("loss", {})
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
    params = prepare_experiment(output_dir, config, student, device, f"{task}_student")
    rng = random.Random(int(config.get("init_rand_seed", 0)))
    start_epoch = 0
    best = {"metric": float("inf") if task == "hpe" else 0.0}
    if config.get("resume"):
        start_epoch, loaded = load_training_checkpoint(student, optimizer, config["resume"], device, strict=False)
        best.update(loaded)
    teacher.eval()
    history_rows = []
    current_epoch = start_epoch - 1
    meter_cls = AverageMeter if task == "hpe" else HARAverageMeter
    loss_fn = compute_hpe_student_loss if task == "hpe" else compute_har_student_loss
    eval_fn = evaluate_hpe_model if task == "hpe" else evaluate_har_model
    save_final = save_final_hpe_eval if task == "hpe" else save_final_har_eval
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            student.train()
            total = meter_cls()
            print(
                f"[{config.get('method', task + '_student')}] epoch {epoch + 1}/{epochs} | "
                f"params={params:,} | student_modalities={'+'.join(student_modalities)} | drop_counts={drop_counts}"
            )
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"{task}-student:{epoch}")):
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break
                batch = move_batch_to_device(batch, device, task)
                selected = sample_missing_modalities(student_modalities, drop_counts, rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_output = teacher(batch["inputs"], teacher_modalities, task=task)
                student_output = student(batch["inputs"], selected)
                loss, _ = loss_fn(student_output, batch["target"], weights, teacher_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.get("grad_clip", 1.0)))
                optimizer.step()
                total.update(float(loss.detach().cpu()), batch["target"].size(0))

            val_metrics = None
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_metrics = eval_fn(student, val_loader, device, eval_modalities, config.get("max_eval_batches"))
                metric = float(val_metrics["mpjpe"] if task == "hpe" else val_metrics["acc"])
                improved = metric < best["metric"] if task == "hpe" else metric >= best["metric"]
                if improved:
                    best["metric"] = metric
                    save_checkpoint(output_dir / "best.pth", student, optimizer, epoch, best, config)
                    print(f"Saved new best student: {output_dir / 'best.pth'} | metric={metric:.6f}")

            save_checkpoint(output_dir / "last.pth", student, optimizer, epoch, best, config)
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", student, optimizer, epoch, best, config)
            row = {
                "epoch": epoch + 1,
                "train_loss": total.avg,
                "val_metric": val_metrics["mpjpe"] if task == "hpe" and val_metrics else (val_metrics["acc"] if val_metrics else "NA"),
                "best_metric": best["metric"],
                "lr": lr,
                "params": params,
                "student_modalities": "+".join(student_modalities),
                "drop_counts": "/".join(str(item) for item in drop_counts),
            }
            history_rows.append(row)
            append_csv_row(output_dir / "epoch_history.csv", row)
            print(f"Epoch {epoch + 1}/{epochs} done | train_loss={total.avg:.6f} | best_metric={best['metric']:.6f}")
    except KeyboardInterrupt:
        save_checkpoint(output_dir / "interrupted.pth", student, optimizer, current_epoch, best, config)
        print(f"Training interrupted. Saved checkpoint to {output_dir / 'interrupted.pth'}")
        raise
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_checkpoint(output_dir / "failed.pth", student, optimizer, current_epoch, best, config)
        print(f"Training failed with {type(exc).__name__}. Saved checkpoint to {output_dir / 'failed.pth'}")
        raise

    best_path = output_dir / "best.pth"
    if best_path.exists():
        load_model_checkpoint(student, best_path, device, strict=False)
    final = save_final(student, val_loader, device, config, output_dir, eval_modalities)
    save_json(output_dir / "final_summary.json", {"best": best, "final": final})
    plot_history(output_dir / "training_curve.png", history_rows)
