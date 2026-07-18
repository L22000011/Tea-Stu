from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn
from tqdm import tqdm

from losses import compute_super_teacher_losses
from utils.hpe_metrics import AverageMeter
from utils.modality import HAR_MODALITIES, HPE_MODALITIES
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


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_super_teacher(
    model: nn.Module,
    hpe_train_loader: Iterable[Dict[str, Any]],
    hpe_val_loader: Iterable[Dict[str, Any]],
    har_train_loader: Iterable[Dict[str, Any]],
    har_val_loader: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    output_dir = ensure_output_dir(config.get("output_dir", "outputs/super_teacher"))
    epochs = int(config.get("training_epochs", 100))
    lr = float(config.get("learning_rate", 1e-4))
    max_train_batches = int(config.get("max_train_batches", 1000))
    eval_every = int(config.get("eval_every", 5))
    save_every = int(config.get("save_every", 5))
    hpe_modalities = config.get("hpe_modalities", HPE_MODALITIES)
    har_modalities = config.get("har_modalities", HAR_MODALITIES)
    weights = config.get("loss", {})
    use_hpe = float(weights.get("lambda_hpe", 1.0)) > 0
    use_har = float(weights.get("lambda_har", 1.0)) > 0
    if not use_hpe and not use_har:
        raise ValueError("At least one of lambda_hpe or lambda_har must be greater than zero.")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    params = prepare_experiment(output_dir, config, model, device, "super_teacher")
    start_epoch = 0
    best = {"hpe_mpjpe": float("inf"), "har_acc": 0.0, "joint_score": float("inf")}
    if config.get("resume"):
        start_epoch, loaded = load_training_checkpoint(model, optimizer, config["resume"], device, strict=False)
        best.update(loaded)

    history_rows = []
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            model.train()
            total = AverageMeter()
            hpe_iter = iter(hpe_train_loader) if use_hpe else None
            har_iter = iter(har_train_loader) if use_har else None
            print(
                f"[{config.get('method', 'Super-Teacher')}] epoch {epoch + 1}/{epochs} | "
                f"params={params:,} | hpe={'+'.join(hpe_modalities)} | har={'+'.join(har_modalities)}"
            )
            for step in tqdm(range(max_train_batches), desc=f"super:{epoch}"):
                optimizer.zero_grad(set_to_none=True)
                batch_count = 0
                step_loss = 0.0
                if use_hpe:
                    hpe_batch, hpe_iter = _next_batch(hpe_iter, hpe_train_loader)
                    hpe_batch = move_batch_to_device(hpe_batch, device, "hpe")
                    hpe_output = model(hpe_batch["inputs"], hpe_modalities, task="hpe")
                    hpe_loss, _ = compute_super_teacher_losses(
                        hpe_output,
                        hpe_batch["target"],
                        None,
                        None,
                        weights,
                    )
                    hpe_loss.backward()
                    step_loss += float(hpe_loss.detach().cpu())
                    batch_count += hpe_batch["target"].size(0)
                    del hpe_output, hpe_batch, hpe_loss
                if use_har:
                    har_batch, har_iter = _next_batch(har_iter, har_train_loader)
                    har_batch = move_batch_to_device(har_batch, device, "har")
                    har_output = model(har_batch["inputs"], har_modalities, task="har")
                    har_loss, _ = compute_super_teacher_losses(
                        None,
                        None,
                        har_output,
                        har_batch["target"],
                        weights,
                    )
                    har_loss.backward()
                    step_loss += float(har_loss.detach().cpu())
                    batch_count += har_batch["target"].size(0)
                    del har_output, har_batch, har_loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
                optimizer.step()
                total.update(step_loss, batch_count)

            hpe_metrics = None
            har_metrics = None
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                if use_hpe:
                    hpe_metrics = evaluate_hpe_model(model, hpe_val_loader, device, hpe_modalities, config.get("max_eval_batches"), task="hpe")
                if use_har:
                    har_metrics = evaluate_har_model(model, har_val_loader, device, har_modalities, config.get("max_eval_batches"), task="har")
                joint_score = None
                if hpe_metrics is not None and har_metrics is not None:
                    joint_score = float(hpe_metrics["mpjpe"]) + (1.0 - float(har_metrics["acc"]))
                if hpe_metrics is not None and hpe_metrics["mpjpe"] < best["hpe_mpjpe"]:
                    best["hpe_mpjpe"] = float(hpe_metrics["mpjpe"])
                    save_checkpoint(output_dir / "best_hpe.pth", model, optimizer, epoch, best, config)
                    print(f"Saved best HPE checkpoint: {output_dir / 'best_hpe.pth'}")
                if har_metrics is not None and har_metrics["acc"] >= best["har_acc"]:
                    best["har_acc"] = float(har_metrics["acc"])
                    save_checkpoint(output_dir / "best_har.pth", model, optimizer, epoch, best, config)
                    print(f"Saved best HAR checkpoint: {output_dir / 'best_har.pth'}")
                if joint_score is not None and joint_score < best["joint_score"]:
                    best["joint_score"] = joint_score
                    save_checkpoint(output_dir / "best_joint.pth", model, optimizer, epoch, best, config)
                    print(f"Saved best joint checkpoint: {output_dir / 'best_joint.pth'} | joint_score={joint_score:.6f}")

            save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, best, config)
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", model, optimizer, epoch, best, config)
            row = {
                "epoch": epoch + 1,
                "train_loss": total.avg,
                "val_hpe_mpjpe": hpe_metrics["mpjpe"] if hpe_metrics else "NA",
                "val_hpe_pa_mpjpe": hpe_metrics["pa_mpjpe"] if hpe_metrics else "NA",
                "val_har_acc": har_metrics["acc"] if har_metrics else "NA",
                "val_har_macro_f1": har_metrics["macro_f1"] if har_metrics else "NA",
                "best_hpe_mpjpe": best["hpe_mpjpe"],
                "best_har_acc": best["har_acc"],
                "best_joint_score": best["joint_score"],
                "lr": lr,
                "params": params,
            }
            history_rows.append(row)
            append_csv_row(output_dir / "epoch_history.csv", row)
            print(f"Epoch {epoch + 1}/{epochs} done | train_loss={total.avg:.6f} | best={best}")
    except KeyboardInterrupt:
        save_checkpoint(output_dir / "interrupted.pth", model, optimizer, current_epoch, best, config)
        print(f"Training interrupted. Saved checkpoint to {output_dir / 'interrupted.pth'}")
        raise
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_checkpoint(output_dir / "failed.pth", model, optimizer, current_epoch, best, config)
        print(f"Training failed with {type(exc).__name__}. Saved checkpoint to {output_dir / 'failed.pth'}")
        raise

    best_path = output_dir / "best_joint.pth"
    if use_hpe and not use_har:
        best_path = output_dir / "best_hpe.pth"
    if use_har and not use_hpe:
        best_path = output_dir / "best_har.pth"
    if best_path.exists():
        load_model_checkpoint(model, best_path, device, strict=False)
    final_hpe = save_final_hpe_eval(model, hpe_val_loader, device, config, output_dir, hpe_modalities, task="hpe") if use_hpe else None
    final_har = save_final_har_eval(model, har_val_loader, device, config, output_dir, har_modalities, task="har") if use_har else None
    save_json(output_dir / "final_summary.json", {"best": best, "final_hpe": final_hpe, "final_har": final_har})
    plot_history(output_dir / "training_curve.png", history_rows)
