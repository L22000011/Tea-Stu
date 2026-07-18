#!/usr/bin/env python3
"""Train or evaluate one Adapted X-Fi-VK task/protocol variant."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent


def normalize_device(name: str) -> str:
    if not name.startswith("cuda"):
        return name
    visible = torch.cuda.device_count()
    if visible == 0:
        raise RuntimeError("CUDA requested but no visible CUDA device exists")
    index = int(name.split(":", 1)[1]) if ":" in name else 0
    if index >= visible:
        if visible == 1:
            return "cuda:0"
        raise RuntimeError(f"Invalid process-local CUDA device {name}; visible count={visible}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=["HPE", "HAR"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.device = normalize_device(args.device)

    project_root = args.project_root.resolve()
    os.chdir(project_root)
    for value in (str(project_root / "scripts"), str(project_root)):
        if value not in sys.path:
            sys.path.insert(0, value)
    sys.path.insert(0, str(THIS_DIR))
    shared = importlib.import_module("_shared")
    engine = importlib.import_module("training.engine")
    modality = importlib.import_module("utils.modality")
    from model import build_model

    namespace = argparse.Namespace(
        dataset=args.dataset,
        config=args.config,
        output_dir=str(args.output_dir.resolve()),
        checkpoint=None,
        resume=str(args.resume.resolve()) if args.resume else None,
        backbone_root=None,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
    )
    config = shared.prepare_config(namespace)
    config["split_to_use"] = args.protocol
    config["training_epochs"] = args.epochs
    config["method"] = f"Adapted-XFi-VK-{args.project}-{args.protocol}"
    config["student_modalities"] = list(modality.ALL_MODALITIES)
    config["teacher_modalities"] = list(modality.ALL_MODALITIES)
    config["eval_modalities"] = list(modality.ALL_MODALITIES)
    config["eval_missing_during_training"] = True
    config["loss"] = {key: 0.0 for key in (
        "lambda_out", "lambda_token", "lambda_bone", "lambda_rel", "lambda_unc",
        "lambda_kd", "temperature", "tau",
    )}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(args.output_dir.resolve())
    device = shared.get_device(args.device)
    train_loader, val_loader = shared.build_dataloaders(args.dataset, config)
    model = build_model(config, args.project).to(device)

    if args.stage in ("train", "all") and (args.force or not (args.output_dir / "final_summary.json").exists()):
        engine.train_student(model, None, train_loader, val_loader, config, device)

    if args.stage in ("eval", "all"):
        checkpoint = args.output_dir / "best.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing baseline checkpoint: {checkpoint}")
        engine.load_model_checkpoint(model, checkpoint, device, strict=False)
        rows = engine.evaluate_combinations(
            model,
            val_loader,
            device,
            combinations=modality.all_nonempty_combinations(modality.ALL_MODALITIES),
            max_batches=args.max_eval_batches,
        )
        for row in rows:
            row.update({"method": config["method"], "split": args.protocol, "protocol": "protocol3"})
        engine.write_csv(args.output_dir / "all_combinations.csv", rows)
        (args.output_dir / "eval_manifest.json").write_text(
            json.dumps({"method": config["method"], "split": args.protocol, "count": len(rows)}, indent=2),
            encoding="utf-8",
        )
        print(f"[Adapted-XFi-VK] saved {len(rows)} combinations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
