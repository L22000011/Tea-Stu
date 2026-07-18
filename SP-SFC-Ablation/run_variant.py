#!/usr/bin/env python3
"""Run one controlled SP-SFC loss-profile variant."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch

PROFILES = ("task_only", "output", "output_token", "full")


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


def profile_weights(project: str, profile: str) -> dict[str, float]:
    if profile not in PROFILES:
        raise ValueError(profile)
    if project == "HPE":
        values = {
            "lambda_out": 0.0,
            "lambda_token": 0.0,
            "lambda_bone": 0.0,
            "lambda_rel": 0.0,
            "lambda_unc": 0.0,
            "temperature": 4.0,
            "tau": 1.0,
        }
        if profile in ("output", "output_token", "full"):
            values["lambda_out"] = 0.5
        if profile in ("output_token", "full"):
            values["lambda_token"] = 0.2
        if profile == "full":
            values.update({"lambda_bone": 0.1, "lambda_rel": 0.05, "lambda_unc": 0.1})
        return values
    values = {
        "lambda_kd": 0.0,
        "lambda_token": 0.0,
        "lambda_rel": 0.0,
        "lambda_unc": 0.0,
        "temperature": 4.0,
        "tau": 1.0,
    }
    if profile in ("output", "output_token", "full"):
        values["lambda_kd"] = 0.7
    if profile in ("output_token", "full"):
        values["lambda_token"] = 0.1
    if profile == "full":
        values.update({"lambda_rel": 0.03, "lambda_unc": 0.05})
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=["HPE", "HAR"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    shared = importlib.import_module("_shared")
    models = importlib.import_module("models")
    engine = importlib.import_module("training.engine")
    modality = importlib.import_module("utils.modality")
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
    config["training_epochs"] = args.epochs
    config["method"] = f"VK-RMD-{args.project}-SP-SFC-{args.profile}"
    config["student_modalities"] = list(modality.ALL_MODALITIES)
    config["teacher_modalities"] = list(modality.ALL_MODALITIES)
    config["eval_modalities"] = list(modality.ALL_MODALITIES)
    config["eval_missing_during_training"] = True
    config["loss"] = profile_weights(args.project, args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(args.output_dir.resolve())
    (args.output_dir / "profile.json").write_text(
        json.dumps({"project": args.project, "profile": args.profile, "loss": config["loss"]}, indent=2),
        encoding="utf-8",
    )
    device = shared.get_device(args.device)
    train_loader, val_loader = shared.build_dataloaders(args.dataset, config)
    student = models.build_model(config).to(device)
    teacher = models.build_model(config).to(device)
    engine.load_model_checkpoint(teacher, args.teacher, device, strict=False)
    teacher.eval()
    if args.resume and not args.force:
        config["resume"] = str(args.resume.resolve())
    engine.train_student(student, teacher, train_loader, val_loader, config, device)
    checkpoint = args.output_dir / "best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    engine.load_model_checkpoint(student, checkpoint, device, strict=False)
    rows = engine.evaluate_combinations(
        student,
        val_loader,
        device,
        combinations=modality.all_nonempty_combinations(modality.ALL_MODALITIES),
        max_batches=args.max_eval_batches,
    )
    for row in rows:
        row.update({"method": config["method"], "profile": args.profile, "project": args.project})
    engine.write_csv(args.output_dir / "all_combinations.csv", rows)
    (args.output_dir / "eval_manifest.json").write_text(
        json.dumps({"project": args.project, "profile": args.profile, "combination_count": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"[SP-SFC] saved {len(rows)} combinations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
