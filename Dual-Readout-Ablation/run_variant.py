#!/usr/bin/env python3
"""Train or evaluate one isolated dual-readout ablation variant."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from readout_modes import VALID_MODES, apply_readout_mode


def project_context(project_root: Path) -> None:
    scripts = project_root / "scripts"
    for value in (str(scripts), str(project_root)):
        if value not in sys.path:
            sys.path.insert(0, value)
    os.chdir(project_root)


def normalize_cuda_device(requested: str) -> str:
    if not requested.startswith("cuda"):
        return requested
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError(
            "CUDA was requested but this process sees no GPU. Check CUDA_VISIBLE_DEVICES and nvidia-smi."
        )
    index = int(requested.split(":", 1)[1]) if ":" in requested else 0
    if index < count:
        return requested
    if count == 1:
        print(
            f"[Device] requested={requested}, visible_cuda_devices=1; using cuda:0. "
            "CUDA_VISIBLE_DEVICES remaps the selected physical GPU to process-local cuda:0.",
            flush=True,
        )
        return "cuda:0"
    raise RuntimeError(
        f"Invalid CUDA device {requested}: this process sees {count} devices (valid indices 0..{count - 1})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=["HPE", "HAR"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--stage", choices=["train", "eval"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    args = parser.parse_args()

    args.device = normalize_cuda_device(args.device)

    project_root = args.project_root.resolve()
    project_context(project_root)
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
    config["method"] = f"VK-RMD-{args.project}-Readout-{args.mode}"
    config["readout_ablation"] = args.mode
    config["student_modalities"] = list(modality.ALL_MODALITIES)
    config["teacher_modalities"] = list(modality.ALL_MODALITIES)
    config["eval_modalities"] = list(modality.ALL_MODALITIES)
    device = shared.get_device(args.device)
    _, val_loader = shared.build_dataloaders(args.dataset, config)

    if args.stage == "train":
        train_loader, val_loader = shared.build_dataloaders(args.dataset, config)
        student = models.build_model(config).to(device)
        apply_readout_mode(student, args.mode)
        teacher = models.build_model(config).to(device)
        engine.load_model_checkpoint(teacher, args.teacher, device, strict=False)
        engine.train_student(student, teacher, train_loader, val_loader, config, device)
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation")
    model = models.build_model(config).to(device)
    apply_readout_mode(model, args.mode)
    engine.load_model_checkpoint(model, args.checkpoint, device, strict=False)
    rows = engine.evaluate_combinations(
        model,
        val_loader,
        device,
        combinations=modality.all_nonempty_combinations(modality.ALL_MODALITIES),
        max_batches=args.max_eval_batches,
    )
    for row in rows:
        row["method"] = config["method"]
        row["split"] = config.get("split_to_use", "unknown")
        row["protocol"] = config.get("protocol", "protocol3")
        row["readout_mode"] = args.mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "all_combinations.csv"
    engine.write_csv(output_csv, rows)
    (args.output_dir / "eval_manifest.json").write_text(
        json.dumps(
            {
                "project": args.project,
                "mode": args.mode,
                "checkpoint": str(args.checkpoint.resolve()),
                "max_eval_batches": args.max_eval_batches,
                "combination_count": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ReadoutAblation] Saved {len(rows)} combinations to {output_csv}")


if __name__ == "__main__":
    main()
