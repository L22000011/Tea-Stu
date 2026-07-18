from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--backbone-root", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def get_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def _resolve_model_paths(config: Dict[str, Any]) -> None:
    config.setdefault("model", {})
    model_cfg = config["model"]
    backbone_root = Path(model_cfg.get("backbone_root", "backbones"))
    if not backbone_root.is_absolute():
        model_cfg["backbone_root"] = str((PROJECT_ROOT / backbone_root).resolve())
    pretrained = model_cfg.get("pretrained_paths", {})
    resolved = {}
    for name, value in pretrained.items():
        path = Path(value)
        resolved[name] = str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
    if resolved:
        model_cfg["pretrained_paths"] = resolved


def prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    config["config_path"] = args.config
    config["dataset_root"] = args.dataset
    config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.resume is not None:
        config["resume"] = args.resume
    if args.max_train_batches is not None:
        config["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        config["max_eval_batches"] = args.max_eval_batches
    if args.backbone_root is not None:
        config.setdefault("model", {})
        config["model"]["backbone_root"] = args.backbone_root
    _resolve_model_paths(config)
    return config


def task_dataset_config(config: Dict[str, Any], task: str) -> Dict[str, Any]:
    task_cfg = dict(config)
    if task == "hpe":
        task_cfg["modality"] = config.get("hpe_modalities", config.get("modality", []))
    elif task == "har":
        task_cfg["modality"] = config.get("har_modalities", config.get("modality", []))
    else:
        raise ValueError(f"Unknown task: {task}")
    return task_cfg


def variant_config(config: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    for variant in config.get("students", []):
        if variant.get("name") == variant_name:
            merged = dict(config)
            merged.update({key: value for key, value in variant.items() if key != "name"})
            merged["variant"] = variant_name
            merged["model"] = dict(config.get("model", {}))
            merged["model"]["modalities"] = variant.get("student_modalities", config.get("student_modalities", []))
            if "output_dir" in variant:
                merged["output_dir"] = variant["output_dir"]
            return merged
    raise ValueError(f"Unknown variant {variant_name}. Available: {[item.get('name') for item in config.get('students', [])]}")
