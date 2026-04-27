from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_dataloaders
from models import build_model
from training.engine import load_model_checkpoint
from utils.config import load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--backbone-root", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    config["config_path"] = args.config
    config["dataset_root"] = args.dataset
    config["device"] = args.device
    config.setdefault("model", {})
    if args.backbone_root is not None:
        config["model"]["backbone_root"] = args.backbone_root
    else:
        backbone_root = Path(config["model"].get("backbone_root", "backbones"))
        if not backbone_root.is_absolute():
            config["model"]["backbone_root"] = str((PROJECT_ROOT / backbone_root).resolve())
    pretrained_paths = config["model"].get("pretrained_paths", {})
    resolved_paths = {}
    for name, path in pretrained_paths.items():
        weight_path = Path(path)
        resolved_paths[name] = str(weight_path if weight_path.is_absolute() else (PROJECT_ROOT / weight_path).resolve())
    if resolved_paths:
        config["model"]["pretrained_paths"] = resolved_paths
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.resume is not None:
        config["resume"] = args.resume
    if args.max_train_batches is not None:
        config["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        config["max_eval_batches"] = args.max_eval_batches
    return config


def get_device(device_name: str = "cuda:0") -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def build_model_from_config(config: Dict[str, Any], device: torch.device):
    model = build_model(config)
    model.to(device)
    return model


def load_optional_checkpoint(model, path: str | None, device: torch.device) -> None:
    if path:
        load_model_checkpoint(model, path, device, strict=False)

