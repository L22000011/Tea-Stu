from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rgb_subset_common import TRAIN_SUBJECTS, VAL_SUBJECTS, read_manifest

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


COCO17_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]


def split_rows(rows: list[dict[str, str]], protocol: str, split: str) -> list[dict[str, str]]:
    if protocol == "random":
        rng = np.random.default_rng(0)
        indices = rng.permutation(len(rows))
        cut = int(np.floor(0.8 * len(rows)))
        chosen = set(indices[:cut] if split == "train" else indices[cut:])
        return [row for idx, row in enumerate(rows) if idx in chosen]
    if protocol == "cross_scene":
        if split == "train":
            return [row for row in rows if row["scene"] in {"E01", "E02", "E03"}]
        return [row for row in rows if row["scene"] == "E04"]
    if protocol == "cross_subject":
        subjects = set(TRAIN_SUBJECTS if split == "train" else VAL_SUBJECTS)
        return [row for row in rows if row["subject"] in subjects]
    raise ValueError(f"Unsupported protocol: {protocol}")


def read_depth(path: Path, image_size: int) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required to read depth images.")
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    if data.shape[2] > 3:
        data = data[:, :, :3]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(data, 1))
    hi = float(np.percentile(data, 99))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    data = cv2.resize(data, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return data.transpose(2, 0, 1).astype(np.float32)


def generated_vk_path(output_root: Path, row: dict[str, str]) -> Path:
    return output_root / row["scene"] / row["subject"] / row["action"] / f"frame{int(row['idx']) + 1:03d}.npy"


class DepthVKDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        depth_path = Path(row["depth_path"])
        vk_path = Path(row["vk_path"])
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")
        if not vk_path.exists():
            raise FileNotFoundError(f"Missing target VK frame: {vk_path}")
        return {
            "depth": read_depth(depth_path, self.image_size),
            "vk": np.load(vk_path).astype(np.float32).reshape(17, 2),
            "row": row,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "depth": torch.tensor(np.stack([item["depth"] for item in batch]), dtype=torch.float32),
        "vk": torch.tensor(np.stack([item["vk"] for item in batch]), dtype=torch.float32),
        "rows": [item["row"] for item in batch],
    }


class DepthVKGenerator(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 34),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(depth)).view(-1, 17, 2)


def keypoint_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = torch.linalg.norm(pred - target, dim=-1)
    mse = F.mse_loss(pred, target).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    span = target.amax(dim=1) - target.amin(dim=1)
    diag = torch.linalg.norm(span, dim=-1).clamp_min(1e-6)
    nme = (diff.mean(dim=1) / diag).mean().item()
    pck05 = (diff <= (0.05 * diag[:, None])).float().mean().item()
    pck10 = (diff <= (0.10 * diag[:, None])).float().mean().item()
    return {"mse": mse, "mae": mae, "nme": nme, "pck05": pck05, "pck10": pck10}


def bone_length_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for i, j in COCO17_BONES:
        pred_len = torch.linalg.norm(pred[:, i] - pred[:, j], dim=-1)
        target_len = torch.linalg.norm(target[:, i] - target[:, j], dim=-1)
        losses.append(F.smooth_l1_loss(pred_len, target_len))
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_nme: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_nme": best_nme,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_nme", math.inf))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    totals = {"mse": 0.0, "mae": 0.0, "nme": 0.0, "pck05": 0.0, "pck10": 0.0}
    seen = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="depth-vk:eval", total=total_batches, dynamic_ncols=True)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            depth = batch["depth"].to(device)
            target = batch["vk"].to(device)
            pred = model(depth)
            metrics = keypoint_metrics(pred, target)
            batch_size = target.size(0)
            seen += batch_size
            for key, value in metrics.items():
                totals[key] += value * batch_size
    return {key: value / max(seen, 1) for key, value in totals.items()}


def export_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_root: Path, force: bool) -> int:
    model.eval()
    exported = 0
    output_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc="depth-vk:export", total=len(loader), dynamic_ncols=True):
            pred = model(batch["depth"].to(device)).detach().cpu().numpy().astype(np.float32)
            for row, vk in zip(batch["rows"], pred):
                path = generated_vk_path(output_root, row)
                if path.exists() and not force:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vk)
                exported += 1
    return exported


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_final_eval(path: Path, protocol: str, metrics: dict[str, float], train_samples: int, val_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["protocol", "variant", "mse", "mae", "nme", "pck05", "pck10", "train_samples", "val_samples"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"protocol": protocol, "variant": "depth_cnn_no_rgb_online", **metrics, "train_samples": train_samples, "val_samples": val_samples})


def train_and_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rows = read_manifest(args.manifest)
    train_rows = split_rows(rows, args.protocol, "train")
    val_rows = split_rows(rows, args.protocol, "val")
    export_rows = rows if args.export_all_manifest else train_rows + val_rows

    output_dir = args.output_dir / "generators" / args.protocol
    generated_root = args.output_dir / "generated_vk" / args.protocol
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        DepthVKDataset(train_rows, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        DepthVKDataset(val_rows, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    export_loader = DataLoader(
        DepthVKDataset(export_rows, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = DepthVKGenerator(args.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_nme = math.inf
    start_epoch = 0
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    history_path = output_dir / "epoch_history.csv"
    if args.resume and last_path.exists():
        start_epoch, best_nme = load_checkpoint(last_path, model, optimizer, device)
        print(f"[Depth-VK] Resumed {last_path} from epoch {start_epoch}", flush=True)
    elif best_path.exists() and not args.force_train:
        load_checkpoint(best_path, model, None, device)
        print(f"[Depth-VK] Reusing existing best checkpoint: {best_path}", flush=True)
        start_epoch = args.epochs

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        total_batches = min(len(train_loader), args.max_train_batches) if args.max_train_batches is not None else len(train_loader)
        progress = tqdm(train_loader, desc=f"depth-vk:train:{epoch + 1}/{args.epochs}", total=total_batches, dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            depth = batch["depth"].to(device)
            target = batch["vk"].to(device)
            pred = model(depth)
            loss_abs = F.smooth_l1_loss(pred, target)
            loss_bone = bone_length_loss(pred, target)
            loss = loss_abs + args.bone_weight * loss_bone
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = target.size(0)
            total_loss += float(loss.item()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{float(loss.item()):.4f}")
        val_metrics = evaluate(model, val_loader, device, args.max_eval_batches)
        train_loss = total_loss / max(seen, 1)
        is_best = val_metrics["nme"] < best_nme
        if is_best:
            best_nme = val_metrics["nme"]
            save_checkpoint(best_path, model, optimizer, epoch, best_nme, args)
        save_checkpoint(last_path, model, optimizer, epoch, best_nme, args)
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", model, optimizer, epoch, best_nme, args)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "best_nme": best_nme,
            "is_best": int(is_best),
        }
        append_history(history_path, row)
        print(f"[Depth-VK] epoch={epoch + 1} | train_loss={train_loss:.6f} | val_nme={val_metrics['nme']:.6f} | best_nme={best_nme:.6f}", flush=True)

    load_checkpoint(best_path, model, None, device)
    final_metrics = evaluate(model, val_loader, device, args.max_eval_batches)
    write_final_eval(output_dir / "final_eval.csv", args.protocol, final_metrics, len(train_rows), len(val_rows))
    exported = export_predictions(model, export_loader, device, generated_root, args.force_export)
    marker = {
        "protocol": args.protocol,
        "generated_root": str(generated_root),
        "generated": exported,
        "final_metrics": final_metrics,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }
    (output_dir / "export_complete.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"[Depth-VK] Export complete | generated={exported}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a lightweight Depth-to-VK generator.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["random", "cross_scene", "cross_subject"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bone-weight", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--export-all-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_and_export(parse_args())


if __name__ == "__main__":
    main()
