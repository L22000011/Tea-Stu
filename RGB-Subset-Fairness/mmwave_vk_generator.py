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


def split_rows(rows: list[dict[str, str]], protocol: str, split: str) -> list[dict[str, str]]:
    if protocol == "cross_scene":
        if split == "train":
            return [row for row in rows if row["scene"] in {"E01", "E02", "E03"}]
        return [row for row in rows if row["scene"] == "E04"]
    if protocol == "cross_subject":
        subjects = set(TRAIN_SUBJECTS if split == "train" else VAL_SUBJECTS)
        return [row for row in rows if row["subject"] in subjects]
    raise ValueError(f"Unsupported protocol: {protocol}")


def mmwave_path(mmwave_root: Path, row: dict[str, str]) -> Path:
    frame_name = f"frame{int(row['idx']) + 1:03d}.bin"
    candidates = [
        mmwave_root / row["scene"] / row["subject"] / row["action"] / frame_name,
        mmwave_root / row["scene"] / row["subject"] / row["action"] / "mmwave" / frame_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def generated_vk_path(output_root: Path, row: dict[str, str]) -> Path:
    return output_root / row["scene"] / row["subject"] / row["action"] / f"frame{int(row['idx']) + 1:03d}.npy"


def read_mmwave(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float64).astype(np.float32)
    if data.size == 0:
        return np.zeros((1, 5), dtype=np.float32)
    valid_size = (data.size // 5) * 5
    if valid_size == 0:
        return np.zeros((1, 5), dtype=np.float32)
    data = data[:valid_size].reshape(-1, 5)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] == 0:
        data = np.zeros((1, 5), dtype=np.float32)
    return data.astype(np.float32)


class MmWaveVKDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], mmwave_root: Path) -> None:
        self.rows = rows
        self.mmwave_root = mmwave_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        mm_path = mmwave_path(self.mmwave_root, row)
        if not mm_path.exists():
            raise FileNotFoundError(f"Missing filtered mmWave frame: {mm_path}")
        vk_path = Path(row["vk_path"])
        if not vk_path.exists():
            raise FileNotFoundError(f"Missing target VK frame: {vk_path}")
        return {
            "mmwave": read_mmwave(mm_path),
            "vk": np.load(vk_path).astype(np.float32).reshape(17, 2),
            "row": row,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    clouds = []
    for item in batch:
        tensor = torch.tensor(item["mmwave"], dtype=torch.float32)
        if tensor.numel() == 0 or tensor.size(0) == 0:
            tensor = torch.zeros(1, 5, dtype=torch.float32)
        clouds.append(tensor)
    return {
        "mmwave": torch.nn.utils.rnn.pad_sequence(clouds, batch_first=True),
        "vk": torch.tensor(np.stack([item["vk"] for item in batch]), dtype=torch.float32),
        "rows": [item["row"] for item in batch],
    }


class PointNetVKGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 34),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.point_mlp(points)
        pooled = features.max(dim=1).values
        return self.head(pooled).view(-1, 17, 2)


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
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="mmwave-vk:eval")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            points = batch["mmwave"].to(device)
            target = batch["vk"].to(device)
            pred = model(points)
            metrics = keypoint_metrics(pred, target)
            batch_size = target.size(0)
            seen += batch_size
            for key, value in metrics.items():
                totals[key] += value * batch_size
    return {key: value / max(seen, 1) for key, value in totals.items()}


def train_and_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rows = read_manifest(args.manifest)
    train_rows = split_rows(rows, args.protocol, "train")
    val_rows = split_rows(rows, args.protocol, "val")
    if args.export_all_manifest:
        export_rows = rows
    else:
        export_rows = train_rows + val_rows

    output_dir = args.output_dir / "generators" / args.protocol
    generated_root = args.output_dir / "generated_vk" / args.protocol
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        MmWaveVKDataset(train_rows, args.mmwave_root),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        MmWaveVKDataset(val_rows, args.mmwave_root),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = PointNetVKGenerator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_nme = math.inf
    start_epoch = 0
    reuse_existing_best = False
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    if args.resume and last_path.exists():
        start_epoch, best_nme = load_checkpoint(last_path, model, optimizer, device)
        print(f"[mmWave-VK] Resumed {last_path} from epoch {start_epoch}")
    elif best_path.exists() and not args.force_train:
        load_checkpoint(best_path, model, None, device)
        reuse_existing_best = True
        print(f"[mmWave-VK] Reusing existing best checkpoint: {best_path}")

    history_path = output_dir / "epoch_history.csv"
    if not history_path.exists() or args.force_train:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_mse", "val_mae", "val_nme", "val_pck05", "val_pck10"])
            writer.writeheader()

    if not reuse_existing_best and (not best_path.exists() or args.force_train or start_epoch < args.epochs):
        for epoch in range(start_epoch, args.epochs):
            model.train()
            loss_total = 0.0
            seen_batches = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"mmwave-vk:train:{epoch + 1}/{args.epochs}")):
                if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                    break
                points = batch["mmwave"].to(device)
                target = batch["vk"].to(device)
                pred = model(points)
                loss = F.smooth_l1_loss(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_total += float(loss.detach().cpu())
                seen_batches += 1

            metrics = evaluate(model, val_loader, device, args.max_eval_batches)
            train_loss = loss_total / max(seen_batches, 1)
            row = {"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}}
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_mse", "val_mae", "val_nme", "val_pck05", "val_pck10"])
                writer.writerow(row)
            print(
                f"[mmWave-VK] epoch {epoch + 1}/{args.epochs} | "
                f"loss={train_loss:.6f} | nme={metrics['nme']:.6f} | pck10={metrics['pck10']:.4f}"
            )
            if metrics["nme"] < best_nme:
                best_nme = metrics["nme"]
                save_checkpoint(best_path, model, optimizer, epoch, best_nme, args)
                print(f"[mmWave-VK] Saved best: {best_path}")
            save_checkpoint(last_path, model, optimizer, epoch, best_nme, args)
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", model, optimizer, epoch, best_nme, args)

    load_checkpoint(best_path, model, None, device)
    final_metrics = evaluate(model, val_loader, device, args.max_eval_batches)
    final_eval = output_dir / "final_eval.csv"
    with final_eval.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol", "mse", "mae", "nme", "pck05", "pck10", "train_samples", "val_samples"])
        writer.writeheader()
        writer.writerow({"protocol": args.protocol, **final_metrics, "train_samples": len(train_rows), "val_samples": len(val_rows)})

    export_dataset = MmWaveVKDataset(export_rows, args.mmwave_root)
    export_loader = DataLoader(
        export_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    model.eval()
    exported = 0
    with torch.no_grad():
        for batch in tqdm(export_loader, desc=f"mmwave-vk:export:{args.protocol}"):
            points = batch["mmwave"].to(device)
            pred = model(points).detach().cpu().numpy().astype(np.float32)
            for row, keypoints in zip(batch["rows"], pred):
                path = generated_vk_path(generated_root, row)
                if path.exists() and not args.force_export:
                    exported += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, keypoints)
                exported += 1

    complete = {
        "protocol": args.protocol,
        "generated_root": str(generated_root),
        "exported": exported,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "metrics": final_metrics,
    }
    (output_dir / "export_complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(f"[mmWave-VK] Export complete | root={generated_root} | samples={exported}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train and export mmWave-generated Visual Keypoints.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mmwave-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=["cross_scene", "cross_subject"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--export-all-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_and_export(parse_args())


if __name__ == "__main__":
    main()
