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


COCO17_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def split_rows(rows: list[dict[str, str]], protocol: str, split: str) -> list[dict[str, str]]:
    if protocol == "cross_scene":
        if split == "train":
            return [row for row in rows if row["scene"] in {"E01", "E02", "E03"}]
        return [row for row in rows if row["scene"] == "E04"]
    if protocol == "cross_subject":
        subjects = set(TRAIN_SUBJECTS if split == "train" else VAL_SUBJECTS)
        return [row for row in rows if row["subject"] in subjects]
    raise ValueError(f"Unsupported protocol: {protocol}")


def row_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return row["scene"], row["subject"], row["action"], int(row["idx"])


def mmwave_path(mmwave_root: Path, row: dict[str, str], idx: int | None = None) -> Path:
    frame_idx = int(row["idx"]) if idx is None else int(idx)
    frame_name = f"frame{frame_idx + 1:03d}.bin"
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
    valid_size = (data.size // 5) * 5
    if valid_size <= 0:
        return np.zeros((1, 5), dtype=np.float32)
    data = data[:valid_size].reshape(-1, 5)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] == 0:
        data = np.zeros((1, 5), dtype=np.float32)
    return data.astype(np.float32)


def sample_points(points: np.ndarray, num_points: int, rng: np.random.Generator | None) -> np.ndarray:
    if points.shape[0] == num_points:
        return points
    if points.shape[0] > num_points:
        if rng is None:
            idx = np.linspace(0, points.shape[0] - 1, num_points).astype(np.int64)
        else:
            idx = rng.choice(points.shape[0], size=num_points, replace=False)
        return points[idx]
    if points.shape[0] == 0:
        return np.zeros((num_points, points.shape[1] if points.ndim == 2 else 5), dtype=np.float32)
    pad_count = num_points - points.shape[0]
    if rng is None:
        idx = np.arange(pad_count) % points.shape[0]
    else:
        idx = rng.choice(points.shape[0], size=pad_count, replace=True)
    return np.concatenate([points, points[idx]], axis=0)


def compute_global_stats(rows: list[dict[str, str]], mmwave_root: Path, max_rows: int) -> dict[str, list[float]]:
    coords: list[np.ndarray] = []
    use_rows = rows[:max_rows] if max_rows > 0 else rows
    for row in tqdm(use_rows, desc="temporal-mmwave-vk:global-stats"):
        path = mmwave_path(mmwave_root, row)
        if not path.exists():
            continue
        points = read_mmwave(path)
        if points.size:
            coords.append(points[:, :5])
    if not coords:
        mean = np.zeros(5, dtype=np.float32)
        std = np.ones(5, dtype=np.float32)
    else:
        stacked = np.concatenate(coords, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = stacked.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-4)
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_frame(points: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray, frame_offset: int) -> np.ndarray:
    points = points.astype(np.float32, copy=True)
    global_feat = (points - global_mean[None, :]) / global_std[None, :]
    xyz = points[:, :3]
    local_center = xyz.mean(axis=0, keepdims=True)
    local_scale = np.maximum(xyz.std(axis=0, keepdims=True).mean(), 1e-4)
    local_xyz = (xyz - local_center) / local_scale
    local_rest = np.tanh(points[:, 3:])
    time = np.full((points.shape[0], 1), frame_offset / 2.0, dtype=np.float32)
    return np.concatenate([global_feat, local_xyz, local_rest, time], axis=-1).astype(np.float32)


class TemporalMmWaveVKDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        all_rows: list[dict[str, str]],
        mmwave_root: Path,
        num_points: int,
        window: int,
        global_stats: dict[str, list[float]],
        augment: bool,
    ) -> None:
        self.rows = rows
        self.mmwave_root = mmwave_root
        self.num_points = num_points
        self.window = window
        self.augment = augment
        self.global_mean = np.array(global_stats["mean"], dtype=np.float32)
        self.global_std = np.array(global_stats["std"], dtype=np.float32)
        self.index = {row_key(row): row for row in all_rows}

    def __len__(self) -> int:
        return len(self.rows)

    def _neighbor_row(self, row: dict[str, str], offset: int) -> dict[str, str]:
        idx = int(row["idx"]) + offset
        key = (row["scene"], row["subject"], row["action"], idx)
        return self.index.get(key, row)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rng = np.random.default_rng() if self.augment else None
        half = self.window // 2
        frames = []
        for offset in range(-half, half + 1):
            nrow = self._neighbor_row(row, offset)
            path = mmwave_path(self.mmwave_root, nrow)
            if not path.exists():
                path = mmwave_path(self.mmwave_root, row)
            points = read_mmwave(path)
            features = normalize_frame(points, self.global_mean, self.global_std, offset)
            features = sample_points(features, self.num_points, rng)
            if self.augment:
                features = features.copy()
                features[:, :5] += np.random.normal(0.0, 0.01, size=features[:, :5].shape).astype(np.float32)
            frames.append(features)
        vk_path = Path(row["vk_path"])
        if not vk_path.exists():
            raise FileNotFoundError(f"Missing target VK frame: {vk_path}")
        return {
            "mmwave": torch.tensor(np.stack(frames, axis=0), dtype=torch.float32),
            "vk": torch.tensor(np.load(vk_path).astype(np.float32).reshape(17, 2), dtype=torch.float32),
            "rows": row,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mmwave": torch.stack([item["mmwave"] for item in batch], dim=0),
        "vk": torch.stack([item["vk"] for item in batch], dim=0),
        "rows": [item["rows"] for item in batch],
    }


class TemporalGlobalLocalVKGenerator(nn.Module):
    def __init__(self, in_dim: int = 11, hidden: int = 192, window: int = 5) -> None:
        super().__init__()
        self.window = window
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.center_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.relative_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 256), nn.ReLU(inplace=True), nn.Linear(256, 34))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, window, num_points, channels = points.shape
        x = points.view(batch * window, num_points, channels)
        point_features = self.point_mlp(x)
        frame_features = point_features.max(dim=1).values.view(batch, window, -1)
        encoded = self.temporal_encoder(frame_features)
        pooled = encoded.mean(dim=1)
        center = self.center_head(pooled)
        relative = self.relative_head(pooled).view(batch, 17, 2)
        relative = relative - relative.mean(dim=1, keepdim=True)
        pred = center[:, None, :] + relative
        return pred, center, relative


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


def bone_lengths(pose: torch.Tensor) -> torch.Tensor:
    idx_a = torch.tensor([a for a, _ in COCO17_BONES], device=pose.device)
    idx_b = torch.tensor([b for _, b in COCO17_BONES], device=pose.device)
    return torch.linalg.norm(pose[:, idx_a] - pose[:, idx_b], dim=-1)


def compute_loss(pred: torch.Tensor, center: torch.Tensor, relative: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_center = target.mean(dim=1)
    target_relative = target - target_center[:, None, :]
    loss_abs = F.smooth_l1_loss(pred, target)
    loss_center = F.smooth_l1_loss(center, target_center)
    loss_relative = F.smooth_l1_loss(relative, target_relative)
    pred_bones = bone_lengths(pred)
    target_bones = bone_lengths(target)
    loss_bone = F.smooth_l1_loss(pred_bones, target_bones)
    # Single-window supervision predicts the center frame. The temporal term stays
    # lightweight by penalizing implausibly large skeleton spread, not global offset.
    loss_temporal = torch.mean(torch.relu(torch.linalg.norm(relative, dim=-1).mean(dim=1) - 120.0))
    return loss_abs + 0.5 * loss_center + 0.5 * loss_relative + 0.1 * loss_bone + 0.05 * loss_temporal


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    totals = {"mse": 0.0, "mae": 0.0, "nme": 0.0, "pck05": 0.0, "pck10": 0.0}
    seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="temporal-mmwave-vk:eval")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pred, _, _ = model(batch["mmwave"].to(device))
            target = batch["vk"].to(device)
            metrics = keypoint_metrics(pred, target)
            batch_size = target.size(0)
            seen += batch_size
            for key, value in metrics.items():
                totals[key] += value * batch_size
    return {key: value / max(seen, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_nme: float, args: argparse.Namespace, global_stats: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_nme": best_nme,
            "args": vars(args),
            "global_stats": global_stats,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[int, float, dict[str, list[float]] | None]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_nme", math.inf)), checkpoint.get("global_stats")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def export_predictions(model: nn.Module, loader: DataLoader, device: torch.device, root: Path, force: bool) -> int:
    model.eval()
    exported = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"temporal-mmwave-vk:export:{root.name}"):
            pred, _, _ = model(batch["mmwave"].to(device))
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            for row, keypoints in zip(batch["rows"], pred_np):
                path = generated_vk_path(root, row)
                if path.exists() and not force:
                    exported += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, keypoints)
                exported += 1
    return exported


def train_and_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rows = read_manifest(args.manifest)
    train_rows = split_rows(rows, args.protocol, "train")
    val_rows = split_rows(rows, args.protocol, "val")
    export_rows = rows if args.export_all_manifest else train_rows + val_rows

    run_dir = args.output_dir / "generators_temporal" / args.protocol
    generated_root = args.output_dir / "generated_vk" / args.protocol
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "global_stats.json"
    if stats_path.exists() and not args.force_train:
        global_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        global_stats = compute_global_stats(train_rows, args.mmwave_root, args.global_stats_rows)
        stats_path.write_text(json.dumps(global_stats, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        TemporalMmWaveVKDataset(train_rows, rows, args.mmwave_root, args.num_points, args.window, global_stats, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        TemporalMmWaveVKDataset(val_rows, rows, args.mmwave_root, args.num_points, args.window, global_stats, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    export_loader = DataLoader(
        TemporalMmWaveVKDataset(export_rows, rows, args.mmwave_root, args.num_points, args.window, global_stats, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = TemporalGlobalLocalVKGenerator(in_dim=11, hidden=args.hidden_dim, window=args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_nme = math.inf
    start_epoch = 0
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    reuse_existing_best = False
    if args.resume and last_path.exists():
        start_epoch, best_nme, loaded_stats = load_checkpoint(last_path, model, optimizer, device)
        if loaded_stats is not None:
            global_stats = loaded_stats
        print(f"[Temporal-mmWave-VK] Resumed {last_path} from epoch {start_epoch}")
    elif best_path.exists() and not args.force_train:
        _, best_nme, loaded_stats = load_checkpoint(best_path, model, None, device)
        if loaded_stats is not None:
            global_stats = loaded_stats
        reuse_existing_best = True
        print(f"[Temporal-mmWave-VK] Reusing existing best checkpoint: {best_path}")

    history_path = run_dir / "epoch_history.csv"
    history_fields = ["epoch", "train_loss", "val_mse", "val_mae", "val_nme", "val_pck05", "val_pck10"]
    if not history_path.exists() or args.force_train:
        write_csv(history_path, [], history_fields)

    if not reuse_existing_best and (not best_path.exists() or args.force_train or start_epoch < args.epochs):
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0.0
            seen_batches = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"temporal-mmwave-vk:train:{epoch + 1}/{args.epochs}")):
                if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                    break
                points = batch["mmwave"].to(device)
                target = batch["vk"].to(device)
                pred, center, relative = model(points)
                loss = compute_loss(pred, center, relative, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                seen_batches += 1

            metrics = evaluate(model, val_loader, device, max_batches=args.max_eval_batches)
            train_loss = total_loss / max(seen_batches, 1)
            write_csv(
                history_path,
                [{"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{key}": value for key, value in metrics.items()}}],
                history_fields,
                append=True,
            )
            print(
                f"[Temporal-mmWave-VK] epoch {epoch + 1}/{args.epochs} | "
                f"loss={train_loss:.6f} | mae={metrics['mae']:.4f} | nme={metrics['nme']:.6f} | pck10={metrics['pck10']:.4f}"
            )
            if metrics["nme"] < best_nme:
                best_nme = metrics["nme"]
                save_checkpoint(best_path, model, optimizer, epoch, best_nme, args, global_stats)
                print(f"[Temporal-mmWave-VK] Saved best: {best_path}")
            save_checkpoint(last_path, model, optimizer, epoch, best_nme, args, global_stats)
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pth", model, optimizer, epoch, best_nme, args, global_stats)

    if not best_path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {best_path}")
    load_checkpoint(best_path, model, None, device)
    metrics = evaluate(model, val_loader, device, max_batches=args.max_eval_batches)
    write_csv(
        run_dir / "final_eval.csv",
        [{"protocol": args.protocol, "variant": "temporal_global_local_no_offset", **metrics, "train_samples": len(train_rows), "val_samples": len(val_rows)}],
        ["protocol", "variant", "mse", "mae", "nme", "pck05", "pck10", "train_samples", "val_samples"],
    )
    exported = export_predictions(model, export_loader, device, generated_root, force=args.force_export)
    complete = {
        "protocol": args.protocol,
        "variant": "temporal_global_local_no_offset",
        "generated_root": str(generated_root),
        "exported": exported,
        "metrics": metrics,
    }
    (run_dir / "export_complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(f"[Temporal-mmWave-VK] Export complete | generated={exported}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Temporal global-local mmWave-to-VK generator.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mmwave-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=["cross_scene", "cross_subject"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--global-stats-rows", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
