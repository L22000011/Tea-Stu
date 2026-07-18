#!/usr/bin/env python
"""Export one synchronized MMFi sample as modality illustration panels.

This script is intentionally lightweight: it does not load any model and does
not modify training outputs. It reads one action frame from MMFi_DATA and saves
VK, depth, LiDAR, mmWave, and WiFi-CSI visual examples under Paper/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COCO17_BONES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="MMFi_DATA root.")
    parser.add_argument("--output-dir", default="Paper/modality_examples", help="Output directory.")
    parser.add_argument("--scene", default=None, help="Example: E01.")
    parser.add_argument("--subject", default=None, help="Example: S01.")
    parser.add_argument("--action", default=None, help="Example: A01.")
    parser.add_argument("--frame", type=int, default=1, help="1-based frame index, default 1.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def list_sorted(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if not p.name.startswith(".")])


def first_existing_file(path: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        files = sorted(path.glob(pattern))
        if files:
            return files[0]
    return None


def choose_sample(dataset: Path, scene: str | None, subject: str | None, action: str | None, frame: int) -> dict[str, Path | int | str]:
    if scene and subject and action:
        action_root = dataset / scene / subject / action
        if not action_root.exists():
            raise FileNotFoundError(f"Sample action directory not found: {action_root}")
    else:
        action_root = None
        for scene_dir in list_sorted(dataset):
            if not scene_dir.is_dir() or not scene_dir.name.startswith("E"):
                continue
            for subject_dir in list_sorted(scene_dir):
                if not subject_dir.is_dir() or not subject_dir.name.startswith("S"):
                    continue
                for action_dir in list_sorted(subject_dir):
                    if not action_dir.is_dir() or not action_dir.name.startswith("A"):
                        continue
                    required = ["rgb", "depth", "lidar", "mmwave", "wifi-csi"]
                    if all((action_dir / name).exists() for name in required):
                        action_root = action_dir
                        break
                if action_root is not None:
                    break
            if action_root is not None:
                break
        if action_root is None:
            raise FileNotFoundError("Could not find an action directory with rgb/depth/lidar/mmwave/wifi-csi.")

    frame_id = max(frame, 1)
    stem = f"frame{frame_id:03d}"
    paths: dict[str, Path | int | str] = {
        "action_root": action_root,
        "scene": action_root.parents[2].name,
        "subject": action_root.parents[1].name,
        "action": action_root.name,
        "frame": frame_id,
    }

    modality_patterns = {
        "vk": ["*.npy"],
        "depth": [f"{stem}.*", "*.png", "*.jpg", "*.jpeg"],
        "lidar": [f"{stem}.*", "*.bin"],
        "mmwave": [f"{stem}.*", "*.bin"],
        "wifi-csi": [f"{stem}.*", "*.mat"],
    }
    folder_map = {"vk": "rgb", "depth": "depth", "lidar": "lidar", "mmwave": "mmwave", "wifi-csi": "wifi-csi"}
    for modality, folder in folder_map.items():
        modality_dir = action_root / folder
        chosen = first_existing_file(modality_dir, modality_patterns[modality])
        if chosen is None:
            raise FileNotFoundError(f"No file found for {modality} in {modality_dir}")
        if modality == "vk":
            # Prefer the same frame if keypoints are stored frame-wise as npy.
            same_frame = first_existing_file(modality_dir, [f"{stem}.npy"])
            if same_frame is not None:
                chosen = same_frame
        paths[modality] = chosen
    return paths


def read_point_cloud(path: Path, dims: int) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.float64).copy()
    if raw.size < dims:
        return np.zeros((0, dims), dtype=np.float32)
    usable = raw[: raw.size - (raw.size % dims)]
    return usable.reshape(-1, dims).astype(np.float32)


def normalize_image(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.nanpercentile(x[finite], 1))
    hi = float(np.nanpercentile(x[finite], 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0, 1)


def load_depth(path: Path) -> np.ndarray:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cv2 failed to read {path}")
        if image.ndim == 3:
            image = image[..., ::-1]
        return image
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path))


def load_csi(path: Path) -> np.ndarray:
    try:
        import scipy.io as scio
    except Exception as exc:
        raise ModuleNotFoundError("scipy is required to read WiFi-CSI .mat files.") from exc
    mat = scio.loadmat(path)
    if "CSIamp" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"CSIamp not found in {path}. Available keys: {keys}")
    return mat["CSIamp"].astype(np.float32)


def save_vk(points: np.ndarray, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(points[:, 0], points[:, 1], s=28, c="#2b7a3d", zorder=3)
    for i, j in COCO17_BONES:
        ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color="#66a85f", linewidth=1.8)
    for idx, (x, y) in enumerate(points):
        ax.text(x, y, str(idx), fontsize=5, color="#1f4f2b")
    ax.set_title("VK: 17 x 2 keypoints", fontsize=9)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_depth(image: np.ndarray, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    if image.ndim == 2:
        ax.imshow(normalize_image(image), cmap="viridis")
    else:
        ax.imshow(normalize_image(image))
    ax.set_title(f"Depth: {tuple(image.shape)}", fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_point_cloud(points: np.ndarray, output: Path, title: str, dpi: int) -> None:
    fig = plt.figure(figsize=(3.4, 3.2))
    ax = fig.add_subplot(111, projection="3d")
    if points.size:
        stride = max(1, math.ceil(points.shape[0] / 3000))
        pts = points[::stride]
        color = pts[:, 2] if pts.shape[1] >= 3 else np.arange(len(pts))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=color, cmap="viridis", alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_csi(csi: np.ndarray, output: Path, dpi: int) -> None:
    data = csi.copy()
    data[~np.isfinite(data)] = np.nan
    if data.ndim >= 3:
        view = np.nanmean(data, axis=tuple(range(2, data.ndim)))
    elif data.ndim == 2:
        view = data
    else:
        view = data.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(normalize_image(view), cmap="magma", aspect="auto")
    ax.set_title(f"WiFi-CSI: CSIamp {tuple(csi.shape)}", fontsize=9)
    ax.set_xlabel("subcarrier/time axis", fontsize=7)
    ax.set_ylabel("antenna/packet axis", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.2)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_overview(images: list[Path], output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(14, 3.2))
    for ax, path in zip(axes, images):
        from PIL import Image

        ax.imshow(Image.open(path))
        ax.axis("off")
        ax.set_title(path.stem.replace("example_", ""), fontsize=9)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = choose_sample(dataset, args.scene, args.subject, args.action, args.frame)
    print(f"[Sample] {paths['scene']}/{paths['subject']}/{paths['action']} frame={paths['frame']}")

    vk = np.load(paths["vk"]).astype(np.float32)
    if vk.shape != (17, 2):
        vk = vk.reshape(-1, 2)[:17]
    depth = load_depth(paths["depth"])
    lidar = read_point_cloud(paths["lidar"], 3)
    mmwave = read_point_cloud(paths["mmwave"], 5)
    csi = load_csi(paths["wifi-csi"])

    outputs = {
        "vk": output_dir / "example_vk_17x2.png",
        "depth": output_dir / "example_depth.png",
        "lidar": output_dir / "example_lidar_pointcloud.png",
        "mmwave": output_dir / "example_mmwave_pointcloud.png",
        "wifi-csi": output_dir / "example_wifi_csi.png",
    }
    save_vk(vk, outputs["vk"], args.dpi)
    save_depth(depth, outputs["depth"], args.dpi)
    save_point_cloud(lidar, outputs["lidar"], f"LiDAR: {lidar.shape[0]} x 3 points", args.dpi)
    save_point_cloud(mmwave[:, :3] if mmwave.shape[1] >= 3 else mmwave, outputs["mmwave"], f"mmWave: {mmwave.shape[0]} x 5 points", args.dpi)
    save_csi(csi, outputs["wifi-csi"], args.dpi)
    save_overview(list(outputs.values()), output_dir / "example_all_modalities_overview.png", args.dpi)

    metadata: dict[str, Any] = {
        "sample": {k: str(v) for k, v in paths.items()},
        "shapes": {
            "vk": list(vk.shape),
            "depth": list(depth.shape),
            "lidar": list(lidar.shape),
            "mmwave": list(mmwave.shape),
            "wifi-csi": list(csi.shape),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {output_dir}")
    print(json.dumps(metadata["shapes"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
