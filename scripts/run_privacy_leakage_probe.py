from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
ALL_ACTIONS = [f"A{i:02d}" for i in range(1, 28)]


def subject_to_scene(subject: str) -> str:
    idx = int(subject[1:])
    if 1 <= idx <= 10:
        return "E01"
    if 11 <= idx <= 20:
        return "E02"
    if 21 <= idx <= 30:
        return "E03"
    if 31 <= idx <= 40:
        return "E04"
    raise ValueError(f"Unknown subject: {subject}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_array(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_vk(path: Path) -> np.ndarray:
    data = finite_array(np.load(path))
    if data.shape != (17, 2):
        fixed = np.zeros((17, 2), dtype=np.float32)
        flat = data.reshape(-1, 2) if data.size >= 2 else np.zeros((0, 2), dtype=np.float32)
        keep = min(17, flat.shape[0])
        if keep:
            fixed[:keep] = flat[:keep]
        data = fixed
    max_abs = np.maximum(np.abs(data).max(axis=0, keepdims=True), 1.0)
    coords = data / max_abs
    center = coords.mean(axis=0, keepdims=True)
    centered = coords - center
    bone_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    lengths = [np.linalg.norm(centered[i] - centered[j]) for i, j in bone_pairs]
    return np.concatenate([centered.reshape(-1), np.asarray(lengths, dtype=np.float32)])


def load_depth(path: Path) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for depth privacy probe.") from exc
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    image = finite_array(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    lo, hi = float(np.min(image)), float(np.max(image))
    if hi - lo > 1e-6:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    hist, _ = np.histogram(image, bins=32, range=(0.0, 1.0), density=True)
    return np.concatenate([small.astype(np.float32), hist.astype(np.float32)])


def load_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for RGB privacy probe.") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = finite_array(image) / 255.0
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(image[..., channel], bins=16, range=(0.0, 1.0), density=True)
        hist_parts.append(finite_array(hist))
    gray = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    texture = np.asarray(
        [
            float(gray.mean()),
            float(gray.std()),
            float(np.abs(gx).mean()) if gx.size else 0.0,
            float(np.abs(gy).mean()) if gy.size else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([small.astype(np.float32)] + hist_parts + [texture])


def load_point_cloud(path: Path, dims: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float64)
    valid = (raw.size // dims) * dims
    if valid == 0:
        data = np.zeros((1, dims), dtype=np.float32)
    else:
        data = finite_array(raw[:valid].reshape(-1, dims))
    stats = [
        data.mean(axis=0),
        data.std(axis=0),
        data.min(axis=0),
        data.max(axis=0),
        np.percentile(data, 25, axis=0),
        np.percentile(data, 75, axis=0),
    ]
    count = np.asarray([math.log1p(data.shape[0])], dtype=np.float32)
    return np.concatenate([item.astype(np.float32).reshape(-1) for item in stats] + [count])


def load_wifi(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".mat":
        try:
            import scipy.io as scio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("scipy is required for .mat WiFi-CSI privacy probe.") from exc
        mat = scio.loadmat(path)
        arrays = [value for value in mat.values() if isinstance(value, np.ndarray) and value.size > 0]
        data = max(arrays, key=lambda arr: arr.size) if arrays else np.zeros((1,), dtype=np.float32)
    else:
        data = np.fromfile(path, dtype=np.float32)
    data = finite_array(np.abs(data).reshape(-1))
    if data.size == 0:
        data = np.zeros((1,), dtype=np.float32)
    quantiles = np.percentile(data, [0, 10, 25, 50, 75, 90, 100]).astype(np.float32)
    stats = np.asarray([data.mean(), data.std(), math.log1p(data.size)], dtype=np.float32)
    hist, _ = np.histogram(data, bins=32, density=True)
    hist = finite_array(hist)
    return np.concatenate([stats, quantiles, hist])


def modality_folder(modality: str) -> str:
    if modality == "vk":
        return "rgb"
    return modality


def modality_feature(path: Path, modality: str) -> np.ndarray:
    if modality == "rgb":
        return load_rgb(path)
    if modality == "vk":
        return load_vk(path)
    if modality == "depth":
        return load_depth(path)
    if modality == "lidar":
        return load_point_cloud(path, 3)
    if modality == "mmwave":
        return load_point_cloud(path, 5)
    if modality == "wifi-csi":
        return load_wifi(path)
    raise ValueError(f"Unsupported modality: {modality}")


def parse_frame_idx(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        return None
    return int(digits) - 1


def exact_or_index_file(files: list[Path], frame_idx: int, preferred_suffix: str | None = None) -> Path | None:
    frame_name = f"frame{frame_idx + 1:03d}"
    for file in files:
        if file.stem == frame_name and (preferred_suffix is None or file.suffix.lower() == preferred_suffix):
            return file
    if not files:
        return None
    file_index = min(max(frame_idx, 0), len(files) - 1)
    return files[file_index]


def collect_samples(
    dataset: Path,
    rgb_root: Path | None,
    modalities: list[str],
    max_per_subject: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    per_subject: dict[str, int] = defaultdict(int)
    subject_actions = [(subject, action) for subject in ALL_SUBJECTS for action in ALL_ACTIONS]
    rng.shuffle(subject_actions)
    total_needed = len(ALL_SUBJECTS) * max_per_subject
    print(
        f"[Collect] target={total_needed} samples | subjects={len(ALL_SUBJECTS)} | "
        f"max_per_subject={max_per_subject} | modalities={','.join(modalities)}",
        flush=True,
    )
    if "rgb" in modalities:
        if rgb_root is None:
            raise ValueError("--rgb-root is required when modalities include rgb.")
        print(f"[Collect] RGB root={rgb_root}", flush=True)
    pbar = tqdm(subject_actions, desc="collect:samples")
    for subject, action in pbar:
        if per_subject[subject] >= max_per_subject:
            continue
        scene = subject_to_scene(subject)
        root = dataset / scene / subject / action
        mmwave_dir = root / "mmwave"
        if not mmwave_dir.exists():
            continue
        if "rgb" in modalities:
            rgb_dir = (rgb_root or dataset) / scene / subject / action / "rgb"
            if not rgb_dir.exists():
                continue
            frames = sorted(path for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        else:
            frames = sorted(mmwave_dir.iterdir())
        if not frames:
            continue
        rng.shuffle(frames)
        for frame_path in frames:
            if per_subject[subject] >= max_per_subject:
                break
            frame_idx = parse_frame_idx(frame_path)
            if frame_idx is None:
                continue
            item = {"subject": subject, "action": action, "frame_idx": frame_idx}
            ok = True
            for modality in modalities:
                if modality == "rgb":
                    item["rgb_path"] = frame_path
                    continue
                folder = root / modality_folder(modality)
                if not folder.exists():
                    ok = False
                    break
                files = sorted(folder.iterdir())
                if not files:
                    ok = False
                    break
                selected = exact_or_index_file(files, frame_idx)
                if selected is None:
                    ok = False
                    break
                item[f"{modality}_path"] = selected
            if ok:
                samples.append(item)
                per_subject[subject] += 1
                if len(samples) % 100 == 0:
                    pbar.set_postfix(samples=len(samples), subject=subject)
                if len(samples) >= total_needed:
                    break
        if len(samples) >= total_needed:
            break
    pbar.close()
    print(f"[Collect] done | collected={len(samples)} samples", flush=True)
    return samples


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def split_indices(
    samples: list[dict[str, Any]],
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if split_mode == "random":
        indices = rng.permutation(len(labels))
        split = int(len(indices) * train_ratio)
        return indices[:split], indices[split:]

    if split_mode != "action-holdout":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    train: list[int] = []
    test: list[int] = []
    by_subject: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (sample, label) in enumerate(zip(samples, labels)):
        by_subject[int(label)][sample["action"]].append(idx)

    for subject_label, by_action in by_subject.items():
        actions = np.array(sorted(by_action.keys()))
        rng.shuffle(actions)
        if len(actions) <= 1:
            indices = [idx for action in actions for idx in by_action[action]]
            rng.shuffle(indices)
            split = max(1, int(len(indices) * train_ratio))
            train.extend(indices[:split])
            test.extend(indices[split:])
            continue
        split = min(max(1, int(len(actions) * train_ratio)), len(actions) - 1)
        train_actions = set(actions[:split].tolist())
        for action, indices in by_action.items():
            if action in train_actions:
                train.extend(indices)
            else:
                test.extend(indices)

    train_arr = np.array(train, dtype=np.int64)
    test_arr = np.array(test, dtype=np.int64)
    rng.shuffle(train_arr)
    rng.shuffle(test_arr)
    return train_arr, test_arr


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    modality: str,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    train_x, test_x = standardize(train_x, test_x)
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.long)
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.long)
    model = torch.nn.Linear(x_train.shape[1], int(y_train.max().item()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    progress = tqdm(range(epochs), desc=f"probe:{modality}")
    for epoch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_acc_now = (model(x_train).argmax(dim=1) == y_train).float().mean().item()
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", train_acc=f"{train_acc_now:.3f}")
    with torch.no_grad():
        train_acc = (model(x_train).argmax(dim=1) == y_train).float().mean().item()
        test_acc = (model(x_test).argmax(dim=1) == y_test).float().mean().item()
    return train_acc, test_acc


def run_probe(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    rgb_root = Path(args.rgb_root).resolve() if args.rgb_root else None
    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    samples = collect_samples(dataset, rgb_root, modalities, args.max_per_subject, args.seed)
    if not samples:
        raise RuntimeError("No samples collected. Check dataset path and modality folders.")

    subject_to_label = {subject: idx for idx, subject in enumerate(ALL_SUBJECTS)}
    rows: list[dict[str, Any]] = []
    output_dir = PROJECT_ROOT / "supplement" / "privacy_leakage_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    for modality in modalities:
        print("=" * 88, flush=True)
        print(f"[PrivacyProbe] Extracting features for modality={modality}", flush=True)
        print("=" * 88, flush=True)
        features = []
        labels = []
        used_samples: list[dict[str, Any]] = []
        used = 0
        for sample in tqdm(samples, desc=f"extract:{modality}"):
            path = sample.get(f"{modality}_path")
            if path is None:
                continue
            try:
                features.append(modality_feature(Path(path), modality))
                labels.append(subject_to_label[sample["subject"]])
                used_samples.append(sample)
                used += 1
            except Exception as exc:  # keep probe robust across optional modalities
                if args.verbose:
                    print(f"[skip] {modality}: {path} ({exc})")
                continue
        print(
            f"[PrivacyProbe] modality={modality} | usable_samples={len(features)} | "
            f"skipped={len(samples) - len(features)}",
            flush=True,
        )
        if not features:
            rows.append({
                "modality": modality,
                "status": "skipped_no_features",
                "num_samples": 0,
                "feature_dim": 0,
                "split_mode": args.split_mode,
                "subject_train_acc": "",
                "subject_test_acc": "",
                "chance_acc": 1.0 / len(ALL_SUBJECTS),
                "chance_percent": f"{100.0 / len(ALL_SUBJECTS):.2f}",
                "privacy_interpretation": "No features available.",
            })
            continue
        max_dim = max(feat.size for feat in features)
        padded = np.zeros((len(features), max_dim), dtype=np.float32)
        for idx, feat in enumerate(features):
            padded[idx, :feat.size] = feat
        labels_arr = np.asarray(labels, dtype=np.int64)
        train_idx, test_idx = split_indices(used_samples, labels_arr, float(args.train_ratio), args.seed, args.split_mode)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"Empty train/test split for modality={modality}; try --split-mode random.")
        print(
            f"[PrivacyProbe] modality={modality} | split_mode={args.split_mode} | "
            f"train={len(train_idx)} | test={len(test_idx)} | chance={1.0 / len(ALL_SUBJECTS):.4f}",
            flush=True,
        )
        train_acc, test_acc = fit_linear_probe(
            padded[train_idx],
            labels_arr[train_idx],
            padded[test_idx],
            labels_arr[test_idx],
            args.epochs,
            args.lr,
            args.seed,
            modality,
        )
        chance = 1.0 / len(ALL_SUBJECTS)
        if test_acc <= chance * 2:
            interpretation = "Low subject leakage under this lightweight probe."
        elif test_acc <= 0.25:
            interpretation = "Measurable subject leakage; avoid privacy guarantee claims."
        else:
            interpretation = "High subject leakage; only reduced visual exposure can be claimed."
        rows.append({
            "modality": modality,
            "status": "done",
            "num_samples": len(features),
            "feature_dim": max_dim,
            "split_mode": args.split_mode,
            "subject_train_acc": f"{train_acc:.6f}",
            "subject_test_acc": f"{test_acc:.6f}",
            "chance_acc": f"{chance:.6f}",
            "chance_percent": f"{chance * 100.0:.2f}",
            "privacy_interpretation": interpretation,
        })
        print(f"[{modality}] subject test acc={test_acc:.4f}, chance={chance:.4f}")

    write_csv(output_dir / "privacy_leakage_probe_subject_id.csv", rows)
    report = [
        "# Privacy Leakage Probe",
        "",
        "This lightweight probe evaluates whether subject identity can be predicted from each representation.",
        "It is a diagnostic privacy-leakage test, not a formal privacy guarantee.",
        "",
        "## Interpretation",
        "",
        "- Accuracy near chance suggests weak subject leakage under this probe.",
        "- Accuracy clearly above chance means the representation still carries identity-related cues.",
        "- Even low leakage does not prove anonymity; it only supports reduced visual exposure.",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        report.append(
            f"- {row['modality']}: status={row['status']}, split={row.get('split_mode', '')}, "
            f"subject_test_acc={row['subject_test_acc']}, chance={row['chance_acc']} "
            f"({row.get('chance_percent', '')}%). "
            f"{row['privacy_interpretation']}"
        )
    (output_dir / "privacy_leakage_probe_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Privacy leakage probe outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight subject-identity privacy leakage probe.")
    parser.add_argument("--dataset", required=True, help="Path to MM-Fi dataset root.")
    parser.add_argument("--rgb-root", default="", help="Optional defaced/raw RGB subset root with E/S/A/rgb/frameXXX.png layout.")
    parser.add_argument("--modalities", default="vk,depth,lidar,mmwave,wifi-csi")
    parser.add_argument("--max-per-subject", type=int, default=80)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--split-mode",
        choices=["action-holdout", "random"],
        default="action-holdout",
        help="action-holdout is stricter: train/test actions are disjoint within each subject.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
