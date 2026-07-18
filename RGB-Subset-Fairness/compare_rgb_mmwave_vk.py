from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


def iter_progress(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generated_path(root: Path, protocol: str, row: dict[str, str]) -> Path:
    frame_name = f"frame{int(row['idx']) + 1:03d}.npy"
    return root / protocol / row["scene"] / row["subject"] / row["action"] / frame_name


def load_vk(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    return arr.reshape(17, 2)


def bbox_diag(vk: np.ndarray) -> float:
    span = vk.max(axis=0) - vk.min(axis=0)
    return float(np.linalg.norm(span).clip(min=1e-6))


def sample_metrics(rgb_vk: np.ndarray, mmwave_vk: np.ndarray) -> dict[str, float]:
    diff = np.linalg.norm(rgb_vk - mmwave_vk, axis=-1)
    diag = bbox_diag(rgb_vk)
    shift = rgb_vk.mean(axis=0) - mmwave_vk.mean(axis=0)
    shifted = mmwave_vk + shift
    shifted_diff = np.linalg.norm(rgb_vk - shifted, axis=-1)
    return {
        "mse": float(np.mean((rgb_vk - mmwave_vk) ** 2)),
        "mae": float(np.mean(np.abs(rgb_vk - mmwave_vk))),
        "mean_joint_error": float(np.mean(diff)),
        "max_joint_error": float(np.max(diff)),
        "nme": float(np.mean(diff) / diag),
        "pck05": float(np.mean(diff <= 0.05 * diag)),
        "pck10": float(np.mean(diff <= 0.10 * diag)),
        "bbox_diag": diag,
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "shifted_mse": float(np.mean((rgb_vk - shifted) ** 2)),
        "shifted_mae": float(np.mean(np.abs(rgb_vk - shifted))),
        "shifted_mean_joint_error": float(np.mean(shifted_diff)),
        "shifted_nme": float(np.mean(shifted_diff) / diag),
        "shifted_pck05": float(np.mean(shifted_diff <= 0.05 * diag)),
        "shifted_pck10": float(np.mean(shifted_diff <= 0.10 * diag)),
    }


def summarize(values: list[float]) -> dict[str, str]:
    if not values:
        return {"mean": "NA", "std": "NA", "min": "NA", "median": "NA", "max": "NA"}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": f"{arr.mean():.6f}",
        "std": f"{arr.std():.6f}",
        "min": f"{arr.min():.6f}",
        "median": f"{np.median(arr):.6f}",
        "max": f"{arr.max():.6f}",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_examples(rows: list[dict[str, str]], generated_root: Path, protocol: str, output_dir: Path, count: int) -> None:
    if count <= 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[compare] matplotlib is not installed; skip example plots.")
        return

    plot_dir = output_dir / "examples" / protocol
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row in rows:
        if saved >= count:
            break
        gen_path = generated_path(generated_root, protocol, row)
        rgb_path = Path(row["vk_path"])
        if not gen_path.exists() or not rgb_path.exists():
            continue
        rgb_vk = load_vk(rgb_path)
        mm_vk = load_vk(gen_path)
        plt.figure(figsize=(4, 4))
        plt.scatter(rgb_vk[:, 0], rgb_vk[:, 1], c="tab:blue", label="RGB-derived VK", s=20)
        plt.scatter(mm_vk[:, 0], mm_vk[:, 1], c="tab:red", label="mmWave-generated VK", s=20, marker="x")
        for idx in range(17):
            plt.plot([rgb_vk[idx, 0], mm_vk[idx, 0]], [rgb_vk[idx, 1], mm_vk[idx, 1]], color="gray", linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title(f"{protocol} | {row['scene']}/{row['subject']}/{row['action']}/frame{int(row['idx']) + 1:03d}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        out_path = plot_dir / f"{saved + 1:03d}_{row['scene']}_{row['subject']}_{row['action']}_frame{int(row['idx']) + 1:03d}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        saved += 1
    print(f"[compare] saved example plots: {saved} -> {plot_dir}")


def compare_protocol(
    manifest_rows: list[dict[str, str]],
    generated_root: Path,
    protocol: str,
    output_dir: Path,
    max_samples: int | None,
    save_examples: int,
) -> dict[str, Any]:
    detail_rows: list[dict[str, Any]] = []
    metric_names = [
        "mse",
        "mae",
        "mean_joint_error",
        "max_joint_error",
        "nme",
        "pck05",
        "pck10",
        "bbox_diag",
        "shift_x",
        "shift_y",
        "shifted_mse",
        "shifted_mae",
        "shifted_mean_joint_error",
        "shifted_nme",
        "shifted_pck05",
        "shifted_pck10",
    ]
    missing_generated = 0
    missing_rgb = 0
    compared = 0
    rows = manifest_rows[: max_samples if max_samples is not None else len(manifest_rows)]
    for row in iter_progress(rows, f"compare:{protocol}"):
        rgb_path = Path(row["vk_path"])
        gen_path = generated_path(generated_root, protocol, row)
        if not rgb_path.exists():
            missing_rgb += 1
            continue
        if not gen_path.exists():
            missing_generated += 1
            continue
        rgb_vk = load_vk(rgb_path)
        mm_vk = load_vk(gen_path)
        metrics = sample_metrics(rgb_vk, mm_vk)
        compared += 1
        detail_rows.append(
            {
                "protocol": protocol,
                "scene": row["scene"],
                "subject": row["subject"],
                "action": row["action"],
                "idx": row["idx"],
                "rgb_vk_path": str(rgb_path),
                "mmwave_vk_path": str(gen_path),
                **{name: f"{metrics[name]:.8f}" for name in metric_names},
            }
        )

    fieldnames = [
        "protocol",
        "scene",
        "subject",
        "action",
        "idx",
        "rgb_vk_path",
        "mmwave_vk_path",
        *metric_names,
    ]
    write_csv(output_dir / f"{protocol}_rgb_vs_mmwave_vk_detail.csv", detail_rows, fieldnames)
    maybe_plot_examples(rows, generated_root, protocol, output_dir, save_examples)

    summary: dict[str, Any] = {
        "protocol": protocol,
        "manifest_samples": len(manifest_rows),
        "checked_samples": len(rows),
        "compared_samples": compared,
        "missing_rgb_vk": missing_rgb,
        "missing_mmwave_vk": missing_generated,
    }
    for name in metric_names:
        stats = summarize([float(row[name]) for row in detail_rows])
        for stat_name, value in stats.items():
            summary[f"{name}_{stat_name}"] = value
    return summary


def count_generated_files(generated_root: Path, protocol: str) -> int:
    root = generated_root / protocol
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("frame*.npy"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare RGB-derived VK and mmWave-generated VK on the same manifest frames.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/rgb_subset_manifest/manifest.csv"))
    parser.add_argument("--generated-root", type=Path, default=Path("RGB-Subset-Fairness/outputs_mmwave_vk/generated_vk"))
    parser.add_argument("--output-dir", type=Path, default=Path("RGB-Subset-Fairness/outputs_mmwave_vk/vk_compare"))
    parser.add_argument("--protocols", nargs="+", choices=["cross_scene", "cross_subject"], default=["cross_scene", "cross_subject"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    count_rows = []
    for protocol in args.protocols:
        generated_count = count_generated_files(args.generated_root, protocol)
        count_rows.append(
            {
                "protocol": protocol,
                "manifest_samples": len(rows),
                "generated_vk_files": generated_count,
                "count_matches_manifest": str(generated_count == len(rows)),
                "note": (
                    "The pipeline exports one generated VK file for each RGB-subset manifest row. "
                    "Sensor-rate differences are handled by the MMFi frame-level synchronization/indexing; "
                    "this script checks the resulting one-to-one frame count."
                ),
            }
        )
        summaries.append(compare_protocol(rows, args.generated_root, protocol, args.output_dir, args.max_samples, args.save_examples))

    summary_fields = sorted({key for row in summaries for key in row.keys()})
    write_csv(args.output_dir / "rgb_vs_mmwave_vk_summary.csv", summaries, summary_fields)
    write_csv(
        args.output_dir / "rgb_vs_mmwave_vk_count_check.csv",
        count_rows,
        ["protocol", "manifest_samples", "generated_vk_files", "count_matches_manifest", "note"],
    )
    print("Saved:", args.output_dir / "rgb_vs_mmwave_vk_summary.csv")
    print("Saved:", args.output_dir / "rgb_vs_mmwave_vk_count_check.csv")


if __name__ == "__main__":
    main()
