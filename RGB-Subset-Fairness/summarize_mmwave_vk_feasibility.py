from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rgb_subset_common import TRAIN_SUBJECTS, VAL_SUBJECTS, read_manifest


def first_existing(repo: Path, names: list[str]) -> Path:
    for name in names:
        path = repo / name
        if path.exists():
            return path
    return repo / names[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_generator_metrics(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if not rows:
        return {}
    return rows[0]


def manifest_counts(manifest_path: Path, protocol: str) -> tuple[int, int]:
    rows = read_manifest(manifest_path) if manifest_path.exists() else []
    if protocol == "cross_scene":
        return (
            sum(1 for row in rows if row["scene"] in {"E01", "E02", "E03"}),
            sum(1 for row in rows if row["scene"] == "E04"),
        )
    train_subjects = set(TRAIN_SUBJECTS)
    val_subjects = set(VAL_SUBJECTS)
    return (
        sum(1 for row in rows if row["subject"] in train_subjects),
        sum(1 for row in rows if row["subject"] in val_subjects),
    )


def summarize_hpe(rows: list[dict[str, str]], modality_key: str = "modality_set", metric_key: str = "mpjpe") -> tuple[str, str, str, str, int]:
    values: list[tuple[str, float]] = []
    for row in rows:
        try:
            values.append((row[modality_key], float(row[metric_key])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return "NA", "NA", "NA", "NA", 0
    avg = sum(value for _, value in values) / len(values)
    best = min(values, key=lambda item: item[1])
    worst = max(values, key=lambda item: item[1])
    full_names = {"vk+depth+lidar+mmwave+wifi-csi", "rgb+depth+lidar+mmwave+wifi-csi"}
    full = next((value for name, value in values if name.lower().replace(" ", "") in full_names), "NA")
    return f"{avg:.6f}", f"{full:.6f}" if isinstance(full, float) else "NA", best[0], worst[0], len(values)


def summarize_har(rows: list[dict[str, str]], modality_key: str = "modality_set", metric_key: str = "acc") -> tuple[str, str, str, str, int]:
    values: list[tuple[str, float]] = []
    for row in rows:
        try:
            values.append((row[modality_key], float(row[metric_key])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return "NA", "NA", "NA", "NA", 0
    avg = sum(value for _, value in values) / len(values)
    best = max(values, key=lambda item: item[1])
    worst = min(values, key=lambda item: item[1])
    full_names = {"vk+depth+lidar+mmwave", "rgb+depth+lidar+mmwave"}
    full = next((value for name, value in values if name.lower().replace(" ", "") in full_names), "NA")
    return f"{avg:.6f}", f"{full:.6f}" if isinstance(full, float) else "NA", best[0], worst[0], len(values)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "method",
        "visual_input",
        "protocol",
        "train_samples",
        "test_samples",
        "combination_count",
        "avg_metric",
        "full_metric",
        "best_subset",
        "worst_subset",
        "vk_generation_nme",
        "vk_generation_pck10",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_row(
    rows: list[dict[str, str]],
    *,
    task: str,
    method: str,
    visual_input: str,
    protocol: str,
    train_samples: int,
    test_samples: int,
    summary: tuple[str, str, str, str, int],
    generator_metrics: dict[str, str] | None = None,
    note: str,
) -> None:
    avg, full, best, worst, count = summary
    generator_metrics = generator_metrics or {}
    rows.append(
        {
            "task": task,
            "method": method,
            "visual_input": visual_input,
            "protocol": protocol,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "combination_count": count,
            "avg_metric": avg,
            "full_metric": full,
            "best_subset": best,
            "worst_subset": worst,
            "vk_generation_nme": generator_metrics.get("nme", "NA"),
            "vk_generation_pck10": generator_metrics.get("pck10", "NA"),
            "note": note,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser("Summarize mmWave-generated VK feasibility results.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo
    hpe_project = first_existing(repo, ["HPE", "MMFi_HPE"])
    har_project = first_existing(repo, ["HAR", "MMFi_HAR"])
    manifest = repo / "outputs" / "rgb_subset_manifest" / "manifest.csv"
    output_root = repo / "RGB-Subset-Fairness" / "outputs_mmwave_vk"

    hpe_rows: list[dict[str, str]] = []
    har_rows: list[dict[str, str]] = []
    for protocol in ["cross_scene", "cross_subject"]:
        train_count, test_count = manifest_counts(manifest, protocol)
        generator_metrics = read_generator_metrics(output_root / "generators" / protocol / "final_eval.csv")

        rgb_hpe = read_rows(repo / "Ori-RGB-HPE" / "outputs_rgb_subset" / protocol / "eval" / "main_eval_latest.csv")
        add_row(
            hpe_rows,
            task="HPE",
            method="X-Fi",
            visual_input="RGB/defaced-RGB",
            protocol=protocol,
            train_samples=train_count,
            test_samples=test_count,
            summary=summarize_hpe(rgb_hpe, "modality", "mpjpe"),
            note="Original RGB-XFi subset baseline when available; lower MPJPE is better.",
        )
        original_vk_hpe = read_rows(hpe_project / "outputs_rgb_subset" / "eval" / f"student_vk_{protocol}_all_combinations.csv")
        add_row(
            hpe_rows,
            task="HPE",
            method="VK-RMD",
            visual_input="Original VK",
            protocol=protocol,
            train_samples=train_count,
            test_samples=test_count,
            summary=summarize_hpe(original_vk_hpe),
            note="Original VK-RMD on the same RGB-available manifest.",
        )
        mmwave_vk_hpe = read_rows(hpe_project / "outputs_mmwave_vk" / "eval" / f"student_vk_{protocol}_all_combinations.csv")
        add_row(
            hpe_rows,
            task="HPE",
            method="VK-RMD-mmWaveVK",
            visual_input="mmWave-generated VK",
            protocol=protocol,
            train_samples=train_count,
            test_samples=test_count,
            summary=summarize_hpe(mmwave_vk_hpe),
            generator_metrics=generator_metrics,
            note="Quick feasibility: downstream VK is generated from filtered mmWave.",
        )

        rgb_har = read_rows(repo / "Ori-RGB-HAR" / "outputs_rgb_subset" / protocol / "eval" / "main_eval_latest.csv")
        add_row(
            har_rows,
            task="HAR",
            method="X-Fi",
            visual_input="RGB/defaced-RGB",
            protocol=protocol,
            train_samples=train_count,
            test_samples=test_count,
            summary=summarize_har(rgb_har, "modality", "accuracy"),
            note="Original RGB-XFi subset baseline when available; higher accuracy is better.",
        )
        original_vk_har = read_rows(har_project / "outputs_rgb_subset" / "eval" / f"student_vk_{protocol}_all_combinations.csv")
        add_row(
            har_rows,
            task="HAR",
            method="VK-RMD",
            visual_input="Original VK",
            protocol=protocol,
            train_samples=train_count,
            test_samples=test_count,
            summary=summarize_har(original_vk_har),
            note="Original VK-RMD on the same RGB-available manifest.",
        )
        mmwave_vk_har = read_rows(har_project / "outputs_mmwave_vk" / "eval" / f"student_vk_{protocol}_all_combinations.csv")
        add_row(
            har_rows,
            task="HAR",
            method="VK-RMD-mmWaveVK",
            visual_input="mmWave-generated VK",
            protocol=protocol,
            train_samples=train_count,
            test_samples=test_count,
            summary=summarize_har(mmwave_vk_har),
            generator_metrics=generator_metrics,
            note="Quick feasibility: downstream VK is generated from filtered mmWave.",
        )

    write_rows(repo / "tables" / "mmwave_vk_feasibility_hpe.csv", hpe_rows)
    write_rows(repo / "tables" / "mmwave_vk_feasibility_har.csv", har_rows)
    report = {
        "hpe_table": str(repo / "tables" / "mmwave_vk_feasibility_hpe.csv"),
        "har_table": str(repo / "tables" / "mmwave_vk_feasibility_har.csv"),
    }
    (repo / "tables" / "mmwave_vk_feasibility_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", report["hpe_table"])
    print("Saved:", report["har_table"])


if __name__ == "__main__":
    main()
