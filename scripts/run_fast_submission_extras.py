from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_dir(*candidates: str) -> Path:
    for name in candidates:
        path = PROJECT_ROOT / name
        if path.exists():
            return path
    return PROJECT_ROOT / candidates[0]


HPE_ROOT = resolve_project_dir("HPE", "MMFi_HPE")
HAR_ROOT = resolve_project_dir("HAR", "MMFi_HAR")


HPE_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
HAR_MODALITIES = ["vk", "depth", "lidar", "mmwave"]


def normalize_modality(name: str) -> str:
    value = name.strip().lower().replace("_", "-")
    aliases = {
        "v": "vk",
        "visual-keypoint": "vk",
        "visual keypoint": "vk",
        "d": "depth",
        "l": "lidar",
        "r": "mmwave",
        "w": "wifi-csi",
        "wifi": "wifi-csi",
        "wifi-csi": "wifi-csi",
        "csi": "wifi-csi",
    }
    return aliases.get(value, value)


def normalize_set(value: str) -> str:
    if not value:
        return ""
    parts = value.replace(",", "+").split("+")
    return "+".join(normalize_modality(part) for part in parts if part.strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def metric_column(rows: list[dict[str, str]], candidates: list[str]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def summarize_hpe_result(path: Path, full_set: str) -> tuple[float, float]:
    rows = read_csv(path)
    column = metric_column(rows, ["mpjpe", "xfi_mpjpe_mm"])
    if column is None:
        return float("nan"), float("nan")
    values = [safe_float(row.get(column)) for row in rows]
    if column == "mpjpe" and values and mean(values) < 10.0:
        values = [value * 1000.0 for value in values]
    avg_value = mean(values)
    full_value = float("nan")
    for row, value in zip(rows, values):
        row_set = normalize_set(row.get("modality_set", row.get("modality", "")))
        if row_set == full_set:
            full_value = value
            break
    return avg_value, full_value


def summarize_har_result(path: Path, full_set: str) -> tuple[float, float]:
    rows = read_csv(path)
    column = metric_column(rows, ["acc", "accuracy", "xfi_acc_pct"])
    if column is None:
        return float("nan"), float("nan")
    values = [safe_float(row.get(column)) for row in rows]
    if column in {"acc", "accuracy"} and values and mean(values) <= 1.0:
        values = [value * 100.0 for value in values]
    avg_value = mean(values)
    full_value = float("nan")
    for row, value in zip(rows, values):
        row_set = normalize_set(row.get("modality_set", row.get("modality", "")))
        if row_set == full_set:
            full_value = value
            break
    return avg_value, full_value


def fmt(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def result_path(project_root: Path, *parts: str) -> Path:
    return project_root.joinpath(*parts)


def build_fair_baseline_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task: str,
        method: str,
        source: Path | None,
        avg_metric: float,
        full_metric: float,
        metric_name: str,
        missing_training: str,
        adaptive_fusion: str,
        kd: str,
        conclusion: str,
    ) -> None:
        rows.append(
            {
                "task": task,
                "method": method,
                "missing_modality_training": missing_training,
                "adaptive_fusion": adaptive_fusion,
                "kd": kd,
                "avg_metric": fmt(avg_metric),
                "full_metric": fmt(full_metric),
                "metric_name": metric_name,
                "source_file": str(source) if source is not None else "manual_official_table",
                "main_conclusion": conclusion,
            }
        )

    hpe_full = "vk+depth+lidar+mmwave+wifi-csi"
    hpe_nv_full = "depth+lidar+mmwave+wifi-csi"
    official_hpe = result_path(HPE_ROOT, "legacy_xfi", "xfi_hpe_table1.csv")
    official_hpe_avg, official_hpe_full = summarize_hpe_result(official_hpe, hpe_full)
    if math.isnan(official_hpe_avg):
        official_hpe_avg, official_hpe_full = 103.70, 83.70
    add(
        "HPE",
        "Official X-Fi/VK table",
        official_hpe if official_hpe.exists() else None,
        official_hpe_avg,
        official_hpe_full,
        "MPJPE mm lower-better",
        "No",
        "X-fusion / reported baseline",
        "No",
        "External reference baseline; not counted as ours.",
    )
    for method, filename, full_set, missing_training, adaptive, kd, conclusion in [
        (
            "Full-modality baseline under missing test",
            "baseline_full_all_combinations.csv",
            hpe_full,
            "No",
            "Standard fusion",
            "No",
            "Shows full-modality training is brittle when modalities are missing.",
        ),
        (
            "Uniform-fusion ablation",
            "ablation_uniform_fusion_all_combinations.csv",
            hpe_full,
            "Yes",
            "No, uniform weights",
            "Yes",
            "Tests whether learned fusion is useful beyond missing-modality training.",
        ),
        (
            "w/o KD ablation",
            "ablation_no_distill_all_combinations.csv",
            hpe_full,
            "Yes",
            "Yes",
            "No",
            "Shows HPE gains are not solely from distillation.",
        ),
        (
            "VK-RMD Student-VK",
            "student_vk_random_all_combinations.csv",
            hpe_full,
            "Yes",
            "Yes",
            "Auxiliary",
            "Main deployable VK setting.",
        ),
        (
            "VK-RMD Student-NV",
            "student_nv_random_nonvisual_combinations.csv",
            hpe_nv_full,
            "Yes",
            "Yes",
            "Auxiliary",
            "Strict non-visual inference setting.",
        ),
    ]:
        path = result_path(HPE_ROOT, "outputs", "eval", filename)
        avg_value, full_value = summarize_hpe_result(path, full_set)
        add("HPE", method, path, avg_value, full_value, "MPJPE mm lower-better", missing_training, adaptive, kd, conclusion)

    har_full = "vk+depth+lidar+mmwave"
    har_nv_full = "depth+lidar+mmwave"
    official_har = result_path(HAR_ROOT, "legacy_xfi", "xfi_har_table6.csv")
    official_har_avg, official_har_full = summarize_har_result(official_har, har_full)
    if math.isnan(official_har_avg):
        reproduced = PROJECT_ROOT / "origin-XFI" / "Ori-HAR" / "outputs" / "main_eval_latest.csv"
        official_har_avg, official_har_full = summarize_har_result(reproduced, har_full)
        official_har = reproduced
    add(
        "HAR",
        "Official/reproduced X-Fi/VK",
        official_har if official_har.exists() else None,
        official_har_avg,
        official_har_full,
        "Accuracy % higher-better",
        "No",
        "X-fusion / reported baseline",
        "No",
        "External reference baseline; not counted as ours.",
    )
    for method, filename, full_set, missing_training, adaptive, kd, conclusion in [
        (
            "Full-modality teacher under missing test",
            "teacher_random_all_combinations.csv",
            har_full,
            "No",
            "Standard full-modality teacher",
            "No",
            "Shows full-modality teacher is not robust to missing modalities.",
        ),
        (
            "Uniform-fusion ablation",
            "ablation_uniform_fusion_all_combinations.csv",
            har_full,
            "Yes",
            "No, uniform weights",
            "Yes",
            "Tests reliability-aware fusion against equal weighting.",
        ),
        (
            "w/o KD ablation",
            "ablation_no_distill_all_combinations.csv",
            har_full,
            "Yes",
            "Yes",
            "No",
            "Shows distillation provides auxiliary benefit for HAR.",
        ),
        (
            "VK-RMD Student-VK",
            "student_vk_random_all_combinations.csv",
            har_full,
            "Yes",
            "Yes",
            "Auxiliary",
            "Main deployable VK setting.",
        ),
        (
            "VK-RMD Student-NV",
            "student_nv_random_nonvisual_combinations.csv",
            har_nv_full,
            "Yes",
            "Yes",
            "Auxiliary",
            "Strict non-visual inference setting.",
        ),
    ]:
        path = result_path(HAR_ROOT, "outputs", "eval", filename)
        avg_value, full_value = summarize_har_result(path, full_set)
        add("HAR", method, path, avg_value, full_value, "Accuracy % higher-better", missing_training, adaptive, kd, conclusion)
    return rows


def combo_parts(modality_set: str) -> list[str]:
    return [part for part in normalize_set(modality_set).split("+") if part]


def load_performance_rows(task: str, method: str, path: Path, full_modalities: list[str]) -> list[dict[str, Any]]:
    rows = read_csv(path)
    output = []
    if task == "HPE":
        column = metric_column(rows, ["mpjpe"])
        if column is None:
            return output
        for row in rows:
            value = safe_float(row.get(column))
            if value < 10:
                value *= 1000.0
            parts = combo_parts(row.get("modality_set", ""))
            output.append(
                {
                    "task": task,
                    "method": method,
                    "modality_set": "+".join(parts),
                    "metric": value,
                    "metric_name": "MPJPE mm",
                    "higher_better": False,
                    "missing_count": len(full_modalities) - len(parts),
                }
            )
    else:
        column = metric_column(rows, ["acc", "accuracy"])
        if column is None:
            return output
        for row in rows:
            value = safe_float(row.get(column))
            if value <= 1.0:
                value *= 100.0
            parts = combo_parts(row.get("modality_set", ""))
            output.append(
                {
                    "task": task,
                    "method": method,
                    "modality_set": "+".join(parts),
                    "metric": value,
                    "metric_name": "Accuracy %",
                    "higher_better": True,
                    "missing_count": len(full_modalities) - len(parts),
                }
            )
    return output


def build_subset_analyses() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    perf_rows = []
    perf_rows.extend(load_performance_rows("HPE", "Student-VK", result_path(HPE_ROOT, "outputs", "eval", "student_vk_random_all_combinations.csv"), HPE_MODALITIES))
    perf_rows.extend(load_performance_rows("HPE", "Student-NV", result_path(HPE_ROOT, "outputs", "eval", "student_nv_random_nonvisual_combinations.csv"), HPE_MODALITIES[1:]))
    perf_rows.extend(load_performance_rows("HAR", "Student-VK", result_path(HAR_ROOT, "outputs", "eval", "student_vk_random_all_combinations.csv"), HAR_MODALITIES))
    perf_rows.extend(load_performance_rows("HAR", "Student-NV", result_path(HAR_ROOT, "outputs", "eval", "student_nv_random_nonvisual_combinations.csv"), HAR_MODALITIES[1:]))

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in perf_rows:
        grouped[(row["task"], row["method"], int(row["missing_count"]))].append(row)
    severity_rows = []
    for (task, method, missing_count), items in sorted(grouped.items()):
        values = [float(item["metric"]) for item in items]
        higher_better = bool(items[0]["higher_better"])
        best = max(values) if higher_better else min(values)
        worst = min(values) if higher_better else max(values)
        severity_rows.append(
            {
                "task": task,
                "method": method,
                "missing_count": missing_count,
                "num_combinations": len(items),
                "avg_metric": fmt(mean(values)),
                "best_metric": fmt(best),
                "worst_metric": fmt(worst),
                "metric_name": items[0]["metric_name"],
                "higher_better": str(higher_better),
            }
        )

    leave_one_rows = []
    for task, method, full_modalities in [
        ("HPE", "Student-VK", HPE_MODALITIES),
        ("HAR", "Student-VK", HAR_MODALITIES),
    ]:
        items = [row for row in perf_rows if row["task"] == task and row["method"] == method]
        full_key = "+".join(full_modalities)
        full_item = next((row for row in items if row["modality_set"] == full_key), None)
        if full_item is None:
            continue
        full_metric = float(full_item["metric"])
        for modality in full_modalities:
            keep = [name for name in full_modalities if name != modality]
            keep_key = "+".join(keep)
            row = next((item for item in items if item["modality_set"] == keep_key), None)
            if row is None:
                continue
            metric = float(row["metric"])
            degradation = metric - full_metric if task == "HPE" else full_metric - metric
            leave_one_rows.append(
                {
                    "task": task,
                    "method": method,
                    "missing_modality": modality,
                    "full_modality_metric": fmt(full_metric),
                    "leave_one_out_metric": fmt(metric),
                    "degradation": fmt(degradation),
                    "metric_name": row["metric_name"],
                    "higher_better": str(row["higher_better"]),
                }
            )

    pattern_rows = []
    by_task_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in perf_rows:
        by_task_method[(row["task"], row["method"])].append(row)
    for (task, method), items in sorted(by_task_method.items()):
        higher_better = bool(items[0]["higher_better"])
        sorted_items = sorted(items, key=lambda item: float(item["metric"]), reverse=higher_better)
        for rank, item in enumerate(sorted_items[:3], start=1):
            pattern_rows.append(
                {
                    "task": task,
                    "method": method,
                    "pattern": "best",
                    "rank": rank,
                    "modality_set": item["modality_set"],
                    "metric": fmt(float(item["metric"])),
                    "metric_name": item["metric_name"],
                }
            )
        for rank, item in enumerate(reversed(sorted_items[-3:]), start=1):
            pattern_rows.append(
                {
                    "task": task,
                    "method": method,
                    "pattern": "worst",
                    "rank": rank,
                    "modality_set": item["modality_set"],
                    "metric": fmt(float(item["metric"])),
                    "metric_name": item["metric_name"],
                }
            )
    return severity_rows, leave_one_rows, pattern_rows


def rank_values(values: list[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(sorted_pairs):
        end = position + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[position][0]:
            end += 1
        avg_rank = (position + 1 + end) / 2.0
        for _, index in sorted_pairs[position:end]:
            ranks[index] = avg_rank
        position = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return float("nan")
    return numerator / (x_den * y_den)


def build_reliability_correlation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows = []
    summary_rows = []
    configs = [
        ("HPE", result_path(HPE_ROOT, "outputs", "eval", "student_vk_reliability_weights.csv"), result_path(HPE_ROOT, "outputs", "eval", "student_vk_random_all_combinations.csv"), "mpjpe", False),
        ("HAR", result_path(HAR_ROOT, "outputs", "eval", "student_vk_reliability_weights.csv"), result_path(HAR_ROOT, "outputs", "eval", "student_vk_random_all_combinations.csv"), "acc", True),
    ]
    for task, weight_path, perf_path, metric_col, higher_better in configs:
        weight_rows = read_csv(weight_path)
        perf_rows = read_csv(perf_path)
        if not weight_rows or not perf_rows:
            continue
        perf_by_set = {}
        for row in perf_rows:
            combo = normalize_set(row.get("modality_set", ""))
            value = safe_float(row.get(metric_col))
            if task == "HPE" and value < 10:
                value *= 1000.0
            if task == "HAR" and value <= 1.0:
                value *= 100.0
            perf_by_set[combo] = value

        grouped_weights: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in weight_rows:
            combo = normalize_set(row.get("modality_set", ""))
            modality = normalize_modality(row.get("modality", ""))
            if combo in perf_by_set and "+" in combo:
                grouped_weights[(combo, modality)].append(safe_float(row.get("weight")))

        modality_weight: dict[str, list[float]] = defaultdict(list)
        modality_score: dict[str, list[float]] = defaultdict(list)
        for (combo, modality), values in grouped_weights.items():
            weight = mean(values)
            metric = perf_by_set[combo]
            score = metric if higher_better else -metric
            modality_weight[modality].append(weight)
            modality_score[modality].append(score)
            detail_rows.append(
                {
                    "task": task,
                    "modality_set": combo,
                    "modality": modality,
                    "mean_weight": fmt(weight, 4),
                    "performance_metric": fmt(metric),
                    "metric_name": "Accuracy %" if task == "HAR" else "MPJPE mm",
                    "higher_better": str(higher_better),
                }
            )

        weights = []
        scores = []
        for modality in sorted(modality_weight):
            weights.append(mean(modality_weight[modality]))
            scores.append(mean(modality_score[modality]))
        if weights:
            spearman = pearson(rank_values(weights), rank_values(scores))
            linear = pearson(weights, scores)
            summary_rows.append(
                {
                    "task": task,
                    "num_modalities": len(weights),
                    "pearson_weight_vs_score": fmt(linear, 4),
                    "spearman_weight_vs_score": fmt(spearman, 4),
                    "interpretation": "Positive means higher learned weight tends to align with better task-level subset performance.",
                }
            )
    return detail_rows, summary_rows


def write_privacy_exposure_table(path: Path) -> None:
    rows = [
        {
            "representation": "RGB",
            "face_appearance": "High",
            "background": "High",
            "body_geometry": "High",
            "gait_action_leakage": "High",
            "formal_privacy_guarantee": "No",
            "our_claim": "Training privileged source only; not used by deployable student.",
        },
        {
            "representation": "VK",
            "face_appearance": "Low",
            "background": "Low",
            "body_geometry": "Medium/High",
            "gait_action_leakage": "Medium/High",
            "formal_privacy_guarantee": "No",
            "our_claim": "Reduced visual exposure structural bridge, not anonymity.",
        },
        {
            "representation": "Depth",
            "face_appearance": "Low/Medium",
            "background": "Medium",
            "body_geometry": "High",
            "gait_action_leakage": "Medium",
            "formal_privacy_guarantee": "No",
            "our_claim": "Non-RGB sensing modality.",
        },
        {
            "representation": "LiDAR/mmWave",
            "face_appearance": "Low",
            "background": "Low/Medium",
            "body_geometry": "Medium",
            "gait_action_leakage": "Medium",
            "formal_privacy_guarantee": "No",
            "our_claim": "Non-RGB sensing modality.",
        },
        {
            "representation": "WiFi-CSI",
            "face_appearance": "Very low visual exposure",
            "background": "Low visual exposure",
            "body_geometry": "Indirect",
            "gait_action_leakage": "Possible",
            "formal_privacy_guarantee": "No",
            "our_claim": "Non-visual sensing modality.",
        },
    ]
    write_csv(path, rows)


def maybe_plot_subset_curve(path: Path, severity_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional on servers
        print(f"[Plot] Skip subset curve because matplotlib is unavailable: {exc}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in severity_rows:
        try:
            grouped[(row["task"], row["method"])].append((int(row["missing_count"]), float(row["avg_metric"])))
        except ValueError:
            continue
    plt.figure(figsize=(8, 4.8))
    for (task, method), values in sorted(grouped.items()):
        values = sorted(values)
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        plt.plot(xs, ys, marker="o", label=f"{task} {method}")
    plt.xlabel("Number of missing modalities")
    plt.ylabel("MPJPE mm / Accuracy %")
    plt.title("Subset severity summary")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def maybe_plot_reliability_heatmap(path: Path, detail_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[Plot] Skip reliability heatmap because matplotlib is unavailable: {exc}")
        return
    if not detail_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the figure compact by plotting only full-modality rows and common multimodal rows.
    selected = [
        row
        for row in detail_rows
        if row["modality_set"] in {"vk+depth+lidar+mmwave+wifi-csi", "vk+depth+lidar+mmwave", "vk+depth", "depth+lidar", "vk+mmwave", "lidar+mmwave"}
    ]
    if not selected:
        selected = detail_rows[:40]
    labels = sorted({row["task"] + ":" + row["modality_set"] for row in selected})
    modalities = sorted({row["modality"] for row in selected})
    matrix = [[float("nan") for _ in modalities] for _ in labels]
    label_index = {label: idx for idx, label in enumerate(labels)}
    modality_index = {modality: idx for idx, modality in enumerate(modalities)}
    for row in selected:
        label = row["task"] + ":" + row["modality_set"]
        matrix[label_index[label]][modality_index[row["modality"]]] = safe_float(row["mean_weight"])
    plt.figure(figsize=(max(6, len(modalities) * 1.1), max(4, len(labels) * 0.45)))
    plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(label="Mean learned reliability proxy")
    plt.xticks(range(len(modalities)), modalities, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title("Reliability weights by subset")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_report(
    path: Path,
    fair_rows: list[dict[str, Any]],
    severity_rows: list[dict[str, Any]],
    leave_one_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    corruption_outputs: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fast Submission Extras Summary",
        "",
        "This report summarizes lightweight experiments and result reorganization for VK-RMD.",
        "",
        "## Fair Missing-Modality Baselines",
        "",
        "The fair-baseline table compares official/reproduced X-Fi, full-modality baselines under missing tests, uniform fusion, w/o KD, Student-VK, and Student-NV.",
        "",
    ]
    for row in fair_rows:
        lines.append(
            f"- {row['task']} | {row['method']} | avg={row['avg_metric']} | full={row['full_metric']} | {row['metric_name']}"
        )
    lines.extend(["", "## Subset Severity", ""])
    for row in severity_rows:
        if row["method"] == "Student-VK":
            lines.append(
                f"- {row['task']} missing={row['missing_count']} | avg={row['avg_metric']} | combos={row['num_combinations']} | {row['metric_name']}"
            )
    lines.extend(["", "## Leave-One-Modality Degradation", ""])
    for row in leave_one_rows:
        lines.append(
            f"- {row['task']} missing {row['missing_modality']}: degradation={row['degradation']} ({row['metric_name']})"
        )
    lines.extend(["", "## Reliability-Performance Correlation", ""])
    for row in corr_rows:
        lines.append(
            f"- {row['task']}: Pearson={row['pearson_weight_vs_score']}, Spearman={row['spearman_weight_vs_score']}"
        )
    lines.extend(["", "## Reliability Corruption Outputs", ""])
    if corruption_outputs:
        for output in corruption_outputs:
            lines.append(f"- {output}")
    else:
        lines.append("- Corruption evaluation was skipped or not completed.")
    lines.extend(
        [
            "",
            "## Paper Usage",
            "",
            "- Use the fair-baseline table to address the concern that VK-RMD only beats weak baselines.",
            "- Use subset severity and leave-one-out analyses to support subset-invariant missing-modality learning.",
            "- Use reliability corruption and correlation as evidence that reliability weights are learned task-level contribution proxies, not physical sensor calibration.",
            "- Keep privacy claims bounded to reduced visual exposure and RGB-free inference.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class RunRecord:
    index: int
    name: str
    status: str
    seconds: float
    outputs: str
    reason: str = ""


def write_status(path: Path, records: list[RunRecord]) -> None:
    rows = [
        {
            "index": record.index,
            "name": record.name,
            "status": record.status,
            "reason": record.reason,
            "seconds": f"{record.seconds:.3f}",
            "outputs": record.outputs,
        }
        for record in records
    ]
    write_csv(path, rows, ["index", "name", "status", "reason", "seconds", "outputs"])


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool = False) -> tuple[str, float, str]:
    print("=" * 88, flush=True)
    print(f"TASK-COMMAND | cwd={cwd}", flush=True)
    print(" ".join(command), flush=True)
    print(f"TASK-LOG | {log_path}", flush=True)
    print("=" * 88, flush=True)
    if dry_run:
        return "dry_run", 0.0, ""
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        process.wait()
    seconds = time.time() - start
    if process.returncode != 0:
        return "failed", seconds, f"returncode={process.returncode}; log={log_path}"
    return "done", seconds, ""


def checkpoint_path(project_root: Path, relative: str) -> Path:
    return project_root / relative


def existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def build_internal_corruption_command(args: argparse.Namespace, project: str, output_csv: Path) -> list[str]:
    project_root = HPE_ROOT if project == "HPE" else HAR_ROOT
    config = "configs/student_vk_missing.yaml"
    checkpoint = existing_path(
        [
            checkpoint_path(project_root, "outputs/student_vk_missing/best.pth"),
            checkpoint_path(project_root, "outputs/student_vk_missing/last.pth"),
        ]
    )
    if checkpoint is None:
        checkpoint = checkpoint_path(project_root, "outputs/student_vk_missing/best.pth")
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--internal-corruption",
        "--project",
        project,
        "--project-root",
        str(project_root),
        "--dataset",
        args.dataset,
        "--config",
        config,
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--corruption-severity",
        str(args.corruption_severity),
        "--output-csv",
        str(output_csv),
    ]


def orchestrate(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "outputs" / "fast_extras_runs" / timestamp
    log_dir = run_dir / "logs"
    tables_dir = PROJECT_ROOT / "tables"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = PROJECT_ROOT / "figures"
    records: list[RunRecord] = []
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": timestamp,
        "dataset": args.dataset,
        "device": args.device,
        "max_eval_batches": args.max_eval_batches,
        "hpe_root": str(HPE_ROOT),
        "har_root": str(HAR_ROOT),
        "description": "Fast non-training extras for VK-RMD paper submission.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    start = time.time()
    fair_rows = build_fair_baseline_table()
    fair_path = tables_dir / "fair_missing_modality_baseline.csv"
    write_csv(fair_path, fair_rows)
    records.append(RunRecord(1, "fair_missing_modality_baseline", "done", time.time() - start, str(fair_path)))
    write_status(run_dir / "status.csv", records)

    start = time.time()
    severity_rows, leave_one_rows, pattern_rows = build_subset_analyses()
    severity_path = tables_dir / "subset_severity_summary.csv"
    leave_one_path = tables_dir / "leave_one_modality_delta.csv"
    pattern_path = tables_dir / "subset_best_worst_patterns.csv"
    write_csv(severity_path, severity_rows)
    write_csv(leave_one_path, leave_one_rows)
    write_csv(pattern_path, pattern_rows)
    maybe_plot_subset_curve(figures_dir / "subset_severity_curve.png", severity_rows)
    records.append(
        RunRecord(
            2,
            "subset_complementarity_analysis",
            "done",
            time.time() - start,
            ";".join(str(path) for path in [severity_path, leave_one_path, pattern_path, figures_dir / "subset_severity_curve.png"]),
        )
    )
    write_status(run_dir / "status.csv", records)

    start = time.time()
    reliability_detail, reliability_summary = build_reliability_correlation()
    reliability_detail_path = tables_dir / "reliability_weight_by_subset.csv"
    reliability_summary_path = tables_dir / "reliability_performance_correlation.csv"
    write_csv(reliability_detail_path, reliability_detail)
    write_csv(reliability_summary_path, reliability_summary)
    maybe_plot_reliability_heatmap(figures_dir / "reliability_weight_heatmap.png", reliability_detail)
    records.append(
        RunRecord(
            3,
            "reliability_performance_correlation",
            "done",
            time.time() - start,
            ";".join(str(path) for path in [reliability_detail_path, reliability_summary_path, figures_dir / "reliability_weight_heatmap.png"]),
        )
    )
    write_status(run_dir / "status.csv", records)

    start = time.time()
    privacy_path = tables_dir / "privacy_exposure_residual_risk.csv"
    write_privacy_exposure_table(privacy_path)
    records.append(RunRecord(4, "privacy_exposure_table", "done", time.time() - start, str(privacy_path)))
    write_status(run_dir / "status.csv", records)

    corruption_outputs: list[Path] = []
    if not args.skip_corruption:
        for index, project in [(5, "HPE"), (6, "HAR")]:
            project_root = HPE_ROOT if project == "HPE" else HAR_ROOT
            output_csv = project_root / "outputs" / "eval" / f"reliability_corruption_{project.lower()}.csv"
            command = build_internal_corruption_command(args, project, output_csv)
            required_checkpoint = Path(command[command.index("--checkpoint") + 1])
            if not required_checkpoint.exists():
                records.append(
                    RunRecord(
                        index,
                        f"{project.lower()}_reliability_corruption",
                        "skipped",
                        0.0,
                        str(output_csv),
                        f"missing_checkpoint={required_checkpoint}",
                    )
                )
                write_status(run_dir / "status.csv", records)
                continue
            if output_csv.exists() and not args.force:
                records.append(
                    RunRecord(
                        index,
                        f"{project.lower()}_reliability_corruption",
                        "skipped",
                        0.0,
                        str(output_csv),
                        "outputs_exist",
                    )
                )
                corruption_outputs.append(output_csv)
                write_status(run_dir / "status.csv", records)
                continue
            status, seconds, reason = run_command(command, PROJECT_ROOT, log_dir / f"{project.lower()}_reliability_corruption.log", dry_run=args.dry_run)
            records.append(RunRecord(index, f"{project.lower()}_reliability_corruption", status, seconds, str(output_csv), reason))
            if output_csv.exists():
                corruption_outputs.append(output_csv)
            write_status(run_dir / "status.csv", records)

    report_path = reports_dir / "fast_extras_summary.md"
    write_report(report_path, fair_rows, severity_rows, leave_one_rows, reliability_summary, corruption_outputs)
    records.append(RunRecord(7, "summary_report", "done", 0.0, str(report_path)))
    write_status(run_dir / "status.csv", records)
    print(f"Fast extras run records: {run_dir}")
    print(f"Summary report: {report_path}")


def corrupt_tensor(tensor, modality: str, severity: float):
    import torch

    output = tensor.clone()
    severity = max(0.0, min(float(severity), 0.95))
    if severity <= 0:
        return output
    if modality == "vk":
        noise_std = 10.0 * severity
        output = output + torch.randn_like(output) * noise_std
        if output.dim() >= 3:
            joint_mask = torch.rand(output.shape[:2], device=output.device) < min(severity, 0.75)
            output[joint_mask] = 0.0
        else:
            output[torch.rand_like(output) < severity] = 0.0
        return output
    tensor_float = output.float()
    scale = torch.std(tensor_float).detach()
    if not torch.isfinite(scale) or float(scale) == 0.0:
        scale = torch.tensor(1.0, device=output.device)
    if modality in {"depth", "wifi-csi"}:
        output = output + torch.randn_like(output) * scale * (0.25 * severity)
    drop_mask = torch.rand(output.shape, device=output.device) < severity
    output[drop_mask] = 0.0
    return output


def average_alpha(output: dict[str, Any], modalities: list[str]) -> dict[str, float]:
    if "alphas" not in output:
        return {modality: float("nan") for modality in modalities}
    alphas = output["alphas"].detach().float().cpu()
    return {modality: float(alphas[:, idx].mean().item()) for idx, modality in enumerate(modalities)}


def evaluate_corruption_internal(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from _shared import build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
    from training.engine import move_batch_to_device

    namespace = argparse.Namespace(
        dataset=args.dataset,
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=None,
        resume=None,
        backbone_root=None,
        max_train_batches=None,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
    )
    config = prepare_config(namespace)
    device = get_device(args.device)
    print("=" * 88, flush=True)
    print("[Corruption] Launch reliability corruption evaluation", flush=True)
    print(f"[Corruption] project={args.project}", flush=True)
    print(f"[Corruption] project_root={project_root}", flush=True)
    print(f"[Corruption] dataset={args.dataset}", flush=True)
    print(f"[Corruption] config={args.config}", flush=True)
    print(f"[Corruption] checkpoint={args.checkpoint}", flush=True)
    print(f"[Corruption] device={args.device}", flush=True)
    print(f"[Corruption] max_eval_batches={args.max_eval_batches}", flush=True)
    print(f"[Corruption] severity={args.corruption_severity}", flush=True)
    print("=" * 88, flush=True)
    print("[Corruption] Building dataloaders...", flush=True)
    _, val_loader = build_dataloaders(args.dataset, config)
    print("[Corruption] Building model and loading checkpoint...", flush=True)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    model.eval()

    if args.project == "HPE":
        from utils.metrics import AverageMeter, compute_metrics

        modalities = HPE_MODALITIES
        meters_template = ["mse", "mpjpe", "pa_mpjpe"]

        def eval_once(corrupted_modality: str | None) -> tuple[dict[str, float], dict[str, float]]:
            stage_name = corrupted_modality or "clean"
            print(f"[Corruption] HPE stage start | {stage_name}", flush=True)
            meters = {name: AverageMeter() for name in meters_template}
            alpha_values = {modality: [] for modality in modalities}
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"hpe_corrupt:{stage_name}", dynamic_ncols=True)):
                    if args.max_eval_batches is not None and batch_idx >= int(args.max_eval_batches):
                        break
                    batch = move_batch_to_device(batch, device)
                    inputs = dict(batch["inputs"])
                    if corrupted_modality is not None:
                        inputs[corrupted_modality] = corrupt_tensor(inputs[corrupted_modality], corrupted_modality, args.corruption_severity)
                    output = model(inputs, modalities)
                    metrics = compute_metrics(output["pose"], batch["target"])
                    batch_size = batch["target"].size(0)
                    for key, value in metrics.items():
                        meters[key].update(value, batch_size)
                    alphas = average_alpha(output, modalities)
                    for modality, value in alphas.items():
                        alpha_values[modality].append(value)
            metric_values = {key: meter.avg for key, meter in meters.items()}
            alpha_means = {modality: mean(values) for modality, values in alpha_values.items()}
            print(
                "[Corruption] HPE stage done | "
                f"{stage_name} | mpjpe={metric_values['mpjpe']:.6f} | "
                f"weights={json.dumps(alpha_means, ensure_ascii=False)}",
                flush=True,
            )
            return metric_values, alpha_means

        clean_metrics, clean_alphas = eval_once(None)
        rows = []
        for index, modality in enumerate(modalities, start=1):
            print(f"[Corruption] HPE corrupt modality {index}/{len(modalities)} | {modality}", flush=True)
            corrupt_metrics, corrupt_alphas = eval_once(modality)
            rows.append(
                {
                    "task": "HPE",
                    "corrupted_modality": modality,
                    "corruption_type": "noise_and_random_zeroing",
                    "severity": args.corruption_severity,
                    "metric_name": "MPJPE",
                    "metric": corrupt_metrics["mpjpe"],
                    "clean_metric": clean_metrics["mpjpe"],
                    "delta_metric": corrupt_metrics["mpjpe"] - clean_metrics["mpjpe"],
                    "weight_clean": clean_alphas.get(modality, float("nan")),
                    "weight_corrupted": corrupt_alphas.get(modality, float("nan")),
                    "delta_weight": corrupt_alphas.get(modality, float("nan")) - clean_alphas.get(modality, float("nan")),
                    "other_weight_clean": mean([value for key, value in clean_alphas.items() if key != modality]),
                    "other_weight_corrupted": mean([value for key, value in corrupt_alphas.items() if key != modality]),
                }
            )
    else:
        from utils.metrics import AverageMeter, classification_metrics

        modalities = HAR_MODALITIES

        def eval_once(corrupted_modality: str | None) -> tuple[dict[str, float], dict[str, float]]:
            stage_name = corrupted_modality or "clean"
            print(f"[Corruption] HAR stage start | {stage_name}", flush=True)
            loss_meter = AverageMeter()
            correct = 0
            total = 0
            all_pred = []
            all_target = []
            alpha_values = {modality: [] for modality in modalities}
            num_classes = int(getattr(model, "num_classes", 27)) if hasattr(model, "num_classes") else 27
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"har_corrupt:{stage_name}", dynamic_ncols=True)):
                    if args.max_eval_batches is not None and batch_idx >= int(args.max_eval_batches):
                        break
                    batch = move_batch_to_device(batch, device)
                    inputs = dict(batch["inputs"])
                    if corrupted_modality is not None:
                        inputs[corrupted_modality] = corrupt_tensor(inputs[corrupted_modality], corrupted_modality, args.corruption_severity)
                    output = model(inputs, modalities)
                    logits = output["logits"]
                    loss = F.cross_entropy(logits, batch["target"])
                    batch_size = batch["target"].size(0)
                    loss_meter.update(float(loss.detach().cpu()), batch_size)
                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == batch["target"]).sum().item()
                    total += batch_size
                    all_pred.append(pred.detach().cpu())
                    all_target.append(batch["target"].detach().cpu())
                    alphas = average_alpha(output, modalities)
                    for modality, value in alphas.items():
                        alpha_values[modality].append(value)
            if total == 0:
                metric_values = {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}
            else:
                preds = torch.cat(all_pred)
                targets = torch.cat(all_target)
                f1 = classification_metrics(torch.nn.functional.one_hot(preds, num_classes=num_classes).float(), targets, num_classes)["macro_f1"]
                metric_values = {"loss": loss_meter.avg, "acc": correct / total, "macro_f1": f1}
            alpha_means = {modality: mean(values) for modality, values in alpha_values.items()}
            print(
                "[Corruption] HAR stage done | "
                f"{stage_name} | acc={metric_values['acc']:.6f} | macro_f1={metric_values['macro_f1']:.6f} | "
                f"weights={json.dumps(alpha_means, ensure_ascii=False)}",
                flush=True,
            )
            return metric_values, alpha_means

        clean_metrics, clean_alphas = eval_once(None)
        rows = []
        for index, modality in enumerate(modalities, start=1):
            print(f"[Corruption] HAR corrupt modality {index}/{len(modalities)} | {modality}", flush=True)
            corrupt_metrics, corrupt_alphas = eval_once(modality)
            rows.append(
                {
                    "task": "HAR",
                    "corrupted_modality": modality,
                    "corruption_type": "noise_and_random_zeroing",
                    "severity": args.corruption_severity,
                    "metric_name": "Accuracy",
                    "metric": corrupt_metrics["acc"],
                    "clean_metric": clean_metrics["acc"],
                    "delta_metric": corrupt_metrics["acc"] - clean_metrics["acc"],
                    "weight_clean": clean_alphas.get(modality, float("nan")),
                    "weight_corrupted": corrupt_alphas.get(modality, float("nan")),
                    "delta_weight": corrupt_alphas.get(modality, float("nan")) - clean_alphas.get(modality, float("nan")),
                    "other_weight_clean": mean([value for key, value in clean_alphas.items() if key != modality]),
                    "other_weight_corrupted": mean([value for key, value in corrupt_alphas.items() if key != modality]),
                }
            )

    write_csv(Path(args.output_csv), rows)
    print(f"[Corruption] Saved {len(rows)} rows to {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run lightweight VK-RMD submission extras without training.")
    parser.add_argument("--dataset", type=str, default="", help="MMFi dataset root; required unless --skip-corruption is used.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--corruption-severity", type=float, default=0.3)
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite corruption outputs if they already exist.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--internal-corruption", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project", choices=["HPE", "HAR"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-csv", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.internal_corruption:
        evaluate_corruption_internal(args)
        return
    if not args.dataset and not args.skip_corruption:
        raise SystemExit("--dataset is required unless --skip-corruption is set.")
    orchestrate(args)


if __name__ == "__main__":
    main()
