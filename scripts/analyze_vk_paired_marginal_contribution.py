from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPE_CSV = ROOT / "HPE" / "outputs" / "eval" / "student_vk_random_all_combinations.csv"
DEFAULT_HAR_CSV = ROOT / "HAR" / "outputs" / "eval" / "student_vk_random_all_combinations.csv"
DEFAULT_OUTPUT_DIR = ROOT / "supplement" / "vk_paired_marginal_contribution"

MODALITY_ORDER = ("vk", "depth", "lidar", "mmwave", "wifi-csi")
MODALITY_LABELS = {
    "vk": "V",
    "depth": "D",
    "lidar": "L",
    "mmwave": "R",
    "wifi-csi": "W",
}
ALIASES = {
    "v": "vk",
    "vk": "vk",
    "visual-keypoints": "vk",
    "visual_keypoints": "vk",
    "d": "depth",
    "depth": "depth",
    "l": "lidar",
    "lidar": "lidar",
    "r": "mmwave",
    "radar": "mmwave",
    "mmwave": "mmwave",
    "mm-wave": "mmwave",
    "w": "wifi-csi",
    "wifi": "wifi-csi",
    "wifi-csi": "wifi-csi",
    "wifi_csi": "wifi-csi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the paired marginal contribution of VK across matched modality subsets."
    )
    parser.add_argument("--hpe-csv", type=Path, default=DEFAULT_HPE_CSV)
    parser.add_argument("--har-csv", type=Path, default=DEFAULT_HAR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return rows


def canonical_subset(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower().replace(",", "+").replace(" ", "")
    items = [item for item in normalized.split("+") if item]
    canonical: list[str] = []
    for item in items:
        if item not in ALIASES:
            raise ValueError(f"Unknown modality name '{item}' in '{value}'")
        modality = ALIASES[item]
        if modality not in canonical:
            canonical.append(modality)
    return tuple(name for name in MODALITY_ORDER if name in canonical)


def subset_label(subset: Iterable[str]) -> str:
    return "+".join(MODALITY_LABELS[name] for name in subset)


def metric_value(row: dict[str, str], key: str, task: str) -> float:
    value = float(row[key])
    if task == "HPE":
        return value * 1000.0 if abs(value) < 10.0 else value
    return value * 100.0 if abs(value) <= 1.0 else value


def build_index(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        subset = canonical_subset(row["modality_set"])
        if subset in index:
            raise ValueError(f"Duplicate modality set: {row['modality_set']}")
        index[subset] = row
    return index


def validate_complete_power_set(
    task: str,
    index: dict[tuple[str, ...], dict[str, str]],
    non_vk_modalities: tuple[str, ...],
) -> None:
    expected_non_vk = {
        tuple(combo)
        for size in range(1, len(non_vk_modalities) + 1)
        for combo in itertools.combinations(non_vk_modalities, size)
    }
    observed_non_vk = {subset for subset in index if "vk" not in subset}
    missing = expected_non_vk - observed_non_vk
    unexpected = observed_non_vk - expected_non_vk
    if missing or unexpected:
        raise ValueError(
            f"Incomplete {task} non-VK power set | "
            f"missing={[subset_label(item) for item in sorted(missing)]} | "
            f"unexpected={[subset_label(item) for item in sorted(unexpected)]}"
        )


def paired_rows(
    task: str,
    rows: list[dict[str, str]],
    metrics: tuple[tuple[str, str, str], ...],
) -> list[dict[str, object]]:
    index = build_index(rows)
    non_vk_modalities = tuple(
        name for name in MODALITY_ORDER if name != "vk" and any(name in subset for subset in index)
    )
    validate_complete_power_set(task, index, non_vk_modalities)
    result: list[dict[str, object]] = []
    non_vk_subsets = sorted(
        (subset for subset in index if "vk" not in subset),
        key=lambda subset: (len(subset), [MODALITY_ORDER.index(name) for name in subset]),
    )
    for subset in non_vk_subsets:
        with_vk = tuple(name for name in MODALITY_ORDER if name == "vk" or name in subset)
        if with_vk not in index:
            raise ValueError(
                f"Missing paired row for {task}: {subset_label(subset)} + VK ({with_vk})"
            )
        base_row = index[subset]
        vk_row = index[with_vk]
        for metric_key, metric_name, unit in metrics:
            if metric_key not in base_row or metric_key not in vk_row:
                raise ValueError(f"Missing metric '{metric_key}' in {task} CSV")
            without_vk = metric_value(base_row, metric_key, task)
            with_vk_value = metric_value(vk_row, metric_key, task)
            if task == "HPE":
                gain = without_vk - with_vk_value
                relative_gain = 100.0 * gain / max(abs(without_vk), 1e-12)
            else:
                gain = with_vk_value - without_vk
                relative_gain = 100.0 * gain / max(abs(without_vk), 1e-12)
            result.append(
                {
                    "task": task,
                    "base_subset": "+".join(subset),
                    "base_subset_label": subset_label(subset),
                    "with_vk_subset": "+".join(with_vk),
                    "with_vk_subset_label": subset_label(with_vk),
                    "base_modality_count": len(subset),
                    "metric": metric_name,
                    "unit": unit,
                    "without_vk": without_vk,
                    "with_vk": with_vk_value,
                    "vk_gain": gain,
                    "relative_gain_percent": relative_gain,
                    "vk_improves": "yes" if gain > 0 else "no",
                }
            )
    return result


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["metric"]), str(row["unit"]))].append(row)
    output: list[dict[str, object]] = []
    for (task, metric, unit), group in grouped.items():
        gains = [float(row["vk_gain"]) for row in group]
        nonzero = [gain for gain in gains if abs(gain) > 1e-12]
        wins = sum(gain > 0 for gain in nonzero)
        losses = sum(gain < 0 for gain in nonzero)
        output.append(
            {
                "task": task,
                "metric": metric,
                "unit": unit,
                "pair_count": len(group),
                "win_count": wins,
                "tie_count": len(gains) - len(nonzero),
                "loss_count": losses,
                "win_rate_percent": 100.0 * wins / max(len(nonzero), 1),
                "mean_vk_gain": statistics.fmean(gains),
                "median_vk_gain": statistics.median(gains),
                "min_vk_gain": min(gains),
                "max_vk_gain": max(gains),
                "exact_sign_consistency_p": exact_two_sided_sign_p(wins, losses),
            }
        )
    return output


def summarize_by_size(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["task"]),
                str(row["metric"]),
                str(row["unit"]),
                int(row["base_modality_count"]),
            )
        ].append(float(row["vk_gain"]))
    output: list[dict[str, object]] = []
    for (task, metric, unit, size), gains in sorted(grouped.items()):
        output.append(
            {
                "task": task,
                "metric": metric,
                "unit": unit,
                "base_modality_count": size,
                "pair_count": len(gains),
                "win_count": sum(gain > 0 for gain in gains),
                "mean_vk_gain": statistics.fmean(gains),
                "median_vk_gain": statistics.median(gains),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def primary_rows(rows: list[dict[str, object]], task: str, metric: str) -> list[dict[str, object]]:
    return [row for row in rows if row["task"] == task and row["metric"] == metric]


def plot_primary(rows: list[dict[str, object]], output_dir: Path) -> None:
    panels = [
        ("HPE", "MPJPE", "MPJPE reduction after adding VK (mm)"),
        ("HAR", "Accuracy", "Accuracy gain after adding VK (percentage points)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), constrained_layout=True)
    colors = {1: "#4C78A8", 2: "#59A14F", 3: "#F28E2B", 4: "#B279A2"}
    for ax, (task, metric, xlabel) in zip(axes, panels):
        panel_rows = primary_rows(rows, task, metric)
        panel_rows.sort(key=lambda row: (int(row["base_modality_count"]), str(row["base_subset_label"])))
        labels = [str(row["base_subset_label"]) for row in panel_rows]
        gains = [float(row["vk_gain"]) for row in panel_rows]
        bar_colors = [colors[int(row["base_modality_count"])] for row in panel_rows]
        y = list(range(len(panel_rows)))
        ax.barh(y, gains, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0.0, color="#444444", linewidth=0.9)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(f"{task}: paired marginal contribution of VK", fontweight="bold")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        span = max(max(abs(value) for value in gains), 1e-6)
        for yi, value in zip(y, gains):
            offset = 0.012 * span
            ax.text(
                value + (offset if value >= 0 else -offset),
                yi,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
    fig.suptitle(
        "Paired comparison of each non-VK subset S against the matched subset S+VK",
        fontsize=12,
        fontweight="bold",
    )
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(output_dir / f"fig_vk_paired_marginal_contribution.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: list[dict[str, object]],
    hpe_csv: Path,
    har_csv: Path,
) -> None:
    lookup = {(row["task"], row["metric"]): row for row in summary}
    hpe = lookup[("HPE", "MPJPE")]
    har = lookup[("HAR", "Accuracy")]
    report = f"""# Paired Marginal Contribution of VK

## Definition

For every non-VK modality subset `S`, this analysis compares the same trained model under `S` and the exactly matched input set `S+VK`. Positive gain always means that adding VK improves the task metric.

## Main result

- HPE MPJPE: VK improves {hpe['win_count']}/{hpe['pair_count']} matched subsets; mean reduction = {float(hpe['mean_vk_gain']):.2f} mm, median reduction = {float(hpe['median_vk_gain']):.2f} mm.
- HAR Accuracy: VK improves {har['win_count']}/{har['pair_count']} matched subsets; mean gain = {float(har['mean_vk_gain']):.2f} percentage points, median gain = {float(har['median_vk_gain']):.2f} percentage points.

## Interpretation

The paired design controls the identity of the non-VK sensor subset: the only input-set difference within each pair is the addition of VK. Therefore, it is stronger evidence for the conditional task value of VK than comparing unrelated modality combinations or reporting an all-combination average.

The result does not prove that VK is a unique semantic center, that the pairs are statistically independent, or that VK provides formal privacy protection. The exact sign-consistency value in the summary CSV is descriptive because modality subsets share sensors and model parameters.

## Inputs

- HPE: `{hpe_csv}`
- HAR: `{har_csv}`
"""
    (output_dir / "vk_paired_marginal_contribution_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hpe_rows = paired_rows(
        "HPE",
        read_csv(args.hpe_csv),
        (("mpjpe", "MPJPE", "mm"), ("pa_mpjpe", "PA-MPJPE", "mm")),
    )
    har_rows = paired_rows(
        "HAR",
        read_csv(args.har_csv),
        (("acc", "Accuracy", "percentage_points"), ("macro_f1", "Macro-F1", "percentage_points")),
    )
    all_rows = hpe_rows + har_rows
    summary = summarize(all_rows)
    by_size = summarize_by_size(all_rows)

    write_csv(output_dir / "vk_paired_marginal_pairs.csv", all_rows)
    write_csv(output_dir / "vk_paired_marginal_summary.csv", summary)
    write_csv(output_dir / "vk_paired_marginal_by_subset_size.csv", by_size)
    plot_primary(all_rows, output_dir)
    write_report(output_dir, summary, args.hpe_csv.resolve(), args.har_csv.resolve())

    for row in summary:
        print(
            f"[{row['task']}] {row['metric']} | pairs={row['pair_count']} | "
            f"wins={row['win_count']} | losses={row['loss_count']} | "
            f"mean_gain={float(row['mean_vk_gain']):.4f} {row['unit']}"
        )
    print(f"Saved paired VK analysis to: {output_dir}")


if __name__ == "__main__":
    main()
