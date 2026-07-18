from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


MODALITY_COLORS = {
    "RGB": "#a23b72",
    "VK": "#2e86ab",
    "Depth": "#3ca370",
    "LiDAR": "#f18f01",
    "mmWave": "#c73e1d",
    "WiFi-CSI": "#6a4c93",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def add_box(ax, xy, width, height, text, facecolor="#ffffff", edgecolor="#222222", fontsize=10):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, start, end, color="#333333"):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=color)
    ax.add_patch(arrow)


def figure_framework(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)

    add_box(ax, (0.3, 3.45), 1.8, 0.85, "Training-only\nRGB", "#f6d6e6", MODALITY_COLORS["RGB"])
    add_box(ax, (0.3, 2.05), 1.8, 0.85, "Visual\nKeypoints", "#d9ecf5", MODALITY_COLORS["VK"])
    add_box(ax, (2.75, 2.75), 2.1, 1.05, "RGB-VK\nPrivileged Teacher", "#f2f2f2")
    add_box(ax, (5.35, 3.45), 1.6, 0.7, "Depth", "#dff2e6", MODALITY_COLORS["Depth"])
    add_box(ax, (5.35, 2.55), 1.6, 0.7, "LiDAR", "#fee8c8", MODALITY_COLORS["LiDAR"])
    add_box(ax, (5.35, 1.65), 1.6, 0.7, "mmWave", "#f7d3cc", MODALITY_COLORS["mmWave"])
    add_box(ax, (5.35, 0.75), 1.6, 0.7, "WiFi-CSI", "#e4dcf1", MODALITY_COLORS["WiFi-CSI"])
    add_box(ax, (7.45, 2.15), 2.0, 1.15, "Random Missing\nModality Student", "#ffffff")
    add_box(ax, (9.9, 2.15), 1.65, 1.15, "Reliability\nFusion", "#fff6d9")
    add_box(ax, (10.05, 3.75), 1.25, 0.75, "HPE", "#e8f4ff")
    add_box(ax, (10.05, 0.95), 1.25, 0.75, "HAR", "#eaf7ea")

    add_arrow(ax, (2.1, 3.88), (2.75, 3.45), MODALITY_COLORS["RGB"])
    add_arrow(ax, (2.1, 2.48), (2.75, 3.05), MODALITY_COLORS["VK"])
    add_arrow(ax, (4.85, 3.25), (7.45, 2.95), "#666666")
    for y in [3.8, 2.9, 2.0, 1.1]:
        add_arrow(ax, (6.95, y), (7.45, 2.72), "#666666")
    add_arrow(ax, (9.45, 2.72), (9.9, 2.72), "#333333")
    add_arrow(ax, (10.75, 3.3), (10.7, 3.75), "#333333")
    add_arrow(ax, (10.75, 2.15), (10.7, 1.7), "#333333")

    ax.text(0.35, 4.85, "Training phase", fontsize=11, fontweight="bold")
    ax.text(5.35, 4.85, "Deployment phase: no raw RGB", fontsize=11, fontweight="bold")
    ax.text(7.55, 1.68, "Available sensors change per sample", fontsize=9, color="#555555")
    save(fig, output_dir / "fig1_overall_framework.png")


def figure_missing_setting(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.8)

    modalities = ["VK", "Depth", "LiDAR", "mmWave", "WiFi-CSI"]
    patterns = [
        ("Full", [1, 1, 1, 1, 1]),
        ("Drop 1", [1, 1, 0, 1, 1]),
        ("Drop 2", [1, 0, 1, 0, 1]),
        ("Drop 3", [0, 1, 0, 1, 0]),
        ("Non-visual", [0, 1, 1, 1, 1]),
        ("Single sensor", [0, 0, 0, 1, 0]),
    ]
    cell_w, cell_h = 1.25, 0.45
    x0, y0 = 2.8, 3.7
    for j, name in enumerate(modalities):
        ax.text(x0 + j * cell_w + cell_w / 2, y0 + 0.45, name, ha="center", va="center", fontsize=9)
    for i, (pattern, mask) in enumerate(patterns):
        y = y0 - i * 0.62
        ax.text(0.45, y + cell_h / 2, pattern, ha="left", va="center", fontsize=10)
        for j, active in enumerate(mask):
            color = "#2b8a3e" if active else "#d9d9d9"
            edge = "#1f5f2a" if active else "#999999"
            rect = Rectangle((x0 + j * cell_w, y), cell_w * 0.86, cell_h, facecolor=color, edgecolor=edge, linewidth=1.0)
            ax.add_patch(rect)
            ax.text(x0 + j * cell_w + cell_w * 0.43, y + cell_h / 2, "on" if active else "off", ha="center", va="center", fontsize=8, color="white" if active else "#333333")
    ax.text(0.45, 4.45, "Missing-modality protocol", fontsize=13, fontweight="bold")
    ax.text(0.45, 0.25, "Training randomly samples modality subsets; evaluation reports all valid combinations.", fontsize=10, color="#444444")
    save(fig, output_dir / "fig2_missing_modality_setting.png")


def figure_main_results(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    hpe_names = ["Official X-Fi/VK\ntable", "Student-VK", "Student-NV"]
    hpe_vals = [103.70, 75.18, 84.68]
    har_names = ["X-Fi/VK\nbaseline", "Student-VK", "Student-NV"]
    har_vals = [69.63, 85.15, 80.62]
    axes[0].bar(hpe_names, hpe_vals, color=["#8c8c8c", "#2e86ab", "#3ca370"])
    axes[0].set_ylabel("Average MPJPE (mm, lower is better)")
    axes[0].set_title("HPE all-combination average")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, val in enumerate(hpe_vals):
        axes[0].text(idx, val + 5, f"{val:.1f}", ha="center", fontsize=9)
    axes[1].bar(har_names, har_vals, color=["#8c8c8c", "#2e86ab", "#3ca370"])
    axes[1].set_ylabel("Average accuracy (%, higher is better)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("HAR all-combination average")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, val in enumerate(har_vals):
        axes[1].text(idx, val + 2, f"{val:.1f}", ha="center", fontsize=9)
    fig.suptitle("Main HPE/HAR Results Under Arbitrary Modality Combinations", fontsize=13, fontweight="bold")
    save(fig, output_dir / "fig3_main_results.png")


def figure_robustness(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    hpe_x_vk = [0, 1, 2, 3, 4]
    hpe_y_vk = [50.17, 53.54, 60.78, 78.11, 124.73]
    hpe_x_nv = [0, 1, 2, 3]
    hpe_y_nv = [51.03, 61.32, 79.43, 124.32]
    har_x_vk = [0, 1, 2, 3]
    har_y_vk = [96.56, 93.83, 87.55, 70.00]
    har_x_nv = [0, 1, 2]
    har_y_nv = [96.12, 90.05, 66.02]
    axes[0].plot(hpe_x_vk, hpe_y_vk, marker="o", linewidth=2, label="HPE Student-VK", color="#2e86ab")
    axes[0].plot(hpe_x_nv, hpe_y_nv, marker="s", linewidth=2, label="HPE Student-NV", color="#3ca370")
    axes[0].set_xlabel("Number of missing modalities")
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title("HPE robustness")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(har_x_vk, har_y_vk, marker="o", linewidth=2, label="HAR Student-VK", color="#2e86ab")
    axes[1].plot(har_x_nv, har_y_nv, marker="s", linewidth=2, label="HAR Student-NV", color="#3ca370")
    axes[1].set_xlabel("Number of missing modalities")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(50, 100)
    axes[1].set_title("HAR robustness")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Performance Degradation as Modalities Become Missing", fontsize=13, fontweight="bold")
    save(fig, output_dir / "fig4_missing_modality_robustness.png")


def figure_ablation(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    hpe_labels = ["VK-RMD", "w/o KD", "Uniform\nfusion"]
    hpe_vals = [75.18, 72.68, 82.77]
    har_labels = ["VK-RMD", "w/o KD", "Uniform\nfusion"]
    har_vals = [85.15, 82.86, 67.46]
    axes[0].bar(hpe_labels, hpe_vals, color=["#2e86ab", "#9aa0a6", "#f18f01"])
    axes[0].set_ylabel("Average MPJPE (mm)")
    axes[0].set_title("HPE ablation")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, val in enumerate(hpe_vals):
        axes[0].text(idx, val + 1.5, f"{val:.1f}", ha="center", fontsize=9)
    axes[1].bar(har_labels, har_vals, color=["#2e86ab", "#9aa0a6", "#f18f01"])
    axes[1].set_ylabel("Average accuracy (%)")
    axes[1].set_ylim(50, 95)
    axes[1].set_title("HAR ablation")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, val in enumerate(har_vals):
        axes[1].text(idx, val + 1.0, f"{val:.1f}", ha="center", fontsize=9)
    fig.suptitle("Ablation Evidence for Reliability-Guided Fusion", fontsize=13, fontweight="bold")
    save(fig, output_dir / "fig5_ablation.png")


def read_reliability_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def figure_reliability(output_dir: Path, reliability_paths: Iterable[Path]) -> None:
    rows = read_reliability_rows(reliability_paths)
    if not rows:
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        ax.set_axis_off()
        ax.text(0.5, 0.72, "Reliability weights are available in model output as 'alphas'.", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(0.5, 0.50, "Run export_reliability_weights.py on the trained checkpoint to create CSV rows:", ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.35, "sample_id, missing_setting, modality_set, modality, weight, label", ha="center", va="center", fontsize=10, family="monospace")
        ax.text(0.5, 0.18, "The figure will become a heatmap after the CSV is available.", ha="center", va="center", fontsize=10, color="#555555")
        save(fig, output_dir / "fig6_reliability_weights.png")
        return

    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    modality_order: List[str] = []
    combo_order: List[str] = []
    for row in rows:
        combo = row.get("modality_set", "")
        modality = row.get("modality", "")
        try:
            weight = float(row.get("weight", "nan"))
        except ValueError:
            continue
        grouped[(combo, modality)].append(weight)
        if combo and combo not in combo_order:
            combo_order.append(combo)
        if modality and modality not in modality_order:
            modality_order.append(modality)
    combo_order = combo_order[:12]
    data = []
    for combo in combo_order:
        data.append([
            sum(grouped[(combo, modality)]) / len(grouped[(combo, modality)]) if grouped[(combo, modality)] else 0.0
            for modality in modality_order
        ])
    fig, ax = plt.subplots(figsize=(11.0, max(4.0, len(combo_order) * 0.35)))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(modality_order)), modality_order, rotation=25, ha="right")
    ax.set_yticks(range(len(combo_order)), combo_order)
    ax.set_title("Learned Reliability Weights by Modality Combination", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Average reliability weight")
    save(fig, output_dir / "fig6_reliability_weights.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper support figures for VK-RMD.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reliability-csv", action="append", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "figures"
    ensure_dir(output_dir)
    reliability_paths = [Path(path) for path in (args.reliability_csv or [])]
    if not reliability_paths:
        reliability_paths = [
            root / "HPE" / "outputs" / "eval" / "student_vk_reliability_weights.csv",
            root / "HAR" / "outputs" / "eval" / "student_vk_reliability_weights.csv",
        ]

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    figure_framework(output_dir)
    figure_missing_setting(output_dir)
    figure_main_results(output_dir)
    figure_robustness(output_dir)
    figure_ablation(output_dir)
    figure_reliability(output_dir, reliability_paths)


if __name__ == "__main__":
    main()
