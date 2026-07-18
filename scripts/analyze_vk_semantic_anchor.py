from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "supplement" / "vk_semantic_anchor_summary.csv"
OUT_DIR = ROOT / "supplement" / "vk_semantic_anchor_analysis"


LABELS = {
    "vk_noise_0.1": "VK noise\n0.1",
    "vk_noise_0.3": "VK noise\n0.3",
    "vk_noise_0.5": "VK noise\n0.5",
    "depth_noise_0.3": "Depth\nnoise",
    "lidar_noise_0.3": "LiDAR\nnoise",
    "mmwave_noise_0.3": "mmWave\nnoise",
    "wifi_csi_noise_0.3": "WiFi-CSI\nnoise",
    "vk_joint_shuffle": "VK joint\nshuffle",
    "vk_sample_mismatch": "VK sample\nmismatch",
}

COLORS = {
    "vk": "#4C78A8",
    "depth": "#F58518",
    "lidar": "#54A24B",
    "mmwave": "#B279A2",
    "wifi-csi": "#72B7B2",
    "none": "#9D9D9D",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def by_task(rows: list[dict[str, str]], task: str) -> dict[str, dict[str, str]]:
    return {row["setting"]: row for row in rows if row["task"] == task}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def hpe_delta_mm(row: dict[str, str]) -> float:
    return f(row, "delta_metric") * 1000.0


def har_acc_drop_pp(row: dict[str, str]) -> float:
    return -f(row, "delta_metric") * 100.0


def build_interpretation_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if row["setting"] == "clean":
            continue
        if row["task"] == "HPE":
            degradation = hpe_delta_mm(row)
            unit = "mm MPJPE increase"
            direction = "worse" if degradation > 0 else "better_or_no_worse"
        else:
            degradation = har_acc_drop_pp(row)
            unit = "percentage-point accuracy drop"
            direction = "worse" if degradation > 0 else "better_or_no_worse"
        output.append(
            {
                "task": row["task"],
                "setting": row["setting"],
                "perturbed_modality": row["perturbed_modality"],
                "perturbation_type": row["perturbation_type"],
                "degradation": f"{degradation:.4f}",
                "unit": unit,
                "interpretation": direction,
            }
        )
    return output


def key_numbers(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    hpe = by_task(rows, "HPE")
    har = by_task(rows, "HAR")
    entries = [
        ("HPE", "clean", "Clean MPJPE", f(hpe["clean"], "metric_value") * 1000.0, "mm"),
        ("HPE", "vk_joint_shuffle", "VK joint shuffle degradation", hpe_delta_mm(hpe["vk_joint_shuffle"]), "mm"),
        ("HPE", "vk_noise_0.3", "VK coordinate noise 0.3 degradation", hpe_delta_mm(hpe["vk_noise_0.3"]), "mm"),
        ("HPE", "depth_noise_0.3", "Depth noise 0.3 degradation", hpe_delta_mm(hpe["depth_noise_0.3"]), "mm"),
        ("HPE", "lidar_noise_0.3", "LiDAR noise 0.3 degradation", hpe_delta_mm(hpe["lidar_noise_0.3"]), "mm"),
        ("HAR", "clean", "Clean accuracy", f(har["clean"], "metric_value") * 100.0, "%"),
        ("HAR", "vk_joint_shuffle", "VK joint shuffle degradation", har_acc_drop_pp(har["vk_joint_shuffle"]), "pp"),
        ("HAR", "vk_noise_0.3", "VK coordinate noise 0.3 degradation", har_acc_drop_pp(har["vk_noise_0.3"]), "pp"),
        ("HAR", "depth_noise_0.3", "Depth noise 0.3 degradation", har_acc_drop_pp(har["depth_noise_0.3"]), "pp"),
        ("HAR", "mmwave_noise_0.3", "mmWave noise 0.3 degradation", har_acc_drop_pp(har["mmwave_noise_0.3"]), "pp"),
    ]
    return [
        {
            "task": task,
            "setting": setting,
            "quantity": quantity,
            "value": f"{value:.4f}",
            "unit": unit,
        }
        for task, setting, quantity, value, unit in entries
    ]


def plot(rows: list[dict[str, str]]) -> None:
    hpe = by_task(rows, "HPE")
    har = by_task(rows, "HAR")
    order_hpe = [
        "vk_noise_0.1",
        "vk_noise_0.3",
        "vk_noise_0.5",
        "depth_noise_0.3",
        "lidar_noise_0.3",
        "mmwave_noise_0.3",
        "wifi_csi_noise_0.3",
        "vk_joint_shuffle",
        "vk_sample_mismatch",
    ]
    order_har = [
        "vk_noise_0.1",
        "vk_noise_0.3",
        "vk_noise_0.5",
        "depth_noise_0.3",
        "lidar_noise_0.3",
        "mmwave_noise_0.3",
        "vk_joint_shuffle",
        "vk_sample_mismatch",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    hpe_values = [hpe_delta_mm(hpe[setting]) for setting in order_hpe]
    hpe_colors = [COLORS[hpe[setting]["perturbed_modality"]] for setting in order_hpe]
    axes[0].bar(range(len(order_hpe)), hpe_values, color=hpe_colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(range(len(order_hpe)))
    axes[0].set_xticklabels([LABELS[setting] for setting in order_hpe], fontsize=8)
    axes[0].set_ylabel("MPJPE increase (mm)")
    axes[0].set_title("HPE diagnostic")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for idx, value in enumerate(hpe_values):
        offset = 1.5 if value >= 0 else -1.5
        va = "bottom" if value >= 0 else "top"
        axes[0].text(idx, value + offset, f"{value:.1f}", ha="center", va=va, fontsize=7)

    har_values = [har_acc_drop_pp(har[setting]) for setting in order_har]
    har_colors = [COLORS[har[setting]["perturbed_modality"]] for setting in order_har]
    axes[1].bar(range(len(order_har)), har_values, color=har_colors, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(order_har)))
    axes[1].set_xticklabels([LABELS[setting] for setting in order_har], fontsize=8)
    axes[1].set_ylabel("Accuracy drop (percentage points)")
    axes[1].set_title("HAR diagnostic")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    for idx, value in enumerate(har_values):
        offset = 0.18 if value >= 0 else -0.18
        va = "bottom" if value >= 0 else "top"
        axes[1].text(idx, value + offset, f"{value:.1f}", ha="center", va=va, fontsize=7)

    legend_items = [
        ("VK perturbation", COLORS["vk"]),
        ("Depth perturbation", COLORS["depth"]),
        ("LiDAR perturbation", COLORS["lidar"]),
        ("mmWave perturbation", COLORS["mmwave"]),
        ("WiFi-CSI perturbation", COLORS["wifi-csi"]),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, ec="black", lw=0.5) for _, color in legend_items]
    fig.legend(handles, [label for label, _ in legend_items], loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        "VK semantic-anchor diagnostic: structural semantics and task-specific physical evidence",
        y=1.08,
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"fig_vk_semantic_anchor_diagnostic.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(rows: list[dict[str, str]]) -> None:
    hpe = by_task(rows, "HPE")
    har = by_task(rows, "HAR")
    report = f"""# VK Semantic-Anchor Diagnostic

## Purpose

This diagnostic evaluates whether VK contributes structure-explicit semantics rather than acting as a replaceable coordinate feature. It should be used as supplementary evidence, not as a claim that VK is the only semantic center.

## Key Numbers

- HPE clean MPJPE: {f(hpe['clean'], 'metric_value') * 1000.0:.2f} mm.
- HPE VK joint shuffle: +{hpe_delta_mm(hpe['vk_joint_shuffle']):.2f} mm MPJPE.
- HPE VK coordinate noise at 0.3: +{hpe_delta_mm(hpe['vk_noise_0.3']):.2f} mm MPJPE.
- HPE Depth noise at 0.3: +{hpe_delta_mm(hpe['depth_noise_0.3']):.2f} mm MPJPE.
- HAR clean accuracy: {f(har['clean'], 'metric_value') * 100.0:.2f}%.
- HAR VK joint shuffle: {har_acc_drop_pp(har['vk_joint_shuffle']):.2f} percentage-point accuracy drop.
- HAR mmWave noise at 0.3: {har_acc_drop_pp(har['mmwave_noise_0.3']):.2f} percentage-point accuracy drop.

## Conclusion

VK joint shuffling degrades both HPE and HAR, indicating that the model uses VK joint identity and skeleton organization as structure-explicit cues. However, the strongest degradation is caused by Depth corruption for HPE and mmWave corruption for HAR. Therefore, the evidence supports the safer conclusion that VK acts as a structure-explicit semantic anchor, while task-specific physical sensors provide critical complementary evidence.

## Recommended Paper Wording

The semantic perturbation diagnostic shows that disrupting VK joint identity degrades both HPE and HAR, suggesting that VK is not used merely as raw coordinates but contributes structure-explicit body semantics. At the same time, the stronger degradation under Depth corruption for HPE and mmWave corruption for HAR indicates task-dependent sensor complementarity. We therefore describe VK as a semantic anchor in the body-latent space, not as the sole semantic center.

## Claims to Avoid

- Do not claim that VK is the only semantic center.
- Do not claim that VK is always the dominant modality.
- Do not claim that VK is noise-free or privacy-preserving.
- Do not claim that this diagnostic provides a formal proof of semantic alignment.
"""
    (OUT_DIR / "vk_semantic_anchor_conclusion.md").write_text(report, encoding="utf-8")


def main() -> None:
    rows = read_rows(INPUT_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(rows)
    write_csv(OUT_DIR / "vk_semantic_anchor_interpretation.csv", build_interpretation_rows(rows))
    write_csv(OUT_DIR / "vk_anchor_key_numbers.csv", key_numbers(rows))
    write_report(rows)
    print(f"Saved VK semantic-anchor analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
