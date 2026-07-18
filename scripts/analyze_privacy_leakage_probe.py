from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "privacy_leakage_probe"
CSV_PATH = IN_DIR / "privacy_leakage_probe_subject_id.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rgb_acc = next((safe_float(row["subject_test_acc"]) for row in rows if row["modality"] == "rgb"), float("nan"))
    out: list[dict[str, Any]] = []
    for row in rows:
        acc = safe_float(row["subject_test_acc"])
        chance = safe_float(row["chance_acc"])
        ratio_to_chance = acc / chance if chance > 0 else float("nan")
        reduction_vs_rgb = 1.0 - acc / rgb_acc if rgb_acc > 0 else float("nan")
        out.append(
            {
                "modality": row["modality"],
                "subject_id_acc_percent": f"{acc * 100.0:.2f}",
                "chance_percent": f"{chance * 100.0:.2f}",
                "times_chance": f"{ratio_to_chance:.2f}",
                "relative_reduction_vs_defaced_rgb_percent": f"{reduction_vs_rgb * 100.0:.2f}" if reduction_vs_rgb == reduction_vs_rgb else "",
                "interpretation": row.get("privacy_interpretation", ""),
            }
        )
    return out


def plot(rows: list[dict[str, str]], png_path: Path, svg_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["rgb", "vk", "depth", "lidar", "mmwave", "wifi-csi"]
    present = [row for name in order for row in rows if row["modality"] == name]
    labels = [row["modality"] for row in present]
    values = [safe_float(row["subject_test_acc"]) * 100.0 for row in present]
    chance = safe_float(present[0]["chance_acc"]) * 100.0 if present else 0.0

    colors = {
        "rgb": "#b23a48",
        "vk": "#2f7f95",
        "depth": "#6b9f71",
        "lidar": "#8c6bb1",
        "mmwave": "#d18f3b",
        "wifi-csi": "#637a91",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, values, color=[colors.get(label, "#777777") for label in labels], width=0.68)
    ax.axhline(chance, color="#222222", linestyle="--", linewidth=1.2, label=f"Chance ({chance:.1f}%)")
    ax.set_ylabel("Subject-ID probe accuracy (%)")
    ax.set_xlabel("Representation / modality")
    ax.set_title("Identity leakage diagnostic under action-holdout split")
    ax.set_ylim(0, max(values + [chance]) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Diagnostic only: high accuracy indicates residual identity leakage; low accuracy does not prove anonymity.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)


def write_report(rows: list[dict[str, str]], summary: list[dict[str, Any]], path: Path) -> None:
    def acc(name: str) -> float:
        return next((safe_float(row["subject_test_acc"]) for row in rows if row["modality"] == name), float("nan"))

    rgb = acc("rgb")
    vk = acc("vk")
    depth = acc("depth")
    lidar = acc("lidar")
    mmwave = acc("mmwave")
    wifi = acc("wifi-csi")
    chance = next((safe_float(row["chance_acc"]) for row in rows), 0.025)

    lines = [
        "# 隐私泄露 Probe 结果分析",
        "",
        "## 1. 实验目的",
        "",
        "该实验使用轻量 subject-ID linear probe 评估不同表示是否携带身份相关信息。它不是形式化隐私证明，也不是匿名性评估；它用于限制论文中的隐私表述边界。",
        "",
        "实验采用 `action-holdout` 划分，即同一 subject 的训练动作和测试动作不同，比随机 frame split 更严格。",
        "",
        "## 2. 关键结果",
        "",
        f"- Chance accuracy: {chance * 100:.2f}%。",
        f"- Defaced RGB subject-ID accuracy: {rgb * 100:.2f}%。",
        f"- VK subject-ID accuracy: {vk * 100:.2f}%。",
        f"- Depth subject-ID accuracy: {depth * 100:.2f}%。",
        f"- LiDAR subject-ID accuracy: {lidar * 100:.2f}%。",
        f"- mmWave subject-ID accuracy: {mmwave * 100:.2f}%。",
        f"- WiFi-CSI subject-ID accuracy: {wifi * 100:.2f}%。",
        "",
        "## 3. 结论",
        "",
        "1. Defaced RGB 仍然具有最高身份泄露风险。即使去脸后，RGB 仍可能保留衣着、体型、背景、场景和外观线索。",
        "",
        f"2. VK 的身份泄露明显低于 defaced RGB：VK 为 {vk * 100:.2f}%，defaced RGB 为 {rgb * 100:.2f}%。这支持 `reduced visual exposure`，但不支持 `privacy-preserving`。",
        "",
        f"3. VK 仍显著高于 chance：{vk * 100:.2f}% vs {chance * 100:.2f}%。这说明 VK 保留身体比例、骨架结构或动作习惯等身份相关线索，不能声称匿名或 identity-free。",
        "",
        f"4. Depth 和 LiDAR 也存在明显身份泄露，分别为 {depth * 100:.2f}% 和 {lidar * 100:.2f}%。因此非 RGB 不等于隐私安全。",
        "",
        "5. mmWave 和 WiFi-CSI 的 subject-ID accuracy 较低但仍高于 chance，说明无线/雷达信号也可能携带身份相关行为或空间线索。",
        "",
        "## 4. 论文推荐写法",
        "",
        "> We evaluate identity leakage using a lightweight subject-identification probe under an action-holdout split. Defaced RGB shows the highest leakage, while VK substantially reduces identity predictability compared with defaced RGB. However, VK remains above chance, indicating that keypoint geometry still carries body-structure and motion-style cues. Therefore, our claim is limited to reduced visual exposure and RGB-free downstream inference, not formal privacy preservation or anonymity.",
        "",
        "中文：",
        "",
        "> 我们使用 action-holdout subject-ID probe 评估不同表示的身份泄露风险。结果显示，defaced RGB 仍然具有最高泄露风险，而 VK 相比 defaced RGB 明显降低了身份可识别性。然而，VK 仍显著高于随机猜测，说明关键点几何仍包含身体结构和动作风格线索。因此，本文只主张降低视觉外观暴露和 RGB-free downstream inference，不主张形式化隐私保护或匿名性。",
        "",
        "## 5. 不应写的结论",
        "",
        "- 不写 VK is privacy-preserving。",
        "- 不写 VK is anonymous / identity-free。",
        "- 不写 non-RGB modalities are privacy-safe。",
        "- 不写该实验证明了隐私安全。",
        "",
        "## 6. 输出文件",
        "",
        "- `privacy_leakage_probe_subject_id_summary.csv`",
        "- `fig_privacy_leakage_subject_id.png`",
        "- `fig_privacy_leakage_subject_id.svg`",
        "",
        "## 7. 汇总表",
        "",
        "| Modality | Subject-ID acc (%) | Chance (%) | Times chance | Reduction vs defaced RGB (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['modality']} | {row['subject_id_acc_percent']} | {row['chance_percent']} | "
            f"{row['times_chance']} | {row['relative_reduction_vs_defaced_rgb_percent']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)
    rows = read_csv(CSV_PATH)
    summary = build_summary(rows)
    write_csv(IN_DIR / "privacy_leakage_probe_subject_id_summary.csv", summary)
    plot(rows, IN_DIR / "fig_privacy_leakage_subject_id.png", IN_DIR / "fig_privacy_leakage_subject_id.svg")
    write_report(rows, summary, IN_DIR / "privacy_leakage_probe_conclusion_中文.md")
    print(f"Privacy leakage analysis written to: {IN_DIR}")


if __name__ == "__main__":
    main()
