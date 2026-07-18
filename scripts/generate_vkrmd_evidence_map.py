from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "supplement" / "VK_RMD_experiment_evidence_map.png"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"Missing row: {criteria}")


def add_card(ax, x, y, w, h, title, body, accent, status="已完成"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#FFFFFF",
            edgecolor="#D9E2E8",
            linewidth=1.2,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.052), w, 0.052,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=accent,
            edgecolor=accent,
            linewidth=0,
            zorder=2,
        )
    )
    ax.text(x + 0.025, y + h - 0.028, title, fontsize=14, fontweight="bold", color="#FFFFFF", va="center", zorder=3)
    ax.text(x + w - 0.025, y + h - 0.028, status, fontsize=9.5, color="#FFFFFF", va="center", ha="right", zorder=3)
    ax.text(x + 0.025, y + h - 0.078, body, fontsize=11.2, color="#1E2933", va="top", linespacing=1.55, zorder=3)


def add_claim(ax, x, y, w, h, title, body, color, boundary=False):
    background = "#FFF8EF" if boundary else "#F3FAF8"
    edge = "#F0B96B" if boundary else "#78BFA8"
    label = "边界" if boundary else "可防守结论"
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=background,
            edgecolor=edge,
            linewidth=1.25,
            zorder=1,
        )
    )
    ax.add_patch(Rectangle((x + 0.018, y + 0.025), 0.009, h - 0.05, facecolor=color, edgecolor="none", zorder=2))
    ax.text(x + 0.045, y + h - 0.043, title, fontsize=13, fontweight="bold", color="#14353A", va="top", zorder=3)
    ax.text(x + w - 0.025, y + h - 0.043, label, fontsize=9.5, color=color, va="top", ha="right", zorder=3)
    ax.text(x + 0.045, y + h - 0.087, body, fontsize=10.8, color="#26343A", va="top", linespacing=1.5, zorder=3)


def main() -> None:
    hpe = read_rows(ROOT / "outputs" / "tables" / "main_hpe_results.csv")
    har = read_rows(ROOT / "outputs" / "tables" / "main_har_results.csv")
    privacy = read_rows(ROOT / "privacy_leakage_probe" / "privacy_leakage_probe_subject_id_summary.csv")
    severity = read_rows(ROOT / "outputs" / "tables" / "subset_severity_summary.csv")
    expert = read_rows(ROOT / "outputs" / "expert_compensation_analysis" / "expert_compensation_summary.csv")
    depth = read_rows(ROOT / "tables" / "depth_vk_feasibility.csv")

    hpe_random = find_row(hpe, method="VK-RMD Student-VK", protocol="random split")
    hpe_scene = find_row(hpe, method="VK-RMD Student-VK", protocol="cross-scene")
    har_random = find_row(har, method="VK-RMD Student-VK", protocol="random split")
    rgb = find_row(privacy, modality="rgb")
    vk = find_row(privacy, modality="vk")
    hpe_full = find_row(severity, task="HPE", method="Student-VK", missing_count="0")
    hpe_severe = find_row(severity, task="HPE", method="Student-VK", missing_count="4")
    har_full = find_row(severity, task="HAR", method="Student-VK", missing_count="0")
    har_severe = find_row(severity, task="HAR", method="Student-VK", missing_count="3")
    comp = find_row(expert, task="HAR", modality_set="vk+mmwave", vk_condition="vk_joint_shuffle")
    depth_random = find_row(depth, protocol="random")
    depth_subject = find_row(depth, protocol="cross_subject")
    depth_scene = find_row(depth, protocol="cross_scene")

    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 12.4), dpi=180, facecolor="#F7FAFC")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    teal = "#167E83"
    green = "#2F956D"
    navy = "#315C89"
    orange = "#D98732"
    coral = "#BF5B50"

    ax.text(0.055, 0.955, "VK-RMD 实验完成度与证据地图", fontsize=25, fontweight="bold", color="#153B4A", va="top")
    ax.text(
        0.055, 0.918,
        "从低视觉暴露结构输入，到随机缺失模态下的全组合感知：已完成证据、可防守结论与当前边界",
        fontsize=12.5, color="#55717E", va="top",
    )
    ax.plot([0.055, 0.945], [0.892, 0.892], color="#C9D8DF", linewidth=1.2)

    ax.text(0.055, 0.861, "已完成实验与直接证据", fontsize=16, fontweight="bold", color="#153B4A", va="center")
    ax.text(0.585, 0.861, "论文中可以怎样解释", fontsize=16, fontweight="bold", color="#153B4A", va="center")

    add_card(
        ax, 0.055, 0.655, 0.47, 0.175,
        "主任务与 X-Fi 对照",
        f"完整模态：HPE MPJPE 83.7 → {float(hpe_random['full_mpjpe_mm']):.1f} mm\n"
        f"完整模态：HAR Acc. 72.2% → {float(har_random['full_accuracy_pct']):.2f}%\n"
        "HPE 31 个、HAR 15 个非空模态组合均已评估。",
        teal,
    )
    add_claim(
        ax, 0.555, 0.655, 0.39, 0.175,
        "主结论：动态子集感知可用",
        "在与 X-Fi 对应的完整模态设置下，VK-RMD 取得更优结果；\n"
        "全组合评估进一步说明模型能够处理不同可用传感器子集。",
        green,
    )

    add_card(
        ax, 0.055, 0.450, 0.47, 0.175,
        "低视觉暴露身份探针",
        f"40 类主体识别：脱敏 RGB {float(rgb['subject_id_acc_percent']):.2f}% → VK {float(vk['subject_id_acc_percent']):.2f}%\n"
        f"相对下降 {float(vk['relative_reduction_vs_defaced_rgb_percent']):.2f}%；VK 仍高于 chance ({float(vk['chance_percent']):.2f}%)。",
        navy,
    )
    add_claim(
        ax, 0.555, 0.450, 0.39, 0.175,
        "主结论：降低外观暴露，不是匿名化",
        "在线推理不处理原始 RGB，VK 显著降低主体可预测性；\n"
        "但 VK、Depth 等仍保留身体几何和行为泄露风险，因此不主张 formal privacy guarantee。",
        navy,
    )

    add_card(
        ax, 0.055, 0.245, 0.47, 0.175,
        "缺失模态压力与 VK 受损补偿",
        f"HPE Student-VK：0 缺失 {float(hpe_full['avg_metric']):.1f} mm，4 缺失 {float(hpe_severe['avg_metric']):.1f} mm\n"
        f"HAR Student-VK：0 缺失 {float(har_full['avg_metric']):.2f}%，3 缺失 {float(har_severe['avg_metric']):.2f}%\n"
        f"VK+mmWave joint shuffle：VK expert {float(comp['vk_expert_metric']) * 100:.2f}%；融合 {float(comp['final_metric']) * 100:.2f}%。",
        green,
    )
    add_claim(
        ax, 0.555, 0.245, 0.39, 0.175,
        "机制结论：物理模态可部分补偿坏 VK",
        "VK 受损后，最终融合优于 VK expert（如 VK+mmWave 提升 8.50 个百分点）。\n"
        "这支撑 chunk experts + subset-adaptive fusion 的互补作用；权重仅是任务贡献代理。",
        green,
    )

    add_card(
        ax, 0.055, 0.075, 0.47, 0.140,
        "非 RGB VK 来源可行性：Depth → VK",
        f"PCK@0.10：random {float(depth_random['pck10']) * 100:.1f}%；cross-subject {float(depth_subject['pck10']) * 100:.1f}%；\n"
        f"cross-scene {float(depth_scene['pck10']) * 100:.1f}%。Depth-VK 下游 HPE/HAR 训练因 GPU OOM 未完成。",
        orange,
    )
    add_claim(
        ax, 0.555, 0.075, 0.39, 0.140,
        "边界：上游 VK 来源与场景泛化仍开放",
        f"Depth 可在 random/cross-subject 下提供结构输入候选；但跨场景 VK 生成不稳定。\n"
        f"主模型 HPE cross-scene 平均 MPJPE 由 {float(hpe_random['avg_mpjpe_mm']):.2f} 增至 {float(hpe_scene['avg_mpjpe_mm']):.2f} mm，不能声称已解决泛化。",
        orange,
        boundary=True,
    )

    ax.text(0.055, 0.036, "总体定位：VK-RMD 的可防守价值是“RGB-free online inference + full-to-subset supervision + subset-adaptive expert fusion”；不是形式化隐私保证，也不是校准传感器可靠性估计。", fontsize=10.3, color="#5A6B74", va="center")
    ax.text(0.055, 0.014, "数据来源：本地 main results、privacy probe、subset severity、expert compensation、Depth-VK feasibility。", fontsize=8.9, color="#7B8B93", va="center")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
