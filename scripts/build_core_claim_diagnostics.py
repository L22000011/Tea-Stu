from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "supplement" / "core_claim_diagnostics"


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


def score_text(value: str) -> int:
    value = (value or "").lower()
    if "very low" in value:
        return 1
    if "low/medium" in value:
        return 2
    if "medium/high" in value:
        return 4
    if "high" in value:
        return 5
    if "medium" in value:
        return 3
    if "low" in value:
        return 1
    if "indirect" in value:
        return 2
    if "possible" in value:
        return 3
    return 0


def build_body_latent_evidence() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "tables" / "body_latent_alignment_from_logs.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        gap = safe_float(row.get("gap"))
        top1 = safe_float(row.get("top1"))
        random_top1 = safe_float(row.get("random"))
        if gap >= 0.01 and top1 >= random_top1 * 2:
            grade = "moderate"
        elif gap > 0 and top1 > random_top1:
            grade = "weak diagnostic"
        else:
            grade = "not supportive"
        out.append(
            {
                "task": row.get("task", ""),
                "target_modality": row.get("target", ""),
                "same_sample_cosine": row.get("same", ""),
                "shuffled_cosine": row.get("shuffled", ""),
                "cosine_gap": row.get("gap", ""),
                "retrieval_top1": row.get("top1", ""),
                "random_top1": row.get("random", ""),
                "evidence_grade": grade,
                "paper_policy": "Use only as weak diagnostic evidence; do not claim strict semantic alignment.",
            }
        )
    return out


def build_privacy_risk() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "tables" / "privacy_exposure_residual_risk.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        face = score_text(row.get("face_appearance", ""))
        bg = score_text(row.get("background", ""))
        body = score_text(row.get("body_geometry", ""))
        gait = score_text(row.get("gait_action_leakage", ""))
        visual_exposure = round((face + bg) / 2.0, 2)
        residual_risk = round((body + gait) / 2.0, 2)
        out.append(
            {
                "representation": row.get("representation", ""),
                "visual_exposure_score_1_low_5_high": visual_exposure,
                "residual_identity_behavior_risk_1_low_5_high": residual_risk,
                "formal_privacy_guarantee": row.get("formal_privacy_guarantee", ""),
                "safe_claim": row.get("our_claim", ""),
                "paper_policy": "Qualitative writing aid only. Do not use these heuristic scores as quantitative privacy metrics; use privacy leakage probe for systematic evaluation.",
            }
        )
    return out


def build_fsg_evidence() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "tables" / "fair_missing_modality_baseline.csv")
    by_task: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row.get("task", ""), {})[row.get("method", "")] = row

    out: list[dict[str, Any]] = []
    for task, methods in by_task.items():
        full = methods.get("VK-RMD Student-VK")
        nokd = methods.get("w/o KD ablation")
        uniform = methods.get("Uniform-fusion ablation")
        if not full:
            continue
        metric = full.get("metric_name", "")
        higher = "Accuracy" in metric
        full_avg = safe_float(full.get("avg_metric"))
        nokd_avg = safe_float(nokd.get("avg_metric")) if nokd else float("nan")
        uniform_avg = safe_float(uniform.get("avg_metric")) if uniform else float("nan")

        if nokd:
            kd_delta = full_avg - nokd_avg
            if higher:
                kd_interpretation = "FSG/KD improves over no-KD." if kd_delta > 0 else "No-KD is competitive or stronger; present FSG as auxiliary."
            else:
                kd_interpretation = "FSG/KD improves over no-KD." if kd_delta < 0 else "No-KD is competitive or stronger; present FSG as auxiliary."
        else:
            kd_delta = float("nan")
            kd_interpretation = "No no-KD row found."

        if uniform:
            fusion_delta = full_avg - uniform_avg
            if higher:
                fusion_interpretation = "Learned fusion improves over uniform." if fusion_delta > 0 else "Uniform fusion is competitive; keep reliability claim modest."
            else:
                fusion_interpretation = "Learned fusion improves over uniform." if fusion_delta < 0 else "Uniform fusion is competitive; keep reliability claim modest."
        else:
            fusion_delta = float("nan")
            fusion_interpretation = "No uniform-fusion row found."

        out.append(
            {
                "task": task,
                "metric_name": metric,
                "full_method_avg": full.get("avg_metric", ""),
                "no_kd_avg": nokd.get("avg_metric", "") if nokd else "",
                "full_minus_no_kd": f"{kd_delta:.4f}" if kd_delta == kd_delta else "",
                "fsg_interpretation": kd_interpretation,
                "uniform_fusion_avg": uniform.get("avg_metric", "") if uniform else "",
                "full_minus_uniform": f"{fusion_delta:.4f}" if fusion_delta == fusion_delta else "",
                "fusion_interpretation": fusion_interpretation,
                "paper_policy": "Do not overclaim FSG as sole source of gains; use as full-to-subset behavior regularization.",
            }
        )
    return out


def build_reviewer_matrix() -> list[dict[str, Any]]:
    body_rows = build_body_latent_evidence()
    fsg_rows = build_fsg_evidence()
    privacy_rows = build_privacy_risk()

    weak_alignment = any(row["evidence_grade"] == "weak diagnostic" for row in body_rows)
    hpe_nokd_strong = any(
        row["task"] == "HPE" and "No-KD is competitive" in row["fsg_interpretation"]
        for row in fsg_rows
    )
    return [
        {
            "reviewer_concern": "Body-latent alignment may be ordinary Transformer fusion.",
            "available_evidence": "Token projection, modality embedding, Transformer interaction, HPE/HAR supervision, VK semantic perturbation, body-latent diagnostic.",
            "evidence_strength": "limited" if weak_alignment else "moderate",
            "safe_response": "Call it task-supervised body-latent tokenization, not strict semantic alignment.",
            "recommended_paper_action": "Use Fig.7/diagnostic only as weak support; emphasize all-combination performance and perturbation evidence.",
        },
        {
            "reviewer_concern": "FSG may be modality dropout plus KD under a new name.",
            "available_evidence": "Full teacher sees all modalities; student samples subsets; HPE/HAR losses include output/logit, token, reliability and structure guidance where implemented.",
            "evidence_strength": "bounded" if hpe_nokd_strong else "moderate",
            "safe_response": "Define FSG as modality-availability asymmetry and full-modality behavior regularization.",
            "recommended_paper_action": "Do not present FSG as the only performance source; keep no-KD out of main table if it distracts, but discuss task-dependent auxiliary role if needed.",
        },
        {
            "reviewer_concern": "Privacy-friendly may be overstated because VK/Depth still leak body and behavior cues.",
            "available_evidence": f"Qualitative exposure table for {len(privacy_rows)} representations; run privacy leakage probe for identity-leakage evidence.",
            "evidence_strength": "conceptual/diagnostic",
            "safe_response": "Use reduced visual exposure and RGB-free inference; explicitly state no anonymity or formal privacy guarantee.",
            "recommended_paper_action": "Add privacy exposure and residual risk table; avoid privacy-preserving wording.",
        },
    ]


def write_privacy_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width = 980
    row_h = 52
    height = 120 + row_h * max(len(rows), 1)
    left = 230
    bar_w = 260
    gap = 80
    colors = ["#e8f3f2", "#b8ded8", "#f0d98c", "#e79a71", "#c95c5c"]

    def bar(x: int, y: int, score: float) -> str:
        filled = int(bar_w * max(0, min(score, 5)) / 5.0)
        color = colors[max(0, min(int(round(score)) - 1, 4))]
        return (
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="18" fill="#eeeeee"/>'
            f'<rect x="{x}" y="{y}" width="{filled}" height="18" fill="{color}"/>'
            f'<text x="{x + bar_w + 8}" y="{y + 14}" font-size="13">{score:.1f}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-size="22" font-weight="700">Visual exposure and residual privacy risk diagnostic</text>',
        '<text x="24" y="58" font-size="14" fill="#555">Heuristic writing aid only; use subject-leakage probe for systematic privacy evaluation.</text>',
        f'<text x="{left}" y="92" font-size="15" font-weight="700">Visual exposure</text>',
        f'<text x="{left + bar_w + gap}" y="92" font-size="15" font-weight="700">Residual identity/behavior risk</text>',
    ]
    for i, row in enumerate(rows):
        y = 115 + i * row_h
        rep = row["representation"]
        exposure = safe_float(row["visual_exposure_score_1_low_5_high"], 0)
        risk = safe_float(row["residual_identity_behavior_risk_1_low_5_high"], 0)
        parts.append(f'<text x="24" y="{y + 16}" font-size="15" font-weight="600">{rep}</text>')
        parts.append(bar(left, y, exposure))
        parts.append(bar(left + bar_w + gap, y, risk))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(
    reviewer_rows: list[dict[str, Any]],
    body_rows: list[dict[str, Any]],
    privacy_rows: list[dict[str, Any]],
    fsg_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    hpe_body = [row for row in body_rows if row["task"] == "HPE"]
    har_body = [row for row in body_rows if row["task"] == "HAR"]
    lines = [
        "# 核心理论质疑的小实验/诊断证据包",
        "",
        "本报告由 `scripts/build_core_claim_diagnostics.py` 生成，只汇总已有结果，不训练、不修改模型。",
        "",
        "## 总结判断",
        "",
        "- body-latent token 对齐目前只能写成 **task-supervised body-latent tokenization**，不能写成严格语义对齐证明。",
        "- FSG 可以写成 **full-modality behavior regularization under modality-availability asymmetry**，不能写成传统大模型蒸馏或唯一性能来源。",
        "- privacy-friendly 必须降调为 **reduced visual exposure / RGB-free inference**，不能写成 anonymity 或 privacy guarantee。",
        "",
        "## 1. Body-latent alignment 证据边界",
        "",
        f"- HPE body-latent diagnostic rows: {len(hpe_body)}.",
        f"- HAR body-latent diagnostic rows: {len(har_body)}.",
        "- 当前 cosine gap 普遍较小，因此 Fig.7/相关表格只能作为弱诊断证据。",
        "- 推荐写法：模型在任务监督下诱导人体状态相关 token 表示，而不是显式证明异构传感器严格语义对齐。",
        "",
        "## 2. FSG 证据边界",
        "",
    ]
    for row in fsg_rows:
        lines.extend(
            [
                f"### {row['task']}",
                f"- Full method avg: {row['full_method_avg']} ({row['metric_name']})",
                f"- No-KD avg: {row['no_kd_avg']}",
                f"- Interpretation: {row['fsg_interpretation']}",
                f"- Fusion interpretation: {row['fusion_interpretation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. Privacy-friendly 证据边界",
            "",
            f"- 已生成 {len(privacy_rows)} 类 representation 的 visual exposure / residual risk 定性汇总。",
            "- 这不是隐私攻击实验，也不是形式化隐私指标，不能作为论文中的量化风险结果。",
            "- 系统性评价应使用 `scripts/run_privacy_leakage_probe.py` 的 subject-identity leakage probe。",
            "- 推荐写法：VK/Depth 减少 raw RGB 外观暴露，但仍保留身体几何和动作线索。",
            "",
            "## 4. 三条审稿质疑回应矩阵",
            "",
        ]
    )
    for row in reviewer_rows:
        lines.extend(
            [
                f"### {row['reviewer_concern']}",
                f"- Evidence strength: {row['evidence_strength']}",
                f"- Available evidence: {row['available_evidence']}",
                f"- Safe response: {row['safe_response']}",
                f"- Paper action: {row['recommended_paper_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 5. 可以直接写入论文的安全表述",
            "",
            "> VK-RMD does not claim explicit raw-space alignment or formal privacy preservation. Instead, it learns a task-supervised body-latent token space under reduced visual exposure. VK and depth provide low-appearance structural anchors, while LiDAR, mmWave and WiFi-CSI provide complementary physical evidence. Full-to-Subset Guidance regularizes randomly missing-modality students with full-modality teacher behavior, but its role is auxiliary and task-dependent rather than the sole source of performance gains.",
            "",
            "中文：",
            "",
            "> VK-RMD 不主张显式 raw-space 对齐或形式化隐私保护。它学习的是 reduced visual exposure 条件下的任务监督 body-latent token 空间。VK 和 Depth 提供低外观结构锚点，LiDAR、mmWave 和 WiFi-CSI 提供互补物理证据。Full-to-Subset Guidance 使用 full-modality teacher 行为约束随机缺失模态 student，但其作用是辅助且任务相关的，而不是唯一性能来源。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    body_rows = build_body_latent_evidence()
    privacy_rows = build_privacy_risk()
    fsg_rows = build_fsg_evidence()
    reviewer_rows = build_reviewer_matrix()

    write_csv(OUT_DIR / "body_latent_claim_evidence.csv", body_rows)
    write_csv(OUT_DIR / "privacy_exposure_qualitative_table.csv", privacy_rows)
    write_csv(OUT_DIR / "fsg_positioning_evidence.csv", fsg_rows)
    write_csv(OUT_DIR / "reviewer_concern_response_matrix.csv", reviewer_rows)
    write_privacy_svg(privacy_rows, OUT_DIR / "fig_privacy_exposure_risk.svg")
    write_report(
        reviewer_rows,
        body_rows,
        privacy_rows,
        fsg_rows,
        OUT_DIR / "core_claim_diagnostics_report.md",
    )
    print(f"Core claim diagnostics written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
