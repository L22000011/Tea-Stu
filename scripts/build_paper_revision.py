from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Deskbook\Tea")
REV = ROOT / "paper_revision"
GEN = REV / "generated"
PAPER = ROOT / "Paper"

RESULT_DIR_NAMES = {
    "outputs",
    "results",
    "logs",
    "figures",
    "tables",
    "eval",
    "ablation",
    "missing",
    "cross_subject",
    "cross_scene",
    "xfi",
    "ori",
    "superteacher",
    "super-teacher",
    "student_vk",
    "student_nv",
    "fsg",
    "reliability",
    "fusion",
    "privacy_leakage_probe",
    "supplement",
    "legacy_xfi",
}

METRIC_KEYS = {
    "mpjpe": "lower_better",
    "pa_mpjpe": "lower_better",
    "pampjpe": "lower_better",
    "mse": "lower_better",
    "acc": "higher_better",
    "accuracy": "higher_better",
    "macro_f1": "higher_better",
    "f1": "higher_better",
    "subject_id_acc": "lower_better",
    "test_acc": "lower_better",
    "avg_mpjpe_mm": "lower_better",
    "full_mpjpe_mm": "lower_better",
    "avg_accuracy_pct": "higher_better",
    "full_accuracy_pct": "higher_better",
}


def ensure_dirs() -> None:
    REV.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def classify_path(path: Path) -> tuple[str, str]:
    p = str(path).lower()
    task = "unknown"
    if "hpe" in p:
        task = "HPE"
    if "har" in p:
        task = "HAR"
    if "privacy" in p or "leakage" in p:
        task = "privacy_probe"
    if "legacy_xfi" in p or "xfi_" in p:
        task = "xfi_baseline"
    if "ori-" in p or "origin-xfi" in p or "ori_" in p:
        task = "ori_baseline"
    if "super-teacher" in p or "superteacher" in p:
        task = "superteacher"
    if "student_vk" in p:
        task = "student_vk_nv" if "student_nv" in p else "full_combination"
    if "student_nv" in p:
        task = "student_vk_nv"
    if "cross_subject" in p or "cross-subject" in p:
        task = "cross_subject"
    if "cross_scene" in p or "cross-scene" in p:
        task = "cross_scene"
    if "ablation" in p:
        task = "ablation"
    if "reliability" in p or "fusion" in p:
        task = "reliability"
    if "subset_severity" in p or "severity" in p:
        task = "missing_strength"
    if "leave_one" in p or "leave-one" in p:
        task = "leave_one_out"
    usage = "discussion"
    if task in {"HPE", "HAR", "full_combination", "student_vk_nv", "missing_strength", "leave_one_out", "xfi_baseline", "privacy_probe"}:
        usage = "main_table"
    if task in {"ablation", "reliability"}:
        usage = "ablation_table" if task == "ablation" else "figure"
    if task in {"superteacher", "ori_baseline", "unknown"}:
        usage = "discussion" if task == "superteacher" else "discard"
    return task, usage


def method_from_path(path: Path, row: dict[str, Any] | None = None) -> str:
    if row:
        for key in ["method", "method_name", "setting", "name", "model"]:
            if row.get(key):
                return str(row[key])
    stem = path.stem
    p = str(path).lower()
    if "student_vk" in p:
        return "VK-RMD Student-VK"
    if "student_nv" in p:
        return "VK-RMD Student-NV"
    if "teacher" in p:
        return "full-modality teacher"
    if "uniform" in p:
        return "uniform fusion"
    if "no_distill" in p or "no-kd" in p or "no_kd" in p:
        return "w/o distillation"
    if "xfi" in p:
        return "X-Fi baseline"
    return stem


def modality_from_row_or_path(path: Path, row: dict[str, Any] | None = None) -> str:
    if row:
        for key in ["modality", "modality_set", "modalities", "input_modalities", "combination"]:
            if row.get(key):
                return str(row[key])
    return ""


def protocol_from_path(path: Path, row: dict[str, Any] | None = None) -> str:
    if row:
        for key in ["protocol", "split_protocol", "expected_split"]:
            if row.get(key):
                return str(row[key])
    p = str(path).lower()
    if "cross_subject" in p or "cross-subject" in p:
        return "cross-subject"
    if "cross_scene" in p or "cross-scene" in p:
        return "cross-scene"
    if "random" in p:
        return "random"
    return ""


def add_index_row(rows: list[dict[str, Any]], path: Path, metric_name: str = "", metric_value: str = "", row: dict[str, Any] | None = None, reason: str = "") -> None:
    task, usage = classify_path(path)
    metric_key = metric_name.lower()
    rows.append(
        {
            "file_path": str(path),
            "file_type": path.suffix.lower().lstrip("."),
            "task": task,
            "method_name": method_from_path(path, row),
            "input_modalities": modality_from_row_or_path(path, row),
            "missing_setting": row.get("missing_count", "") if row else "",
            "split_protocol": protocol_from_path(path, row),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_direction": METRIC_KEYS.get(metric_key, ""),
            "baseline_reference": row.get("baseline_reference", "") if row else "",
            "improvement_or_drop": row.get("delta", row.get("improvement_or_drop", "")) if row else "",
            "is_ours": "no" if task in {"xfi_baseline", "ori_baseline"} else ("no" if task == "unknown" else "yes"),
            "is_xfi_or_ori_baseline": "yes" if task in {"xfi_baseline", "ori_baseline"} else "no",
            "is_superteacher_exploration": "yes" if task == "superteacher" else "no",
            "recommended_usage": usage,
            "reason": reason or usage_reason(task, usage),
            "source_confidence": "high" if path.suffix.lower() in {".csv", ".json", ".tex"} else "medium",
        }
    )


def usage_reason(task: str, usage: str) -> str:
    if task == "superteacher":
        return "Exploratory only; not part of the main contribution."
    if task == "ori_baseline":
        return "Reproduced/legacy baseline; avoid using anomalous values as the main comparison."
    if usage == "main_table":
        return "Directly supports VK replacement, FSG, missing-modality robustness, or baseline comparison."
    if usage == "ablation_table":
        return "Supports mechanism/ablation analysis."
    if usage == "figure":
        return "Supports mechanism visualization or diagnostic analysis."
    return "Parsed but not central to the main story."


def parse_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            found = False
            for record in reader:
                for key in fieldnames:
                    lk = key.lower()
                    if lk in METRIC_KEYS and record.get(key) not in {None, ""}:
                        add_index_row(rows, path, lk, str(record[key]), record)
                        found = True
            if not found:
                add_index_row(rows, path, reason="CSV parsed but no recognized metric column was found.")
    except Exception as exc:
        add_index_row(rows, path, reason=f"Could not parse CSV: {exc}")


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten_json(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj[:20]):
            out.update(flatten_json(value, f"{prefix}.{i}" if prefix else str(i)))
    else:
        out[prefix] = obj
    return out


def parse_json(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_json(obj)
        found = False
        for key, value in flat.items():
            lk = key.split(".")[-1].lower()
            if lk in METRIC_KEYS and isinstance(value, (int, float, str)):
                add_index_row(rows, path, lk, str(value), None)
                found = True
        if not found:
            add_index_row(rows, path, reason="JSON parsed but no recognized metric key was found.")
    except Exception as exc:
        add_index_row(rows, path, reason=f"Could not parse JSON: {exc}")


LOG_METRIC_RE = re.compile(
    r"(?P<name>mpjpe|pampjpe|pa_mpjpe|mse|acc|accuracy|macro_f1|f1|loss|best_metric)\s*[:=]\s*(?P<value>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_text(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = list(LOG_METRIC_RE.finditer(text))
        if not matches:
            add_index_row(rows, path, reason="Text/log parsed but no recognized metric pattern was found.")
            return
        for match in matches[:20]:
            add_index_row(rows, path, match.group("name").lower(), match.group("value"), None, "Metric extracted from text/log pattern.")
    except Exception as exc:
        add_index_row(rows, path, reason=f"Could not parse text/log: {exc}")


def build_experiment_index() -> None:
    rows: list[dict[str, Any]] = []
    exts = {".csv", ".json", ".txt", ".log", ".md", ".tex", ".xlsx", ".pkl", ".npy", ".npz", ".pt"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        low_parts = {part.lower() for part in path.parts}
        if not (low_parts & RESULT_DIR_NAMES) and path.name not in {"experiment_index.csv", "result_index.csv"}:
            continue
        if path.suffix.lower() == ".csv":
            parse_csv(path, rows)
        elif path.suffix.lower() == ".json":
            parse_json(path, rows)
        elif path.suffix.lower() in {".txt", ".log", ".md", ".tex"}:
            parse_text(path, rows)
        else:
            add_index_row(rows, path, reason="Binary/model/table file indexed by metadata only; not parsed for metric values.")

    out = REV / "experiment_index.csv"
    fields = [
        "file_path",
        "file_type",
        "task",
        "method_name",
        "input_modalities",
        "missing_setting",
        "split_protocol",
        "metric_name",
        "metric_value",
        "metric_direction",
        "baseline_reference",
        "improvement_or_drop",
        "is_ours",
        "is_xfi_or_ori_baseline",
        "is_superteacher_exploration",
        "recommended_usage",
        "reason",
        "source_confidence",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_figures_index() -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    rows: list[dict[str, str]] = []
    for base in [ROOT / "figures", ROOT / "tables", ROOT / "supplement", ROOT / "Paper"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in image_exts:
                continue
            name = path.name.lower()
            content = "unknown"
            section = "Discussion"
            insert = "no"
            need_redraw = "no"
            reason = "Indexed figure candidate."
            if "fig1" in name or "framework" in name or "alignment" in name:
                content, section, insert = "framework/alignment", "Method", "yes"
            if "fig2" in name or "protocol" in name:
                content, section, insert = "missing-modality protocol", "Method", "yes"
            if "fig3" in name or "main" in name:
                content, section, insert = "main HPE/HAR results", "Results", "yes"
            if "fig4" in name or "severity" in name:
                content, section, insert = "missing-modality severity", "Results", "yes"
            if "fig5" in name or "ablation" in name or "baseline" in name:
                content, section, insert = "ablation/fair baseline", "Results", "yes"
            if "fig6" in name or "reliability_proxy" in name or "weights" in name:
                content, section, insert = "reliability proxy", "Results/Discussion", "yes"
            if "fig7" in name or "body_latent" in name:
                content, section, insert = "body-latent diagnostic", "Discussion", "no"
                reason = "Diagnostic only; weak evidence for strict alignment."
            if "fig8" in name or "corruption" in name:
                content, section, insert = "corruption diagnostic", "Discussion", "yes"
                reason = "Use cautiously as learned proxy diagnostic, not physical reliability proof."
            if "fig9" in name or "semantic_anchor" in name:
                content, section, insert = "VK semantic-anchor diagnostic", "Supplementary/Discussion", "no"
            if "unverified" in name:
                insert, need_redraw, reason = "no", "yes", "Unverified earlier figure; use verified replacement if available."
            rows.append(
                {
                    "figure_path": str(path),
                    "file_name": path.name,
                    "content_type": content,
                    "current_quality": "usable" if insert == "yes" else "diagnostic/optional",
                    "suggested_section": section,
                    "suggested_caption": caption_for(content),
                    "insert_to_main_text": insert,
                    "need_redraw": need_redraw,
                    "reason": reason,
                }
            )
    fields = [
        "figure_path",
        "file_name",
        "content_type",
        "current_quality",
        "suggested_section",
        "suggested_caption",
        "insert_to_main_text",
        "need_redraw",
        "reason",
    ]
    with (REV / "figures_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def caption_for(content: str) -> str:
    return {
        "framework/alignment": "VK-RMD framework: VK replaces downstream RGB and anchors body-token learning with non-RGB sensors.",
        "missing-modality protocol": "Full-to-Subset Guidance protocol with full-modality teacher and random-subset student.",
        "main HPE/HAR results": "Main HPE/HAR performance summary.",
        "missing-modality severity": "Performance under increasing missing-modality severity.",
        "ablation/fair baseline": "Fair baseline and ablation comparison.",
        "reliability proxy": "Learned modality contribution proxy across modality subsets.",
        "body-latent diagnostic": "Diagnostic evidence for body-latent token correspondence.",
        "corruption diagnostic": "Reliability proxy response under synthetic modality corruption.",
        "VK semantic-anchor diagnostic": "VK semantic-anchor perturbation diagnostic.",
    }.get(content, "Candidate figure.")


def copy_generated_tables() -> None:
    src = PAPER / "generated"
    if src.exists():
        for path in src.glob("*.tex"):
            shutil.copy2(path, GEN / path.name)


def write_extra_tables() -> None:
    notation = r"""\begin{table}[H]
\centering
\caption{模态符号与在 VK-RMD 中的角色。本文所有组合表均使用缩写：V=VK，D=Depth，L=LiDAR，R=mmWave，W=WiFi-CSI。}
\label{tab:notation_cn}
\scriptsize
\begin{adjustbox}{width=\linewidth}
\begin{tabular}{llll}
\toprule
符号 & 表示 & 外观暴露 & 在 VK-RMD 中的角色 \\
\midrule
V & VK & 低 & 低外观关节结构输入，替代下游 RGB \\
D & Depth & 低/中 & 关键几何证据，HPE 中最敏感 \\
L & LiDAR & 低 & 稀疏三维几何补充 \\
R & mmWave & 低 & 动态与动作线索，HAR 中最敏感 \\
W & WiFi-CSI & 很低 & 非视觉无线感知补充，仅用于 HPE \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    privacy = r"""\begin{table}[H]
\centering
\caption{Action-holdout 主体身份泄露探针。ID Acc. 越低表示身份可恢复风险越低；随机水平为 2.50\%。该实验不构成形式化隐私证明。}
\label{tab:leakage_probe_rebuilt}
\scriptsize
\begin{adjustbox}{width=\linewidth}
\begin{tabular}{lccc}
\toprule
输入表示 & ID Acc.$\downarrow$ & 随机倍数 & 相对脱敏 RGB 降低 \\
\midrule
脱敏 RGB & 67.53\% & 27.01$\times$ & 0.00\% \\
V & 15.43\% & 6.17$\times$ & 77.15\% \\
D & 43.80\% & 17.52$\times$ & 35.14\% \\
L & 36.57\% & 14.63$\times$ & 45.84\% \\
R & 9.72\% & 3.89$\times$ & 85.60\% \\
W & 8.65\% & 3.46$\times$ & 87.19\% \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    (GEN / "modality_notation_table_cn.tex").write_text(notation, encoding="utf-8")
    (GEN / "privacy_probe_table_cn.tex").write_text(privacy, encoding="utf-8")


def build_tables_rebuilt() -> None:
    pieces = [
        "modality_notation_table_cn.tex",
        "privacy_probe_table_cn.tex",
        "xfi_hpe_direct_table_cn.tex",
        "xfi_har_direct_table_cn.tex",
        "fsg_severity_table_cn.tex",
        "leave_one_out_table_cn.tex",
        "protocol_generalization_table_cn.tex",
        "ablation_core_table_cn.tex",
    ]
    text = "\n\n".join((GEN / p).read_text(encoding="utf-8") for p in pieces if (GEN / p).exists())
    (REV / "tables_rebuilt.tex").write_text(text, encoding="utf-8")


def build_revised_tex() -> None:
    tex = (PAPER / "IoTJ_VK_RMD_CN.tex").read_text(encoding="utf-8")
    tex = tex.replace(r"\input{generated/xfi_hpe_direct_table_cn.tex}", r"\input{generated/xfi_hpe_direct_table_cn.tex}")
    # Make revision independent: copy generated tables into paper_revision/generated and keep relative generated/ inputs.
    if r"\input{generated/modality_notation_table_cn.tex}" not in tex:
        marker = r"\end{itemize}" + "\n\n" + r"\section{相关工作}"
        tex = tex.replace(
            marker,
            r"\end{itemize}" + "\n\n" + r"\input{generated/modality_notation_table_cn.tex}" + "\n\n" + r"\section{相关工作}",
            1,
        )
    # Replace the hand-written leakage probe table with a generated copy while keeping qualitative exposure table.
    tex = re.sub(
        r"\\begin\{table\}\[t\]\s*\\centering\s*\\caption\{Action-holdout 划分下的主体身份泄露探针.*?\\end\{table\}",
        r"\\input{generated/privacy_probe_table_cn.tex}",
        tex,
        flags=re.S,
    )
    # Relative figure paths from paper_revision still point to sibling tables directory.
    tex = tex.replace(r"\balance" + "\n\n" + r"{\footnotesize", r"\balance" + "\n\n" + r"{\footnotesize")
    (REV / "iotj_revised.tex").write_text(tex, encoding="utf-8")


def extract_bib() -> None:
    tex = (REV / "iotj_revised.tex").read_text(encoding="utf-8")
    m = re.search(r"\\begin\{thebibliography\}\{99\}(.*?)\\end\{thebibliography\}", tex, re.S)
    body = m.group(1).strip() if m else ""
    content = "% References extracted from iotj_revised.tex. Verify metadata, venue names, pages and DOI before submission.\n"
    content += "% This is a verification list, not a final BibTeX database.\n\n"
    content += body + "\n"
    (REV / "references_to_verify.bib").write_text(content, encoding="utf-8")


def write_reports() -> None:
    missing = """# Missing Data Report

## 可用且已进入主文的结果
- X-Fi 风格 HPE/HAR 逐组合表：来自 `HPE/HAR/legacy_xfi` 与 `student_vk_random_all_combinations.csv`。
- FSG 缺失强度表：来自 Student-VK 与 Student-NV all-combination CSV。
- Leave-one-out 表：由 full-modality 结果与移除单一模态组合重算。
- 跨协议表：来自 `tables/main_hpe_results.csv` 与 `tables/main_har_results.csv`。
- 消融表：来自 `tables/ablation_results.csv`，保留 uniform fusion 与 w/o distillation 的谨慎解释。
- 隐私泄露探针：来自已整理结果，主文只声称 reduced visual exposure，不声称匿名。

## 缺失或不建议强写的结果
- 缺少 formal privacy guarantee、identity attack 的完整协议；因此不能写 privacy-preserving 或 anonymous。
- body-latent alignment 诊断较弱，只能作为 discussion，不应作为严格语义对齐证明。
- reliability corruption 结果不是单调完美响应，只能说明 learned contribution proxy，不是校准物理可靠性。
- SuperTeacher 结果不稳定，不进入主贡献。
- Ori-XFI 244 mm 异常复现结果不进入主表，避免误导性比较。
"""
    (REV / "missing_data_report.md").write_text(missing, encoding="utf-8")

    summary = """# Revision Summary

## 生成文件
- `iotj_revised.tex`：基于当前 IoTJ 中文稿生成的修订主文。
- `tables_rebuilt.tex`：集中保存重建表格片段。
- `experiment_index.csv`：递归扫描结果文件并解析指标。
- `figures_index.csv`：递归扫描图像并建议正文/讨论位置。
- `references_to_verify.bib`：从主文提取的 40 条参考文献候选，需人工核验元数据。

## 主要修改
- 增加 Table I：模态符号与角色说明，统一 V/D/L/R/W 缩写。
- 保留并重建 X-Fi 风格逐组合 HPE/HAR 主表。
- 将 FSG 缺失强度表、leave-one-out 表、跨协议泛化表和核心消融表放入正文核心结果。
- 将身份泄露探针作为 VK 替代 RGB 的主要动机证据。
- 降调不安全表述：不声称 VK 匿名、不声称 reliability 是物理可靠性、不声称 FSG/KD 对所有任务统一提升。

## 进入主文的结果
- Privacy probe、Student-VK/NV、X-Fi direct comparison、FSG severity、leave-one-out、cross-protocol、uniform fusion/w-o distillation。

## 建议 supplementary/discussion 的结果
- reliability corruption diagnostic、body-latent diagnostic、VK semantic-anchor perturbation、SuperTeacher exploration。
"""
    (REV / "revision_summary.md").write_text(summary, encoding="utf-8")

    risk = """# Reviewer Risk Notes

## 1. VK 创新与 X-Fi 的区分
风险：审稿人可能认为只是将 RGB 换成 VK 并重新训练。
应对：主文强调 V 替代下游 RGB、身份泄露探针、FSG full-to-subset 可用性约束，以及逐组合缺失评估。

## 2. FSG 证据边界
风险：HPE w/o distillation 结果竞争力强，不能声称 teacher guidance 是唯一收益来源。
应对：写成完整模态行为约束和缺失子集训练范式；对 HAR 有辅助收益，对 HPE 不做统一提升承诺。

## 3. Reliability proxy 解释
风险：权重被误读为真实传感器可靠性。
应对：统一写 learned modality contribution proxy；corruption 只作为诊断，不作为强证明。

## 4. 隐私友好表述
风险：VK/Depth 仍可能泄露体型、步态、动作习惯。
应对：仅使用 reduced visual exposure / RGB-free inference；明确非 formal privacy。

## 5. 表格密度
风险：正文表格多，可能显得结果堆叠。
应对：每张表绑定一条主线：V 替代 RGB 或 FSG 缺失模态，而不是散放。
"""
    (REV / "reviewer_risk_notes.md").write_text(risk, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    # Ensure upstream generated X-Fi-style tables are fresh.
    import subprocess

    subprocess.run(["python", str(ROOT / "scripts" / "build_xfi_style_tables.py")], check=True)
    copy_generated_tables()
    write_extra_tables()
    build_tables_rebuilt()
    build_experiment_index()
    build_figures_index()
    build_revised_tex()
    extract_bib()
    write_reports()
    print(f"Wrote revision package to {REV}")


if __name__ == "__main__":
    main()
