from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HPE_ROOT = ROOT / "HPE"
HAR_ROOT = ROOT / "HAR"
REPORT_ROOT = ROOT / "reports"

HPE_OFFICIAL_ROWS = [
    ("vk", 93.9, 60.3), ("depth", 101.8, 48.4), ("lidar", 167.1, 103.2), ("mmwave", 127.4, 69.8), ("wifi-csi", 225.6, 105.3),
    ("vk+depth", 86.1, 48.1), ("vk+lidar", 93.0, 59.7), ("vk+mmwave", 88.8, 57.3), ("vk+wifi-csi", 93.0, 59.5),
    ("depth+lidar", 102.5, 48.4), ("depth+mmwave", 98.0, 47.3), ("depth+wifi-csi", 101.8, 48.1),
    ("lidar+mmwave", 109.8, 63.4), ("lidar+wifi-csi", 159.5, 102.7), ("mmwave+wifi-csi", 117.2, 62.7),
    ("vk+depth+lidar", 84.8, 48.2), ("vk+depth+mmwave", 83.4, 47.3), ("vk+depth+wifi-csi", 85.3, 48.1),
    ("vk+lidar+mmwave", 88.4, 57.2), ("vk+lidar+wifi-csi", 93.0, 59.7), ("vk+mmwave+wifi-csi", 88.5, 57.1),
    ("depth+lidar+mmwave", 96.0, 47.3), ("depth+lidar+wifi-csi", 102.0, 48.1), ("depth+mmwave+wifi-csi", 97.0, 47.1),
    ("lidar+mmwave+wifi-csi", 107.4, 63.1), ("vk+depth+lidar+mmwave", 83.5, 47.6), ("vk+depth+lidar+wifi-csi", 86.0, 48.2),
    ("vk+depth+mmwave+wifi-csi", 84.0, 47.6), ("vk+lidar+mmwave+wifi-csi", 88.6, 57.1), ("depth+lidar+mmwave+wifi-csi", 97.6, 47.4),
    ("vk+depth+lidar+mmwave+wifi-csi", 83.7, 47.6),
]

# Official X-Fi MM-Fi HAR values used by the local report generator. If you later replace this CSV with publisher-extracted
# numbers, the summarizer will preserve the existing file and use your version.
HAR_OFFICIAL_ROWS = [
    ("vk", 26.5), ("depth", 48.1), ("lidar", 52.7), ("mmwave", 85.7),
    ("vk+depth", 45.3), ("vk+lidar", 35.2), ("vk+mmwave", 73.4), ("depth+lidar", 51.6), ("depth+mmwave", 79.8), ("lidar+mmwave", 88.7),
    ("vk+depth+lidar", 48.7), ("vk+depth+mmwave", 70.7), ("vk+lidar+mmwave", 77.8), ("depth+lidar+mmwave", 80.5),
    ("vk+depth+lidar+mmwave", 72.2),
]

HPE_EXPECTED = {
    "models": [
        "teacher_full/best.pth", "student_vk_missing/best.pth", "student_nv_missing/best.pth", "baseline_full/best.pth",
        "ablation/no_kd/best.pth", "ablation/output_kd/best.pth", "ablation/output_token_kd/best.pth", "ablation/output_bone_kd/best.pth",
        "ablation/full_structural_kd/best.pth", "ablation/uniform/best.pth", "ablation/attention/best.pth", "ablation/uncertainty/best.pth",
        "ablation/mlp_vk/best.pth", "ablation/skeleton_prompt/best.pth", "teacher_full_cross_scene_split/best.pth",
        "student_vk_missing_cross_scene_split/best.pth", "student_nv_missing_cross_scene/best.pth", "baseline_full_cross_scene_split/best.pth",
        "teacher_full_cross_subject_split/best.pth", "student_vk_missing_cross_subject_split/best.pth", "student_nv_missing_cross_subject/best.pth",
        "baseline_full_cross_subject_split/best.pth",
    ],
    "evals": [
        "teacher_all_combinations.csv", "all_combinations.csv", "student_vk_missing_modality_summary.csv", "student_vk_noise_robustness.csv",
        "student_nv_nonvisual_combinations.csv", "baseline_full_all_combinations.csv", "ablation_distillation_no_kd_all_combinations.csv",
        "ablation_distillation_output_kd_all_combinations.csv", "ablation_distillation_output_token_kd_all_combinations.csv",
        "ablation_distillation_output_bone_kd_all_combinations.csv", "ablation_distillation_full_structural_kd_all_combinations.csv",
        "ablation_reliability_uniform_all_combinations.csv", "ablation_reliability_attention_all_combinations.csv",
        "ablation_reliability_uncertainty_all_combinations.csv", "ablation_encoder_mlp_vk_all_combinations.csv",
        "ablation_encoder_skeleton_prompt_all_combinations.csv", "teacher_cross_scene_all_combinations.csv", "student_vk_cross_scene_all_combinations.csv",
        "student_nv_cross_scene_nonvisual_combinations.csv", "baseline_cross_scene_all_combinations.csv", "teacher_cross_subject_all_combinations.csv",
        "student_vk_cross_subject_all_combinations.csv", "student_nv_cross_subject_nonvisual_combinations.csv", "baseline_cross_subject_all_combinations.csv",
    ],
}

HAR_EXPECTED = {
    "models": [
        "teacher_full/best.pth", "student_vk_missing/best.pth", "student_nv_missing/best.pth", "ablation_no_distill/best.pth",
        "ablation_uniform_fusion/best.pth", "teacher_full_cross_scene/best.pth", "student_vk_missing_cross_scene/best.pth",
        "student_nv_missing_cross_scene/best.pth", "teacher_full_cross_subject/best.pth", "student_vk_missing_cross_subject/best.pth",
        "student_nv_missing_cross_subject/best.pth",
    ],
    "evals": [
        "teacher_random_all_combinations.csv", "student_vk_random_all_combinations.csv", "student_nv_random_nonvisual_combinations.csv",
        "all_combinations.csv", "student_vk_all_combinations.csv", "student_nv_nonvisual_combinations.csv", "ablation_no_distill_all_combinations.csv",
        "ablation_uniform_fusion_all_combinations.csv", "teacher_cross_scene_all_combinations.csv", "student_vk_cross_scene_all_combinations.csv",
        "student_nv_cross_scene_nonvisual_combinations.csv", "teacher_cross_subject_all_combinations.csv", "student_vk_cross_subject_all_combinations.csv",
        "student_nv_cross_subject_nonvisual_combinations.csv",
    ],
}

@dataclass
class ProjectSpec:
    name: str
    root: Path
    report_dir: Path
    official_csv: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["status"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct_or_value(value: str | None) -> float:
    if value in (None, "", "NA", "nan"):
        return math.nan
    number = float(value)
    return number * 100.0 if abs(number) <= 1.0 else number


def mm_or_value(value: str) -> float:
    number = float(value)
    return number * 1000.0 if abs(number) < 10.0 else number


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.2f}"


def tex_escape(value: object) -> str:
    text = str(value)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(repl.get(char, char) for char in text)


def ensure_official_tables(root: Path) -> None:
    hpe_path = root / "HPE" / "legacy_xfi" / "xfi_hpe_table1.csv"
    har_path = root / "HAR" / "legacy_xfi" / "xfi_har_table6.csv"
    if not hpe_path.exists():
        write_csv(hpe_path, [{"modality_set": m, "xfi_mpjpe_mm": mp, "xfi_pa_mpjpe_mm": pa, "source": "X-Fi ICLR 2025 Table 1"} for m, mp, pa in HPE_OFFICIAL_ROWS], ["modality_set", "xfi_mpjpe_mm", "xfi_pa_mpjpe_mm", "source"])
    if not har_path.exists():
        write_csv(har_path, [{"modality_set": m, "xfi_acc_pct": acc, "source": "X-Fi ICLR 2025 Table 2 MM-Fi HAR"} for m, acc in HAR_OFFICIAL_ROWS], ["modality_set", "xfi_acc_pct", "source"])


def audit_project(spec: ProjectSpec, expected: dict[str, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in expected["models"]:
        path = spec.root / "outputs" / relative
        rows.append({"project": spec.name, "type": "model", "item": relative, "status": "DONE" if path.exists() else "MISSING_MODEL", "path": path, "bytes": path.stat().st_size if path.exists() else ""})
    for relative in expected["evals"]:
        path = spec.root / "outputs" / "eval" / relative
        rows.append({"project": spec.name, "type": "eval", "item": relative, "status": "DONE" if path.exists() else "MISSING_EVAL", "path": path, "bytes": path.stat().st_size if path.exists() else ""})
    return rows


def collect_hpe_results(spec: ProjectSpec) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    main: list[dict[str, object]] = []
    robust: list[dict[str, object]] = []
    for path in sorted((spec.root / "outputs" / "eval").glob("*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        keys = set(rows[0].keys())
        if {"modality_set", "mpjpe", "pa_mpjpe"}.issubset(keys):
            for row in rows:
                main.append({"project": "HPE", "file": path.name, "method": row.get("method", path.stem), "split": row.get("split", ""), "modality_set": row.get("modality_set", ""), "mpjpe_mm": fmt(mm_or_value(row["mpjpe"])), "pa_mpjpe_mm": fmt(mm_or_value(row["pa_mpjpe"])), "mse": row.get("mse", ""), "params": row.get("params", "")})
        if "noise_std" in keys or "drop_prob" in keys:
            for row in rows:
                out = dict(row)
                out["project"] = "HPE"
                out["file"] = path.name
                if "mpjpe" in out:
                    out["mpjpe_mm"] = fmt(mm_or_value(out["mpjpe"]))
                if "pa_mpjpe" in out:
                    out["pa_mpjpe_mm"] = fmt(mm_or_value(out["pa_mpjpe"]))
                robust.append(out)
    for path in sorted((spec.root / "outputs").glob("*/final_eval.csv")):
        for row in read_csv(path):
            if "mpjpe" in row:
                main.append({"project": "HPE", "file": str(path.relative_to(spec.root / "outputs")), "method": row.get("method", path.parent.name), "split": row.get("split", ""), "modality_set": row.get("modality_set", ""), "mpjpe_mm": fmt(mm_or_value(row["mpjpe"])), "pa_mpjpe_mm": fmt(mm_or_value(row["pa_mpjpe"])), "mse": row.get("mse", ""), "params": row.get("params", "")})
    return main, robust


def collect_har_results(spec: ProjectSpec) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in sorted((spec.root / "outputs" / "eval").glob("*.csv")):
        rows = read_csv(path)
        if not rows or "acc" not in rows[0]:
            continue
        for row in rows:
            out.append({"project": "HAR", "file": path.name, "method": row.get("method", path.stem), "split": row.get("split", ""), "modality_set": row.get("modality_set", ""), "acc_pct": fmt(pct_or_value(row.get("acc"))), "macro_f1_pct": fmt(pct_or_value(row.get("macro_f1"))), "loss": row.get("loss", ""), "params": row.get("params", "")})
    for path in sorted((spec.root / "outputs").glob("*/final_eval.csv")):
        for row in read_csv(path):
            if "acc" in row:
                out.append({"project": "HAR", "file": str(path.relative_to(spec.root / "outputs")), "method": row.get("method", path.parent.name), "split": row.get("split", ""), "modality_set": row.get("modality_set", ""), "acc_pct": fmt(pct_or_value(row.get("acc"))), "macro_f1_pct": fmt(pct_or_value(row.get("macro_f1"))), "loss": row.get("loss", ""), "params": row.get("params", "")})
    return out


def compare_hpe(result_csv: Path, official_csv: Path) -> list[dict[str, object]]:
    if not result_csv.exists() or not official_csv.exists():
        return []
    official = {row["modality_set"]: row for row in read_csv(official_csv)}
    out = []
    for row in read_csv(result_csv):
        modality = row.get("modality_set", "")
        if modality not in official or "mpjpe" not in row:
            continue
        ours_mpjpe, ours_pa = mm_or_value(row["mpjpe"]), mm_or_value(row["pa_mpjpe"])
        xfi_mpjpe, xfi_pa = float(official[modality]["xfi_mpjpe_mm"]), float(official[modality]["xfi_pa_mpjpe_mm"])
        out.append({"project": "HPE", "method": row.get("method", result_csv.stem), "split": row.get("split", ""), "source_file": result_csv.name, "modality_set": modality, "ours_mpjpe_mm": fmt(ours_mpjpe), "official_mpjpe_mm": fmt(xfi_mpjpe), "delta_mpjpe_mm": fmt(ours_mpjpe - xfi_mpjpe), "mpjpe_trend": "IMPROVED" if ours_mpjpe < xfi_mpjpe else "DEGRADED" if ours_mpjpe > xfi_mpjpe else "TIE", "ours_pa_mpjpe_mm": fmt(ours_pa), "official_pa_mpjpe_mm": fmt(xfi_pa), "delta_pa_mpjpe_mm": fmt(ours_pa - xfi_pa), "pa_mpjpe_trend": "IMPROVED" if ours_pa < xfi_pa else "DEGRADED" if ours_pa > xfi_pa else "TIE"})
    return out


def compare_har(result_csv: Path, official_csv: Path) -> list[dict[str, object]]:
    if not result_csv.exists() or not official_csv.exists():
        return []
    official = {row["modality_set"]: row for row in read_csv(official_csv)}
    out = []
    for row in read_csv(result_csv):
        modality = row.get("modality_set", "")
        if modality not in official or "acc" not in row:
            continue
        ours_acc, xfi_acc = pct_or_value(row["acc"]), float(official[modality]["xfi_acc_pct"])
        out.append({"project": "HAR", "method": row.get("method", result_csv.stem), "split": row.get("split", ""), "source_file": result_csv.name, "modality_set": modality, "ours_acc_pct": fmt(ours_acc), "official_acc_pct": fmt(xfi_acc), "delta_acc_pct": fmt(ours_acc - xfi_acc), "acc_trend": "IMPROVED" if ours_acc > xfi_acc else "DEGRADED" if ours_acc < xfi_acc else "TIE", "ours_macro_f1_pct": fmt(pct_or_value(row.get("macro_f1")))})
    return out


def tex_color(delta: float, improved: bool) -> str:
    if not math.isfinite(delta):
        return "NA"
    color = "blue" if improved else "red" if abs(delta) > 1e-9 else "gray"
    sign = "+" if delta > 0 else ""
    return rf"\textcolor{{{color}}}{{{sign}{delta:.2f}}}"


def write_tex(path: Path, hpe_rows: list[dict[str, object]], har_rows: list[dict[str, object]], audit_rows: list[dict[str, object]]) -> None:
    hpe_done = sum(1 for r in audit_rows if r["project"] == "HPE" and r["status"] == "DONE")
    hpe_total = sum(1 for r in audit_rows if r["project"] == "HPE")
    har_done = sum(1 for r in audit_rows if r["project"] == "HAR" and r["status"] == "DONE")
    har_total = sum(1 for r in audit_rows if r["project"] == "HAR")
    lines = [
        r"\documentclass[10pt]{article}", r"\usepackage[margin=0.55in]{geometry}", r"\usepackage{booktabs}", r"\usepackage{xcolor}", r"\usepackage{longtable}", r"\begin{document}",
        r"\section*{HPE and HAR First-Round Result Summary}",
        r"Blue indicates improvement over official X-Fi; red indicates degradation. HPE uses MPJPE/PA-MPJPE where lower is better. HAR uses accuracy where higher is better.",
        r"\subsection*{Completion Audit}", r"\begin{tabular}{lrr}", r"\toprule", r"Project & Done & Total\\", r"\midrule", rf"HPE & {hpe_done} & {hpe_total}\\", rf"HAR & {har_done} & {har_total}\\", r"\bottomrule", r"\end{tabular}",
        r"\subsection*{HPE Official Comparison}", r"\scriptsize", r"\begin{longtable}{llllrrrr}", r"\toprule", r"Method & Split & Source & Modality & Ours MPJPE & X-Fi MPJPE & $\Delta$ & PA $\Delta$\\", r"\midrule", r"\endfirsthead", r"\toprule", r"Method & Split & Source & Modality & Ours MPJPE & X-Fi MPJPE & $\Delta$ & PA $\Delta$\\", r"\midrule", r"\endhead",
    ]
    for r in hpe_rows:
        d, pa = float(r["delta_mpjpe_mm"]), float(r["delta_pa_mpjpe_mm"])
        lines.append(rf"{tex_escape(r['method'])} & {tex_escape(r['split'])} & {tex_escape(r['source_file'])} & {tex_escape(r['modality_set'])} & {r['ours_mpjpe_mm']} & {r['official_mpjpe_mm']} & {tex_color(d, d < 0)} & {tex_color(pa, pa < 0)}\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\subsection*{HAR Official Comparison}", r"\begin{longtable}{llllrrrr}", r"\toprule", r"Method & Split & Source & Modality & Ours Acc. & X-Fi Acc. & $\Delta$ & Macro-F1\\", r"\midrule", r"\endfirsthead", r"\toprule", r"Method & Split & Source & Modality & Ours Acc. & X-Fi Acc. & $\Delta$ & Macro-F1\\", r"\midrule", r"\endhead"]
    for r in har_rows:
        d = float(r["delta_acc_pct"])
        lines.append(rf"{tex_escape(r['method'])} & {tex_escape(r['split'])} & {tex_escape(r['source_file'])} & {tex_escape(r['modality_set'])} & {r['ours_acc_pct']} & {r['official_acc_pct']} & {tex_color(d, d > 0)} & {r['ours_macro_f1_pct']}\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\normalsize", r"\end{document}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hpe_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\documentclass[10pt]{article}", r"\usepackage[margin=0.55in]{geometry}", r"\usepackage{booktabs}", r"\usepackage{xcolor}", r"\usepackage{longtable}", r"\begin{document}",
        r"\section*{HPE Official X-Fi Comparison}",
        r"Blue indicates lower error than official X-Fi; red indicates higher error.",
        r"\scriptsize", r"\begin{longtable}{llllrrrr}", r"\toprule", r"Method & Split & Source & Modality & Ours MPJPE & X-Fi MPJPE & $\Delta$ & PA $\Delta$\\", r"\midrule", r"\endfirsthead", r"\toprule", r"Method & Split & Source & Modality & Ours MPJPE & X-Fi MPJPE & $\Delta$ & PA $\Delta$\\", r"\midrule", r"\endhead",
    ]
    for r in rows:
        d, pa = float(r["delta_mpjpe_mm"]), float(r["delta_pa_mpjpe_mm"])
        lines.append(rf"{tex_escape(r['method'])} & {tex_escape(r['split'])} & {tex_escape(r['source_file'])} & {tex_escape(r['modality_set'])} & {r['ours_mpjpe_mm']} & {r['official_mpjpe_mm']} & {tex_color(d, d < 0)} & {tex_color(pa, pa < 0)}\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\end{document}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_har_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\documentclass[10pt]{article}", r"\usepackage[margin=0.55in]{geometry}", r"\usepackage{booktabs}", r"\usepackage{xcolor}", r"\usepackage{longtable}", r"\begin{document}",
        r"\section*{HAR Official X-Fi Comparison}",
        r"Blue indicates higher accuracy than official X-Fi; red indicates lower accuracy.",
        r"\scriptsize", r"\begin{longtable}{llllrrrr}", r"\toprule", r"Method & Split & Source & Modality & Ours Acc. & X-Fi Acc. & $\Delta$ & Macro-F1\\", r"\midrule", r"\endfirsthead", r"\toprule", r"Method & Split & Source & Modality & Ours Acc. & X-Fi Acc. & $\Delta$ & Macro-F1\\", r"\midrule", r"\endhead",
    ]
    for r in rows:
        d = float(r["delta_acc_pct"])
        lines.append(rf"{tex_escape(r['method'])} & {tex_escape(r['split'])} & {tex_escape(r['source_file'])} & {tex_escape(r['modality_set'])} & {r['ours_acc_pct']} & {r['official_acc_pct']} & {tex_color(d, d > 0)} & {r['ours_macro_f1_pct']}\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\end{document}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_tex(tex_path: Path) -> Path | None:
    engine = next((n for n in ("pdflatex", "xelatex") if shutil.which(n)), None)
    if engine is None:
        return None
    try:
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tex_path.parent}", str(tex_path)], check=True, capture_output=True, text=True, timeout=120)
    except Exception as exc:
        tex_path.with_suffix(".latex_error.txt").write_text(str(exc), encoding="utf-8")
        return None
    pdf = tex_path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser("Summarize first-round HPE and HAR results.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reports", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    ensure_official_tables(args.root)
    hpe = ProjectSpec("HPE", args.root / "HPE", args.reports / "HPE", args.root / "HPE" / "legacy_xfi" / "xfi_hpe_table1.csv")
    har = ProjectSpec("HAR", args.root / "HAR", args.reports / "HAR", args.root / "HAR" / "legacy_xfi" / "xfi_har_table6.csv")
    hpe_audit, har_audit = audit_project(hpe, HPE_EXPECTED), audit_project(har, HAR_EXPECTED)
    all_audit = hpe_audit + har_audit
    write_csv(args.reports / "completion_audit.csv", all_audit)
    write_csv(hpe.report_dir / "completion_audit.csv", hpe_audit)
    write_csv(har.report_dir / "completion_audit.csv", har_audit)
    hpe_main, hpe_robust = collect_hpe_results(hpe)
    har_main = collect_har_results(har)
    write_csv(hpe.report_dir / "hpe_main_results.csv", hpe_main)
    write_csv(hpe.report_dir / "hpe_robustness_results.csv", hpe_robust)
    write_csv(har.report_dir / "har_main_results.csv", har_main)
    hpe_cmp: list[dict[str, object]] = []
    for name in ["teacher_all_combinations.csv", "all_combinations.csv", "baseline_full_all_combinations.csv"]:
        hpe_cmp.extend(compare_hpe(hpe.root / "outputs" / "eval" / name, hpe.official_csv))
    har_cmp: list[dict[str, object]] = []
    for name in ["teacher_random_all_combinations.csv", "student_vk_random_all_combinations.csv", "all_combinations.csv", "ablation_no_distill_all_combinations.csv", "ablation_uniform_fusion_all_combinations.csv"]:
        har_cmp.extend(compare_har(har.root / "outputs" / "eval" / name, har.official_csv))
    write_csv(hpe.report_dir / "hpe_official_comparison.csv", hpe_cmp)
    write_csv(har.report_dir / "har_official_comparison.csv", har_cmp)
    write_csv(args.reports / "official_comparison_all.csv", hpe_cmp + har_cmp)
    hpe_tex = hpe.report_dir / "hpe_official_comparison.tex"
    har_tex = har.report_dir / "har_official_comparison.tex"
    write_hpe_tex(hpe_tex, hpe_cmp)
    write_har_tex(har_tex, har_cmp)
    hpe_pdf = compile_tex(hpe_tex)
    har_pdf = compile_tex(har_tex)
    tex_path = args.reports / "hpe_har_official_comparison.tex"
    write_tex(tex_path, hpe_cmp, har_cmp, all_audit)
    pdf = compile_tex(tex_path)
    print("Generated reports:")
    for p in [args.reports / "completion_audit.csv", hpe.report_dir / "hpe_main_results.csv", hpe.report_dir / "hpe_robustness_results.csv", hpe.report_dir / "hpe_official_comparison.csv", har.report_dir / "har_main_results.csv", har.report_dir / "har_official_comparison.csv", args.reports / "official_comparison_all.csv", hpe_tex, hpe_pdf, har_tex, har_pdf, tex_path, pdf]:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
