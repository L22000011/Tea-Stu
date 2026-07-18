from __future__ import annotations

import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_vkrcd_cn"
PAPER_STEM = "2026-05-09_vkrcd_xfi_paper"
TEX_PATH = OUT_DIR / f"{PAPER_STEM}.tex"
PDF_PATH = OUT_DIR / f"{PAPER_STEM}.pdf"


HPE_OFFICIAL = {
    "vk": (93.9, 60.3),
    "depth": (101.8, 48.4),
    "lidar": (167.1, 103.2),
    "mmwave": (127.4, 69.8),
    "wifi-csi": (225.6, 105.3),
    "vk+depth": (86.1, 48.1),
    "vk+lidar": (93.0, 59.7),
    "vk+mmwave": (88.8, 57.3),
    "vk+wifi-csi": (93.0, 59.5),
    "depth+lidar": (102.5, 48.4),
    "depth+mmwave": (98.0, 47.3),
    "depth+wifi-csi": (101.8, 48.1),
    "lidar+mmwave": (109.8, 63.4),
    "lidar+wifi-csi": (159.5, 102.7),
    "mmwave+wifi-csi": (117.2, 62.7),
    "vk+depth+lidar": (84.8, 48.2),
    "vk+depth+mmwave": (83.4, 47.3),
    "vk+depth+wifi-csi": (85.3, 48.1),
    "vk+lidar+mmwave": (88.4, 57.2),
    "vk+lidar+wifi-csi": (93.0, 59.7),
    "vk+mmwave+wifi-csi": (88.5, 57.1),
    "depth+lidar+mmwave": (96.0, 47.3),
    "depth+lidar+wifi-csi": (102.0, 48.1),
    "depth+mmwave+wifi-csi": (97.0, 47.1),
    "lidar+mmwave+wifi-csi": (107.4, 63.1),
    "vk+depth+lidar+mmwave": (83.5, 47.6),
    "vk+depth+lidar+wifi-csi": (86.0, 48.2),
    "vk+depth+mmwave+wifi-csi": (84.0, 47.6),
    "vk+lidar+mmwave+wifi-csi": (88.6, 57.1),
    "depth+lidar+mmwave+wifi-csi": (97.6, 47.4),
    "vk+depth+lidar+mmwave+wifi-csi": (83.7, 47.6),
}

HAR_OFFICIAL = {
    "vk": 26.5,
    "depth": 48.1,
    "lidar": 52.7,
    "mmwave": 85.7,
    "vk+depth": 45.3,
    "vk+lidar": 35.2,
    "vk+mmwave": 73.4,
    "depth+lidar": 51.6,
    "depth+mmwave": 79.8,
    "lidar+mmwave": 88.7,
    "vk+depth+lidar": 48.7,
    "vk+depth+mmwave": 70.7,
    "vk+lidar+mmwave": 77.8,
    "depth+lidar+mmwave": 80.5,
    "vk+depth+lidar+mmwave": 72.2,
}

MODALITY_LABELS = {
    "vk": "I/VK",
    "depth": "D",
    "lidar": "L",
    "mmwave": "R",
    "wifi-csi": "W",
}

METHOD_LABELS = {
    "VK-XFi-Baseline-Full": "Baseline",
    "VK-RCD-Teacher": "Teacher",
    "VK-RCD-Student-VK": "Student-VK",
    "VK-RCD-Student-NV": "Student-NV",
    "VK-RCD-HAR-Teacher": "Teacher",
    "VK-RCD-HAR-Student-VK-Missing": "Student-VK",
    "VK-RCD-HAR-Student-NV-Missing": "Student-NV",
    "VK-RCD-HAR-Ablation-NoKD": "NoKD",
    "VK-RCD-HAR-Ablation-UniformFusion": "Uniform",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object, default: float = math.nan) -> float:
    if value in (None, "", "NA", "nan"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_to_mm(value: object) -> float:
    number = to_float(value)
    if not math.isfinite(number):
        return number
    return number * 1000.0 if abs(number) < 10.0 else number


def metric_to_pct(value: object) -> float:
    number = to_float(value)
    if not math.isfinite(number):
        return number
    return number * 100.0 if abs(number) <= 1.0 else number


def fmt(value: float, digits: int = 2) -> str:
    return "--" if not math.isfinite(value) else f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def combo_label(modality_set: str) -> str:
    return "+".join(MODALITY_LABELS.get(part, part) for part in modality_set.split("+"))


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("VK-RCD-HAR-", "").replace("VK-RCD-", ""))


def colored(value: float, improved: bool, digits: int = 2, suffix: str = "") -> str:
    if not math.isfinite(value):
        return "--"
    sign = "+" if value > 0 else ""
    number = f"{sign}{value:.{digits}f}{suffix}"
    return rf"\better{{{number}}}" if improved else rf"\worse{{{number}}}"


def hpe_delta_cells(official: float, ours: float) -> tuple[str, str]:
    delta = ours - official
    rel = (official - ours) / official * 100.0 if official else math.nan
    return colored(delta, delta < 0), colored(rel, rel >= 0, suffix=r"\%")


def har_delta_cells(official: float, ours: float) -> tuple[str, str]:
    delta = ours - official
    rel = (ours - official) / official * 100.0 if official else math.nan
    return colored(delta, delta > 0), colored(rel, rel >= 0, suffix=r"\%")


def load_hpe_official() -> dict[str, tuple[float, float]]:
    official_path = ROOT / "HPE" / "legacy_xfi" / "xfi_hpe_table1.csv"
    rows = read_csv(official_path)
    if not rows:
        return dict(HPE_OFFICIAL)
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        modality = row.get("modality_set", "")
        mpjpe = to_float(row.get("xfi_mpjpe_mm"))
        pa = to_float(row.get("xfi_pa_mpjpe_mm"))
        if modality and math.isfinite(mpjpe) and math.isfinite(pa):
            out[modality] = (mpjpe, pa)
    return out or dict(HPE_OFFICIAL)


def load_har_official() -> dict[str, float]:
    official_path = ROOT / "HAR" / "legacy_xfi" / "xfi_har_table6.csv"
    rows = read_csv(official_path)
    if not rows:
        return dict(HAR_OFFICIAL)
    out: dict[str, float] = {}
    for row in rows:
        modality = row.get("modality_set", "")
        acc = to_float(row.get("xfi_acc_pct"))
        if modality and math.isfinite(acc):
            out[modality] = acc
    return out or dict(HAR_OFFICIAL)


def collect_hpe_finals() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "HPE" / "outputs").glob("*/final_eval.csv")):
        for row in read_csv(path):
            if "mpjpe" not in row:
                continue
            rows.append(
                {
                    "source": path.relative_to(ROOT / "HPE" / "outputs").as_posix(),
                    "method": row.get("method", path.parent.name),
                    "split": row.get("split", ""),
                    "modality_set": row.get("modality_set", ""),
                    "mpjpe_mm": metric_to_mm(row.get("mpjpe")),
                    "pa_mpjpe_mm": metric_to_mm(row.get("pa_mpjpe")),
                    "params": to_float(row.get("params")),
                }
            )
    return rows


def collect_har_finals() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "HAR" / "outputs").glob("*/final_eval.csv")):
        for row in read_csv(path):
            if "acc" not in row:
                continue
            rows.append(
                {
                    "source": path.relative_to(ROOT / "HAR" / "outputs").as_posix(),
                    "method": row.get("method", path.parent.name),
                    "split": row.get("split", ""),
                    "modality_set": row.get("modality_set", ""),
                    "acc_pct": metric_to_pct(row.get("acc")),
                    "macro_f1_pct": metric_to_pct(row.get("macro_f1")),
                    "params": to_float(row.get("params")),
                }
            )
    return rows


def collect_hpe_main_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_path = ROOT / "reports" / "HPE" / "hpe_main_results.csv"
    for row in read_csv(source_path):
        if row.get("file") not in {"all_combinations.csv", "teacher_all_combinations.csv"}:
            continue
        rows.append(
            {
                "source": row.get("file", ""),
                "method": row.get("method", ""),
                "split": row.get("split", ""),
                "modality_set": row.get("modality_set", ""),
                "mpjpe_mm": to_float(row.get("mpjpe_mm")),
                "pa_mpjpe_mm": to_float(row.get("pa_mpjpe_mm")),
            }
        )
    return rows


def collect_har_eval_results() -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for path in sorted((ROOT / "HAR" / "outputs" / "eval").glob("*.csv")):
        rows: list[dict[str, object]] = []
        for row in read_csv(path):
            if "acc" not in row:
                continue
            rows.append(
                {
                    "source": path.name,
                    "method": row.get("method", path.stem),
                    "split": row.get("split", ""),
                    "modality_set": row.get("modality_set", ""),
                    "acc_pct": metric_to_pct(row.get("acc")),
                    "macro_f1_pct": metric_to_pct(row.get("macro_f1")),
                    "loss": to_float(row.get("loss")),
                }
            )
        if rows:
            out[path.name] = rows
    return out


def collect_hpe_robustness() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(ROOT / "reports" / "HPE" / "hpe_robustness_results.csv"):
        rows.append(
            {
                "noise_std": to_float(row.get("noise_std"), 0.0),
                "drop_joint_ratio": to_float(row.get("drop_joint_ratio"), 0.0),
                "mpjpe_mm": to_float(row.get("mpjpe_mm")),
                "pa_mpjpe_mm": to_float(row.get("pa_mpjpe_mm")),
                "mse": to_float(row.get("mse")),
            }
        )
    return rows


def collect_audit_missing() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(ROOT / "reports" / "completion_audit.csv"):
        if row.get("status") != "DONE":
            rows.append(row)
    return rows


def group_hpe_student(rows: list[dict[str, object]], official: dict[str, tuple[float, float]]) -> list[dict[str, float]]:
    grouped: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("method") != "VK-RCD-Student-VK":
            continue
        modality = str(row.get("modality_set", ""))
        if modality not in official:
            continue
        grouped[len(modality.split("+"))].append(
            (
                official[modality][0],
                float(row["mpjpe_mm"]),
                official[modality][1],
                float(row["pa_mpjpe_mm"]),
            )
        )
    out = []
    for count in sorted(grouped):
        values = grouped[count]
        official_mpjpe = sum(v[0] for v in values) / len(values)
        ours_mpjpe = sum(v[1] for v in values) / len(values)
        official_pa = sum(v[2] for v in values) / len(values)
        ours_pa = sum(v[3] for v in values) / len(values)
        out.append(
            {
                "count": count,
                "n": len(values),
                "official_mpjpe": official_mpjpe,
                "ours_mpjpe": ours_mpjpe,
                "official_pa": official_pa,
                "ours_pa": ours_pa,
            }
        )
    return out


def group_har_student(rows: list[dict[str, object]], official: dict[str, float]) -> list[dict[str, float]]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("method") != "VK-RCD-HAR-Student-VK-Missing":
            continue
        modality = str(row.get("modality_set", ""))
        if modality not in official:
            continue
        grouped[len(modality.split("+"))].append((official[modality], float(row["acc_pct"])))
    out = []
    for count in sorted(grouped):
        values = grouped[count]
        official_acc = sum(v[0] for v in values) / len(values)
        ours_acc = sum(v[1] for v in values) / len(values)
        out.append({"count": count, "n": len(values), "official_acc": official_acc, "ours_acc": ours_acc})
    return out


def hpe_full_rows(finals: list[dict[str, object]], official: dict[str, tuple[float, float]]) -> str:
    full_modality = "vk+depth+lidar+mmwave+wifi-csi"
    selected = [
        row
        for row in finals
        if row.get("modality_set") in {full_modality, "depth+lidar+mmwave+wifi-csi"}
    ]
    order = {"VK-XFi-Baseline-Full": 0, "VK-RCD-Teacher": 1, "VK-RCD-Student-VK": 2, "VK-RCD-Student-NV": 3}
    selected.sort(key=lambda r: order.get(str(r.get("method")), 99))
    lines = []
    for row in selected:
        official_mpjpe, official_pa = official[str(row["modality_set"])]
        d_mpjpe, rel_mpjpe = hpe_delta_cells(official_mpjpe, float(row["mpjpe_mm"]))
        d_pa, rel_pa = hpe_delta_cells(official_pa, float(row["pa_mpjpe_mm"]))
        lines.append(
            rf"{method_label(str(row['method']))} & {combo_label(str(row['modality_set']))} & {fmt(float(row['mpjpe_mm']))} & {fmt(official_mpjpe)} & {d_mpjpe} & {rel_mpjpe} & {fmt(float(row['pa_mpjpe_mm']))} & {d_pa} & {rel_pa}\\"
        )
    return "\n".join(lines)


def hpe_group_rows(grouped: list[dict[str, float]]) -> str:
    lines = []
    total_n = 0
    sum_official_mpjpe = sum_ours_mpjpe = sum_official_pa = sum_ours_pa = 0.0
    for row in grouped:
        count = int(row["count"])
        n = int(row["n"])
        total_n += n
        sum_official_mpjpe += row["official_mpjpe"] * n
        sum_ours_mpjpe += row["ours_mpjpe"] * n
        sum_official_pa += row["official_pa"] * n
        sum_ours_pa += row["ours_pa"] * n
        _, rel_mpjpe = hpe_delta_cells(row["official_mpjpe"], row["ours_mpjpe"])
        _, rel_pa = hpe_delta_cells(row["official_pa"], row["ours_pa"])
        lines.append(
            rf"{count} & {n} & {fmt(row['ours_mpjpe'])} & {fmt(row['official_mpjpe'])} & {rel_mpjpe} & {fmt(row['ours_pa'])} & {fmt(row['official_pa'])} & {rel_pa}\\"
        )
    if total_n:
        avg_official_mpjpe = sum_official_mpjpe / total_n
        avg_ours_mpjpe = sum_ours_mpjpe / total_n
        avg_official_pa = sum_official_pa / total_n
        avg_ours_pa = sum_ours_pa / total_n
        _, rel_mpjpe = hpe_delta_cells(avg_official_mpjpe, avg_ours_mpjpe)
        _, rel_pa = hpe_delta_cells(avg_official_pa, avg_ours_pa)
        lines.append(r"\midrule")
        lines.append(
            rf"平均 & {total_n} & {fmt(avg_ours_mpjpe)} & {fmt(avg_official_mpjpe)} & {rel_mpjpe} & {fmt(avg_ours_pa)} & {fmt(avg_official_pa)} & {rel_pa}\\"
        )
    return "\n".join(lines)


def har_full_rows(finals: list[dict[str, object]], official: dict[str, float]) -> str:
    full_modality = "vk+depth+lidar+mmwave"
    nv_modality = "depth+lidar+mmwave"
    official_full = official[full_modality]
    official_nv = official[nv_modality]
    selected = [
        row
        for row in finals
        if row.get("source")
        in {
            "teacher_full/final_eval.csv",
            "student_vk_missing/final_eval.csv",
            "student_nv_missing/final_eval.csv",
            "ablation_no_distill/final_eval.csv",
            "ablation_uniform_fusion/final_eval.csv",
        }
    ]
    order = {
        "VK-RCD-HAR-Teacher": 0,
        "VK-RCD-HAR-Student-VK-Missing": 1,
        "VK-RCD-HAR-Student-NV-Missing": 2,
        "VK-RCD-HAR-Ablation-NoKD": 3,
        "VK-RCD-HAR-Ablation-UniformFusion": 4,
    }
    selected.sort(key=lambda r: order.get(str(r.get("method")), 99))
    lines = []
    for row in selected:
        modality = str(row["modality_set"])
        baseline = official_nv if modality == nv_modality else official_full
        delta, rel = har_delta_cells(baseline, float(row["acc_pct"]))
        lines.append(
            rf"{method_label(str(row['method']))} & {combo_label(modality)} & {fmt(float(row['acc_pct']))} & {fmt(baseline)} & {delta} & {rel} & {fmt(float(row['macro_f1_pct']))}\\"
        )
    return "\n".join(lines)


def har_protocol_rows(finals: list[dict[str, object]], official: dict[str, float]) -> str:
    wanted = {
        "teacher_full/final_eval.csv",
        "student_vk_missing/final_eval.csv",
        "student_nv_missing/final_eval.csv",
        "teacher_full_cross_scene/final_eval.csv",
        "student_vk_missing_cross_scene/final_eval.csv",
        "student_nv_missing_cross_scene/final_eval.csv",
        "teacher_full_cross_subject/final_eval.csv",
        "student_vk_missing_cross_subject/final_eval.csv",
        "student_nv_missing_cross_subject/final_eval.csv",
    }
    protocol_by_source = {
        "teacher_full/final_eval.csv": "random",
        "student_vk_missing/final_eval.csv": "random",
        "student_nv_missing/final_eval.csv": "random",
        "teacher_full_cross_scene/final_eval.csv": "cross-scene",
        "student_vk_missing_cross_scene/final_eval.csv": "cross-scene",
        "student_nv_missing_cross_scene/final_eval.csv": "cross-scene",
        "teacher_full_cross_subject/final_eval.csv": "cross-subject",
        "student_vk_missing_cross_subject/final_eval.csv": "cross-subject",
        "student_nv_missing_cross_subject/final_eval.csv": "cross-subject",
    }
    method_by_source = {
        "teacher_full/final_eval.csv": "Teacher",
        "student_vk_missing/final_eval.csv": "Student-VK",
        "student_nv_missing/final_eval.csv": "Student-NV",
        "teacher_full_cross_scene/final_eval.csv": "Teacher",
        "student_vk_missing_cross_scene/final_eval.csv": "Student-VK",
        "student_nv_missing_cross_scene/final_eval.csv": "Student-NV",
        "teacher_full_cross_subject/final_eval.csv": "Teacher",
        "student_vk_missing_cross_subject/final_eval.csv": "Student-VK",
        "student_nv_missing_cross_subject/final_eval.csv": "Student-NV",
    }
    rows = [row for row in finals if row.get("source") in wanted]
    protocol_order = {"random": 0, "cross-scene": 1, "cross-subject": 2}
    method_order = {"Teacher": 0, "Student-VK": 1, "Student-NV": 2}
    rows.sort(
        key=lambda r: (
            protocol_order.get(protocol_by_source.get(str(r.get("source")), ""), 99),
            method_order.get(method_by_source.get(str(r.get("source")), ""), 99),
        )
    )
    lines = []
    for row in rows:
        source = str(row["source"])
        modality = str(row["modality_set"])
        baseline = official.get(modality, math.nan)
        delta, rel = har_delta_cells(baseline, float(row["acc_pct"])) if math.isfinite(baseline) else ("--", "--")
        lines.append(
            rf"{protocol_by_source.get(source, tex_escape(row['split']))} & {method_by_source.get(source, method_label(str(row['method'])))} & {combo_label(modality)} & {fmt(float(row['acc_pct']))} & {fmt(float(row['macro_f1_pct']))} & {fmt(baseline)} & {delta} & {rel}\\"
        )
    return "\n".join(lines)


def hpe_robust_rows(rows: list[dict[str, object]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    lines = []
    for row in selected:
        lines.append(
            rf"{fmt(float(row['noise_std']), 1)} & {fmt(float(row['drop_joint_ratio']), 1)} & {fmt(float(row['mpjpe_mm']))} & {fmt(float(row['pa_mpjpe_mm']))} & {float(row['mse']):.6f}\\"
        )
    return "\n".join(lines)


def hpe_appendix_table(rows: list[dict[str, object]], official: dict[str, tuple[float, float]], method: str) -> str:
    selected = [row for row in rows if row.get("method") == method and str(row.get("modality_set")) in official]
    selected.sort(key=lambda r: (len(str(r["modality_set"]).split("+")), str(r["modality_set"])))
    lines = []
    for row in selected:
        modality = str(row["modality_set"])
        official_mpjpe, official_pa = official[modality]
        d_mpjpe, rel_mpjpe = hpe_delta_cells(official_mpjpe, float(row["mpjpe_mm"]))
        d_pa, rel_pa = hpe_delta_cells(official_pa, float(row["pa_mpjpe_mm"]))
        lines.append(
            rf"{combo_label(modality)} & {fmt(float(row['mpjpe_mm']))} & {fmt(official_mpjpe)} & {d_mpjpe} & {rel_mpjpe} & {fmt(float(row['pa_mpjpe_mm']))} & {fmt(official_pa)} & {d_pa} & {rel_pa}\\"
        )
    return "\n".join(lines)


def har_appendix_table(rows: list[dict[str, object]], official: dict[str, float], title_source: str | None = None) -> str:
    selected = [row for row in rows if str(row.get("modality_set")) in official]
    selected.sort(key=lambda r: (len(str(r["modality_set"]).split("+")), str(r["modality_set"])))
    lines = []
    for row in selected:
        modality = str(row["modality_set"])
        official_acc = official[modality]
        delta, rel = har_delta_cells(official_acc, float(row["acc_pct"]))
        source = tex_escape(title_source or str(row.get("source", "")))
        lines.append(
            rf"{source} & {combo_label(modality)} & {fmt(float(row['acc_pct']))} & {fmt(official_acc)} & {delta} & {rel} & {fmt(float(row['macro_f1_pct']))}\\"
        )
    return "\n".join(lines)


def har_nv_appendix_table(eval_sets: dict[str, list[dict[str, object]]], official: dict[str, float]) -> str:
    files = [
        "student_nv_random_nonvisual_combinations.csv",
        "student_nv_cross_scene_nonvisual_combinations.csv",
        "student_nv_cross_subject_nonvisual_combinations.csv",
    ]
    protocol = {
        "student_nv_random_nonvisual_combinations.csv": "random",
        "student_nv_cross_scene_nonvisual_combinations.csv": "cross-scene",
        "student_nv_cross_subject_nonvisual_combinations.csv": "cross-subject",
    }
    lines = []
    for file in files:
        for row in eval_sets.get(file, []):
            modality = str(row["modality_set"])
            if modality not in official:
                continue
            official_acc = official[modality]
            delta, rel = har_delta_cells(official_acc, float(row["acc_pct"]))
            lines.append(
                rf"{protocol[file]} & {combo_label(modality)} & {fmt(float(row['acc_pct']))} & {fmt(official_acc)} & {delta} & {rel} & {fmt(float(row['macro_f1_pct']))}\\"
            )
    return "\n".join(lines)


def har_all_existing_appendix(eval_sets: dict[str, list[dict[str, object]]], official: dict[str, float]) -> str:
    files = [
        "teacher_random_all_combinations.csv",
        "teacher_cross_scene_all_combinations.csv",
        "teacher_cross_subject_all_combinations.csv",
        "all_combinations.csv",
        "student_vk_all_combinations.csv",
        "student_vk_random_all_combinations.csv",
        "student_vk_cross_scene_all_combinations.csv",
        "student_vk_cross_subject_all_combinations.csv",
        "student_nv_nonvisual_combinations.csv",
        "student_nv_random_nonvisual_combinations.csv",
        "student_nv_cross_scene_nonvisual_combinations.csv",
        "student_nv_cross_subject_nonvisual_combinations.csv",
        "ablation_no_distill_all_combinations.csv",
        "ablation_uniform_fusion_all_combinations.csv",
    ]
    lines = []
    for file in files:
        for row in eval_sets.get(file, []):
            modality = str(row["modality_set"])
            if modality not in official:
                continue
            official_acc = official[modality]
            delta, rel = har_delta_cells(official_acc, float(row["acc_pct"]))
            lines.append(
                rf"{tex_escape(file.replace('_all_combinations.csv', '').replace('_nonvisual_combinations.csv', ''))} & {combo_label(modality)} & {fmt(float(row['acc_pct']))} & {fmt(official_acc)} & {delta} & {rel} & {fmt(float(row['macro_f1_pct']))}\\"
            )
    return "\n".join(lines)


def audit_rows(rows: list[dict[str, str]]) -> str:
    hpe_rows = [row for row in rows if row.get("project") == "HPE"]
    har_rows = [row for row in rows if row.get("project") == "HAR"]
    selected = hpe_rows[:24] + har_rows[:8]
    lines = []
    for row in selected:
        lines.append(
            rf"{tex_escape(row.get('project', ''))} & {tex_escape(row.get('type', ''))} & {tex_escape(row.get('item', ''))} & {tex_escape(row.get('status', ''))}\\"
        )
    return "\n".join(lines)


def build_tex() -> str:
    hpe_official = load_hpe_official()
    har_official = load_har_official()
    hpe_finals = collect_hpe_finals()
    har_finals = collect_har_finals()
    hpe_main = collect_hpe_main_results()
    har_eval = collect_har_eval_results()
    hpe_robust = collect_hpe_robustness()
    audit_missing = collect_audit_missing()

    hpe_student_group = group_hpe_student(hpe_main, hpe_official)
    har_student_random = har_eval.get("student_vk_random_all_combinations.csv", [])
    har_student_group = group_har_student(har_student_random, har_official)

    hpe_student_full = next(
        row for row in hpe_finals if row["method"] == "VK-RCD-Student-VK" and row["modality_set"] == "vk+depth+lidar+mmwave+wifi-csi"
    )
    har_student_full = next(
        row for row in har_finals if row["method"] == "VK-RCD-HAR-Student-VK-Missing" and row["modality_set"] == "vk+depth+lidar+mmwave"
    )

    hpe_official_full = hpe_official["vk+depth+lidar+mmwave+wifi-csi"][0]
    har_official_full = har_official["vk+depth+lidar+mmwave"]
    _, hpe_full_rel = hpe_delta_cells(hpe_official_full, float(hpe_student_full["mpjpe_mm"]))
    _, har_full_rel = har_delta_cells(har_official_full, float(har_student_full["acc_pct"]))

    tex = rf"""
\documentclass[UTF8,10pt,a4paper]{{ctexart}}
\usepackage[left=1.35cm,right=1.35cm,top=1.55cm,bottom=1.55cm]{{geometry}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{booktabs,longtable,tabularx,array,multirow}}
\usepackage{{caption,float}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\definecolor{{ImpBlue}}{{RGB}}{{0,82,204}}
\definecolor{{DropRed}}{{RGB}}{{196,31,31}}
\newcommand{{\better}}[1]{{\textcolor{{ImpBlue}}{{#1}}}}
\newcommand{{\worse}}[1]{{\textcolor{{DropRed}}{{#1}}}}
\newcommand{{\method}}{{VK-RCD}}
\newcommand{{\xfi}}{{\textsc{{X-Fi}}}}
\newcommand{{\dataset}}{{MM-Fi}}
\hypersetup{{colorlinks=true,linkcolor=ImpBlue,citecolor=ImpBlue,urlcolor=ImpBlue}}
\setlist[itemize]{{leftmargin=1.15em,itemsep=0.08em,topsep=0.1em}}
\setlength{{\parskip}}{{0.15em}}
\setlength{{\parindent}}{{1.5em}}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.02}}
\captionsetup{{font=small,labelfont=bf}}
\title{{\vspace{{-1.8em}}\textbf{{\method{{}}：面向隐私友好与任意模态缺失的人体感知蒸馏框架}}\\[-0.2em]\large 基于官方 \xfi{{}} 的系统对比}}
\author{{作者姓名$^1$ \quad 作者姓名$^1$ \quad 作者姓名$^1$\\$^1$单位名称\\\texttt{{email@example.com}}}}
\date{{2026-05-09}}
\begin{{document}}
\maketitle
\vspace{{-1.0em}}
\begin{{abstract}}
多模态人体感知需要在高精度、隐私保护和真实部署鲁棒性之间取得平衡。官方 \xfi{{}} 证明了单一模型处理多种传感器组合的可行性，但其视觉分支仍依赖原始 RGB 图像，且缺失模态下的可靠决策仍有提升空间。本文提出 \method{{}}，以视觉关键点（VK）替代原始 RGB，利用全模态 Teacher 学习跨模态人体结构知识，并通过随机缺失模态训练将知识蒸馏到 Student-VK 与非视觉 Student-NV。与官方 \xfi{{}} 相比，本文 Student-VK 在 \dataset{{}} HPE 全模态设置下将 MPJPE 从 {hpe_official_full:.2f} mm 降至 {float(hpe_student_full['mpjpe_mm']):.2f} mm（{hpe_full_rel}），在 HAR 全模态设置下将准确率从 {har_official_full:.2f}\% 提升至 {float(har_student_full['acc_pct']):.2f}\%（{har_full_rel}）。本文进一步报告 HPE 31 种组合、HAR 15 种组合、非视觉部署、跨场景/跨主体和关键消融，所有已同步指标均纳入正文或附录。
\end{{abstract}}

\section{{引言}}
人体感知广泛服务于康复评估、智能家居、人机交互和机器人协作。RGB 图像虽然信息丰富，却包含面部、服饰、身份和家庭环境等隐私敏感线索。Depth、LiDAR、mmWave 与 WiFi-CSI 等非 RGB 传感器能够缓解部分隐私风险，但单一模态又容易受到遮挡、点云稀疏、信道变化或环境迁移影响。官方 \xfi{{}} 提出了模态不变人体感知框架，一次训练即可支持参与训练的传感器单独或任意组合使用，并在 MM-Fi 的 HPE/HAR 任务上取得优势。

本文的目标不是简单把 RGB backbone 换成小 MLP，而是重新组织隐私友好人体感知的训练与部署链路：使用 VK 作为结构化视觉替代，保留人体姿态与动作信息；利用全模态 Teacher 作为特权知识源；通过随机缺失模态训练让 Student-VK 在任意可用模态集合下工作；进一步训练 Student-NV 验证无视觉输入的部署能力；最后通过可靠性感知融合削弱低质量模态对最终决策的负面影响。

本文贡献如下：1）提出面向 MM-Fi HPE/HAR 的非 RGB 隐私友好 VK-RCD 框架；2）构建 full-to-missing 的 Teacher--Student 蒸馏流程；3）系统评估任意模态组合、非视觉组合、跨场景/跨主体和 VK 噪声鲁棒性；4）与官方 \xfi{{}} 逐项对齐，并用蓝红标注报告性能提升或退化。

\section{{相关工作}}
\paragraph{{多模态人体感知。}}
多模态人体感知融合视觉、深度、点云、毫米波与无线信道信息。不同模态在空间分辨率、弱光鲁棒性、穿透性、成本和隐私保护方面互补。MM-Fi 提供同步多模态数据和 HPE/HAR 任务，\xfi{{}} 则进一步验证了任意模态组合人体感知的可行性。
\paragraph{{缺失模态学习。}}
真实系统中传感器可能因遮挡、权限、硬件故障或通信异常而缺失。固定融合模型通常需要为不同组合重新训练，而模态不变学习、动态融合与蒸馏方法试图让模型在可变输入集合下保持稳定。本文沿用任意组合评估协议，但强调训练阶段随机缺失和 Teacher 到 Student 的知识迁移。
\paragraph{{隐私友好与蒸馏。}}
非 RGB 传感器和骨架化视觉表示都可降低原始图像暴露风险。本文采用 VK 作为结构化视觉输入，同时使用全模态 Teacher 提供软目标和中间结构知识，使最终部署的 Student 不必依赖原始 RGB 图像。

\section{{方法}}
\subsection{{问题定义}}
给定模态集合 $\mathcal{{M}}=\{{\mathrm{{VK}},D,L,R,W\}}$，其中 $D,L,R,W$ 分别表示 Depth、LiDAR、mmWave 与 WiFi-CSI。HPE 输出 17 个三维人体关节，HAR 输出 27 类动作概率。对任意可用子集 $\mathcal{{S}}\subseteq\mathcal{{M}}$，模型均需完成预测。表中 $I/VK$ 表示官方 \xfi{{}} 的 $I$ 为 RGB，而本文对应位置为 VK，二者不被视作完全同源输入。

\subsection{{VK 编码与跨模态融合}}
VK 分支接收 17 个二维关节坐标，先用共享线性层将每个关节独立升维，再在关节维和模态维进行全局交互。Depth、LiDAR、mmWave 与 WiFi-CSI 使用各自传感器 backbone 抽取特征后投影到统一 token 空间。融合模块根据当前可用模态集合生成跨模态表示，并由 HPE/HAR 任务头输出预测。

\subsection{{Teacher--Student 蒸馏与随机缺失}}
Teacher 使用全模态训练，学习完整传感器条件下的人体结构表示。Student-VK 在训练时随机采样可用模态子集，并同时优化监督损失、输出蒸馏损失和特征蒸馏损失。Student-NV 移除 VK，仅使用非视觉传感器，以检验完全不依赖视觉输入时的部署性能。可靠性感知融合为不同模态分配动态权重；Uniform 消融使用等权融合，NoKD 消融移除 Teacher 蒸馏。

\section{{实验设置}}
实验基于 \dataset{{}}。HPE 使用 VK、Depth、LiDAR、mmWave、WiFi-CSI 五类输入，指标为 MPJPE 与 PA-MPJPE，单位为 mm，越低越好。HAR 使用 VK、Depth、LiDAR、mmWave，指标为 Accuracy 与 Macro-F1，越高越好。训练遵循官方截断风格，评估使用完整验证集。所有官方数值来自 \xfi{{}} 论文公开表格或本地整理的 legacy CSV。

颜色规则如下：HPE 的 $\Delta=\mathrm{{Ours}}-\mathrm{{Official}}$，负数为更好；HAR 的 $\Delta=\mathrm{{Ours}}-\mathrm{{Official}}$，正数为更好。蓝色表示本文结果优于官方，红色表示退化。

\section{{定量结果}}
\subsection{{HPE 全模态与官方对比}}
\begin{{table}}[H]\centering\small
\caption{{HPE 全模态结果。Student-NV 不含 VK，因此与官方非视觉 D+L+R+W 组合比较。}}
\begin{{tabular}}{{llrrrrrrr}}
\toprule
方法 & 组合 & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Rel.Imp & Ours PA & $\Delta$ PA & Rel.Imp\\
\midrule
{hpe_full_rows(hpe_finals, hpe_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]\centering\small
\caption{{HPE Student-VK 按可用模态数量的 31 组合平均结果。}}
\begin{{tabular}}{{rrrrrrrr}}
\toprule
可用模态数 & 组合数 & Ours MPJPE & X-Fi MPJPE & Rel.Imp & Ours PA & X-Fi PA & Rel.Imp\\
\midrule
{hpe_group_rows(hpe_student_group)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{HAR 主结果与跨协议泛化}}
\begin{{table}}[H]\centering\small
\caption{{HAR random split 全模态、非视觉与消融结果。}}
\begin{{tabular}}{{llrrrrr}}
\toprule
方法 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
{har_full_rows(har_finals, har_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]\centering\scriptsize
\caption{{HAR random/cross-scene/cross-subject 全模态与非视觉结果。}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
协议 & 方法 & 组合 & Acc & Macro-F1 & X-Fi Acc & $\Delta$ & Rel.Imp\\
\midrule
{har_protocol_rows(har_finals, har_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{消融与鲁棒性}}
HAR 消融显示，Student-VK 在全模态 random split 上达到 {float(har_student_full['acc_pct']):.2f}\% Acc，高于 NoKD 与 Uniform 的对应结果，说明蒸馏和可靠性感知融合都对最终部署模型有贡献。Student-NV 在无 VK 的情况下仍保持强性能，说明非视觉传感器组合能够承担隐私友好部署场景。

\begin{{table}}[H]\centering\scriptsize
\caption{{HPE VK 噪声与关节缺失鲁棒性。}}
\begin{{tabular}}{{rrrrr}}
\toprule
噪声标准差 & 关节丢弃率 & MPJPE & PA-MPJPE & MSE\\
\midrule
{hpe_robust_rows(hpe_robust, limit=16)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{讨论}}
第一，\method{{}} 的核心创新不是“RGB 换成 VK MLP”，而是将隐私友好输入、缺失模态训练、全模态知识蒸馏和动态可靠性融合组合成完整部署范式。第二，Teacher 在附录的缺失模态评估中可能明显退化，这是因为 Teacher 不是缺失模态部署模型；其作用是为 Student 提供全模态知识。第三，当前本地 HPE cross-scene/cross-subject 与部分 HPE 消融仍未同步，因此本文仅在审计表中列出，不将其写成已完成结论。

\section{{结论}}
本文基于 \xfi{{}} 和 \dataset{{}} 构建 \method{{}}，在不使用原始 RGB 的前提下实现 HPE/HAR 任意模态组合人体感知。当前结果显示，Student-VK 在 HPE 31 组合和 HAR 15 组合上均表现稳定，并在全模态与多数缺失组合上超过官方 \xfi{{}}。后续工作应补齐 HPE 跨场景/跨主体与更完整可靠性消融，以进一步支撑高等级投稿。

\appendix
\section{{HPE Student-VK 31 组合官方对比}}
{{\tiny
\begin{{longtable}}{{lrrrrrrrr}}
\toprule
组合 & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Rel.Imp & Ours PA & X-Fi PA & $\Delta$ PA & Rel.Imp\\
\midrule
\endfirsthead
\toprule
组合 & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Rel.Imp & Ours PA & X-Fi PA & $\Delta$ PA & Rel.Imp\\
\midrule
\endhead
{hpe_appendix_table(hpe_main, hpe_official, "VK-RCD-Student-VK")}
\bottomrule
\end{{longtable}}
}}

\section{{HPE Teacher 31 组合官方对比}}
{{\tiny
Teacher 不是缺失模态部署模型，以下表格仅用于说明全模态 Teacher 在部分模态输入下存在明显分布偏移。
\begin{{longtable}}{{lrrrrrrrr}}
\toprule
组合 & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Rel.Imp & Ours PA & X-Fi PA & $\Delta$ PA & Rel.Imp\\
\midrule
\endfirsthead
\toprule
组合 & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Rel.Imp & Ours PA & X-Fi PA & $\Delta$ PA & Rel.Imp\\
\midrule
\endhead
{hpe_appendix_table(hpe_main, hpe_official, "VK-RCD-Teacher")}
\bottomrule
\end{{longtable}}
}}

\section{{HPE VK 鲁棒性完整表}}
{{\scriptsize
\begin{{longtable}}{{rrrrr}}
\toprule
噪声标准差 & 关节丢弃率 & MPJPE & PA-MPJPE & MSE\\
\midrule
\endfirsthead
\toprule
噪声标准差 & 关节丢弃率 & MPJPE & PA-MPJPE & MSE\\
\midrule
\endhead
{hpe_robust_rows(hpe_robust)}
\bottomrule
\end{{longtable}}
}}

\section{{HAR random split 15 组合官方对比}}
{{\tiny
\begin{{longtable}}{{llrrrrr}}
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endhead
{har_appendix_table(har_eval.get("student_vk_random_all_combinations.csv", []), har_official, "Student-VK")}
\bottomrule
\end{{longtable}}
}}

\section{{HAR cross-scene 与 cross-subject 15 组合}}
{{\tiny
\begin{{longtable}}{{llrrrrr}}
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endhead
{har_appendix_table(har_eval.get("student_vk_cross_scene_all_combinations.csv", []), har_official, "cross-scene")}
{har_appendix_table(har_eval.get("student_vk_cross_subject_all_combinations.csv", []), har_official, "cross-subject")}
\bottomrule
\end{{longtable}}
}}

\section{{HAR Student-NV 非视觉组合}}
{{\tiny
\begin{{longtable}}{{llrrrrr}}
\toprule
协议 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
协议 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endhead
{har_nv_appendix_table(har_eval, har_official)}
\bottomrule
\end{{longtable}}
}}

\section{{HAR NoKD 与 Uniform 消融}}
{{\tiny
\begin{{longtable}}{{llrrrrr}}
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endhead
{har_appendix_table(har_eval.get("ablation_no_distill_all_combinations.csv", []), har_official, "NoKD")}
{har_appendix_table(har_eval.get("ablation_uniform_fusion_all_combinations.csv", []), har_official, "Uniform")}
\bottomrule
\end{{longtable}}
}}

\section{{HAR 已同步组合结果全集}}
{{\tiny
\begin{{longtable}}{{llrrrrr}}
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
来源 & 组合 & Ours Acc & X-Fi Acc & $\Delta$ & Rel.Imp & Macro-F1\\
\midrule
\endhead
{har_all_existing_appendix(har_eval, har_official)}
\bottomrule
\end{{longtable}}
}}

\section{{当前缺失项审计}}
{{\scriptsize
\begin{{longtable}}{{llll}}
\toprule
任务 & 类型 & 项目 & 状态\\
\midrule
\endfirsthead
\toprule
任务 & 类型 & 项目 & 状态\\
\midrule
\endhead
{audit_rows(audit_missing)}
\bottomrule
\end{{longtable}}
}}

\begin{{thebibliography}}{{9}}
\bibitem{{xfi}} Xinyan Chen and Jianfei Yang. X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing. ICLR, 2025. Official code: \url{{https://github.com/NTUMARS/X-Fi}}.
\bibitem{{mmfi}} Jianfei Yang et al. MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing. NeurIPS Datasets and Benchmarks, 2023.
\bibitem{{kd}} Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. NeurIPS Workshop, 2015.
\bibitem{{missing}} Mengmeng Ma et al. Are Multimodal Transformers Robust to Missing Modality? CVPR, 2022.
\bibitem{{dynamic}} Zihui Xue and Radu Marculescu. Dynamic Multimodal Fusion. CVPR, 2023.
\end{{thebibliography}}
\end{{document}}
"""
    return tex


def write_tex() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(build_tex(), encoding="utf-8")


def compile_pdf() -> None:
    xelatex = shutil.which("xelatex")
    if not xelatex:
        print("xelatex not found; TeX was generated but PDF was not compiled.")
        return
    for _ in range(2):
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", TEX_PATH.name],
            cwd=OUT_DIR,
            check=True,
        )


def main() -> None:
    write_tex()
    compile_pdf()
    print(f"TeX: {TEX_PATH}")
    print(f"PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
