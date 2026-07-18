from __future__ import annotations

import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_vkrcd_cn"
PAPER_STEM = "2026-05-22_vkrcd_experiments_cn"
TEX_PATH = OUT_DIR / f"{PAPER_STEM}.tex"
PDF_PATH = OUT_DIR / f"{PAPER_STEM}.pdf"

MODALITY_ORDER = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
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
    "VK-RCD-Ablation": "Ablation",
    "VK-RCD-HAR-Teacher": "Teacher",
    "VK-RCD-HAR-Student-VK-Missing": "Student-VK",
    "VK-RCD-HAR-Student-NV-Missing": "Student-NV",
    "VK-RCD-HAR-Ablation-NoKD": "NoKD",
    "VK-RCD-HAR-Ablation-UniformFusion": "Uniform",
}

HPE_OFFICIAL_FALLBACK = {
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

HAR_OFFICIAL_FALLBACK = {
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or str(value).strip() in {"", "NA", "nan"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_to_mm(value: object) -> float:
    val = to_float(value)
    if not math.isfinite(val):
        return val
    return val * 1000.0 if abs(val) < 10 else val


def metric_to_pct(value: object) -> float:
    val = to_float(value)
    if not math.isfinite(val):
        return val
    return val * 100.0 if abs(val) <= 1.5 else val


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def fmt_int(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{int(round(value)):,}"


def tex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def normalize_combo(combo: str) -> str:
    items = [item.strip() for item in combo.split("+") if item.strip()]
    order = {name: idx for idx, name in enumerate(MODALITY_ORDER)}
    return "+".join(sorted(items, key=lambda x: order.get(x, 99)))


def combo_label(combo: str) -> str:
    combo = normalize_combo(combo)
    return "+".join(MODALITY_LABELS.get(part, tex_escape(part)) for part in combo.split("+") if part)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, tex_escape(method))


def protocol_label(split: str) -> str:
    mapping = {
        "random_split": "random",
        "cross_scene_split": "cross-scene",
        "cross_subject_split": "cross-subject",
    }
    return mapping.get(split, tex_escape(split))


def color_signed(value: float, better_when_positive: bool = True, suffix: str = "") -> str:
    if not math.isfinite(value):
        return "--"
    better = value >= 0 if better_when_positive else value <= 0
    macro = "better" if better else "worse"
    return rf"\{macro}{{{value:+.2f}{suffix}}}"


def hpe_imp_cell(official: float, ours: float) -> str:
    if not math.isfinite(official) or official == 0 or not math.isfinite(ours):
        return "--"
    return color_signed((official - ours) / official * 100.0, True, r"\%")


def har_imp_cell(official: float, ours: float) -> str:
    if not math.isfinite(official) or official == 0 or not math.isfinite(ours):
        return "--"
    return color_signed((ours - official) / official * 100.0, True, r"\%")


def delta_lower_better_cell(value: float) -> str:
    return color_signed(value, False)


def delta_higher_better_cell(value: float) -> str:
    return color_signed(value, True)


def load_hpe_official() -> dict[str, tuple[float, float]]:
    path = ROOT / "HPE" / "legacy_xfi" / "xfi_hpe_table1.csv"
    rows = read_csv(path)
    if not rows:
        return dict(HPE_OFFICIAL_FALLBACK)
    table: dict[str, tuple[float, float]] = {}
    for row in rows:
        combo = normalize_combo(row.get("modality_set", ""))
        table[combo] = (to_float(row.get("xfi_mpjpe_mm")), to_float(row.get("xfi_pa_mpjpe_mm")))
    return table or dict(HPE_OFFICIAL_FALLBACK)


def load_har_official() -> dict[str, float]:
    path = ROOT / "HAR" / "legacy_xfi" / "xfi_har_table6.csv"
    rows = read_csv(path)
    if not rows:
        return dict(HAR_OFFICIAL_FALLBACK)
    table: dict[str, float] = {}
    for row in rows:
        combo = normalize_combo(row.get("modality_set", ""))
        table[combo] = to_float(row.get("xfi_acc_pct"))
    return table or dict(HAR_OFFICIAL_FALLBACK)


def collect_hpe_finals() -> dict[str, dict[str, object]]:
    finals: dict[str, dict[str, object]] = {}
    base = ROOT / "HPE" / "outputs"
    for path in sorted(base.rglob("final_eval.csv")):
        for row in read_csv(path):
            if "mpjpe" not in row:
                continue
            rel = path.relative_to(base).as_posix()
            finals[rel] = {
                "source": rel,
                "method": row.get("method", path.parent.name),
                "split": row.get("split", ""),
                "protocol": row.get("protocol", ""),
                "modality_set": normalize_combo(row.get("modality_set", "")),
                "mse": to_float(row.get("mse")),
                "mpjpe_mm": metric_to_mm(row.get("mpjpe")),
                "pa_mpjpe_mm": metric_to_mm(row.get("pa_mpjpe")),
                "params": to_float(row.get("params")),
                "fps": to_float(row.get("fps")),
                "peak_memory": to_float(row.get("peak_memory")),
            }
    return finals


def collect_har_finals() -> dict[str, dict[str, object]]:
    finals: dict[str, dict[str, object]] = {}
    base = ROOT / "HAR" / "outputs"
    for path in sorted(base.rglob("final_eval.csv")):
        for row in read_csv(path):
            if "acc" not in row:
                continue
            rel = path.relative_to(base).as_posix()
            finals[rel] = {
                "source": rel,
                "method": row.get("method", path.parent.name),
                "split": row.get("split", ""),
                "protocol": row.get("protocol", ""),
                "modality_set": normalize_combo(row.get("modality_set", "")),
                "loss": to_float(row.get("loss")),
                "acc_pct": metric_to_pct(row.get("acc")),
                "macro_f1_pct": metric_to_pct(row.get("macro_f1")),
                "params": to_float(row.get("params")),
                "fps": to_float(row.get("fps")),
                "peak_memory": to_float(row.get("peak_memory")),
            }
    return finals


def collect_hpe_eval() -> dict[str, list[dict[str, object]]]:
    evals: dict[str, list[dict[str, object]]] = {}
    base = ROOT / "HPE" / "outputs" / "eval"
    for path in sorted(base.glob("*.csv")):
        rows: list[dict[str, object]] = []
        for row in read_csv(path):
            item = dict(row)
            if "modality_set" in item:
                item["modality_set"] = normalize_combo(str(item["modality_set"]))
            if "mpjpe" in item:
                item["mpjpe_mm"] = metric_to_mm(item.get("mpjpe"))
            if "pa_mpjpe" in item:
                item["pa_mpjpe_mm"] = metric_to_mm(item.get("pa_mpjpe"))
            if "mse" in item:
                item["mse_float"] = to_float(item.get("mse"))
            if "params" in item:
                item["params_float"] = to_float(item.get("params"))
            if "fps" in item:
                item["fps_float"] = to_float(item.get("fps"))
            if "peak_memory" in item:
                item["peak_memory_float"] = to_float(item.get("peak_memory"))
            rows.append(item)
        evals[path.name] = rows
    return evals


def collect_har_eval() -> dict[str, list[dict[str, object]]]:
    evals: dict[str, list[dict[str, object]]] = {}
    base = ROOT / "HAR" / "outputs" / "eval"
    for path in sorted(base.glob("*.csv")):
        rows: list[dict[str, object]] = []
        for row in read_csv(path):
            item = dict(row)
            if "modality_set" in item:
                item["modality_set"] = normalize_combo(str(item["modality_set"]))
            if "acc" in item:
                item["acc_pct"] = metric_to_pct(item.get("acc"))
            if "macro_f1" in item:
                item["macro_f1_pct"] = metric_to_pct(item.get("macro_f1"))
            if "loss" in item:
                item["loss_float"] = to_float(item.get("loss"))
            if "params" in item:
                item["params_float"] = to_float(item.get("params"))
            rows.append(item)
        evals[path.name] = rows
    return evals


def by_combo(rows: list[dict[str, object]], combo: str) -> dict[str, object] | None:
    combo = normalize_combo(combo)
    for row in rows:
        if row.get("modality_set") == combo:
            return row
    return None


def avg(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return sum(valid) / len(valid) if valid else math.nan


def best_worst_combo(rows: list[dict[str, object]], key: str, higher_better: bool) -> tuple[str, float, str, float]:
    valid = [(str(row.get("modality_set", "")), to_float(row.get(key))) for row in rows]
    valid = [(combo, value) for combo, value in valid if combo and math.isfinite(value)]
    if not valid:
        return "--", math.nan, "--", math.nan
    best = max(valid, key=lambda item: item[1]) if higher_better else min(valid, key=lambda item: item[1])
    worst = min(valid, key=lambda item: item[1]) if higher_better else max(valid, key=lambda item: item[1])
    return best[0], best[1], worst[0], worst[1]


def hpe_random_main_table(finals: dict[str, dict[str, object]], official: dict[str, tuple[float, float]]) -> str:
    sources = [
        ("Baseline", "baseline_full/final_eval.csv"),
        ("Teacher", "teacher_full/final_eval.csv"),
        ("Student-VK", "student_vk_missing/final_eval.csv"),
        ("Student-NV", "student_nv_missing/final_eval.csv"),
    ]
    lines = []
    for label, source in sources:
        row = finals.get(source)
        if not row:
            continue
        combo = str(row["modality_set"])
        off_mpjpe, off_pa = official.get(combo, (math.nan, math.nan))
        lines.append(
            rf"{label} & {combo_label(combo)} & {fmt(float(row['mpjpe_mm']))} & {fmt(off_mpjpe)} & {hpe_imp_cell(off_mpjpe, float(row['mpjpe_mm']))} & "
            rf"{fmt(float(row['pa_mpjpe_mm']))} & {fmt(off_pa)} & {hpe_imp_cell(off_pa, float(row['pa_mpjpe_mm']))}\\"
        )
    return "\n".join(lines)


def hpe_cross_scene_table(
    finals: dict[str, dict[str, object]], hpe_eval: dict[str, list[dict[str, object]]]
) -> str:
    rows = []
    pairs = [
        ("Teacher", "vk+depth+lidar+mmwave+wifi-csi", finals.get("teacher_full/final_eval.csv"), finals.get("teacher_full_cross_scene_split/final_eval.csv")),
        ("Student-VK", "vk+depth+lidar+mmwave+wifi-csi", finals.get("student_vk_missing/final_eval.csv"), finals.get("student_vk_missing_cross_scene_split/final_eval.csv")),
    ]
    nv_cross = by_combo(hpe_eval.get("student_nv_cross_scene_nonvisual_combinations.csv", []), "depth+lidar+mmwave+wifi-csi")
    pairs.append(("Student-NV", "depth+lidar+mmwave+wifi-csi", finals.get("student_nv_missing/final_eval.csv"), nv_cross))
    for label, combo, random_row, cross_row in pairs:
        if not random_row or not cross_row:
            continue
        random_mpjpe = float(random_row["mpjpe_mm"])
        cross_mpjpe = float(cross_row["mpjpe_mm"])
        random_pa = float(random_row["pa_mpjpe_mm"])
        cross_pa = float(cross_row["pa_mpjpe_mm"])
        rows.append(
            rf"{label} & {combo_label(combo)} & {fmt(random_mpjpe)} & {fmt(cross_mpjpe)} & {delta_lower_better_cell(cross_mpjpe - random_mpjpe)} & "
            rf"{fmt(random_pa)} & {fmt(cross_pa)} & {delta_lower_better_cell(cross_pa - random_pa)}\\"
        )
    return "\n".join(rows)


def hpe_combo_summary(rows: list[dict[str, object]], official: dict[str, tuple[float, float]]) -> str:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        combo = str(row.get("modality_set", ""))
        if combo in official:
            groups[len(combo.split("+"))].append(row)
    lines = []
    all_rows = []
    for count in sorted(groups):
        group = groups[count]
        ours_mpjpe = avg([float(row["mpjpe_mm"]) for row in group])
        ours_pa = avg([float(row["pa_mpjpe_mm"]) for row in group])
        official_mpjpe = avg([official[str(row["modality_set"])][0] for row in group])
        official_pa = avg([official[str(row["modality_set"])][1] for row in group])
        best_combo, best_value, worst_combo, worst_value = best_worst_combo(group, "mpjpe_mm", False)
        lines.append(
            rf"{count} & {len(group)} & {fmt(ours_mpjpe)} & {fmt(official_mpjpe)} & {hpe_imp_cell(official_mpjpe, ours_mpjpe)} & "
            rf"{fmt(ours_pa)} & {fmt(official_pa)} & {hpe_imp_cell(official_pa, ours_pa)} & "
            rf"{combo_label(best_combo)} / {fmt(best_value)} & {combo_label(worst_combo)} / {fmt(worst_value)}\\"
        )
        all_rows.extend(group)
    if all_rows:
        ours_mpjpe = avg([float(row["mpjpe_mm"]) for row in all_rows])
        ours_pa = avg([float(row["pa_mpjpe_mm"]) for row in all_rows])
        official_mpjpe = avg([official[str(row["modality_set"])][0] for row in all_rows])
        official_pa = avg([official[str(row["modality_set"])][1] for row in all_rows])
        best_combo, best_value, worst_combo, worst_value = best_worst_combo(all_rows, "mpjpe_mm", False)
        lines.append(
            rf"\midrule 平均 & {len(all_rows)} & {fmt(ours_mpjpe)} & {fmt(official_mpjpe)} & {hpe_imp_cell(official_mpjpe, ours_mpjpe)} & "
            rf"{fmt(ours_pa)} & {fmt(official_pa)} & {hpe_imp_cell(official_pa, ours_pa)} & "
            rf"{combo_label(best_combo)} / {fmt(best_value)} & {combo_label(worst_combo)} / {fmt(worst_value)}\\"
        )
    return "\n".join(lines)


def hpe_ablation_table(
    finals: dict[str, dict[str, object]], hpe_eval: dict[str, list[dict[str, object]]]
) -> str:
    reference = finals.get("student_vk_missing/final_eval.csv")
    if not reference:
        return ""
    ref_mpjpe = float(reference["mpjpe_mm"])
    ref_pa = float(reference["pa_mpjpe_mm"])
    candidates: list[tuple[str, dict[str, object] | None, str]] = [
        ("Student-VK 完整目标", reference, "随机缺失 + 蒸馏 + 可靠性融合"),
        ("NoKD", finals.get("ablation/no_kd/final_eval.csv") or by_combo(hpe_eval.get("ablation_no_distill_all_combinations.csv", []), "vk+depth+lidar+mmwave+wifi-csi"), "移除 Teacher 蒸馏"),
        ("Output KD", finals.get("ablation/output_kd/final_eval.csv"), "仅输出蒸馏"),
        ("Output+Token KD", finals.get("ablation/output_token_kd/final_eval.csv"), "输出与 token 蒸馏"),
        ("Output+Bone KD", finals.get("ablation/output_bone_kd/final_eval.csv"), "输出与骨架结构蒸馏"),
        ("Full Structural KD", finals.get("ablation/full_structural_kd/final_eval.csv"), "完整结构蒸馏变体"),
        ("Uniform Fusion", by_combo(hpe_eval.get("ablation_uniform_fusion_all_combinations.csv", []), "vk+depth+lidar+mmwave+wifi-csi"), "等权融合"),
    ]
    lines = []
    for label, row, note in candidates:
        if not row:
            continue
        mpjpe = float(row["mpjpe_mm"])
        pa = float(row["pa_mpjpe_mm"])
        lines.append(
            rf"{tex_escape(label)} & {fmt(mpjpe)} & {delta_lower_better_cell(mpjpe - ref_mpjpe)} & {fmt(pa)} & {delta_lower_better_cell(pa - ref_pa)} & {tex_escape(note)}\\"
        )
    return "\n".join(lines)


def hpe_robustness_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    clean = None
    for row in rows:
        if to_float(row.get("noise_std")) == 0 and to_float(row.get("drop_joint_ratio")) == 0:
            clean = row
            break
    clean_mpjpe = float(clean["mpjpe_mm"]) if clean else math.nan
    clean_pa = float(clean["pa_mpjpe_mm"]) if clean else math.nan
    lines = []
    rows = sorted(rows, key=lambda r: (to_float(r.get("noise_std")), to_float(r.get("drop_joint_ratio"))))
    for row in rows:
        mpjpe = float(row["mpjpe_mm"])
        pa = float(row["pa_mpjpe_mm"])
        lines.append(
            rf"{fmt(to_float(row.get('noise_std')), 1)} & {fmt(to_float(row.get('drop_joint_ratio')), 1)} & {fmt(mpjpe)} & {delta_lower_better_cell(mpjpe - clean_mpjpe)} & {fmt(pa)} & {delta_lower_better_cell(pa - clean_pa)}\\"
        )
    return "\n".join(lines)


def hpe_complexity_table(hpe_eval: dict[str, list[dict[str, object]]]) -> str:
    files = [
        ("Baseline", "baseline_full_model_complexity.csv"),
        ("Teacher", "teacher_full_model_complexity.csv"),
        ("Student-VK", "student_vk_model_complexity.csv"),
        ("Student-NV", "student_nv_model_complexity.csv"),
    ]
    lines = []
    for label, file_name in files:
        rows = hpe_eval.get(file_name, [])
        if not rows:
            continue
        row = rows[0]
        params_m = to_float(row.get("params")) / 1_000_000.0
        memory_gb = to_float(row.get("peak_memory")) / (1024.0**3)
        lines.append(
            rf"{label} & {combo_label(str(row.get('modality_set', '')))} & {fmt(params_m)}M & {fmt(to_float(row.get('fps')))} & {fmt(memory_gb)} GB\\"
        )
    return "\n".join(lines)


def har_main_table(har_eval: dict[str, list[dict[str, object]]], official: dict[str, float]) -> str:
    source_specs = [
        ("random", "Teacher", "teacher_random_all_combinations.csv", "vk+depth+lidar+mmwave"),
        ("random", "Student-VK", "student_vk_random_all_combinations.csv", "vk+depth+lidar+mmwave"),
        ("random", "Student-NV", "student_nv_random_nonvisual_combinations.csv", "depth+lidar+mmwave"),
        ("cross-scene", "Teacher", "teacher_cross_scene_all_combinations.csv", "vk+depth+lidar+mmwave"),
        ("cross-scene", "Student-VK", "student_vk_cross_scene_all_combinations.csv", "vk+depth+lidar+mmwave"),
        ("cross-scene", "Student-NV", "student_nv_cross_scene_nonvisual_combinations.csv", "depth+lidar+mmwave"),
        ("cross-subject", "Teacher", "teacher_cross_subject_all_combinations.csv", "vk+depth+lidar+mmwave"),
        ("cross-subject", "Student-VK", "student_vk_cross_subject_all_combinations.csv", "vk+depth+lidar+mmwave"),
        ("cross-subject", "Student-NV", "student_nv_cross_subject_nonvisual_combinations.csv", "depth+lidar+mmwave"),
    ]
    lines = []
    for protocol, label, source, combo_name in source_specs:
        row = by_combo(har_eval.get(source, []), combo_name)
        if not row:
            continue
        combo = str(row["modality_set"])
        official_acc = official.get(combo, math.nan)
        lines.append(
            rf"{protocol} & {label} & {combo_label(combo)} & {fmt(float(row['acc_pct']))} & {fmt(float(row['macro_f1_pct']))} & "
            rf"{fmt(official_acc)} & {har_imp_cell(official_acc, float(row['acc_pct']))}\\"
        )
    return "\n".join(lines)


def har_combo_summary(har_eval: dict[str, list[dict[str, object]]], official: dict[str, float]) -> str:
    specs = [
        ("random", "Student-VK", "student_vk_random_all_combinations.csv"),
        ("cross-scene", "Student-VK", "student_vk_cross_scene_all_combinations.csv"),
        ("cross-subject", "Student-VK", "student_vk_cross_subject_all_combinations.csv"),
        ("random", "Student-NV", "student_nv_random_nonvisual_combinations.csv"),
        ("cross-scene", "Student-NV", "student_nv_cross_scene_nonvisual_combinations.csv"),
        ("cross-subject", "Student-NV", "student_nv_cross_subject_nonvisual_combinations.csv"),
    ]
    lines = []
    for protocol, method, file_name in specs:
        rows = [row for row in har_eval.get(file_name, []) if str(row.get("modality_set", "")) in official]
        if not rows:
            continue
        acc = avg([float(row["acc_pct"]) for row in rows])
        f1 = avg([float(row["macro_f1_pct"]) for row in rows])
        official_avg = avg([official[str(row["modality_set"])] for row in rows])
        best_combo, best_value, worst_combo, worst_value = best_worst_combo(rows, "acc_pct", True)
        lines.append(
            rf"{protocol} & {method} & {len(rows)} & {fmt(acc)} & {fmt(official_avg)} & {har_imp_cell(official_avg, acc)} & {fmt(f1)} & "
            rf"{combo_label(best_combo)} / {fmt(best_value)} & {combo_label(worst_combo)} / {fmt(worst_value)}\\"
        )
    return "\n".join(lines)


def har_ablation_table(finals: dict[str, dict[str, object]], har_eval: dict[str, list[dict[str, object]]]) -> str:
    reference = finals.get("student_vk_missing/final_eval.csv")
    if not reference:
        return ""
    ref_acc = float(reference["acc_pct"])
    specs = [
        ("Student-VK", finals.get("student_vk_missing/final_eval.csv"), "student_vk_random_all_combinations.csv"),
        ("NoKD", finals.get("ablation_no_distill/final_eval.csv"), "ablation_no_distill_all_combinations.csv"),
        ("Uniform", finals.get("ablation_uniform_fusion/final_eval.csv"), "ablation_uniform_fusion_all_combinations.csv"),
    ]
    lines = []
    for label, final_row, combo_file in specs:
        if not final_row:
            continue
        rows = har_eval.get(combo_file, [])
        avg_acc = avg([float(row["acc_pct"]) for row in rows if "acc_pct" in row])
        avg_f1 = avg([float(row["macro_f1_pct"]) for row in rows if "macro_f1_pct" in row])
        acc = float(final_row["acc_pct"])
        lines.append(
            rf"{label} & {fmt(acc)} & {delta_higher_better_cell(acc - ref_acc)} & {fmt(float(final_row['macro_f1_pct']))} & {fmt(avg_acc)} & {fmt(avg_f1)}\\"
        )
    return "\n".join(lines)


def hpe_full_combo_table(rows: list[dict[str, object]], official: dict[str, tuple[float, float]]) -> str:
    selected = [row for row in rows if str(row.get("modality_set", "")) in official]
    selected.sort(key=lambda r: (len(str(r["modality_set"]).split("+")), str(r["modality_set"])))
    lines = []
    for row in selected:
        combo = str(row["modality_set"])
        off_mpjpe, off_pa = official[combo]
        mpjpe = float(row["mpjpe_mm"])
        pa = float(row["pa_mpjpe_mm"])
        lines.append(
            rf"{combo_label(combo)} & {fmt(mpjpe)} & {fmt(off_mpjpe)} & {hpe_imp_cell(off_mpjpe, mpjpe)} & {fmt(pa)} & {fmt(off_pa)} & {hpe_imp_cell(off_pa, pa)}\\"
        )
    return "\n".join(lines)


def har_full_combo_table(rows: list[dict[str, object]], official: dict[str, float], source_label: str | None = None) -> str:
    selected = [row for row in rows if str(row.get("modality_set", "")) in official]
    selected.sort(key=lambda r: (len(str(r["modality_set"]).split("+")), str(r["modality_set"])))
    lines = []
    for row in selected:
        combo = str(row["modality_set"])
        official_acc = official[combo]
        acc = float(row["acc_pct"])
        prefix = f"{tex_escape(source_label)} & " if source_label else ""
        lines.append(
            rf"{prefix}{combo_label(combo)} & {fmt(acc)} & {fmt(official_acc)} & {har_imp_cell(official_acc, acc)} & {fmt(float(row['macro_f1_pct']))}\\"
        )
    return "\n".join(lines)


def eval_file_index(hpe_eval: dict[str, list[dict[str, object]]], har_eval: dict[str, list[dict[str, object]]]) -> str:
    roles = {
        "HPE/student_vk_random_all_combinations.csv": "附录全表 + 主文组合汇总",
        "HPE/student_vk_cross_scene_all_combinations.csv": "附录全表 + 主文跨场景",
        "HPE/student_nv_random_nonvisual_combinations.csv": "附录全表",
        "HPE/student_nv_nonvisual_combinations.csv": "重复 random 非视觉表，索引记录",
        "HPE/student_nv_cross_scene_nonvisual_combinations.csv": "附录全表 + 主文跨场景",
        "HPE/student_vk_noise_robustness.csv": "主文鲁棒性表",
        "HPE/teacher_random_all_combinations.csv": "Teacher 诊断，不作为部署结论",
        "HPE/teacher_cross_scene_all_combinations.csv": "Teacher 跨场景诊断",
        "HAR/student_vk_random_all_combinations.csv": "附录全表 + 主文组合汇总",
        "HAR/student_vk_cross_scene_all_combinations.csv": "附录全表 + 主文组合汇总",
        "HAR/student_vk_cross_subject_all_combinations.csv": "附录全表 + 主文组合汇总",
        "HAR/student_nv_random_nonvisual_combinations.csv": "附录全表 + 主文组合汇总",
        "HAR/student_nv_cross_scene_nonvisual_combinations.csv": "附录全表 + 主文组合汇总",
        "HAR/student_nv_cross_subject_nonvisual_combinations.csv": "附录全表 + 主文组合汇总",
    }
    lines = []
    for project, files in [("HPE", hpe_eval), ("HAR", har_eval)]:
        for name, rows in sorted(files.items()):
            key = f"{project}/{name}"
            role = roles.get(key, "主文摘要或同步索引")
            lines.append(rf"{project} & {tex_escape(name)} & {len(rows)} & {tex_escape(role)}\\")
    return "\n".join(lines)


def audit_table(hpe_eval: dict[str, list[dict[str, object]]], har_finals: dict[str, dict[str, object]]) -> str:
    required_missing = [
        ("HPE", "eval", "teacher_cross_subject_all_combinations.csv"),
        ("HPE", "eval", "student_vk_cross_subject_all_combinations.csv"),
        ("HPE", "eval", "student_nv_cross_subject_nonvisual_combinations.csv"),
        ("HPE", "eval", "baseline_cross_scene_all_combinations.csv"),
        ("HPE", "eval", "baseline_cross_subject_all_combinations.csv"),
        ("HPE", "eval", "ablation_reliability_attention_all_combinations.csv"),
        ("HPE", "eval", "ablation_reliability_uncertainty_all_combinations.csv"),
        ("HPE", "eval", "ablation_encoder_mlp_vk_all_combinations.csv"),
        ("HPE", "eval", "ablation_encoder_skeleton_prompt_all_combinations.csv"),
    ]
    lines = []
    for project, item_type, item in required_missing:
        if project == "HPE" and item in hpe_eval:
            continue
        lines.append(rf"{project} & {item_type} & {tex_escape(item)} & 未同步或未完成\\")

    scene = har_finals.get("student_vk_missing_cross_scene/final_eval.csv")
    if scene and scene.get("split") != "cross_scene_split":
        lines.append(
            rf"HAR & metadata & {tex_escape('student_vk_missing_cross_scene/final_eval.csv')} & 目录为 cross-scene，但 CSV split={tex_escape(scene.get('split', ''))}\\"
        )
    teacher_subject = har_finals.get("teacher_full_cross_subject/final_eval.csv")
    if teacher_subject and teacher_subject.get("method") != "VK-RCD-HAR-Teacher":
        lines.append(
            rf"HAR & metadata & {tex_escape('teacher_full_cross_subject/final_eval.csv')} & 文件名为 Teacher，但 method={tex_escape(teacher_subject.get('method', ''))}\\"
        )
    student_subject = ROOT / "HAR" / "outputs" / "student_vk_missing_cross_subject" / "best.pth"
    if not student_subject.exists():
        lines.append(
            rf"HAR & model & {tex_escape('student_vk_missing_cross_subject/best.pth')} & 缺少 best.pth，但 final/eval CSV 已同步\\"
        )
    return "\n".join(lines) or r"All & audit & -- & 当前索引未发现缺失或 metadata 异常\\"


def build_tex() -> str:
    hpe_official = load_hpe_official()
    har_official = load_har_official()
    hpe_finals = collect_hpe_finals()
    har_finals = collect_har_finals()
    hpe_eval = collect_hpe_eval()
    har_eval = collect_har_eval()

    hpe_student_vk_random = hpe_eval.get("student_vk_random_all_combinations.csv") or hpe_eval.get("all_combinations.csv", [])
    hpe_student_vk_cross_scene = hpe_eval.get("student_vk_cross_scene_all_combinations.csv", [])
    hpe_student_nv_random = hpe_eval.get("student_nv_random_nonvisual_combinations.csv") or hpe_eval.get("student_nv_nonvisual_combinations.csv", [])
    hpe_student_nv_cross_scene = hpe_eval.get("student_nv_cross_scene_nonvisual_combinations.csv", [])
    hpe_diagnostic_tables = [
        ("Teacher random", hpe_eval.get("teacher_random_all_combinations.csv", []) or hpe_eval.get("teacher_all_combinations.csv", [])),
        ("Baseline random", hpe_eval.get("baseline_full_all_combinations.csv", [])),
        ("NoKD random", hpe_eval.get("ablation_no_distill_all_combinations.csv", [])),
        ("Uniform random", hpe_eval.get("ablation_uniform_fusion_all_combinations.csv", [])),
    ]

    har_student_vk_tables = [
        ("random Student-VK", har_eval.get("student_vk_random_all_combinations.csv", [])),
        ("cross-scene Student-VK", har_eval.get("student_vk_cross_scene_all_combinations.csv", [])),
        ("cross-subject Student-VK", har_eval.get("student_vk_cross_subject_all_combinations.csv", [])),
    ]
    har_student_nv_tables = [
        ("random Student-NV", har_eval.get("student_nv_random_nonvisual_combinations.csv", [])),
        ("cross-scene Student-NV", har_eval.get("student_nv_cross_scene_nonvisual_combinations.csv", [])),
        ("cross-subject Student-NV", har_eval.get("student_nv_cross_subject_nonvisual_combinations.csv", [])),
    ]

    har_vk_appendix = "\n".join(
        har_full_combo_table(rows, har_official, label) for label, rows in har_student_vk_tables if rows
    )
    har_nv_appendix = "\n".join(
        har_full_combo_table(rows, har_official, label) for label, rows in har_student_nv_tables if rows
    )
    hpe_diagnostic_appendix = "\n".join(
        hpe_prefixed_combo_table(label, rows, hpe_official) for label, rows in hpe_diagnostic_tables if rows
    )
    har_diagnostic_tables = [
        ("random Teacher", har_eval.get("teacher_random_all_combinations.csv", []) or har_eval.get("all_combinations.csv", [])),
        ("cross-scene Teacher", har_eval.get("teacher_cross_scene_all_combinations.csv", [])),
        ("cross-subject Teacher", har_eval.get("teacher_cross_subject_all_combinations.csv", [])),
        ("random NoKD", har_eval.get("ablation_no_distill_all_combinations.csv", [])),
        ("random Uniform", har_eval.get("ablation_uniform_fusion_all_combinations.csv", [])),
    ]
    har_diagnostic_appendix = "\n".join(
        har_full_combo_table(rows, har_official, label) for label, rows in har_diagnostic_tables if rows
    )

    return rf"""\documentclass[UTF8,10pt]{{ctexart}}
\usepackage[a4paper,margin=1.45cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,longtable,tabularx,array,float,xcolor,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}}
\definecolor{{goodblue}}{{RGB}}{{0,72,180}}
\definecolor{{badred}}{{RGB}}{{210,40,40}}
\newcommand{{\better}}[1]{{\textcolor{{goodblue}}{{#1}}}}
\newcommand{{\worse}}[1]{{\textcolor{{badred}}{{#1}}}}
\renewcommand{{\arraystretch}}{{0.92}}
\setlength{{\tabcolsep}}{{3.2pt}}
\title{{VK-RCD 实验结果汇总（中文实验部分）}}
\author{{作者信息待补充}}
\date{{2026-05-22}}
\begin{{document}}
\maketitle

\section{{实验设置}}
本文仅汇总实验部分，不包含完整引言、相关工作与方法推导。实验基于 MM-Fi/X-Fi 风格多模态协议：HPE 使用 VK、Depth、LiDAR、mmWave、WiFi-CSI，报告 MPJPE 与 PA-MPJPE，单位为 mm，越低越好；HAR 使用 VK、Depth、LiDAR、mmWave，报告 Accuracy 与 Macro-F1，越高越好。表中 I/VK 表示官方 X-Fi 的 I 为 RGB，本文对应位置使用 VK，不将二者视为完全同源输入。Teacher 是全模态特权知识源；缺失模态部署结论主要依据 Student-VK 与 Student-NV。

HPE 的相对提升定义为 $(\mathrm{{Official}}-\mathrm{{Ours}})/\mathrm{{Official}}$；HAR 的相对提升定义为 $(\mathrm{{Ours}}-\mathrm{{Official}})/\mathrm{{Official}}$。\better{{蓝色}} 表示优于官方或优于参考项，\worse{{红色}} 表示退化。官方 X-Fi 数值来自本地 legacy CSV 与公开代码仓库 \url{{https://github.com/NTUMARS/X-Fi}}。

\section{{HPE 定量结果}}
\subsection{{Random Split 主结果}}
\begin{{table}}[H]\centering\scriptsize
\caption{{HPE random split 全模态/非视觉主结果与官方 X-Fi 对比。Student-NV 不含 VK，因此与官方 D+L+R+W 对比。}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
方法 & 组合 & MPJPE & X-Fi MPJPE & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
{hpe_random_main_table(hpe_finals, hpe_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Cross-Scene 迁移}}
\begin{{table}}[H]\centering\scriptsize
\caption{{HPE cross-scene 与 random split 对比。$\Delta$ 为 cross-scene 减 random，越小越好。}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
方法 & 组合 & Random MPJPE & Cross MPJPE & $\Delta$ & Random PA & Cross PA & $\Delta$\\
\midrule
{hpe_cross_scene_table(hpe_finals, hpe_eval)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{任意模态组合汇总}}
\begin{{table}}[H]\centering\tiny
\caption{{HPE Student-VK 31 种组合按可用模态数量汇总。Best/Worst 使用 MPJPE 排序。}}
\begin{{tabular}}{{rrrrrrrrll}}
\toprule
可用数 & 组合数 & MPJPE & X-Fi & Imp & PA & X-Fi PA & Imp & Best & Worst\\
\midrule
{hpe_combo_summary(hpe_student_vk_random, hpe_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{HAR 定量结果}}
\subsection{{Random、Cross-Scene 与 Cross-Subject 主结果}}
\begin{{table}}[H]\centering\scriptsize
\caption{{HAR 主结果与官方 X-Fi 对比。为避免 final\_eval metadata 混淆，本表统一取 all-combinations 评估中的完整组合行。}}
\begin{{tabular}}{{lllrrrr}}
\toprule
协议 & 方法 & 组合 & Acc & Macro-F1 & X-Fi Acc & Imp\\
\midrule
{har_main_table(har_eval, har_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{HAR 组合鲁棒性汇总}}
\begin{{table}}[H]\centering\tiny
\caption{{HAR Student-VK 15 组合与 Student-NV 7 个非视觉组合的协议级汇总。Best/Worst 使用 Acc 排序。}}
\begin{{tabular}}{{llrrrrrll}}
\toprule
协议 & 方法 & 组合数 & Avg Acc & X-Fi Avg & Imp & Avg F1 & Best & Worst\\
\midrule
{har_combo_summary(har_eval, har_official)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{消融、鲁棒性与复杂度}}
\subsection{{HPE 消融}}
\begin{{table}}[H]\centering\scriptsize
\caption{{HPE 全模态消融。$\Delta$ 相对 Student-VK 完整目标，MPJPE/PA 越低越好。}}
\begin{{tabular}}{{lrrrrp{{5.4cm}}}}
\toprule
设置 & MPJPE & $\Delta$ & PA-MPJPE & $\Delta$ & 说明\\
\midrule
{hpe_ablation_table(hpe_finals, hpe_eval)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{HAR 消融}}
\begin{{table}}[H]\centering\scriptsize
\caption{{HAR random split 消融。Full $\Delta$ 相对 Student-VK，Acc 越高越好。}}
\begin{{tabular}}{{lrrrrr}}
\toprule
设置 & Full Acc & $\Delta$ & Full F1 & Avg-15 Acc & Avg-15 F1\\
\midrule
{har_ablation_table(har_finals, har_eval)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{HPE VK 噪声与关节缺失鲁棒性}}
\begin{{table}}[H]\centering\scriptsize
\caption{{Student-VK 在 VK 噪声与关节随机丢弃下的鲁棒性。$\Delta$ 相对无噪声/无丢弃。}}
\begin{{tabular}}{{rrrrrr}}
\toprule
噪声标准差 & 关节丢弃率 & MPJPE & $\Delta$ & PA-MPJPE & $\Delta$\\
\midrule
{hpe_robustness_table(hpe_eval.get("student_vk_noise_robustness.csv", []))}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{模型复杂度}}
\begin{{table}}[H]\centering\scriptsize
\caption{{HPE 已同步模型复杂度。HAR 当前 final/eval CSV 含参数量，但未同步有效 FPS/显存。}}
\begin{{tabular}}{{llrrr}}
\toprule
模型 & 组合 & Params & FPS & Peak Memory\\
\midrule
{hpe_complexity_table(hpe_eval)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{实验结论与不足}}
\begin{{enumerate}}
\item 在 HPE random split 中，Student-VK 在全模态下达到约 50.17 mm MPJPE，相比官方全模态 83.70 mm 有明显下降；Student-NV 在不使用 VK 的 D+L+R+W 下也优于官方非视觉组合，说明非 RGB 部署路线具备可行性。
\item Teacher 的全模态精度较强，但其缺失模态组合结果只用于诊断；面向任意模态缺失部署时，应优先报告随机缺失训练后的 Student-VK 与 Student-NV。
\item HPE cross-scene 明显高于 random split，说明环境迁移仍是 HPE 的主要挑战；目前 HPE cross-subject 结果未完整同步，不能写成已完成结论。
\item HAR 在 random、cross-scene、cross-subject 的 Student-VK/Student-NV 结果整体较强，尤其 Student-NV 说明无视觉输入也能保持较高动作识别性能。
\item 消融结果表明蒸馏策略与可靠性融合需要分任务讨论：HAR 中 NoKD/Uniform 仍强，但低于 Student-VK；HPE 中部分蒸馏变体可能优于默认 Student-VK，需要后续用统一随机种子和重复实验确认显著性。
\end{{enumerate}}

\appendix
\section{{HPE Student-VK Random 31 组合全表}}
{{\tiny
\begin{{longtable}}{{lrrrrrr}}
\toprule
组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endfirsthead
\toprule
组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endhead
{hpe_full_combo_table(hpe_student_vk_random, hpe_official)}
\bottomrule
\end{{longtable}}
}}

\section{{HPE Student-VK Cross-Scene 31 组合全表}}
{{\tiny
\begin{{longtable}}{{lrrrrrr}}
\toprule
组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endfirsthead
\toprule
组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endhead
{hpe_full_combo_table(hpe_student_vk_cross_scene, hpe_official)}
\bottomrule
\end{{longtable}}
}}

\section{{HPE Student-NV 非视觉组合全表}}
{{\tiny
\begin{{longtable}}{{llrrrrrr}}
\toprule
协议 & 组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endfirsthead
\toprule
协议 & 组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endhead
{hpe_prefixed_combo_table("random", hpe_student_nv_random, hpe_official)}
{hpe_prefixed_combo_table("cross-scene", hpe_student_nv_cross_scene, hpe_official)}
\bottomrule
\end{{longtable}}
}}

\section{{HAR Student-VK 15 组合全表}}
{{\tiny
\begin{{longtable}}{{llrrrr}}
\toprule
协议/模型 & 组合 & Acc & X-Fi Acc & Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
协议/模型 & 组合 & Acc & X-Fi Acc & Imp & Macro-F1\\
\midrule
\endhead
{har_vk_appendix}
\bottomrule
\end{{longtable}}
}}

\section{{HAR Student-NV 非视觉 7 组合全表}}
{{\tiny
\begin{{longtable}}{{llrrrr}}
\toprule
协议/模型 & 组合 & Acc & X-Fi Acc & Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
协议/模型 & 组合 & Acc & X-Fi Acc & Imp & Macro-F1\\
\midrule
\endhead
{har_nv_appendix}
\bottomrule
\end{{longtable}}
}}

\section{{HPE Teacher/Baseline/消融诊断组合表}}
{{\tiny
\begin{{longtable}}{{llrrrrrr}}
\toprule
模型/设置 & 组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endfirsthead
\toprule
模型/设置 & 组合 & MPJPE & X-Fi & Imp & PA-MPJPE & X-Fi PA & Imp\\
\midrule
\endhead
{hpe_diagnostic_appendix}
\bottomrule
\end{{longtable}}
}}

\section{{HAR Teacher/消融诊断组合表}}
{{\tiny
\begin{{longtable}}{{llrrrr}}
\toprule
模型/设置 & 组合 & Acc & X-Fi Acc & Imp & Macro-F1\\
\midrule
\endfirsthead
\toprule
模型/设置 & 组合 & Acc & X-Fi Acc & Imp & Macro-F1\\
\midrule
\endhead
{har_diagnostic_appendix}
\bottomrule
\end{{longtable}}
}}

\section{{同步结果索引}}
{{\tiny
\begin{{longtable}}{{llrl}}
\toprule
任务 & CSV & 行数 & 本文使用方式\\
\midrule
\endfirsthead
\toprule
任务 & CSV & 行数 & 本文使用方式\\
\midrule
\endhead
{eval_file_index(hpe_eval, har_eval)}
\bottomrule
\end{{longtable}}
}}

\section{{缺失与 Metadata 审计}}
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
{audit_table(hpe_eval, har_finals)}
\bottomrule
\end{{longtable}}
}}

\end{{document}}
"""


def hpe_prefixed_combo_table(prefix: str, rows: list[dict[str, object]], official: dict[str, tuple[float, float]]) -> str:
    selected = [row for row in rows if str(row.get("modality_set", "")) in official]
    selected.sort(key=lambda r: (len(str(r["modality_set"]).split("+")), str(r["modality_set"])))
    lines = []
    for row in selected:
        combo = str(row["modality_set"])
        off_mpjpe, off_pa = official[combo]
        mpjpe = float(row["mpjpe_mm"])
        pa = float(row["pa_mpjpe_mm"])
        lines.append(
            rf"{tex_escape(prefix)} & {combo_label(combo)} & {fmt(mpjpe)} & {fmt(off_mpjpe)} & {hpe_imp_cell(off_mpjpe, mpjpe)} & {fmt(pa)} & {fmt(off_pa)} & {hpe_imp_cell(off_pa, pa)}\\"
        )
    return "\n".join(lines) + ("\n" if lines else "")


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
