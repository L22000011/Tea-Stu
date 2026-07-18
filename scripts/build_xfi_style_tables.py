from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_GEN = ROOT / "Paper" / "generated"

HPE_FULL = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
HAR_FULL = ["vk", "depth", "lidar", "mmwave"]
HPE_NV = ["depth", "lidar", "mmwave", "wifi-csi"]
HAR_NV = ["depth", "lidar", "mmwave"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def modality_label_short(name: str) -> str:
    mapping = {
        "vk": "V",
        "depth": "D",
        "lidar": "L",
        "mmwave": "R",
        "wifi-csi": "W",
    }
    return "+".join(mapping.get(part, part) for part in name.split("+"))


def parts(key: str) -> list[str]:
    return key.split("+") if key else []


def make_key(items: list[str]) -> str:
    return "+".join(items)


def delta_cell(delta: float, lower_is_better: bool) -> str:
    if abs(delta) < 0.05:
        return r"\neutral{0.0}"
    improved = delta < 0 if lower_is_better else delta > 0
    macro = r"\gain" if improved else r"\loss"
    arrow = r"$\downarrow$" if delta < 0 else r"$\uparrow$"
    return f"{macro}{{{arrow} {abs(delta):.1f}}}"


def gray_cell(value: str) -> str:
    return rf"\cellcolor{{gray!12}}{value}"


def append_grouped_row(lines: list[str], key: str, previous_count: int | None) -> int:
    count = len(parts(key))
    if previous_count is not None and count != previous_count:
        lines.append(r"\midrule")
    return count


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}"


def build_hpe_table(chinese: bool = False) -> str:
    xfi_rows = read_rows(ROOT / "HPE" / "legacy_xfi" / "xfi_hpe_table1.csv")
    ours_rows = {
        r["modality_set"]: r
        for r in read_rows(ROOT / "HPE" / "outputs" / "eval" / "student_vk_random_all_combinations.csv")
    }
    caption = r"\cnHpeDirectCaption" if chinese else (
        "Direct HPE comparison with the original X-Fi table entries. "
        "MPJPE and PA-MPJPE are in millimeters; lower is better. "
        "Green downward deltas indicate improvement over X-Fi, and red upward deltas indicate degradation."
    )
    first_col = r"\cnModalitySetShort" if chinese else "Modality"
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:xfi_hpe_direct}",
        r"\scriptsize",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{{first_col}}} & \multicolumn{{3}}{{c}}{{MPJPE$\downarrow$}} & \multicolumn{{3}}{{c}}{{PA-MPJPE$\downarrow$}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & X-Fi & VK-RMD & $\Delta$ & X-Fi & VK-RMD & $\Delta$ \\",
        r"\midrule",
    ]
    previous_count = None
    for row in xfi_rows:
        key = row["modality_set"]
        previous_count = append_grouped_row(lines, key, previous_count)
        ours = ours_rows[key]
        xfi_mpjpe = float(row["xfi_mpjpe_mm"])
        xfi_pa = float(row["xfi_pa_mpjpe_mm"])
        ours_mpjpe = float(ours["mpjpe"]) * 1000.0
        ours_pa = float(ours["pa_mpjpe"]) * 1000.0
        lines.append(
            f"{tex_escape(modality_label_short(key))} & {xfi_mpjpe:.1f} & {gray_cell(f'{ours_mpjpe:.1f}')} & "
            f"{delta_cell(ours_mpjpe - xfi_mpjpe, True)} & {xfi_pa:.1f} & "
            f"{gray_cell(f'{ours_pa:.1f}')} & {delta_cell(ours_pa - xfi_pa, True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""]
    return "\n".join(lines)


def build_har_table(chinese: bool = False) -> str:
    xfi_rows = read_rows(ROOT / "HAR" / "legacy_xfi" / "xfi_har_table6.csv")
    ours_rows = {
        r["modality_set"]: r
        for r in read_rows(ROOT / "HAR" / "outputs" / "eval" / "student_vk_random_all_combinations.csv")
    }
    caption = r"\cnHarDirectCaption" if chinese else (
        "Direct HAR comparison with the original X-Fi table entries. "
        "Accuracy is reported in percent; higher is better. "
        "Green upward deltas indicate improvement over X-Fi, and red downward deltas indicate degradation."
    )
    first_col = r"\cnModalitySetShort" if chinese else "Modality"
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:xfi_har_direct}",
        r"\scriptsize",
        r"\begin{adjustbox}{width=0.82\textwidth}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"{first_col} & X-Fi Acc.$\uparrow$ & VK-RMD Acc.$\uparrow$ & $\Delta$ Acc. \\",
        r"\midrule",
    ]
    previous_count = None
    for row in xfi_rows:
        key = row["modality_set"]
        previous_count = append_grouped_row(lines, key, previous_count)
        ours = ours_rows[key]
        xfi_acc = float(row["xfi_acc_pct"])
        ours_acc = float(ours["acc"]) * 100.0
        lines.append(
            f"{tex_escape(modality_label_short(key))} & {xfi_acc:.1f} & {gray_cell(f'{ours_acc:.1f}')} & "
            f"{delta_cell(ours_acc - xfi_acc, False)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""]
    return "\n".join(lines)


def grouped_hpe(path: Path, full_modalities: list[str]) -> dict[int, tuple[float | None, float | None]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"mpjpe": [], "pa": []})
    for row in read_rows(path):
        missing = len(full_modalities) - len(parts(row["modality_set"]))
        buckets[missing]["mpjpe"].append(float(row["mpjpe"]) * 1000.0)
        buckets[missing]["pa"].append(float(row["pa_mpjpe"]) * 1000.0)
    return {k: (mean(v["mpjpe"]), mean(v["pa"])) for k, v in buckets.items()}


def grouped_har(path: Path, full_modalities: list[str]) -> dict[int, tuple[float | None, float | None]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"acc": [], "f1": []})
    for row in read_rows(path):
        missing = len(full_modalities) - len(parts(row["modality_set"]))
        buckets[missing]["acc"].append(float(row["acc"]) * 100.0)
        buckets[missing]["f1"].append(float(row["macro_f1"]) * 100.0)
    return {k: (mean(v["acc"]), mean(v["f1"])) for k, v in buckets.items()}


def build_fsg_severity_table(chinese: bool = False) -> str:
    hpe_vk = grouped_hpe(ROOT / "HPE" / "outputs" / "eval" / "student_vk_random_all_combinations.csv", HPE_FULL)
    hpe_nv = grouped_hpe(ROOT / "HPE" / "outputs" / "eval" / "student_nv_random_nonvisual_combinations.csv", HPE_NV)
    har_vk = grouped_har(ROOT / "HAR" / "outputs" / "eval" / "student_vk_random_all_combinations.csv", HAR_FULL)
    har_nv = grouped_har(ROOT / "HAR" / "outputs" / "eval" / "student_nv_random_nonvisual_combinations.csv", HAR_NV)
    caption = (
        "FSG 缺失强度核心结果。HPE 为 MPJPE/PA-MPJPE，越低越好；HAR 为 Acc./F1，越高越好。"
        if chinese
        else "Core FSG severity results grouped by the number of missing modalities."
    )
    missing_col = "缺失数" if chinese else "Missing"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:severity_cn}" if chinese else r"\label{tab:severity}",
        r"\scriptsize",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{{missing_col}}} & \multicolumn{{4}}{{c}}{{HPE$\downarrow$}} & \multicolumn{{4}}{{c}}{{HAR$\uparrow$}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r" & VK MPJPE & VK PA & NV MPJPE & NV PA & VK Acc. & VK F1 & NV Acc. & NV F1 \\",
        r"\midrule",
    ]
    for missing in range(0, 5):
        hv_m, hv_p = hpe_vk.get(missing, (None, None))
        hn_m, hn_p = hpe_nv.get(missing, (None, None))
        av_a, av_f = har_vk.get(missing, (None, None))
        an_a, an_f = har_nv.get(missing, (None, None))
        lines.append(
            f"{missing} & {fmt(hv_m)} & {fmt(hv_p)} & {fmt(hn_m)} & {fmt(hn_p)} & "
            f"{fmt(av_a)} & {fmt(av_f)} & {fmt(an_a)} & {fmt(an_f)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""]
    return "\n".join(lines)


def lookup_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["modality_set"]: row for row in read_rows(path)}


def build_leave_one_out_table(chinese: bool = False) -> str:
    hpe = lookup_rows(ROOT / "HPE" / "outputs" / "eval" / "student_vk_random_all_combinations.csv")
    har = lookup_rows(ROOT / "HAR" / "outputs" / "eval" / "student_vk_random_all_combinations.csv")
    hpe_full = hpe[make_key(HPE_FULL)]
    har_full = har[make_key(HAR_FULL)]
    hpe_full_m = float(hpe_full["mpjpe"]) * 1000.0
    hpe_full_p = float(hpe_full["pa_mpjpe"]) * 1000.0
    har_full_a = float(har_full["acc"]) * 100.0
    har_full_f = float(har_full["macro_f1"]) * 100.0
    caption = (
        "Leave-one-out 关键模态分析。HPE 数值为移除该模态后的误差增加量；HAR 数值为移除该模态后的准确率/F1 下降量。"
        if chinese
        else "Leave-one-out key-modality analysis."
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:loo_cn}" if chinese else r"\label{tab:loo}",
        r"\scriptsize",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Removed & HPE $\Delta$MPJPE & HPE $\Delta$PA & HAR $\Delta$Acc. & HAR $\Delta$F1 \\",
        r"\midrule",
    ]
    all_modalities = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    for modality in all_modalities:
        hpe_key = make_key([m for m in HPE_FULL if m != modality])
        hpe_row = hpe[hpe_key]
        d_m = float(hpe_row["mpjpe"]) * 1000.0 - hpe_full_m
        d_p = float(hpe_row["pa_mpjpe"]) * 1000.0 - hpe_full_p
        if modality in HAR_FULL:
            har_key = make_key([m for m in HAR_FULL if m != modality])
            har_row = har[har_key]
            d_a = har_full_a - float(har_row["acc"]) * 100.0
            d_f = har_full_f - float(har_row["macro_f1"]) * 100.0
            har_cells = f"{d_a:.1f} & {d_f:.1f}"
        else:
            har_cells = "-- & --"
        lines.append(f"{modality_label_short(modality)} & {d_m:.1f} & {d_p:.1f} & {har_cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(lines)


def build_protocol_table(chinese: bool = False) -> str:
    hpe = read_rows(ROOT / "tables" / "main_hpe_results.csv")
    har = read_rows(ROOT / "tables" / "main_har_results.csv")
    hpe_rows = {(r["method"], r["protocol"]): r for r in hpe if "VK-RMD" in r["method"]}
    har_rows = {(r["method"], r["protocol"]): r for r in har if "VK-RMD" in r["method"]}
    caption = "跨协议泛化核心结果。HPE 为平均/全模态 MPJPE，HAR 为平均/全模态 Accuracy。" if chinese else "Cross-protocol generalization."
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:protocol_cn}" if chinese else r"\label{tab:protocol}",
        r"\scriptsize",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Protocol & Model & HPE Avg.$\downarrow$ & HPE Full$\downarrow$ & HAR Avg.$\uparrow$ & HAR Full$\uparrow$ \\",
        r"\midrule",
    ]
    protocols = ["random split", "cross-subject", "cross-scene"]
    for protocol in protocols:
        for model in ["VK-RMD Student-VK", "VK-RMD Student-NV"]:
            h = hpe_rows.get((model, protocol))
            a = har_rows.get((model, protocol))
            lines.append(
                f"{tex_escape(protocol)} & {model.replace('Student-', '')} & "
                f"{h['avg_mpjpe_mm'] if h else '--'} & {h['full_mpjpe_mm'] if h else '--'} & "
                f"{a['avg_accuracy_pct'] if a else '--'} & {a['full_accuracy_pct'] if a else '--'} \\\\"
            )
        lines.append(r"\midrule" if protocol != protocols[-1] else r"")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""]
    return "\n".join([line for line in lines if line != ""])


def build_ablation_table(chinese: bool = False) -> str:
    rows = [r for r in read_rows(ROOT / "tables" / "ablation_results.csv") if r["main_text_policy"] in {"main text", "main ablation", "discussion/ablation"}]
    caption = "核心消融结果。HPE 数值为 MPJPE，HAR 数值为 Accuracy。" if chinese else "Core ablation results."
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:ablation_cn}" if chinese else r"\label{tab:ablation}",
        r"\scriptsize",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Task & Setting & Avg. & Full \\",
        r"\midrule",
    ]
    for row in rows:
        setting = row["setting"].replace("VK-RMD Student-VK", "VK-RMD")
        avg_metric = row["avg_metric"].replace("%", r"\%")
        full_metric = row["full_metric"].replace("%", r"\%")
        lines.append(f"{row['task']} & {tex_escape(setting)} & {avg_metric} & {full_metric} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    outputs = {
        "xfi_hpe_direct_table.tex": build_hpe_table(chinese=False),
        "xfi_har_direct_table.tex": build_har_table(chinese=False),
        "xfi_hpe_direct_table_cn.tex": build_hpe_table(chinese=True),
        "xfi_har_direct_table_cn.tex": build_har_table(chinese=True),
        "fsg_severity_table.tex": build_fsg_severity_table(chinese=False),
        "fsg_severity_table_cn.tex": build_fsg_severity_table(chinese=True),
        "leave_one_out_table.tex": build_leave_one_out_table(chinese=False),
        "leave_one_out_table_cn.tex": build_leave_one_out_table(chinese=True),
        "protocol_generalization_table_cn.tex": build_protocol_table(chinese=True),
        "ablation_core_table_cn.tex": build_ablation_table(chinese=True),
    }
    for name, content in outputs.items():
        path = PAPER_GEN / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
