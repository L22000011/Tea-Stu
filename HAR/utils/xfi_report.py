from __future__ import annotations

import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XFI_TABLE = PROJECT_ROOT / "legacy_xfi" / "xfi_har_table6.csv"

MODALITY_ALIASES = {
    "i": "vk",
    "rgb": "vk",
    "vk": "vk",
    "visual_keypoint": "vk",
    "visual-keypoint": "vk",
    "d": "depth",
    "depth": "depth",
    "l": "lidar",
    "lidar": "lidar",
    "r": "mmwave",
    "radar": "mmwave",
    "mmwave": "mmwave",
}


def canonicalize_modality_set(value: str) -> str:
    parts = [part.strip() for part in value.replace(",", "+").split("+") if part.strip()]
    return "+".join(MODALITY_ALIASES.get(part.lower(), part.lower()) for part in parts)


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _acc_to_pct(value: str) -> float:
    number = float(value)
    return number * 100.0 if abs(number) <= 1.0 else number


def _fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.2f}"


def _tex_escape(value: str) -> str:
    repl = {
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
    return "".join(repl.get(char, char) for char in value)


def _comparison_rows(result_csv: Path, xfi_csv: Path) -> List[Dict[str, object]]:
    xfi_rows = {
        canonicalize_modality_set(row["modality_set"]): row
        for row in _read_csv(xfi_csv)
        if row.get("modality_set")
    }
    rows: List[Dict[str, object]] = []
    for row in _read_csv(result_csv):
        modality_set = canonicalize_modality_set(row.get("modality_set", ""))
        if modality_set not in xfi_rows or "acc" not in row:
            continue
        ours_acc = _acc_to_pct(row["acc"])
        xfi_acc = float(xfi_rows[modality_set]["xfi_acc_pct"])
        ours_f1 = _acc_to_pct(row.get("macro_f1", "nan")) if row.get("macro_f1") not in (None, "", "NA") else math.nan
        rows.append(
            {
                "modality_set": modality_set,
                "ours_acc": ours_acc,
                "xfi_acc": xfi_acc,
                "delta_acc": ours_acc - xfi_acc,
                "ours_macro_f1": ours_f1,
            }
        )
    return rows


def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    total = len(rows)
    improved = sum(1 for row in rows if float(row["delta_acc"]) > 0)
    return {
        "total": total,
        "improved": improved,
        "avg_delta_acc": sum(float(row["delta_acc"]) for row in rows) / total if total else 0.0,
    }


def _trend_tex(delta: float) -> str:
    if abs(delta) < 1e-6:
        return r"\textcolor{gray}{=}"
    if delta > 0:
        return r"\textcolor{blue}{$\uparrow$}"
    return r"\textcolor{red}{$\downarrow$}"


def _delta_tex(delta: float) -> str:
    color = "blue" if delta > 0 else "red" if delta < 0 else "gray"
    sign = "+" if delta > 0 else ""
    return rf"\textcolor{{{color}}}{{{sign}{delta:.2f}}}"


def write_latex_report(
    result_csv: str | Path,
    tex_path: str | Path | None = None,
    xfi_csv: str | Path = DEFAULT_XFI_TABLE,
) -> Path:
    result_csv = Path(result_csv)
    xfi_csv = Path(xfi_csv)
    tex_path = Path(tex_path) if tex_path is not None else result_csv.with_suffix(".xfi_compare.tex")
    rows = _comparison_rows(result_csv, xfi_csv)
    summary = _summary(rows)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\section*{HAR Comparison with X-Fi}",
        rf"Result CSV: \texttt{{{_tex_escape(str(result_csv))}}}\\",
        rf"X-Fi baseline CSV: \texttt{{{_tex_escape(str(xfi_csv))}}}\\",
        r"Metrics are reported as percentages. Higher accuracy is better. "
        r"Blue up arrows indicate improvement over X-Fi; red down arrows indicate degradation.",
        "",
        r"\subsection*{Summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Compared & Improved & Degraded & Avg. Delta (Ours-X-Fi)\\",
        r"\midrule",
        rf"Accuracy & {int(summary['total'])} & {int(summary['improved'])} & "
        rf"{int(summary['total'] - summary['improved'])} & {_delta_tex(summary['avg_delta_acc'])}\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\subsection*{Per-Modality Results}",
        r"\begin{longtable}{lrrrr}",
        r"\toprule",
        r"Modality & Ours Acc. & X-Fi Acc. & $\Delta$ & Ours Macro-F1\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Modality & Ours Acc. & X-Fi Acc. & $\Delta$ & Ours Macro-F1\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        delta = float(row["delta_acc"])
        lines.append(
            rf"{_tex_escape(str(row['modality_set']))} & "
            rf"{_fmt(float(row['ours_acc']))} & {_fmt(float(row['xfi_acc']))} & "
            rf"{_delta_tex(delta)} {_trend_tex(delta)} & {_fmt(float(row['ours_macro_f1']))}\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{document}"])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _make_pdf_stream(lines: Sequence[str]) -> bytes:
    commands = ["BT", "/F1 9 Tf", "36 806 Td", "12 TL"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands)).encode("latin-1", errors="replace")


def write_fallback_pdf(result_csv: str | Path, pdf_path: str | Path, xfi_csv: str | Path = DEFAULT_XFI_TABLE) -> Path:
    rows = _comparison_rows(Path(result_csv), Path(xfi_csv))
    summary = _summary(rows)
    text_lines = [
        "HAR Comparison with X-Fi",
        f"Result CSV: {result_csv}",
        "Metrics: percentages. Higher is better. UP=improved, DOWN=degraded.",
        f"Summary Accuracy: {int(summary['improved'])}/{int(summary['total'])} improved, avg delta {summary['avg_delta_acc']:.2f} pct.",
        "",
        "Modality | Ours Acc | X-Fi Acc | Delta | Ours Macro-F1",
    ]
    for row in rows:
        delta = float(row["delta_acc"])
        text_lines.append(
            f"{row['modality_set']} | {_fmt(float(row['ours_acc']))} | {_fmt(float(row['xfi_acc']))} | "
            f"{delta:+.2f} {'UP' if delta > 0 else 'DOWN' if delta < 0 else '='} | {_fmt(float(row['ours_macro_f1']))}"
        )

    pages = [text_lines[index:index + 58] for index in range(0, len(text_lines), 58)] or [[]]
    built: List[bytes] = []
    page_ids = []
    for page in pages:
        page_id = len(built) + 1
        content_id = len(built) + 2
        page_ids.append(page_id)
        stream = _make_pdf_stream(page)
        built.append(
            (
                f"<< /Type /Page /Parent {2 * len(pages) + 2} 0 R /MediaBox [0 0 612 842] "
                f"/Resources << /Font << /F1 {2 * len(pages) + 1} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        built.append(b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream")
    font_id = len(built) + 1
    pages_id = font_id + 1
    catalog_id = pages_id + 1
    built.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    built.append(f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {len(page_ids)} >>".encode("latin-1"))
    built.append(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in enumerate(built, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(built) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend((f"trailer\n<< /Size {len(built) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode("latin-1"))
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(bytes(pdf))
    return pdf_path


def compile_latex_or_fallback(tex_path: str | Path, result_csv: str | Path, xfi_csv: str | Path) -> Path:
    tex_path = Path(tex_path)
    pdf_path = tex_path.with_suffix(".pdf")
    engine = next((name for name in ("pdflatex", "xelatex") if shutil.which(name)), None)
    if engine is not None:
        try:
            subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tex_path.parent}", str(tex_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if pdf_path.exists():
                return pdf_path
        except Exception as exc:
            tex_path.with_suffix(".latex_error.txt").write_text(str(exc), encoding="utf-8")
    return write_fallback_pdf(result_csv, pdf_path, xfi_csv)


def generate_xfi_har_report(result_csv: str | Path, xfi_csv: str | Path = DEFAULT_XFI_TABLE) -> Dict[str, str]:
    result_csv = Path(result_csv)
    xfi_csv = Path(xfi_csv)
    tex_path = write_latex_report(result_csv, xfi_csv=xfi_csv)
    pdf_path = compile_latex_or_fallback(tex_path, result_csv, xfi_csv)
    rows = _comparison_rows(result_csv, xfi_csv)
    summary = _summary(rows)
    summary_csv = result_csv.with_suffix(".xfi_compare_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["modality_set", "ours_acc_pct", "xfi_acc_pct", "delta_acc_pct", "trend", "ours_macro_f1_pct"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "modality_set": row["modality_set"],
                    "ours_acc_pct": _fmt(float(row["ours_acc"])),
                    "xfi_acc_pct": _fmt(float(row["xfi_acc"])),
                    "delta_acc_pct": _fmt(float(row["delta_acc"])),
                    "trend": "up_good" if float(row["delta_acc"]) > 0 else "down_bad",
                    "ours_macro_f1_pct": _fmt(float(row["ours_macro_f1"])),
                }
            )
    return {
        "tex": str(tex_path),
        "pdf": str(pdf_path),
        "summary_csv": str(summary_csv),
        "compared": str(int(summary["total"])),
        "acc_improved": str(int(summary["improved"])),
        "avg_delta_acc_pct": f"{summary['avg_delta_acc']:.2f}",
    }
