from __future__ import annotations

import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XFI_TABLE = PROJECT_ROOT / "legacy_xfi" / "xfi_hpe_table1.csv"

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
    "w": "wifi-csi",
    "wifi": "wifi-csi",
    "wifi-csi": "wifi-csi",
    "csi": "wifi-csi",
}


def canonicalize_modality_set(value: str) -> str:
    parts = [part.strip() for part in value.replace(",", "+").split("+") if part.strip()]
    canonical = []
    for part in parts:
        key = part.lower()
        canonical.append(MODALITY_ALIASES.get(key, key))
    return "+".join(canonical)


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_to_mm(value: str) -> float:
    number = float(value)
    # Local HPE code reports meters; X-Fi Table 1 reports millimeters.
    return number * 1000.0 if abs(number) < 10.0 else number


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.2f}"


def _tex_escape(value: str) -> str:
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
    return "".join(replacements.get(char, char) for char in value)


def _build_comparison_rows(result_csv: Path, xfi_csv: Path) -> List[Dict[str, object]]:
    result_rows = _read_csv(result_csv)
    xfi_rows = {
        canonicalize_modality_set(row["modality_set"]): row
        for row in _read_csv(xfi_csv)
        if row.get("modality_set")
    }

    comparison = []
    for row in result_rows:
        modality_set = canonicalize_modality_set(row.get("modality_set", ""))
        if not modality_set or modality_set not in xfi_rows:
            continue
        xfi = xfi_rows[modality_set]
        ours_mpjpe = _metric_to_mm(row["mpjpe"])
        ours_pa = _metric_to_mm(row["pa_mpjpe"])
        xfi_mpjpe = float(xfi["xfi_mpjpe_mm"])
        xfi_pa = float(xfi["xfi_pa_mpjpe_mm"])
        comparison.append(
            {
                "modality_set": modality_set,
                "ours_mpjpe": ours_mpjpe,
                "xfi_mpjpe": xfi_mpjpe,
                "delta_mpjpe": ours_mpjpe - xfi_mpjpe,
                "ours_pa": ours_pa,
                "xfi_pa": xfi_pa,
                "delta_pa": ours_pa - xfi_pa,
            }
        )
    return comparison


def _trend_tex(delta: float) -> str:
    if abs(delta) < 1e-6:
        return r"\textcolor{gray}{=}"
    if delta < 0:
        return r"\textcolor{blue}{$\downarrow$}"
    return r"\textcolor{red}{$\uparrow$}"


def _delta_tex(delta: float) -> str:
    color = "blue" if delta < 0 else "red" if delta > 0 else "gray"
    sign = "+" if delta > 0 else ""
    return rf"\textcolor{{{color}}}{{{sign}{delta:.2f}}}"


def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    total = len(rows)
    mpjpe_better = sum(1 for row in rows if float(row["delta_mpjpe"]) < 0)
    pa_better = sum(1 for row in rows if float(row["delta_pa"]) < 0)
    return {
        "total": total,
        "mpjpe_better": mpjpe_better,
        "pa_better": pa_better,
        "avg_delta_mpjpe": sum(float(row["delta_mpjpe"]) for row in rows) / total if total else 0.0,
        "avg_delta_pa": sum(float(row["delta_pa"]) for row in rows) / total if total else 0.0,
    }


def write_latex_report(
    result_csv: str | Path,
    tex_path: str | Path | None = None,
    xfi_csv: str | Path = DEFAULT_XFI_TABLE,
) -> Path:
    result_csv = Path(result_csv)
    xfi_csv = Path(xfi_csv)
    tex_path = Path(tex_path) if tex_path is not None else result_csv.with_suffix(".xfi_compare.tex")
    rows = _build_comparison_rows(result_csv, xfi_csv)
    summary = _summary(rows)

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\section*{HPE Comparison with X-Fi Table 1}",
        rf"Result CSV: \texttt{{{_tex_escape(str(result_csv))}}}\\",
        rf"X-Fi baseline CSV: \texttt{{{_tex_escape(str(xfi_csv))}}}\\",
        r"Metrics are reported in millimeters. Lower MPJPE/PA-MPJPE is better. "
        r"Blue down arrows indicate improvement over X-Fi; red up arrows indicate degradation.",
        "",
        r"\subsection*{Summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Compared & Improved & Degraded & Avg. Delta (Ours-X-Fi)\\",
        r"\midrule",
        rf"MPJPE & {int(summary['total'])} & {int(summary['mpjpe_better'])} & "
        rf"{int(summary['total'] - summary['mpjpe_better'])} & {_delta_tex(summary['avg_delta_mpjpe'])}\\",
        rf"PA-MPJPE & {int(summary['total'])} & {int(summary['pa_better'])} & "
        rf"{int(summary['total'] - summary['pa_better'])} & {_delta_tex(summary['avg_delta_pa'])}\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\subsection*{Per-Modality Results}",
        r"\scriptsize",
        r"\begin{longtable}{p{0.24\linewidth}rrrrrr}",
        r"\toprule",
        r"Modality & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Ours PA & X-Fi PA & $\Delta$\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Modality & Ours MPJPE & X-Fi MPJPE & $\Delta$ & Ours PA & X-Fi PA & $\Delta$\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            rf"{_tex_escape(str(row['modality_set']))} & "
            rf"{_fmt(float(row['ours_mpjpe']))} & {_fmt(float(row['xfi_mpjpe']))} & "
            rf"{_delta_tex(float(row['delta_mpjpe']))} {_trend_tex(float(row['delta_mpjpe']))} & "
            rf"{_fmt(float(row['ours_pa']))} & {_fmt(float(row['xfi_pa']))} & "
            rf"{_delta_tex(float(row['delta_pa']))} {_trend_tex(float(row['delta_pa']))}\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\normalsize",
            r"\end{document}",
        ]
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _make_pdf_stream(lines: Sequence[str]) -> bytes:
    commands = ["BT", "/F1 8 Tf", "36 806 Td", "10 TL"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands)).encode("latin-1", errors="replace")


def write_fallback_pdf(
    result_csv: str | Path,
    pdf_path: str | Path,
    xfi_csv: str | Path = DEFAULT_XFI_TABLE,
) -> Path:
    rows = _build_comparison_rows(Path(result_csv), Path(xfi_csv))
    summary = _summary(rows)
    text_lines = [
        "HPE Comparison with X-Fi Table 1",
        f"Result CSV: {result_csv}",
        "Metrics: millimeters. Lower is better. DOWN=improved, UP=degraded.",
        (
            f"Summary MPJPE: {int(summary['mpjpe_better'])}/{int(summary['total'])} improved, "
            f"avg delta {summary['avg_delta_mpjpe']:.2f} mm"
        ),
        (
            f"Summary PA-MPJPE: {int(summary['pa_better'])}/{int(summary['total'])} improved, "
            f"avg delta {summary['avg_delta_pa']:.2f} mm"
        ),
        "",
        "Modality | Ours MPJPE | X-Fi MPJPE | Delta | Ours PA | X-Fi PA | Delta",
    ]
    for row in rows:
        mp = float(row["delta_mpjpe"])
        pa = float(row["delta_pa"])
        text_lines.append(
            f"{row['modality_set']} | {_fmt(float(row['ours_mpjpe']))} | {_fmt(float(row['xfi_mpjpe']))} | "
            f"{mp:+.2f} {'DOWN' if mp < 0 else 'UP' if mp > 0 else '='} | "
            f"{_fmt(float(row['ours_pa']))} | {_fmt(float(row['xfi_pa']))} | "
            f"{pa:+.2f} {'DOWN' if pa < 0 else 'UP' if pa > 0 else '='}"
        )

    pages = [text_lines[index:index + 58] for index in range(0, len(text_lines), 58)] or [[]]
    objects: List[bytes] = []
    page_ids = []
    content_ids = []
    for page in pages:
        page_ids.append(0)
        content_ids.append(0)
        objects.append(b"")
        objects.append(_make_pdf_stream(page))
    font_id = len(objects) + 1
    pages_id = font_id + 1
    catalog_id = pages_id + 1

    built: List[bytes] = []
    for idx, page in enumerate(pages):
        page_obj_id = idx * 2 + 1
        content_obj_id = idx * 2 + 2
        page_ids[idx] = page_obj_id
        content_ids[idx] = content_obj_id
        stream = objects[idx * 2 + 1]
        built.append(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_obj_id} 0 R >>"
            ).encode("latin-1")
        )
        built.append(
            b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream"
        )
    built.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    built.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1"))
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
    pdf.extend(
        (
            f"trailer\n<< /Size {len(built) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("latin-1")
    )
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(bytes(pdf))
    return pdf_path


def compile_latex_or_fallback(tex_path: str | Path, result_csv: str | Path, xfi_csv: str | Path) -> Path:
    tex_path = Path(tex_path)
    pdf_path = tex_path.with_suffix(".pdf")
    engine = next((name for name in ("pdflatex", "xelatex") if shutil.which(name)), None)
    if engine is not None:
        command = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tex_path.parent}",
            str(tex_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            if pdf_path.exists():
                return pdf_path
        except Exception as exc:
            log_path = tex_path.with_suffix(".latex_error.txt")
            log_path.write_text(str(exc), encoding="utf-8")
    return write_fallback_pdf(result_csv, pdf_path, xfi_csv)


def generate_xfi_hpe_report(
    result_csv: str | Path,
    xfi_csv: str | Path = DEFAULT_XFI_TABLE,
) -> Dict[str, str]:
    result_csv = Path(result_csv)
    xfi_csv = Path(xfi_csv)
    tex_path = write_latex_report(result_csv, xfi_csv=xfi_csv)
    pdf_path = compile_latex_or_fallback(tex_path, result_csv, xfi_csv)
    rows = _build_comparison_rows(result_csv, xfi_csv)
    summary = _summary(rows)
    summary_csv = result_csv.with_suffix(".xfi_compare_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "modality_set",
            "ours_mpjpe_mm",
            "xfi_mpjpe_mm",
            "delta_mpjpe_mm",
            "mpjpe_trend",
            "ours_pa_mpjpe_mm",
            "xfi_pa_mpjpe_mm",
            "delta_pa_mpjpe_mm",
            "pa_mpjpe_trend",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "modality_set": row["modality_set"],
                    "ours_mpjpe_mm": _fmt(float(row["ours_mpjpe"])),
                    "xfi_mpjpe_mm": _fmt(float(row["xfi_mpjpe"])),
                    "delta_mpjpe_mm": _fmt(float(row["delta_mpjpe"])),
                    "mpjpe_trend": "down_good" if float(row["delta_mpjpe"]) < 0 else "up_bad",
                    "ours_pa_mpjpe_mm": _fmt(float(row["ours_pa"])),
                    "xfi_pa_mpjpe_mm": _fmt(float(row["xfi_pa"])),
                    "delta_pa_mpjpe_mm": _fmt(float(row["delta_pa"])),
                    "pa_mpjpe_trend": "down_good" if float(row["delta_pa"]) < 0 else "up_bad",
                }
            )
    return {
        "tex": str(tex_path),
        "pdf": str(pdf_path),
        "summary_csv": str(summary_csv),
        "compared": str(int(summary["total"])),
        "mpjpe_improved": str(int(summary["mpjpe_better"])),
        "pa_mpjpe_improved": str(int(summary["pa_better"])),
        "avg_delta_mpjpe_mm": f"{summary['avg_delta_mpjpe']:.2f}",
        "avg_delta_pa_mpjpe_mm": f"{summary['avg_delta_pa']:.2f}",
    }
