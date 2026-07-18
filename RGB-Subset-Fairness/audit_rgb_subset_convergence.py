from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


XFI_HPE_RE = re.compile(r"Epoch:\s*(?P<epoch>\d+),\s*Loss:\s*(?P<loss>[-+0-9.eE]+)")
XFI_HAR_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*Accuracy:\s*(?P<acc>[-+0-9.eE]+),\s*Loss:\s*(?P<loss>[-+0-9.eE]+)"
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "INF", "-INF"}:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def infer_task_protocol(path: Path) -> tuple[str, str]:
    text = str(path).lower().replace("\\", "/")
    task = "HPE" if "hpe" in text else "HAR" if "har" in text else "unknown"
    protocol = "cross_scene" if "cross_scene" in text else "cross_subject" if "cross_subject" in text else "unknown"
    return task, protocol


def summarize_curve(losses: list[float]) -> dict[str, str]:
    if not losses:
        return {
            "epochs_observed": "0",
            "first_loss": "",
            "last_loss": "",
            "loss_delta": "",
            "normalized_drop": "",
            "last3_rel_drift": "",
            "convergence_status": "not_assessable",
        }
    first = losses[0]
    last = losses[-1]
    delta = first - last
    scale = max(abs(first), abs(last), 1.0)
    normalized_drop = delta / scale
    if len(losses) >= 4:
        previous = losses[-4:-1]
        prev_mean = sum(previous) / len(previous)
        last3_rel_drift = abs(last - prev_mean) / max(abs(prev_mean), 1.0)
    elif len(losses) >= 2:
        prev_mean = losses[-2]
        last3_rel_drift = abs(last - prev_mean) / max(abs(prev_mean), 1.0)
    else:
        last3_rel_drift = float("nan")

    if len(losses) < 3:
        status = "too_few_epochs"
    elif normalized_drop > 0.05 and math.isfinite(last3_rel_drift) and last3_rel_drift <= 0.05:
        status = "stable_fixed_budget"
    elif normalized_drop > 0.05:
        status = "improving_not_converged"
    elif normalized_drop > 0.0:
        status = "weakly_improving"
    else:
        status = "not_improving_or_noisy"

    return {
        "epochs_observed": str(len(losses)),
        "first_loss": f"{first:.8f}",
        "last_loss": f"{last:.8f}",
        "loss_delta": f"{delta:.8f}",
        "normalized_drop": f"{normalized_drop:.6f}",
        "last3_rel_drift": "" if not math.isfinite(last3_rel_drift) else f"{last3_rel_drift:.6f}",
        "convergence_status": status,
    }


def parse_xfi_log(path: Path) -> dict[str, str] | None:
    task, protocol = infer_task_protocol(path)
    losses: list[float] = []
    accs: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        hpe = XFI_HPE_RE.search(line)
        har = XFI_HAR_RE.search(line)
        match = har or hpe
        if match is None:
            continue
        loss = to_float(match.groupdict().get("loss"))
        if loss is not None:
            losses.append(loss)
        acc = to_float(match.groupdict().get("acc"))
        if acc is not None:
            accs.append(acc)
    if not losses:
        return None
    row = {
        "source": "xfi_log",
        "method": "RGB-XFi",
        "task": task,
        "protocol": protocol,
        "stage": "train",
        "metric_hint": f"last_train_acc={accs[-1]:.6f}" if accs else "",
        "evidence_path": str(path),
        "note": "Fixed-budget diagnostic; not a full convergence proof.",
    }
    row.update(summarize_curve(losses))
    return row


def parse_epoch_history(path: Path) -> dict[str, str] | None:
    task, protocol = infer_task_protocol(path)
    losses: list[float] = []
    metric_hints: list[str] = []
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            loss = to_float(row.get("train_loss"))
            if loss is not None:
                losses.append(loss)
            for key in ["val_mpjpe", "val_pa_mpjpe", "val_acc", "val_macro_f1", "best_mpjpe", "best_acc"]:
                value = to_float(row.get(key))
                if value is not None:
                    metric_hints.append(f"{key}={value:.6f}")
    if not losses:
        return None
    row = {
        "source": "epoch_history",
        "method": "VK-RMD",
        "task": task,
        "protocol": protocol,
        "stage": "train",
        "metric_hint": metric_hints[-1] if metric_hints else "",
        "evidence_path": str(path),
        "note": "Fixed-budget diagnostic; use together with identical update budget and sample manifest.",
    }
    row.update(summarize_curve(losses))
    return row


def collect_rows(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for log_path in sorted((repo / "outputs" / "rgb_subset_runs").glob("**/logs/*.log")):
        parsed = parse_xfi_log(log_path)
        if parsed is not None:
            rows.append(parsed)

    for project_name in ["HPE", "HAR", "MMFi_HPE", "MMFi_HAR"]:
        project = repo / project_name
        if not project.exists():
            continue
        for history_path in sorted(project.glob("outputs_rgb_subset/**/epoch_history.csv")):
            parsed = parse_epoch_history(history_path)
            if parsed is not None:
                rows.append(parsed)
    return rows


def write_outputs(repo: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    tables_dir = repo / "tables"
    reports_dir = repo / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "rgb_subset_convergence_audit.csv"
    md_path = reports_dir / "rgb_subset_convergence_audit.md"
    fieldnames = [
        "method",
        "task",
        "protocol",
        "stage",
        "epochs_observed",
        "first_loss",
        "last_loss",
        "loss_delta",
        "normalized_drop",
        "last3_rel_drift",
        "convergence_status",
        "metric_hint",
        "source",
        "evidence_path",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RGB-Subset Fixed-Budget Convergence Audit",
        "",
        "This report checks whether the short RGB-available subset experiments show usable optimization evidence.",
        "It should be described as a fixed-budget fairness diagnostic, not as a full convergence benchmark.",
        "",
        "| Method | Task | Protocol | Epochs | First loss | Last loss | Drop | Drift | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| {method} | {task} | {protocol} | {epochs_observed} | {first_loss} | {last_loss} | "
                "{normalized_drop} | {last3_rel_drift} | {convergence_status} |".format(**row)
            )
    else:
        lines.append("| none | none | none | 0 | | | | | no training curves found |")
    lines.extend(
        [
            "",
            "## How to write this in the paper",
            "",
            "Use cautious wording:",
            "",
            "> We conduct a fixed-budget RGB-available subset comparison to audit fairness under identical sample availability and cross protocols. "
            "Both RGB-XFi and VK-RMD are trained with the same update budget, and convergence diagnostics are reported to verify that optimization proceeds normally. "
            "This subset study is not used as the primary full-convergence benchmark.",
            "",
            "Avoid claiming that 10 epochs are fully converged unless the task-specific curves clearly plateau.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit convergence for RGB-subset fixed-budget experiments.")
    parser.add_argument("--repo", type=Path, default=repo_root_from_script())
    args = parser.parse_args()
    repo = args.repo.resolve()
    rows = collect_rows(repo)
    csv_path, md_path = write_outputs(repo, rows)
    print(f"Convergence audit rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
