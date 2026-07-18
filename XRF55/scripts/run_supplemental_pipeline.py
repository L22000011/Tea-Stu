from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    index: int
    title: str
    name: str
    command: List[str]
    outputs: tuple[Path, ...]
    required_inputs: tuple[Path, ...] = ()


def rel(path: str) -> Path:
    return PROJECT_ROOT / path


def outputs_exist(outputs: Iterable[Path]) -> bool:
    return all(path.exists() for path in outputs)


def append_status(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "name", "title", "status", "reason", "returncode", "seconds", "outputs"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_steps(args: argparse.Namespace) -> list[Step]:
    python = sys.executable
    return [
        Step(1, "Teacher missing-count summary", "teacher_missing_summary",
             [python, "-u", str(rel("scripts/summarize_missing_counts.py")), "--input-csv", str(rel("outputs/eval/teacher_all_combinations.csv")), "--output-csv", str(rel("outputs/eval/teacher_all_combinations_by_missing.csv")), "--output-md", str(rel("outputs/eval/teacher_all_combinations_by_missing.md"))],
             (rel("outputs/eval/teacher_all_combinations_by_missing.csv"), rel("outputs/eval/teacher_all_combinations_by_missing.md")),
             (rel("outputs/eval/teacher_all_combinations.csv"),)),
        Step(2, "Student missing-count summary", "student_missing_summary",
             [python, "-u", str(rel("scripts/summarize_missing_counts.py")), "--input-csv", str(rel("outputs/eval/student_all_combinations.csv")), "--output-csv", str(rel("outputs/eval/student_all_combinations_by_missing.csv")), "--output-md", str(rel("outputs/eval/student_all_combinations_by_missing.md"))],
             (rel("outputs/eval/student_all_combinations_by_missing.csv"), rel("outputs/eval/student_all_combinations_by_missing.md")),
             (rel("outputs/eval/student_all_combinations.csv"),)),
        Step(3, "Baseline missing-count summary", "baseline_missing_summary",
             [python, "-u", str(rel("scripts/summarize_missing_counts.py")), "--input-csv", str(rel("outputs/eval/baseline_all_combinations.csv")), "--output-csv", str(rel("outputs/eval/baseline_all_combinations_by_missing.csv")), "--output-md", str(rel("outputs/eval/baseline_all_combinations_by_missing.md"))],
             (rel("outputs/eval/baseline_all_combinations_by_missing.csv"), rel("outputs/eval/baseline_all_combinations_by_missing.md")),
             (rel("outputs/eval/baseline_all_combinations.csv"),)),
        Step(4, "No-distillation missing-count summary", "ablation_no_distill_missing_summary",
             [python, "-u", str(rel("scripts/summarize_missing_counts.py")), "--input-csv", str(rel("outputs/eval/ablation_no_distill_all_combinations.csv")), "--output-csv", str(rel("outputs/eval/ablation_no_distill_by_missing.csv")), "--output-md", str(rel("outputs/eval/ablation_no_distill_by_missing.md"))],
             (rel("outputs/eval/ablation_no_distill_by_missing.csv"), rel("outputs/eval/ablation_no_distill_by_missing.md")),
             (rel("outputs/eval/ablation_no_distill_all_combinations.csv"),)),
        Step(5, "Uniform-fusion missing-count summary", "ablation_uniform_missing_summary",
             [python, "-u", str(rel("scripts/summarize_missing_counts.py")), "--input-csv", str(rel("outputs/eval/ablation_uniform_all_combinations.csv")), "--output-csv", str(rel("outputs/eval/ablation_uniform_by_missing.csv")), "--output-md", str(rel("outputs/eval/ablation_uniform_by_missing.md"))],
             (rel("outputs/eval/ablation_uniform_by_missing.csv"), rel("outputs/eval/ablation_uniform_by_missing.md")),
             (rel("outputs/eval/ablation_uniform_all_combinations.csv"),)),
        Step(6, "No-semantic missing-count summary", "ablation_no_semantic_missing_summary",
             [python, "-u", str(rel("scripts/summarize_missing_counts.py")), "--input-csv", str(rel("outputs/eval/ablation_no_semantic_all_combinations.csv")), "--output-csv", str(rel("outputs/eval/ablation_no_semantic_by_missing.csv")), "--output-md", str(rel("outputs/eval/ablation_no_semantic_by_missing.md"))],
             (rel("outputs/eval/ablation_no_semantic_by_missing.csv"), rel("outputs/eval/ablation_no_semantic_by_missing.md")),
             (rel("outputs/eval/ablation_no_semantic_all_combinations.csv"),)),
        Step(7, "Teacher complexity export", "teacher_complexity",
             [python, "-u", str(rel("scripts/export_complexity.py")), "--dataset", args.dataset, "--config", str(rel("configs/teacher_full.yaml")), "--checkpoint", str(rel("outputs/teacher_full/best.pth")), "--device", args.device, "--output-csv", str(rel("outputs/eval/teacher_complexity.csv")), "--output-md", str(rel("outputs/eval/teacher_complexity.md"))],
             (rel("outputs/eval/teacher_complexity.csv"), rel("outputs/eval/teacher_complexity.md")),
             (rel("outputs/teacher_full/best.pth"),)),
        Step(8, "Student complexity export", "student_complexity",
             [python, "-u", str(rel("scripts/export_complexity.py")), "--dataset", args.dataset, "--config", str(rel("configs/student_missing.yaml")), "--checkpoint", str(rel("outputs/student_missing/best.pth")), "--device", args.device, "--output-csv", str(rel("outputs/eval/student_complexity.csv")), "--output-md", str(rel("outputs/eval/student_complexity.md"))],
             (rel("outputs/eval/student_complexity.csv"), rel("outputs/eval/student_complexity.md")),
             (rel("outputs/student_missing/best.pth"),)),
        Step(9, "Baseline complexity export", "baseline_complexity",
             [python, "-u", str(rel("scripts/export_complexity.py")), "--dataset", args.dataset, "--config", str(rel("configs/baseline_full.yaml")), "--checkpoint", str(rel("outputs/baseline_full/best.pth")), "--device", args.device, "--output-csv", str(rel("outputs/eval/baseline_complexity.csv")), "--output-md", str(rel("outputs/eval/baseline_complexity.md"))],
             (rel("outputs/eval/baseline_complexity.csv"), rel("outputs/eval/baseline_complexity.md")),
             (rel("outputs/baseline_full/best.pth"),)),
    ]


def run_step(step: Step, args: argparse.Namespace, log_dir: Path, status_csv: Path) -> bool:
    print("=" * 88)
    print(f"[{step.index}/9] {step.title}")
    print("command=", " ".join(str(part) for part in step.command))
    print("outputs=", ";".join(str(path) for path in step.outputs))
    print("=" * 88)

    if args.dry_run:
        append_status(status_csv, {
            "index": step.index,
            "name": step.name,
            "title": step.title,
            "status": "dry_run",
            "reason": "",
            "returncode": 0,
            "seconds": 0.0,
            "outputs": ";".join(str(path) for path in step.outputs),
        })
        print("DRY-RUN | command not executed")
        return True

    missing_required = [path for path in step.required_inputs if not path.exists()]
    if missing_required:
        reason = "missing_required=" + ",".join(str(path) for path in missing_required)
        append_status(status_csv, {
            "index": step.index,
            "name": step.name,
            "title": step.title,
            "status": "skipped",
            "reason": reason,
            "returncode": 0,
            "seconds": 0.0,
            "outputs": ";".join(str(path) for path in step.outputs),
        })
        print(f"SKIP | {reason}")
        return True

    if outputs_exist(step.outputs) and not args.force:
        append_status(status_csv, {
            "index": step.index,
            "name": step.name,
            "title": step.title,
            "status": "skipped",
            "reason": "outputs_exist",
            "returncode": 0,
            "seconds": 0.0,
            "outputs": ";".join(str(path) for path in step.outputs),
        })
        print("SKIP | outputs_exist")
        return True

    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_path = log_dir / f"{step.name}.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            step.command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
    seconds = time.perf_counter() - start
    status = "done" if process.returncode == 0 else "failed"
    append_status(status_csv, {
        "index": step.index,
        "name": step.name,
        "title": step.title,
        "status": status,
        "reason": "",
        "returncode": process.returncode,
        "seconds": round(seconds, 3),
        "outputs": ";".join(str(path) for path in step.outputs),
    })
    print(f"{status.upper()} | returncode={process.returncode} | log={log_path}")
    return process.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run the supplemental XRF55 experiments.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = rel(f"outputs/supplemental_runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir = run_dir / "logs"
    status_csv = run_dir / "status.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = build_steps(args)
    failed = []
    for step in steps:
        ok = run_step(step, args, log_dir, status_csv)
        if not ok:
            failed.append(step.name)
            if not args.continue_on_error:
                break
    if failed:
        print("[Supplemental] Failed steps:")
        for name in failed:
            print(f"  - {name}")
        raise SystemExit(1)
    print(f"[Supplemental] Finished. Records: {run_dir}")


if __name__ == "__main__":
    main()
