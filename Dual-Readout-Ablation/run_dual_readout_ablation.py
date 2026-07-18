#!/usr/bin/env python3
"""Orchestrate the quick, matched chunk/global/dual readout ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODES = ("chunk_only", "global_only", "dual")


def resolve_project(repo: Path, task: str) -> Path:
    candidates = [repo / task, repo / f"MMFi_{task}"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate {task} project; checked: {candidates}")


def append_status(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> tuple[str, float, int]:
    printable = subprocess.list2cmdline(command)
    print(f"command= {printable}", flush=True)
    if dry_run:
        return "dry-run", 0.0, 0
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    return ("done" if code == 0 else "failed"), time.time() - start, code


def summarize(output_root: Path) -> None:
    rows = []
    for task in ("HPE", "HAR"):
        for mode in MODES:
            path = output_root / task / mode / "all_combinations.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                data = list(csv.DictReader(handle))
            if task == "HPE":
                values = [1000.0 * float(row["mpjpe"]) for row in data]
                pa = [1000.0 * float(row["pa_mpjpe"]) for row in data]
                row = {"task": task, "mode": mode, "combination_count": len(data), "metric": "MPJPE_mm", "all_subset_metric": sum(values) / len(values), "secondary_metric": sum(pa) / len(pa), "secondary_name": "PA_MPJPE_mm"}
            else:
                values = [100.0 * float(row["acc"]) for row in data]
                f1 = [100.0 * float(row["macro_f1"]) for row in data]
                row = {"task": task, "mode": mode, "combination_count": len(data), "metric": "Accuracy_pct", "all_subset_metric": sum(values) / len(values), "secondary_metric": sum(f1) / len(f1), "secondary_name": "Macro_F1_pct"}
            rows.append(row)
    if not rows:
        return
    with (output_root / "dual_readout_ablation_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=["HPE", "HAR"], default=["HPE", "HAR"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output_root = (args.output_root or repo / "outputs" / "dual_readout_ablation").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / timestamp
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.csv"
    entry = Path(__file__).resolve().parent / "run_variant.py"

    manifest = vars(args).copy()
    manifest.update({"repo_root": str(repo), "output_root": str(output_root), "timestamp": timestamp})
    for key, value in list(manifest.items()):
        if isinstance(value, Path):
            manifest[key] = str(value)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    task_index = 0
    total = len(args.tasks) * len(args.modes) * 2
    for task in args.tasks:
        project = resolve_project(repo, task)
        config = project / "configs" / "student_vk_missing.yaml"
        teacher = project / "outputs" / "teacher_full" / "best.pth"
        if not teacher.exists() and not args.dry_run:
            raise FileNotFoundError(f"Missing teacher checkpoint: {teacher}")
        for mode in args.modes:
            variant_dir = output_root / task / mode
            best = variant_dir / "best.pth"
            last = variant_dir / "last.pth"
            train_complete = variant_dir / "final_summary.json"
            eval_csv = variant_dir / "all_combinations.csv"
            for stage in ("train", "eval"):
                task_index += 1
                print("\n" + "=" * 88)
                print(f"[{task_index}/{total}] {task} | {mode} | {stage}")
                print("=" * 88, flush=True)
                if stage == "train" and train_complete.exists() and best.exists() and not args.force:
                    append_status(status_path, {"task": task, "mode": mode, "stage": stage, "status": "skip_existing", "seconds": 0.0, "returncode": 0})
                    continue
                if stage == "eval" and eval_csv.exists() and not args.force:
                    append_status(status_path, {"task": task, "mode": mode, "stage": stage, "status": "skip_existing", "seconds": 0.0, "returncode": 0})
                    continue
                command = [
                    sys.executable, "-u", str(entry), "--project", task, "--project-root", str(project),
                    "--dataset", args.dataset, "--config", str(config), "--teacher", str(teacher),
                    "--mode", mode, "--stage", stage, "--output-dir", str(variant_dir),
                    "--device", args.device, "--epochs", str(args.epochs),
                    "--max-train-batches", str(args.max_train_batches), "--max-eval-batches", str(args.max_eval_batches),
                ]
                if stage == "train" and last.exists() and not args.force:
                    command.extend(["--resume", str(last)])
                if stage == "eval":
                    command.extend(["--checkpoint", str(best)])
                    if not best.exists() and not args.dry_run:
                        append_status(status_path, {"task": task, "mode": mode, "stage": stage, "status": "skip_missing_checkpoint", "seconds": 0.0, "returncode": 0})
                        continue
                status, seconds, code = run(command, repo, logs / f"{task.lower()}_{mode}_{stage}.log", args.dry_run)
                append_status(status_path, {"task": task, "mode": mode, "stage": stage, "status": status, "seconds": round(seconds, 2), "returncode": code})
                if code != 0:
                    raise SystemExit(code)
    summarize(output_root)
    print(f"Dual-readout ablation outputs: {output_root}")
    print(f"Run records: {run_dir}")


if __name__ == "__main__":
    main()
