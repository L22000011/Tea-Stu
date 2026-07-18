#!/usr/bin/env python3
"""Orchestrate the minimal four-profile SP-SFC ablation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROFILES = ("task_only", "output", "output_token", "full")


def project_path(repo: Path, task: str) -> Path:
    for candidate in (repo / task, repo / f"MMFi_{task}"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot locate {task} project")


def append_status(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=["HPE", "HAR"], default=["HPE", "HAR"])
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_root = (args.output_root or repo / "outputs" / "sp_sfc_ablation").resolve()
    run_dir = repo / "outputs" / "sp_sfc_ablation_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    status_path = run_dir / "status.csv"
    entry = Path(__file__).resolve().parent / "run_variant.py"
    total = len(args.tasks) * len(args.profiles)
    index = 0
    for task in args.tasks:
        project = project_path(repo, task)
        config = project / "configs" / "student_vk_missing.yaml"
        teacher = project / "outputs" / "teacher_full" / "best.pth"
        if not teacher.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher}")
        for profile in args.profiles:
            index += 1
            out_dir = output_root / task / profile
            print(f"[{index}/{total}] SP-SFC | {task} | {profile}", flush=True)
            command = [
                sys.executable, "-u", str(entry), "--project", task,
                "--project-root", str(project), "--dataset", args.dataset,
                "--config", str(config), "--teacher", str(teacher),
                "--profile", profile, "--output-dir", str(out_dir),
                "--device", args.device, "--epochs", str(args.epochs),
                "--max-train-batches", str(args.max_train_batches),
                "--max-eval-batches", str(args.max_eval_batches),
            ]
            if args.force:
                command.append("--force")
            elif (out_dir / "last.pth").exists() and not (out_dir / "final_summary.json").exists():
                command.extend(["--resume", str((out_dir / "last.pth").resolve())])
            if not args.force and (out_dir / "all_combinations.csv").exists():
                print(f"[skip] completed result exists: {out_dir}", flush=True)
                append_status(status_path, {"task": task, "profile": profile, "status": "skipped", "seconds": 0.0, "returncode": 0})
                continue
            print("command=", subprocess.list2cmdline(command), flush=True)
            started = time.time()
            if args.dry_run:
                code = 0
                status = "dry-run"
            else:
                code = subprocess.run(command, cwd=repo).returncode
                status = "done" if code == 0 else "failed"
            append_status(status_path, {"task": task, "profile": profile, "status": status, "seconds": round(time.time() - started, 2), "returncode": code})
            if code != 0:
                raise SystemExit(code)
    print(f"SP-SFC ablation outputs: {output_root}")


if __name__ == "__main__":
    main()
