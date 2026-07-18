#!/usr/bin/env python3
"""Run the matched Adapted X-Fi-VK baseline on the formal protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROTOCOLS = {
    "random": "random_split",
    "cross_subject": "cross_subject_split",
    "cross_scene": "cross_scene_split",
}


def resolve_project(repo: Path, task: str) -> Path:
    for candidate in (repo / task, repo / f"MMFi_{task}"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot locate {task} project under {repo}")


def status_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(args: argparse.Namespace, repo: Path, task: str, protocol_name: str) -> None:
    project_root = resolve_project(repo, task)
    config_path = project_root / "configs" / "student_vk_missing.yaml"
    output_dir = (args.output_root / task / protocol_name).resolve()
    command = [
        sys.executable, "-u", str(repo / "Matched-XFi-VK" / "run_variant.py"),
        "--project", task, "--project-root", str(project_root), "--dataset", args.dataset,
        "--config", str(config_path), "--protocol", PROTOCOLS[protocol_name],
        "--output-dir", str(output_dir), "--stage", args.stage, "--device", args.device,
        "--epochs", str(args.epochs), "--max-train-batches", str(args.max_train_batches),
        "--max-eval-batches", str(args.max_eval_batches),
    ]
    if args.force:
        command.append("--force")
    elif (output_dir / "last.pth").exists() and not (output_dir / "final_summary.json").exists():
        command.extend(["--resume", str((output_dir / "last.pth").resolve())])
    print("command=", subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    completed = subprocess.run(command, cwd=repo)
    if completed.returncode != 0:
        raise RuntimeError(f"Adapted X-Fi-VK failed with return code {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=["HPE", "HAR"], default=["HPE", "HAR"])
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS), default=["random"])
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    args.output_root = (args.output_root or repo / "outputs" / "adapted_xfi_vk_baseline").resolve()
    run_dir = repo / "outputs" / "adapted_xfi_vk_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    status_path = run_dir / "status.csv"
    total = len(args.tasks) * len(args.protocols)
    index = 0
    for task in args.tasks:
        for protocol_name in args.protocols:
            index += 1
            print(f"[{index}/{total}] Adapted-XFi-VK | {task} | {protocol_name}", flush=True)
            started = time.time()
            try:
                run_one(args, repo, task, protocol_name)
                status_row(status_path, {"task": task, "protocol": protocol_name, "status": "done", "seconds": round(time.time() - started, 2)})
            except Exception as exc:
                status_row(status_path, {"task": task, "protocol": protocol_name, "status": "failed", "seconds": round(time.time() - started, 2), "error": repr(exc)})
                raise
    print(f"Adapted X-Fi-VK outputs: {args.output_root}")


if __name__ == "__main__":
    main()
