from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "Ori-HAR"
BASE_CONFIG = PROJECT / "config.yaml"
GENERATED_CONFIG_DIR = PROJECT / "generated_configs"
OUTPUT_ROOT = PROJECT / "outputs_cross_baseline"
RUN_ROOT = OUTPUT_ROOT / "runs"


@dataclass
class Task:
    name: str
    command: list[str]
    cwd: Path
    required: list[Path]
    outputs: list[Path]
    log_path: Path


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "name", "status", "reason", "returncode", "seconds", "outputs"])
        writer.writeheader()
        writer.writerows(rows)


def run_task(index: int, task: Task, rows: list[dict[str, object]], status_csv: Path, env: dict[str, str], dry_run: bool, force: bool) -> None:
    print("\n" + "=" * 88)
    print(f"[{index}] {task.name}")
    print("command=", " ".join(task.command))
    print("outputs=", "; ".join(str(item) for item in task.outputs))
    print("=" * 88)

    missing = [str(path) for path in task.required if not path.exists()]
    if missing:
        reason = "missing_required=" + ",".join(missing)
        print("SKIP |", reason)
        rows.append({"index": index, "name": task.name, "status": "skipped", "reason": reason, "returncode": "", "seconds": 0, "outputs": ";".join(str(item) for item in task.outputs)})
        write_status(status_csv, rows)
        return

    if not force and task.outputs and all(path.exists() for path in task.outputs):
        print("SKIP | outputs_exist")
        rows.append({"index": index, "name": task.name, "status": "skipped", "reason": "outputs_exist", "returncode": "", "seconds": 0, "outputs": ";".join(str(item) for item in task.outputs)})
        write_status(status_csv, rows)
        return

    if dry_run:
        print("DRY-RUN | command not executed")
        rows.append({"index": index, "name": task.name, "status": "dry_run", "reason": "", "returncode": "", "seconds": 0, "outputs": ";".join(str(item) for item in task.outputs)})
        write_status(status_csv, rows)
        return

    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    with task.log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(task.command) + "\n\n")
        log.flush()
        proc = subprocess.run(task.command, cwd=str(task.cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    seconds = (datetime.now() - started).total_seconds()
    status = "done" if proc.returncode == 0 else "failed"
    print(f"TASK-END | {status} | seconds={seconds:.1f} | returncode={proc.returncode}")
    rows.append({"index": index, "name": task.name, "status": status, "reason": "", "returncode": proc.returncode, "seconds": f"{seconds:.3f}", "outputs": ";".join(str(item) for item in task.outputs)})
    write_status(status_csv, rows)


def make_config(split: str, epochs: int) -> Path:
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["split_to_use"] = split
    config["training_epoch"] = int(epochs)
    out = GENERATED_CONFIG_DIR / f"xfi_vk_har_{split}.yaml"
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return out


def build_tasks(dataset: str, splits: Iterable[str], epochs: int, max_train_batches: int, run_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for split in splits:
        config = make_config(split, epochs)
        split_dir = OUTPUT_ROOT / split
        ckpt_dir = split_dir / "checkpoints"
        eval_dir = split_dir / "eval"
        best = ckpt_dir / "best.pth"
        last = ckpt_dir / "last.pth"
        train_cmd = [
            sys.executable, "-u", "run.py",
            "--dataset", dataset,
            "--config", str(config),
            "--save-dir", str(ckpt_dir),
            "--epochs", str(epochs),
            "--max-train-batches", str(max_train_batches),
        ]
        if last.exists() and not best.exists():
            train_cmd.append("--resume")
        tasks.append(Task(
            name=f"har_{split}_train",
            command=train_cmd,
            cwd=PROJECT,
            required=[config],
            outputs=[best],
            log_path=run_dir / "logs" / f"har_{split}_train.log",
        ))
        tasks.append(Task(
            name=f"har_{split}_eval15",
            command=[
                sys.executable, "-u", "validate_all.py",
                "--dataset", dataset,
                "--config", str(config),
                "--pt_weights", str(best),
                "--outputs-dir", str(eval_dir),
            ],
            cwd=PROJECT,
            required=[config, best],
            outputs=[eval_dir / "main_eval_latest.csv"],
            log_path=run_dir / "logs" / f"har_{split}_eval15.log",
        ))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser("Run adapted X-Fi/VK HAR cross-scene and cross-subject baselines.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", default=["cross_scene_split", "cross_subject_split"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    status_csv = run_dir / "status.csv"
    manifest = {
        "project": "Ori-HAR adapted X-Fi/VK baseline",
        "dataset": args.dataset,
        "gpu": args.gpu,
        "epochs": args.epochs,
        "max_train_batches": args.max_train_batches,
        "splits": args.splits,
        "run_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tasks = build_tasks(args.dataset, args.splits, args.epochs, args.max_train_batches, run_dir)
    rows: list[dict[str, object]] = []
    for idx, task in enumerate(tasks, start=1):
        run_task(idx, task, rows, status_csv, env, args.dry_run, args.force)
    print(f"HAR adapted X-Fi/VK baseline records: {run_dir}")


if __name__ == "__main__":
    main()
