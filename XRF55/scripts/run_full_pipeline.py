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
    kind: str
    command: List[str]
    outputs: tuple[Path, ...]
    required_inputs: tuple[Path, ...] = ()


def rel(path: str) -> Path:
    return PROJECT_ROOT / path


def checkpoint(path: str) -> Path:
    return rel(path) / "best.pth"


def outputs_exist(outputs: Iterable[Path]) -> bool:
    return all(path.exists() for path in outputs)


def build_steps(args: argparse.Namespace) -> list[Step]:
    python = sys.executable
    common_train = ["--dataset", args.dataset, "--device", args.device]
    if args.max_train_batches is not None:
        common_train += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_eval_batches is not None:
        common_train += ["--max-eval-batches", str(args.max_eval_batches)]

    common_eval = ["--dataset", args.dataset, "--device", args.device]
    if args.max_eval_batches is not None:
        common_eval += ["--max-eval-batches", str(args.max_eval_batches)]

    steps = [
        Step(1, "Teacher full-modality training", "teacher_train", "train",
             [python, "-u", str(rel("scripts/train_teacher_full.py")), "--config", str(rel("configs/teacher_full.yaml")), "--output-dir", str(rel("outputs/teacher_full")), *common_train],
             (checkpoint("outputs/teacher_full"),)),
        Step(2, "Teacher 7-combination evaluation", "teacher_eval7", "eval",
             [python, "-u", str(rel("scripts/eval_all_combinations.py")), "--config", str(rel("configs/teacher_full.yaml")), "--checkpoint", str(rel("outputs/teacher_full/best.pth")), "--output-csv", str(rel("outputs/eval/teacher_all_combinations.csv")), "--output-md", str(rel("outputs/eval/teacher_all_combinations.md")), *common_eval],
             (rel("outputs/eval/teacher_all_combinations.csv"), rel("outputs/eval/teacher_all_combinations.md")),
             (rel("outputs/teacher_full/best.pth"),)),
        Step(3, "Student arbitrary-missing training", "student_train", "train",
             [python, "-u", str(rel("scripts/train_student_missing.py")), "--config", str(rel("configs/student_missing.yaml")), "--output-dir", str(rel("outputs/student_missing")), "--teacher", str(rel("outputs/teacher_full/best.pth")), *common_train],
             (checkpoint("outputs/student_missing"),),
             (rel("outputs/teacher_full/best.pth"),)),
        Step(4, "Student 7-combination evaluation", "student_eval7", "eval",
             [python, "-u", str(rel("scripts/eval_all_combinations.py")), "--config", str(rel("configs/student_missing.yaml")), "--checkpoint", str(rel("outputs/student_missing/best.pth")), "--output-csv", str(rel("outputs/eval/student_all_combinations.csv")), "--output-md", str(rel("outputs/eval/student_all_combinations.md")), *common_eval],
             (rel("outputs/eval/student_all_combinations.csv"), rel("outputs/eval/student_all_combinations.md")),
             (rel("outputs/student_missing/best.pth"),)),
        Step(5, "Baseline full-modality training", "baseline_train", "train",
             [python, "-u", str(rel("scripts/train_baseline_full.py")), "--config", str(rel("configs/baseline_full.yaml")), "--output-dir", str(rel("outputs/baseline_full")), *common_train],
             (checkpoint("outputs/baseline_full"),)),
        Step(6, "Baseline 7-combination evaluation", "baseline_eval7", "eval",
             [python, "-u", str(rel("scripts/eval_all_combinations.py")), "--config", str(rel("configs/baseline_full.yaml")), "--checkpoint", str(rel("outputs/baseline_full/best.pth")), "--output-csv", str(rel("outputs/eval/baseline_all_combinations.csv")), "--output-md", str(rel("outputs/eval/baseline_all_combinations.md")), *common_eval],
             (rel("outputs/eval/baseline_all_combinations.csv"), rel("outputs/eval/baseline_all_combinations.md")),
             (rel("outputs/baseline_full/best.pth"),)),
        Step(7, "Ablation no-distillation training", "ablation_no_distill_train", "train",
             [python, "-u", str(rel("scripts/ablate_no_distillation.py")), "--config", str(rel("configs/ablation_no_distill.yaml")), "--output-dir", str(rel("outputs/ablation_no_distill")), *common_train],
             (checkpoint("outputs/ablation_no_distill"),)),
        Step(8, "Ablation no-distillation evaluation", "ablation_no_distill_eval7", "eval",
             [python, "-u", str(rel("scripts/eval_all_combinations.py")), "--config", str(rel("configs/ablation_no_distill.yaml")), "--checkpoint", str(rel("outputs/ablation_no_distill/best.pth")), "--output-csv", str(rel("outputs/eval/ablation_no_distill_all_combinations.csv")), "--output-md", str(rel("outputs/eval/ablation_no_distill_all_combinations.md")), *common_eval],
             (rel("outputs/eval/ablation_no_distill_all_combinations.csv"), rel("outputs/eval/ablation_no_distill_all_combinations.md")),
             (rel("outputs/ablation_no_distill/best.pth"),)),
        Step(9, "Ablation uniform-fusion training", "ablation_uniform_train", "train",
             [python, "-u", str(rel("scripts/ablate_uniform_fusion.py")), "--config", str(rel("configs/ablation_uniform_fusion.yaml")), "--output-dir", str(rel("outputs/ablation_uniform_fusion")), "--teacher", str(rel("outputs/teacher_full/best.pth")), *common_train],
             (checkpoint("outputs/ablation_uniform_fusion"),),
             (rel("outputs/teacher_full/best.pth"),)),
        Step(10, "Ablation uniform-fusion evaluation", "ablation_uniform_eval7", "eval",
             [python, "-u", str(rel("scripts/eval_all_combinations.py")), "--config", str(rel("configs/ablation_uniform_fusion.yaml")), "--checkpoint", str(rel("outputs/ablation_uniform_fusion/best.pth")), "--output-csv", str(rel("outputs/eval/ablation_uniform_all_combinations.csv")), "--output-md", str(rel("outputs/eval/ablation_uniform_all_combinations.md")), *common_eval],
             (rel("outputs/eval/ablation_uniform_all_combinations.csv"), rel("outputs/eval/ablation_uniform_all_combinations.md")),
             (rel("outputs/ablation_uniform_fusion/best.pth"),)),
        Step(11, "Ablation no-semantic training", "ablation_no_semantic_train", "train",
             [python, "-u", str(rel("scripts/ablate_no_semantic.py")), "--config", str(rel("configs/ablation_no_semantic.yaml")), "--output-dir", str(rel("outputs/ablation_no_semantic")), "--teacher", str(rel("outputs/teacher_full/best.pth")), *common_train],
             (checkpoint("outputs/ablation_no_semantic"),),
             (rel("outputs/teacher_full/best.pth"),)),
        Step(12, "Ablation no-semantic evaluation", "ablation_no_semantic_eval7", "eval",
             [python, "-u", str(rel("scripts/eval_all_combinations.py")), "--config", str(rel("configs/ablation_no_semantic.yaml")), "--checkpoint", str(rel("outputs/ablation_no_semantic/best.pth")), "--output-csv", str(rel("outputs/eval/ablation_no_semantic_all_combinations.csv")), "--output-md", str(rel("outputs/eval/ablation_no_semantic_all_combinations.md")), *common_eval],
             (rel("outputs/eval/ablation_no_semantic_all_combinations.csv"), rel("outputs/eval/ablation_no_semantic_all_combinations.md")),
             (rel("outputs/ablation_no_semantic/best.pth"),)),
    ]
    return steps


def append_status(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "name", "title", "kind", "status", "reason", "returncode", "seconds", "outputs"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_step(step: Step, args: argparse.Namespace, log_dir: Path, status_csv: Path) -> bool:
    print("=" * 88)
    print(f"[{step.index}/12] {step.title}")
    print(f"name={step.name} | kind={step.kind}")
    if step.kind == "train":
        print(f"max_train_batches={args.max_train_batches}; evaluation remains full unless max_eval_batches is set.")
    else:
        print("Full validation evaluation is used unless max_eval_batches is set.")
    print("command=", " ".join(str(part) for part in step.command))
    print("outputs=", ";".join(str(path) for path in step.outputs))
    print("=" * 88)

    if args.dry_run:
        append_status(status_csv, {
            "index": step.index,
            "name": step.name,
            "title": step.title,
            "kind": step.kind,
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
            "kind": step.kind,
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
            "kind": step.kind,
            "status": "skipped",
            "reason": "outputs_exist",
            "returncode": 0,
            "seconds": 0.0,
            "outputs": ";".join(str(path) for path in step.outputs),
        })
        print("SKIP | outputs_exist")
        return True

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.name}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
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
        "kind": step.kind,
        "status": status,
        "reason": "",
        "returncode": process.returncode,
        "seconds": round(seconds, 3),
        "outputs": ";".join(str(path) for path in step.outputs),
    })
    print(f"{status.upper()} | returncode={process.returncode} | log={log_path}")
    return process.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run the complete XRF55 experiment pipeline.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = rel(f"outputs/pipeline_runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir = run_dir / "logs"
    status_csv = run_dir / "status.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = build_steps(args)
    print(f"[Pipeline] dataset={args.dataset}")
    print(f"[Pipeline] gpu={args.gpu} | device={args.device}")
    print(f"[Pipeline] steps={len(steps)} | max_train_batches={args.max_train_batches}")
    if args.max_eval_batches is None:
        print("[Pipeline] evaluation uses the full validation set.")
    else:
        print(f"[Pipeline] max_eval_batches={args.max_eval_batches}")

    failed = []
    for step in steps:
        ok = run_step(step, args, log_dir, status_csv)
        if not ok:
            failed.append(step.name)
            if not args.continue_on_error:
                break
    if failed:
        print("[Pipeline] Failed steps:")
        for name in failed:
            print(f"  - {name}")
        raise SystemExit(1)
    print(f"[Pipeline] Finished. Records: {run_dir}")


if __name__ == "__main__":
    main()
