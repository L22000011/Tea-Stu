from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = OUTPUT_DIR / "eval"


@dataclass
class Task:
    name: str
    title: str
    command: list[str]
    kind: str
    required: list[Path | None] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    note: str = ""


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *args]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def yaml_value(path: Path | None, key: str) -> str | None:
    if path is None or not path.exists():
        return None
    prefix = f"{key}:"
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def output_dir_from_config(config: Path | None, fallback: str) -> Path:
    value = yaml_value(config, "output_dir")
    return PROJECT_ROOT / (value or fallback)


def add_resume(command: list[str], output_dir: Path, resume: bool = True) -> list[str]:
    last = output_dir / "last.pth"
    if resume and last.exists() and "--resume" not in command:
        return [*command, "--resume", str(last)]
    return command


def first_existing(candidates: Sequence[str]) -> Path:
    resolved = [PROJECT_ROOT / candidate for candidate in candidates]
    for path in resolved:
        if path.exists():
            return path
    return resolved[0]


def train_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config_name: str,
    device: str,
    max_train_batches: int,
    output_dir: Path,
    teacher_rel: str | None = None,
    max_eval_batches: int | None = None,
) -> Task:
    config = CONFIG_DIR / config_name
    command = py(
        script,
        "--dataset", dataset,
        "--config", str(config),
        "--device", device,
        "--max-train-batches", str(max_train_batches),
    )
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    required: list[Path | None] = [config]
    if teacher_rel is not None:
        teacher = PROJECT_ROOT / teacher_rel
        command.extend(["--teacher", str(teacher)])
        required.append(teacher)
    command = add_resume(command, output_dir, resume=True)
    return Task(
        name=name,
        title=title,
        command=command,
        kind="train",
        required=required,
        outputs=[output_dir / "best.pth"],
        note=f"max_train_batches={max_train_batches}; evaluation remains full unless max_eval_batches is provided.",
    )


def eval_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config_name: str,
    checkpoint_rel: str,
    output_csv_rel: str,
    output_md_rel: str,
    device: str,
    max_eval_batches: int | None = None,
) -> Task:
    config = CONFIG_DIR / config_name
    checkpoint = PROJECT_ROOT / checkpoint_rel
    output_csv = PROJECT_ROOT / output_csv_rel
    output_md = PROJECT_ROOT / output_md_rel
    command = py(
        script,
        "--dataset", dataset,
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--output-csv", str(output_csv),
        "--output-md", str(output_md),
    )
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(
        name=name,
        title=title,
        command=command,
        kind="eval",
        required=[config, checkpoint],
        outputs=[output_csv, output_md],
        note="Evaluates the complete modality-combination set for the target protocol.",
    )


def report_task(name: str, title: str, csv_candidates: Sequence[str]) -> Task:
    csv_path = first_existing(csv_candidates)
    summary_csv = csv_path.with_suffix(".xfi_compare_summary.csv")
    command = py("generate_xfi_report.py", "--csv", str(csv_path))
    return Task(
        name=name,
        title=title,
        command=command,
        kind="report",
        required=[csv_path],
        outputs=[summary_csv],
        note="Generates TeX/PDF/summary CSV against the official HAR X-Fi table.",
    )


def build_tasks(args: argparse.Namespace) -> list[Task]:
    cross_scene_config = CONFIG_DIR / "baseline_cross_scene.yaml"
    cross_subject_config = CONFIG_DIR / "baseline_cross_subject.yaml"
    cross_scene_output = output_dir_from_config(cross_scene_config, "outputs/baseline_cross_scene")
    cross_subject_output = output_dir_from_config(cross_subject_config, "outputs/baseline_cross_subject")
    dataset = args.dataset
    device = args.device
    max_train_batches = args.max_train_batches
    max_eval_batches = args.max_eval_batches
    return [
        train_task(
            "cross_scene_baseline_train",
            "Cross-scene | Full-modality baseline training",
            "train_baseline_full.py",
            dataset,
            "baseline_cross_scene.yaml",
            device,
            max_train_batches,
            cross_scene_output,
            max_eval_batches=max_eval_batches,
        ),
        eval_task(
            "cross_scene_baseline_eval15",
            "Cross-scene | Baseline 15-combination evaluation",
            "eval_all_combinations.py",
            dataset,
            "baseline_cross_scene.yaml",
            "outputs/baseline_cross_scene/best.pth",
            "outputs/eval/baseline_cross_scene_all_combinations.csv",
            "outputs/eval/baseline_cross_scene_all_combinations.md",
            device,
            max_eval_batches=max_eval_batches,
        ),
        train_task(
            "cross_subject_baseline_train",
            "Cross-subject | Full-modality baseline training",
            "train_baseline_full.py",
            dataset,
            "baseline_cross_subject.yaml",
            device,
            max_train_batches,
            cross_subject_output,
            max_eval_batches=max_eval_batches,
        ),
        eval_task(
            "cross_subject_baseline_eval15",
            "Cross-subject | Baseline 15-combination evaluation",
            "eval_all_combinations.py",
            dataset,
            "baseline_cross_subject.yaml",
            "outputs/baseline_cross_subject/best.pth",
            "outputs/eval/baseline_cross_subject_all_combinations.csv",
            "outputs/eval/baseline_cross_subject_all_combinations.md",
            device,
            max_eval_batches=max_eval_batches,
        ),
        report_task(
            "random_teacher_xfi_report",
            "Random split | Teacher vs official X-Fi HAR report",
            [
                "outputs/eval/teacher_random_all_combinations.csv",
                "outputs/eval/all_combinations.csv",
            ],
        ),
        report_task(
            "random_student_vk_xfi_report",
            "Random split | Student-VK vs official X-Fi HAR report",
            [
                "outputs/eval/student_vk_random_all_combinations.csv",
                "outputs/eval/student_vk_all_combinations.csv",
            ],
        ),
        report_task(
            "random_student_nv_xfi_report",
            "Random split | Student-NV vs official X-Fi HAR report",
            [
                "outputs/eval/student_nv_random_nonvisual_combinations.csv",
                "outputs/eval/student_nv_nonvisual_combinations.csv",
            ],
        ),
        report_task(
            "random_baseline_xfi_report",
            "Random split | Baseline vs official X-Fi HAR report",
            ["outputs/eval/baseline_full_all_combinations.csv"],
        ),
        report_task(
            "ablation_no_distill_xfi_report",
            "Random split | No-distillation ablation report",
            ["outputs/eval/ablation_no_distill_all_combinations.csv"],
        ),
        report_task(
            "ablation_uniform_fusion_xfi_report",
            "Random split | Uniform-fusion ablation report",
            ["outputs/eval/ablation_uniform_fusion_all_combinations.csv"],
        ),
    ]


def missing_reason(task: Task) -> str | None:
    missing = [path for path in task.required if path is None or not path.exists()]
    if missing:
        return "missing_required=" + ",".join("None" if path is None else str(path) for path in missing)
    return None


def should_skip(task: Task, force: bool) -> tuple[bool, str]:
    reason = missing_reason(task)
    if reason:
        return True, reason
    if not force and task.outputs and all(path.exists() for path in task.outputs):
        return True, "outputs_exist"
    return False, ""


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_task_header(index: int, total: int, task: Task) -> None:
    print("\n" + "=" * 88, flush=True)
    print(f"[{index}/{total}] {task.title}", flush=True)
    print(f"name={task.name} | kind={task.kind}", flush=True)
    if task.note:
        print(task.note, flush=True)
    print("command=", " ".join(task.command), flush=True)
    print("outputs=", "; ".join(str(path) for path in task.outputs), flush=True)
    print("=" * 88, flush=True)


def run_task(task: Task, env: dict[str, str], log_dir: Path) -> tuple[str, float, int]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.name}.log"
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 88 + "\n")
        log.write(f"Task: {task.title}\n")
        log.write("Command: " + " ".join(task.command) + "\n")
        log.flush()
        process = subprocess.Popen(
            task.command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    elapsed = time.time() - start
    return ("done" if code == 0 else "failed", elapsed, code)


def write_manifest(run_dir: Path, args: argparse.Namespace, tasks: list[Task]) -> None:
    manifest = {
        "dataset": args.dataset,
        "gpu": args.gpu,
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "task_count": len(tasks),
        "tasks": [
            {
                "name": task.name,
                "title": task.title,
                "kind": task.kind,
                "command": task.command,
                "required": [None if path is None else str(path) for path in task.required],
                "outputs": [str(path) for path in task.outputs],
            }
            for task in tasks
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Run the supplemental HAR experiments not covered by random split main runs or the current remaining pipeline.")
    parser.add_argument("--dataset", type=str, default="/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "supplemental_runs" / run_id
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    write_manifest(run_dir, args, tasks)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("HAR supplemental experiment pipeline", flush=True)
    print(f"project={PROJECT_ROOT}", flush=True)
    print(f"dataset={args.dataset}", flush=True)
    print(f"gpu={args.gpu} | device={args.device}", flush=True)
    print(f"max_train_batches={args.max_train_batches} | max_eval_batches={args.max_eval_batches}", flush=True)
    print(f"records={run_dir}", flush=True)

    rows: list[dict[str, str]] = []
    for index, task in enumerate(tasks, start=1):
        print_task_header(index, len(tasks), task)
        skip, reason = should_skip(task, args.force)
        if skip:
            print(f"SKIP | {reason}", flush=True)
            status, elapsed, code = "skipped", 0.0, 0
        elif args.dry_run:
            print("DRY-RUN | command not executed", flush=True)
            status, elapsed, code = "dry_run", 0.0, 0
        else:
            status, elapsed, code = run_task(task, env, log_dir)
            print(f"TASK-END | {status} | seconds={elapsed:.1f} | returncode={code}", flush=True)
        rows.append({
            "index": str(index),
            "name": task.name,
            "title": task.title,
            "kind": task.kind,
            "status": status,
            "reason": reason if skip else "",
            "returncode": str(code),
            "seconds": f"{elapsed:.3f}",
            "outputs": ";".join(str(path) for path in task.outputs),
        })
        write_status(run_dir / "status.csv", rows)
        if status == "failed" and args.stop_on_fail:
            break
    print(f"Supplemental records: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
