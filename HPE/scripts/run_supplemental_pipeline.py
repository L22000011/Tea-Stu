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


def first_existing(candidates: Sequence[str]) -> Path:
    resolved = [PROJECT_ROOT / candidate for candidate in candidates]
    for path in resolved:
        if path.exists():
            return path
    return resolved[0]


def eval_missing_task(
    name: str,
    title: str,
    dataset: str,
    config_name: str,
    checkpoint_rel: str,
    output_rel: str,
    device: str,
    nonvisual: bool = False,
    max_eval_batches: int | None = None,
) -> Task:
    config = CONFIG_DIR / config_name
    checkpoint = PROJECT_ROOT / checkpoint_rel
    output_csv = PROJECT_ROOT / output_rel
    command = py(
        "eval_missing_modality.py",
        "--dataset", dataset,
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--output-csv", str(output_csv),
    )
    if nonvisual:
        command.append("--nonvisual")
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(
        name=name,
        title=title,
        command=command,
        kind="eval",
        required=[config, checkpoint],
        outputs=[output_csv],
        note="Grouped summary by missing-modality count.",
    )


def export_metrics_task(
    name: str,
    title: str,
    dataset: str,
    config_name: str,
    checkpoint_rel: str,
    modality_set: str,
    output_rel: str,
    device: str,
    max_eval_batches: int | None = None,
) -> Task:
    config = CONFIG_DIR / config_name
    checkpoint = PROJECT_ROOT / checkpoint_rel
    output_csv = PROJECT_ROOT / output_rel
    command = py(
        "export_metrics.py",
        "--dataset", dataset,
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--modality-set", modality_set,
        "--output-csv", str(output_csv),
    )
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(
        name=name,
        title=title,
        command=command,
        kind="export",
        required=[config, checkpoint],
        outputs=[output_csv],
        note="Exports params, throughput, and peak memory.",
    )


def robustness_task(
    dataset: str,
    device: str,
    max_eval_batches: int | None = None,
) -> Task:
    config = CONFIG_DIR / "student_vk_missing.yaml"
    checkpoint = PROJECT_ROOT / "outputs/student_vk_missing/best.pth"
    output_csv = PROJECT_ROOT / "outputs/eval/student_vk_noise_robustness.csv"
    command = py(
        "eval_robustness_vk_noise.py",
        "--dataset", dataset,
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--output-csv", str(output_csv),
    )
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(
        name="random_student_vk_vk_noise",
        title="Random split | Student-VK VK noise and dropped-joint robustness",
        command=command,
        kind="eval",
        required=[config, checkpoint],
        outputs=[output_csv],
        note="Robustness under noisy and partially missing VK joints.",
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
        note="Generates TeX/PDF/summary CSV against the official X-Fi table.",
    )


def build_tasks(args: argparse.Namespace) -> list[Task]:
    dataset = args.dataset
    device = args.device
    max_eval_batches = args.max_eval_batches
    return [
        eval_missing_task(
            "random_teacher_missing_summary",
            "Random split | Teacher grouped by missing-modality count",
            dataset,
            "teacher_full.yaml",
            "outputs/teacher_full/best.pth",
            "outputs/eval/teacher_missing_modality_summary.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        eval_missing_task(
            "random_student_vk_missing_summary",
            "Random split | Student-VK grouped by missing-modality count",
            dataset,
            "student_vk_missing.yaml",
            "outputs/student_vk_missing/best.pth",
            "outputs/eval/student_vk_missing_modality_summary.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        eval_missing_task(
            "random_baseline_missing_summary",
            "Random split | Baseline grouped by missing-modality count",
            dataset,
            "baseline_full.yaml",
            "outputs/baseline_full/best.pth",
            "outputs/eval/baseline_missing_modality_summary.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        eval_missing_task(
            "random_student_nv_missing_summary",
            "Random split | Student-NV grouped by missing non-visual modalities",
            dataset,
            "student_nv_missing.yaml",
            "outputs/student_nv_missing/best.pth",
            "outputs/eval/student_nv_missing_modality_summary.csv",
            device,
            nonvisual=True,
            max_eval_batches=max_eval_batches,
        ),
        robustness_task(dataset, device, max_eval_batches=max_eval_batches),
        export_metrics_task(
            "random_teacher_complexity",
            "Random split | Teacher complexity export",
            dataset,
            "teacher_full.yaml",
            "outputs/teacher_full/best.pth",
            "vk,depth,lidar,mmwave,wifi-csi",
            "outputs/eval/teacher_full_model_complexity.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        export_metrics_task(
            "random_student_vk_complexity",
            "Random split | Student-VK complexity export",
            dataset,
            "student_vk_missing.yaml",
            "outputs/student_vk_missing/best.pth",
            "vk,depth,lidar,mmwave,wifi-csi",
            "outputs/eval/student_vk_model_complexity.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        export_metrics_task(
            "random_student_nv_complexity",
            "Random split | Student-NV complexity export",
            dataset,
            "student_nv_missing.yaml",
            "outputs/student_nv_missing/best.pth",
            "depth,lidar,mmwave,wifi-csi",
            "outputs/eval/student_nv_model_complexity.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        export_metrics_task(
            "random_baseline_complexity",
            "Random split | Baseline complexity export",
            dataset,
            "baseline_full.yaml",
            "outputs/baseline_full/best.pth",
            "vk,depth,lidar,mmwave,wifi-csi",
            "outputs/eval/baseline_full_model_complexity.csv",
            device,
            max_eval_batches=max_eval_batches,
        ),
        report_task(
            "random_teacher_xfi_report",
            "Random split | Teacher vs official X-Fi report",
            ["outputs/eval/teacher_all_combinations.csv"],
        ),
        report_task(
            "random_student_vk_xfi_report",
            "Random split | Student-VK vs official X-Fi report",
            ["outputs/eval/all_combinations.csv"],
        ),
        report_task(
            "random_student_nv_xfi_report",
            "Random split | Student-NV vs official X-Fi report",
            ["outputs/eval/student_nv_nonvisual_combinations.csv"],
        ),
        report_task(
            "random_baseline_xfi_report",
            "Random split | Baseline vs official X-Fi report",
            ["outputs/eval/baseline_full_all_combinations.csv"],
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
    parser = argparse.ArgumentParser("Run the supplemental HPE experiments not covered by random split main runs or the remaining pipeline.")
    parser.add_argument("--dataset", type=str, default="/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
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

    print("HPE supplemental experiment pipeline", flush=True)
    print(f"project={PROJECT_ROOT}", flush=True)
    print(f"dataset={args.dataset}", flush=True)
    print(f"gpu={args.gpu} | device={args.device}", flush=True)
    print(f"max_eval_batches={args.max_eval_batches}", flush=True)
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
