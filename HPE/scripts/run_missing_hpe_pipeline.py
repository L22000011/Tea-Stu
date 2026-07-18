from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
    output_dir: Path | None = None
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


def output_dir_from_config(config: Path, fallback: str) -> Path:
    value = yaml_value(config, "output_dir")
    return PROJECT_ROOT / (value or fallback)


def shell_join(command: Sequence[str]) -> str:
    return " ".join(str(part) if re.fullmatch(r"[A-Za-z0-9_./:=+,-]+", str(part)) else "'" + str(part).replace("'", "'\\''") + "'" for part in command)


def apply_runtime_overrides(config: Path, run_dir: Path, args: argparse.Namespace) -> Path:
    if args.batch_size is None and args.num_workers is None:
        return config
    text = read_text(config)
    if args.batch_size is not None:
        text = re.sub(r"(?m)^(\s*batch_size:\s*)\d+", rf"\g<1>{args.batch_size}", text, count=1)
    if args.num_workers is not None:
        text = re.sub(r"(?m)^(\s*num_workers:\s*)\d+", rf"\g<1>{args.num_workers}", text, count=1)
    config_dir = run_dir / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / config.name
    out.write_text(text, encoding="utf-8")
    return out


def add_resume(command: list[str], output_dir: Path | None) -> list[str]:
    if output_dir is None:
        return command
    last = output_dir / "last.pth"
    if last.exists() and "--resume" not in command:
        return [*command, "--resume", str(last)]
    return command


def train_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config: Path,
    output_dir: Path,
    device: str,
    max_train_batches: int,
    teacher_rel: str | None = None,
    max_eval_batches: int | None = None,
    extra_args: Sequence[str] | None = None,
    outputs: Sequence[Path] | None = None,
) -> Task:
    command = py(script, "--dataset", dataset, "--config", str(config), "--device", device, "--max-train-batches", str(max_train_batches))
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    if extra_args:
        command.extend(extra_args)
    required: list[Path | None] = [config]
    if teacher_rel is not None:
        teacher = PROJECT_ROOT / teacher_rel
        command.extend(["--teacher", str(teacher)])
        required.append(teacher)
    command = add_resume(command, output_dir)
    return Task(
        name=name,
        title=title,
        command=command,
        kind="train",
        required=required,
        outputs=list(outputs) if outputs is not None else [output_dir / "best.pth"],
        output_dir=output_dir,
        note=f"Train cap per epoch: {max_train_batches}; rerun this pipeline to resume from last.pth when available.",
    )


def eval_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config: Path,
    checkpoint_rel: str,
    output_rel: str,
    device: str,
    max_eval_batches: int | None = None,
) -> Task:
    checkpoint = PROJECT_ROOT / checkpoint_rel
    output_csv = PROJECT_ROOT / output_rel
    command = py(script, "--dataset", dataset, "--config", str(config), "--checkpoint", str(checkpoint), "--device", device, "--output-csv", str(output_csv))
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(
        name=name,
        title=title,
        command=command,
        kind="eval",
        required=[config, checkpoint],
        outputs=[output_csv],
        note="Full validation evaluation unless max_eval_batches is provided.",
    )


def build_tasks(args: argparse.Namespace, run_dir: Path) -> list[Task]:
    dataset = args.dataset
    device = args.device
    max_train_batches = args.max_train_batches
    max_eval_batches = args.max_eval_batches

    cfg_ablation = apply_runtime_overrides(CONFIG_DIR / "ablation.yaml", run_dir, args)
    cfg_teacher_scene = apply_runtime_overrides(CONFIG_DIR / "teacher_cross_scene.yaml", run_dir, args)
    cfg_student_vk_scene = apply_runtime_overrides(CONFIG_DIR / "student_vk_cross_scene.yaml", run_dir, args)
    cfg_student_nv_scene = apply_runtime_overrides(CONFIG_DIR / "student_nv_cross_scene.yaml", run_dir, args)
    cfg_baseline_scene = apply_runtime_overrides(CONFIG_DIR / "baseline_cross_scene.yaml", run_dir, args)
    cfg_teacher_subject = apply_runtime_overrides(CONFIG_DIR / "teacher_cross_subject.yaml", run_dir, args)
    cfg_student_vk_subject = apply_runtime_overrides(CONFIG_DIR / "student_vk_cross_subject.yaml", run_dir, args)
    cfg_student_nv_subject = apply_runtime_overrides(CONFIG_DIR / "student_nv_cross_subject.yaml", run_dir, args)
    cfg_baseline_subject = apply_runtime_overrides(CONFIG_DIR / "baseline_cross_subject.yaml", run_dir, args)

    ablation_dir = output_dir_from_config(CONFIG_DIR / "ablation.yaml", "outputs/ablation")
    scene_teacher_dir = output_dir_from_config(CONFIG_DIR / "teacher_cross_scene.yaml", "outputs/teacher_full_cross_scene_split")
    scene_student_vk_dir = output_dir_from_config(CONFIG_DIR / "student_vk_cross_scene.yaml", "outputs/student_vk_missing_cross_scene_split")
    scene_student_nv_dir = output_dir_from_config(CONFIG_DIR / "student_nv_cross_scene.yaml", "outputs/student_nv_missing_cross_scene")
    scene_baseline_dir = output_dir_from_config(CONFIG_DIR / "baseline_cross_scene.yaml", "outputs/baseline_full_cross_scene_split")
    subject_teacher_dir = output_dir_from_config(CONFIG_DIR / "teacher_cross_subject.yaml", "outputs/teacher_full_cross_subject_split")
    subject_student_vk_dir = output_dir_from_config(CONFIG_DIR / "student_vk_cross_subject.yaml", "outputs/student_vk_missing_cross_subject_split")
    subject_student_nv_dir = output_dir_from_config(CONFIG_DIR / "student_nv_cross_subject.yaml", "outputs/student_nv_missing_cross_subject")
    subject_baseline_dir = output_dir_from_config(CONFIG_DIR / "baseline_cross_subject.yaml", "outputs/baseline_full_cross_subject_split")

    scene_teacher_rel = str(scene_teacher_dir.relative_to(PROJECT_ROOT) / "best.pth")
    subject_teacher_rel = str(subject_teacher_dir.relative_to(PROJECT_ROOT) / "best.pth")

    tasks: list[Task] = [
        train_task("ablation_uniform_train", "Ablation | Uniform-fusion reliability checkpoint", "ablate_reliability.py", dataset, cfg_ablation, ablation_dir / "uniform", device, max_train_batches, teacher_rel="outputs/teacher_full/best.pth", max_eval_batches=max_eval_batches, extra_args=["--variants", "uniform"], outputs=[ablation_dir / "uniform" / "best.pth"]),
        eval_task("ablation_uniform_eval31", "Ablation | Uniform-fusion 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_ablation, "outputs/ablation/uniform/best.pth", "outputs/eval/ablation_uniform_fusion_all_combinations.csv", device, max_eval_batches),
        train_task("ablation_encoder_train", "Ablation | VK encoder variants", "ablate_encoder.py", dataset, cfg_ablation, ablation_dir, device, max_train_batches, max_eval_batches=max_eval_batches, extra_args=["--variants", "mlp_vk,skeleton_prompt"], outputs=[ablation_dir / "mlp_vk" / "best.pth", ablation_dir / "skeleton_prompt" / "best.pth"]),
        eval_task("ablation_encoder_mlp_vk_eval31", "Ablation | MLP VK encoder 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_ablation, "outputs/ablation/mlp_vk/best.pth", "outputs/eval/ablation_encoder_mlp_vk_all_combinations.csv", device, max_eval_batches),
        eval_task("ablation_encoder_skeleton_prompt_eval31", "Ablation | Skeleton-prompt VK encoder 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_ablation, "outputs/ablation/skeleton_prompt/best.pth", "outputs/eval/ablation_encoder_skeleton_prompt_all_combinations.csv", device, max_eval_batches),

        train_task("cross_scene_teacher_train", "Cross-scene | Teacher checkpoint archive", "train_teacher_full.py", dataset, cfg_teacher_scene, scene_teacher_dir, device, max_train_batches, max_eval_batches=max_eval_batches),
        eval_task("cross_scene_teacher_eval31", "Cross-scene | Teacher 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_teacher_scene, scene_teacher_rel, "outputs/eval/teacher_cross_scene_all_combinations.csv", device, max_eval_batches),
        train_task("cross_scene_student_vk_train", "Cross-scene | Student-VK checkpoint archive", "train_student_vk_missing.py", dataset, cfg_student_vk_scene, scene_student_vk_dir, device, max_train_batches, teacher_rel=scene_teacher_rel, max_eval_batches=max_eval_batches),
        eval_task("cross_scene_student_vk_eval31", "Cross-scene | Student-VK 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_student_vk_scene, str(scene_student_vk_dir.relative_to(PROJECT_ROOT) / "best.pth"), "outputs/eval/student_vk_cross_scene_all_combinations.csv", device, max_eval_batches),
        train_task("cross_scene_student_nv_train", "Cross-scene | Student-NV checkpoint archive", "train_student_nv_missing.py", dataset, cfg_student_nv_scene, scene_student_nv_dir, device, max_train_batches, teacher_rel=scene_teacher_rel, max_eval_batches=max_eval_batches),
        eval_task("cross_scene_student_nv_eval15", "Cross-scene | Student-NV non-visual combination evaluation", "eval_nonvisual_combinations.py", dataset, cfg_student_nv_scene, str(scene_student_nv_dir.relative_to(PROJECT_ROOT) / "best.pth"), "outputs/eval/student_nv_cross_scene_nonvisual_combinations.csv", device, max_eval_batches),
        train_task("cross_scene_baseline_train", "Cross-scene | Baseline checkpoint", "train_baseline_full.py", dataset, cfg_baseline_scene, scene_baseline_dir, device, max_train_batches, max_eval_batches=max_eval_batches),
        eval_task("cross_scene_baseline_eval31", "Cross-scene | Baseline 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_baseline_scene, str(scene_baseline_dir.relative_to(PROJECT_ROOT) / "best.pth"), "outputs/eval/baseline_cross_scene_all_combinations.csv", device, max_eval_batches),

        train_task("cross_subject_teacher_train", "Cross-subject | Teacher checkpoint", "train_teacher_full.py", dataset, cfg_teacher_subject, subject_teacher_dir, device, max_train_batches, max_eval_batches=max_eval_batches),
        eval_task("cross_subject_teacher_eval31", "Cross-subject | Teacher 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_teacher_subject, subject_teacher_rel, "outputs/eval/teacher_cross_subject_all_combinations.csv", device, max_eval_batches),
        train_task("cross_subject_student_vk_train", "Cross-subject | Student-VK checkpoint", "train_student_vk_missing.py", dataset, cfg_student_vk_subject, subject_student_vk_dir, device, max_train_batches, teacher_rel=subject_teacher_rel, max_eval_batches=max_eval_batches),
        eval_task("cross_subject_student_vk_eval31", "Cross-subject | Student-VK 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_student_vk_subject, str(subject_student_vk_dir.relative_to(PROJECT_ROOT) / "best.pth"), "outputs/eval/student_vk_cross_subject_all_combinations.csv", device, max_eval_batches),
        train_task("cross_subject_student_nv_train", "Cross-subject | Student-NV checkpoint", "train_student_nv_missing.py", dataset, cfg_student_nv_subject, subject_student_nv_dir, device, max_train_batches, teacher_rel=subject_teacher_rel, max_eval_batches=max_eval_batches),
        eval_task("cross_subject_student_nv_eval15", "Cross-subject | Student-NV non-visual combination evaluation", "eval_nonvisual_combinations.py", dataset, cfg_student_nv_subject, str(subject_student_nv_dir.relative_to(PROJECT_ROOT) / "best.pth"), "outputs/eval/student_nv_cross_subject_nonvisual_combinations.csv", device, max_eval_batches),
        train_task("cross_subject_baseline_train", "Cross-subject | Baseline checkpoint", "train_baseline_full.py", dataset, cfg_baseline_subject, subject_baseline_dir, device, max_train_batches, max_eval_batches=max_eval_batches),
        eval_task("cross_subject_baseline_eval31", "Cross-subject | Baseline 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_baseline_subject, str(subject_baseline_dir.relative_to(PROJECT_ROOT) / "best.pth"), "outputs/eval/baseline_cross_subject_all_combinations.csv", device, max_eval_batches),
    ]
    return tasks


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


def recovery_command(args: argparse.Namespace) -> str:
    command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "run_missing_hpe_pipeline.py"), "--dataset", args.dataset, "--gpu", args.gpu, "--device", args.device, "--max-train-batches", str(args.max_train_batches)]
    if args.max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    return f"CUDA_VISIBLE_DEVICES={args.gpu} " + shell_join(command)


def individual_recovery_command(task: Task, args: argparse.Namespace) -> str:
    command = add_resume(list(task.command), task.output_dir)
    return f"CUDA_VISIBLE_DEVICES={args.gpu} " + shell_join(command)


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
    print("command=", shell_join(task.command), flush=True)
    print("outputs=", "; ".join(str(path) for path in task.outputs), flush=True)
    print("=" * 88, flush=True)


def run_task(task: Task, env: dict[str, str], log_dir: Path) -> tuple[str, float, int, bool]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.name}.log"
    start = time.time()
    saw_oom = False
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 88 + "\n")
        log.write(f"Task: {task.title}\n")
        log.write("Command: " + shell_join(task.command) + "\n")
        log.flush()
        process = subprocess.Popen(task.command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            if "out of memory" in line.lower() or "cuda outofmemory" in line.lower():
                saw_oom = True
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    elapsed = time.time() - start
    return ("done" if code == 0 else "failed", elapsed, code, saw_oom)


def write_manifest(run_dir: Path, args: argparse.Namespace, tasks: list[Task]) -> None:
    manifest = {
        "purpose": "Run only missing HPE experiments and checkpoint archives.",
        "recovery_command": recovery_command(args),
        "dataset": args.dataset,
        "gpu": args.gpu,
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "batch_size_override": args.batch_size,
        "num_workers_override": args.num_workers,
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
    (run_dir / "recovery_commands.sh").write_text("#!/usr/bin/env bash\nset -e\n" + recovery_command(args) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Run missing HPE experiments with resume-aware checkpointing.")
    parser.add_argument("--dataset", type=str, default="/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Optional runtime config override. Use 8 or 4 if OOM persists.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional runtime config override.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "missing_runs" / run_id
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, run_dir)
    write_manifest(run_dir, args, tasks)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("HPE missing-experiment pipeline", flush=True)
    print(f"project={PROJECT_ROOT}", flush=True)
    print(f"dataset={args.dataset}", flush=True)
    print(f"gpu={args.gpu} | device={args.device}", flush=True)
    print(f"max_train_batches={args.max_train_batches} | max_eval_batches={args.max_eval_batches}", flush=True)
    print(f"batch_size_override={args.batch_size} | num_workers_override={args.num_workers}", flush=True)
    print(f"records={run_dir}", flush=True)
    print("Recovery after interruption/OOM: rerun the same command, or run bash " + str(run_dir / "recovery_commands.sh"), flush=True)

    rows: list[dict[str, str]] = []
    failed_path = run_dir / "failed_tasks.jsonl"
    for index, task in enumerate(tasks, start=1):
        print_task_header(index, len(tasks), task)
        skip, reason = should_skip(task, args.force)
        saw_oom = False
        if skip:
            print(f"SKIP | {reason}", flush=True)
            status, elapsed, code = "skipped", 0.0, 0
        elif args.dry_run:
            print("DRY-RUN | command not executed", flush=True)
            status, elapsed, code = "dry_run", 0.0, 0
        else:
            status, elapsed, code, saw_oom = run_task(task, env, log_dir)
            print(f"TASK-END | {status} | seconds={elapsed:.1f} | returncode={code}", flush=True)
            if status == "failed":
                one = individual_recovery_command(task, args)
                print("RECOVERY | free GPU memory, then rerun the full pipeline or this task:", flush=True)
                print(one, flush=True)
                with open(failed_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"index": index, "name": task.name, "oom": saw_oom, "task_recovery_command": one, "pipeline_recovery_command": recovery_command(args)}, ensure_ascii=False) + "\n")
        rows.append({
            "index": str(index),
            "name": task.name,
            "title": task.title,
            "kind": task.kind,
            "status": status,
            "reason": reason if skip else "",
            "returncode": str(code),
            "oom_detected": str(saw_oom),
            "seconds": f"{elapsed:.3f}",
            "outputs": ";".join(str(path) for path in task.outputs),
        })
        write_status(run_dir / "status.csv", rows)
        if status == "failed" and args.stop_on_fail:
            break
    print(f"Missing-run records: {run_dir}", flush=True)
    print("Resume command: " + recovery_command(args), flush=True)


if __name__ == "__main__":
    main()
