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
from typing import Iterable

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
    expected_split: str | None = None
    config: Path | None = None
    note: str = ""


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


def config_path(candidates: Iterable[str], expected_split: str) -> Path | None:
    for name in candidates:
        path = CONFIG_DIR / name
        if path.exists() and yaml_value(path, "split_to_use") == expected_split:
            return path
    return None


def output_dir_from_config(config: Path | None, fallback: str) -> Path:
    value = yaml_value(config, "output_dir")
    return PROJECT_ROOT / (value or fallback)


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *args]


def add_resume(command: list[str], output_dir: Path, resume: bool) -> list[str]:
    last = output_dir / "last.pth"
    if resume and last.exists() and "--resume" not in command:
        return [*command, "--resume", str(last)]
    return command


def train_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config: Path | None,
    device: str,
    max_train_batches: int,
    output_dir: Path,
    expected_split: str,
    teacher: Path | None = None,
    resume: bool = True,
) -> Task:
    required: list[Path | None] = [config]
    command = py(script, "--dataset", dataset, "--config", str(config), "--device", device, "--max-train-batches", str(max_train_batches))
    if teacher is not None:
        command.extend(["--teacher", str(teacher)])
        required.append(teacher)
    command = add_resume(command, output_dir, resume)
    return Task(
        name=name,
        title=title,
        command=command,
        kind="train",
        required=required,
        outputs=[output_dir / "best.pth"],
        expected_split=expected_split,
        config=config,
        note=f"max_train_batches={max_train_batches}; evaluation remains full unless the config sets max_eval_batches.",
    )


def eval_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config: Path | None,
    checkpoint: Path,
    device: str,
    output_csv: Path,
    expected_split: str,
) -> Task:
    return Task(
        name=name,
        title=title,
        command=py(script, "--dataset", dataset, "--config", str(config), "--checkpoint", str(checkpoint), "--device", device, "--output-csv", str(output_csv)),
        kind="eval",
        required=[config, checkpoint],
        outputs=[output_csv],
        expected_split=expected_split,
        config=config,
        note="Full validation evaluation; no max_train_batches is passed to evaluation scripts.",
    )


def build_random_tasks(args: argparse.Namespace) -> list[Task]:
    split = "random_split"
    dataset, device = args.dataset, args.device
    teacher = OUTPUT_DIR / "teacher_full" / "best.pth"
    student_nv = OUTPUT_DIR / "student_nv_missing" / "best.pth"
    cfg_student_nv = config_path(["student_nv_missing.yaml"], split)
    cfg_baseline = config_path(["baseline_full.yaml"], split)
    cfg_ablation = config_path(["ablation.yaml"], split)
    baseline_dir = output_dir_from_config(cfg_baseline, "outputs/baseline_full")
    ablation_dir = output_dir_from_config(cfg_ablation, "outputs/ablation")

    tasks: list[Task] = [
        eval_task("random_student_nv_nonvisual_eval", "Random split | Student-NV non-visual modality combinations", "eval_nonvisual_combinations.py", dataset, cfg_student_nv, student_nv, device, EVAL_DIR / "student_nv_nonvisual_combinations.csv", split),
        train_task("random_baseline_train", "Random split | Full-modality baseline training", "train_baseline_full.py", dataset, cfg_baseline, device, args.max_train_batches, baseline_dir, split),
        eval_task("random_baseline_all_combinations", "Random split | Baseline 31-combination evaluation", "eval_all_combinations.py", dataset, cfg_baseline, baseline_dir / "best.pth", device, EVAL_DIR / "baseline_full_all_combinations.csv", split),
        train_task("random_distillation_ablation_train", "Random split | Distillation-loss ablation training", "ablate_distillation.py", dataset, cfg_ablation, device, args.max_train_batches, ablation_dir, split, teacher=teacher),
    ]
    distill_variants = ["no_kd", "output_kd", "output_token_kd", "output_bone_kd", "full_structural_kd"]
    tasks[-1].outputs = [ablation_dir / variant / "best.pth" for variant in distill_variants]
    for variant in distill_variants:
        tasks.append(eval_task(f"random_distill_{variant}_eval", f"Random split | Distillation ablation eval: {variant}", "eval_all_combinations.py", dataset, cfg_ablation, ablation_dir / variant / "best.pth", device, EVAL_DIR / f"ablation_distillation_{variant}_all_combinations.csv", split))

    reliability_variants = ["uniform", "attention", "uncertainty"]
    tasks.append(train_task("random_reliability_ablation_train", "Random split | Reliability-fusion ablation training", "ablate_reliability.py", dataset, cfg_ablation, device, args.max_train_batches, ablation_dir, split, teacher=teacher))
    tasks[-1].outputs = [ablation_dir / variant / "best.pth" for variant in reliability_variants]
    for variant in reliability_variants:
        tasks.append(eval_task(f"random_reliability_{variant}_eval", f"Random split | Reliability ablation eval: {variant}", "eval_all_combinations.py", dataset, cfg_ablation, ablation_dir / variant / "best.pth", device, EVAL_DIR / f"ablation_reliability_{variant}_all_combinations.csv", split))

    encoder_variants = ["mlp_vk", "skeleton_prompt"]
    tasks.append(train_task("random_encoder_ablation_train", "Random split | VK encoder ablation training", "ablate_encoder.py", dataset, cfg_ablation, device, args.max_train_batches, ablation_dir, split))
    tasks[-1].outputs = [ablation_dir / variant / "best.pth" for variant in encoder_variants]
    for variant in encoder_variants:
        tasks.append(eval_task(f"random_encoder_{variant}_eval", f"Random split | VK encoder ablation eval: {variant}", "eval_all_combinations.py", dataset, cfg_ablation, ablation_dir / variant / "best.pth", device, EVAL_DIR / f"ablation_encoder_{variant}_all_combinations.csv", split))
    return tasks


def build_protocol_tasks(args: argparse.Namespace, tag: str, split: str) -> list[Task]:
    dataset, device = args.dataset, args.device
    teacher_cfg = config_path([f"teacher_{tag}.yaml"], split)
    student_vk_cfg = config_path([f"student_vk_{tag}.yaml"], split)
    student_nv_cfg = config_path([f"student_nv_{tag}.yaml"], split)
    baseline_cfg = config_path([f"baseline_{tag}.yaml"], split)
    teacher_dir = output_dir_from_config(teacher_cfg, f"outputs/teacher_full_{tag}_split")
    student_vk_dir = output_dir_from_config(student_vk_cfg, f"outputs/student_vk_missing_{tag}_split")
    student_nv_dir = output_dir_from_config(student_nv_cfg, f"outputs/student_nv_missing_{tag}")
    baseline_dir = output_dir_from_config(baseline_cfg, f"outputs/baseline_full_{tag}_split")
    teacher_ckpt = teacher_dir / "best.pth"

    return [
        train_task(f"{tag}_teacher_train", f"{tag} | Teacher full-modality training", "train_teacher_full.py", dataset, teacher_cfg, device, args.max_train_batches, teacher_dir, split),
        eval_task(f"{tag}_teacher_all_combinations", f"{tag} | Teacher 31-combination evaluation", "eval_all_combinations.py", dataset, teacher_cfg, teacher_ckpt, device, EVAL_DIR / f"teacher_{tag}_all_combinations.csv", split),
        train_task(f"{tag}_student_vk_train", f"{tag} | Student-VK missing-modality training", "train_student_vk_missing.py", dataset, student_vk_cfg, device, args.max_train_batches, student_vk_dir, split, teacher=teacher_ckpt),
        eval_task(f"{tag}_student_vk_all_combinations", f"{tag} | Student-VK 31-combination evaluation", "eval_all_combinations.py", dataset, student_vk_cfg, student_vk_dir / "best.pth", device, EVAL_DIR / f"student_vk_{tag}_all_combinations.csv", split),
        train_task(f"{tag}_student_nv_train", f"{tag} | Student-NV non-visual missing-modality training", "train_student_nv_missing.py", dataset, student_nv_cfg, device, args.max_train_batches, student_nv_dir, split, teacher=teacher_ckpt),
        eval_task(f"{tag}_student_nv_nonvisual_combinations", f"{tag} | Student-NV non-visual combination evaluation", "eval_nonvisual_combinations.py", dataset, student_nv_cfg, student_nv_dir / "best.pth", device, EVAL_DIR / f"student_nv_{tag}_nonvisual_combinations.csv", split),
        train_task(f"{tag}_baseline_train", f"{tag} | Full-modality baseline training", "train_baseline_full.py", dataset, baseline_cfg, device, args.max_train_batches, baseline_dir, split),
        eval_task(f"{tag}_baseline_all_combinations", f"{tag} | Baseline 31-combination evaluation", "eval_all_combinations.py", dataset, baseline_cfg, baseline_dir / "best.pth", device, EVAL_DIR / f"baseline_{tag}_all_combinations.csv", split),
    ]


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    tasks.extend(build_random_tasks(args))
    tasks.extend(build_protocol_tasks(args, "cross_scene", "cross_scene_split"))
    tasks.extend(build_protocol_tasks(args, "cross_subject", "cross_subject_split"))
    return tasks


def missing_reason(task: Task) -> str | None:
    missing = [path for path in task.required if path is None or not path.exists()]
    if missing:
        return "missing_required=" + ",".join("None" if path is None else str(path) for path in missing)
    if task.config is not None and task.expected_split is not None:
        actual = yaml_value(task.config, "split_to_use")
        if actual != task.expected_split:
            return f"split_mismatch={task.config}: expected {task.expected_split}, got {actual}"
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
    print(f"name={task.name} | kind={task.kind} | expected_split={task.expected_split}", flush=True)
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
        process = subprocess.Popen(task.command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
        "task_count": len(tasks),
        "tasks": [
            {
                "name": task.name,
                "title": task.title,
                "kind": task.kind,
                "expected_split": task.expected_split,
                "command": task.command,
                "required": [None if path is None else str(path) for path in task.required],
                "outputs": [str(path) for path in task.outputs],
            }
            for task in tasks
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Run the required HPE experiment pipeline sequentially.")
    parser.add_argument("--dataset", type=str, default="/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=1000, help="Training-iteration cap per epoch. Evaluation stays full by default.")
    parser.add_argument("--force", action="store_true", help="Run tasks even when their expected outputs already exist.")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "pipeline_runs" / run_id
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    write_manifest(run_dir, args, tasks)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("HPE required experiment pipeline", flush=True)
    print(f"project={PROJECT_ROOT}", flush=True)
    print(f"dataset={args.dataset}", flush=True)
    print(f"gpu={args.gpu} | device={args.device}", flush=True)
    print(f"max_train_batches={args.max_train_batches} for training tasks only", flush=True)
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
            "expected_split": task.expected_split or "",
            "status": status,
            "reason": reason if skip else "",
            "returncode": str(code),
            "seconds": f"{elapsed:.3f}",
            "outputs": ";".join(str(path) for path in task.outputs),
        })
        write_status(run_dir / "status.csv", rows)
        if status == "failed" and args.stop_on_fail:
            break
    print(f"Pipeline records: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
