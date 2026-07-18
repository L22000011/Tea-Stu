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
from typing import Iterable, Sequence

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


def output_dir_from_config(config_name: str, fallback: str) -> Path:
    config = CONFIG_DIR / config_name
    value = yaml_value(config, "output_dir")
    return PROJECT_ROOT / (value or fallback)


def add_resume(command: list[str], output_dir: Path) -> list[str]:
    last = output_dir / "last.pth"
    if last.exists() and "--resume" not in command:
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
    output_dir: Path,
    device: str,
    max_train_batches: int,
    teacher_rel: str | None = None,
    max_eval_batches: int | None = None,
    outputs: Iterable[Path] | None = None,
    extra_args: Sequence[str] | None = None,
) -> Task:
    config = CONFIG_DIR / config_name
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
    expected_outputs = list(outputs) if outputs is not None else [output_dir / "best.pth"]
    return Task(
        name=name,
        title=title,
        command=command,
        kind="train",
        required=required,
        outputs=expected_outputs,
        note=f"Training uses max_train_batches={max_train_batches}; evaluation is full unless max_eval_batches is set.",
    )


def eval_task(
    name: str,
    title: str,
    script: str,
    dataset: str,
    config_name: str,
    checkpoint_rel: str,
    output_rel: str,
    device: str,
    max_eval_batches: int | None = None,
) -> Task:
    config = CONFIG_DIR / config_name
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
        note="Full validation evaluation; no training cap is passed to evaluation scripts.",
    )


def missing_summary_task(
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
    command = py("eval_missing_modality.py", "--dataset", dataset, "--config", str(config), "--checkpoint", str(checkpoint), "--device", device, "--output-csv", str(output_csv))
    if nonvisual:
        command.append("--nonvisual")
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(name=name, title=title, command=command, kind="eval", required=[config, checkpoint], outputs=[output_csv], note="Grouped summary by missing-modality count.")


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
    command = py("export_metrics.py", "--dataset", dataset, "--config", str(config), "--checkpoint", str(checkpoint), "--device", device, "--modality-set", modality_set, "--output-csv", str(output_csv))
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(name=name, title=title, command=command, kind="export", required=[config, checkpoint], outputs=[output_csv], note="Exports parameter count, throughput, and memory footprint.")


def report_task(name: str, title: str, csv_candidates: Sequence[str]) -> Task:
    csv_path = first_existing(csv_candidates)
    summary_csv = csv_path.with_suffix(".xfi_compare_summary.csv")
    command = py("generate_xfi_report.py", "--csv", str(csv_path))
    return Task(name=name, title=title, command=command, kind="report", required=[csv_path], outputs=[summary_csv], note="Generates comparison assets against the official X-Fi table.")


def robustness_task(dataset: str, device: str, max_eval_batches: int | None = None) -> Task:
    config = CONFIG_DIR / "student_vk_missing.yaml"
    checkpoint = PROJECT_ROOT / "outputs/student_vk_missing/best.pth"
    output_csv = PROJECT_ROOT / "outputs/eval/student_vk_noise_robustness.csv"
    command = py("eval_robustness_vk_noise.py", "--dataset", dataset, "--config", str(config), "--checkpoint", str(checkpoint), "--device", device, "--output-csv", str(output_csv))
    if max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    return Task(name="random_student_vk_vk_noise", title="Random split | Student-VK VK noise and joint-drop robustness", command=command, kind="eval", required=[config, checkpoint], outputs=[output_csv], note="Stress-tests noisy VK coordinates and missing VK joints.")


def build_random_tasks(args: argparse.Namespace) -> list[Task]:
    dataset, device = args.dataset, args.device
    max_eval_batches = args.max_eval_batches
    ablation_root = output_dir_from_config("ablation.yaml", "outputs/ablation")
    distill_outputs = [ablation_root / "no_kd" / "best.pth"]
    reliability_outputs = [ablation_root / "uniform" / "best.pth"]

    tasks: list[Task] = [
        eval_task("random_teacher_eval31", "Random split | Teacher 31-combination evaluation", "eval_all_combinations.py", dataset, "teacher_full.yaml", "outputs/teacher_full/best.pth", "outputs/eval/teacher_random_all_combinations.csv", device, max_eval_batches),
        eval_task("random_student_vk_eval31", "Random split | Student-VK 31-combination evaluation", "eval_all_combinations.py", dataset, "student_vk_missing.yaml", "outputs/student_vk_missing/best.pth", "outputs/eval/student_vk_random_all_combinations.csv", device, max_eval_batches),
        eval_task("random_student_nv_eval15", "Random split | Student-NV non-visual combination evaluation", "eval_nonvisual_combinations.py", dataset, "student_nv_missing.yaml", "outputs/student_nv_missing/best.pth", "outputs/eval/student_nv_random_nonvisual_combinations.csv", device, max_eval_batches),
        eval_task("random_baseline_eval31", "Random split | Baseline 31-combination evaluation", "eval_all_combinations.py", dataset, "baseline_full.yaml", "outputs/baseline_full/best.pth", "outputs/eval/baseline_full_all_combinations.csv", device, max_eval_batches),
        train_task("random_distillation_ablation_train", "Random split | No-distillation ablation training", "ablate_distillation.py", dataset, "ablation.yaml", ablation_root, device, args.max_train_batches, teacher_rel="outputs/teacher_full/best.pth", max_eval_batches=max_eval_batches, outputs=distill_outputs, extra_args=["--variants", "no_kd"]),
        eval_task("random_ablation_no_distill_eval31", "Random split | No-distillation ablation 31-combination evaluation", "eval_all_combinations.py", dataset, "ablation.yaml", "outputs/ablation/no_kd/best.pth", "outputs/eval/ablation_no_distill_all_combinations.csv", device, max_eval_batches),
        train_task("random_reliability_ablation_train", "Random split | Uniform-fusion ablation training", "ablate_reliability.py", dataset, "ablation.yaml", ablation_root, device, args.max_train_batches, teacher_rel="outputs/teacher_full/best.pth", max_eval_batches=max_eval_batches, outputs=reliability_outputs, extra_args=["--variants", "uniform"]),
        eval_task("random_ablation_uniform_fusion_eval31", "Random split | Uniform-fusion ablation 31-combination evaluation", "eval_all_combinations.py", dataset, "ablation.yaml", "outputs/ablation/uniform/best.pth", "outputs/eval/ablation_uniform_fusion_all_combinations.csv", device, max_eval_batches),
        missing_summary_task("random_teacher_missing_summary", "Random split | Teacher missing-modality summary", dataset, "teacher_full.yaml", "outputs/teacher_full/best.pth", "outputs/eval/teacher_missing_modality_summary.csv", device, max_eval_batches=max_eval_batches),
        missing_summary_task("random_student_vk_missing_summary", "Random split | Student-VK missing-modality summary", dataset, "student_vk_missing.yaml", "outputs/student_vk_missing/best.pth", "outputs/eval/student_vk_missing_modality_summary.csv", device, max_eval_batches=max_eval_batches),
        missing_summary_task("random_student_nv_missing_summary", "Random split | Student-NV missing non-visual summary", dataset, "student_nv_missing.yaml", "outputs/student_nv_missing/best.pth", "outputs/eval/student_nv_missing_modality_summary.csv", device, nonvisual=True, max_eval_batches=max_eval_batches),
        missing_summary_task("random_baseline_missing_summary", "Random split | Baseline missing-modality summary", dataset, "baseline_full.yaml", "outputs/baseline_full/best.pth", "outputs/eval/baseline_missing_modality_summary.csv", device, max_eval_batches=max_eval_batches),
        robustness_task(dataset, device, max_eval_batches=max_eval_batches),
        export_metrics_task("random_teacher_complexity", "Random split | Teacher complexity export", dataset, "teacher_full.yaml", "outputs/teacher_full/best.pth", "vk,depth,lidar,mmwave,wifi-csi", "outputs/eval/teacher_full_model_complexity.csv", device, max_eval_batches),
        export_metrics_task("random_student_vk_complexity", "Random split | Student-VK complexity export", dataset, "student_vk_missing.yaml", "outputs/student_vk_missing/best.pth", "vk,depth,lidar,mmwave,wifi-csi", "outputs/eval/student_vk_model_complexity.csv", device, max_eval_batches),
        export_metrics_task("random_student_nv_complexity", "Random split | Student-NV complexity export", dataset, "student_nv_missing.yaml", "outputs/student_nv_missing/best.pth", "depth,lidar,mmwave,wifi-csi", "outputs/eval/student_nv_model_complexity.csv", device, max_eval_batches),
        export_metrics_task("random_baseline_complexity", "Random split | Baseline complexity export", dataset, "baseline_full.yaml", "outputs/baseline_full/best.pth", "vk,depth,lidar,mmwave,wifi-csi", "outputs/eval/baseline_full_model_complexity.csv", device, max_eval_batches),
    ]
    return tasks


def build_protocol_tasks(args: argparse.Namespace, tag: str) -> list[Task]:
    dataset, device = args.dataset, args.device
    max_eval_batches = args.max_eval_batches
    teacher_cfg = f"teacher_{tag}.yaml"
    student_vk_cfg = f"student_vk_{tag}.yaml"
    student_nv_cfg = f"student_nv_{tag}.yaml"
    baseline_cfg = f"baseline_{tag}.yaml"
    teacher_dir = output_dir_from_config(teacher_cfg, f"outputs/teacher_full_{tag}_split")
    student_vk_dir = output_dir_from_config(student_vk_cfg, f"outputs/student_vk_missing_{tag}_split")
    student_nv_dir = output_dir_from_config(student_nv_cfg, f"outputs/student_nv_missing_{tag}")
    baseline_dir = output_dir_from_config(baseline_cfg, f"outputs/baseline_full_{tag}_split")
    teacher_rel = str(teacher_dir.relative_to(PROJECT_ROOT) / "best.pth")
    student_vk_rel = str(student_vk_dir.relative_to(PROJECT_ROOT) / "best.pth")
    student_nv_rel = str(student_nv_dir.relative_to(PROJECT_ROOT) / "best.pth")
    baseline_rel = str(baseline_dir.relative_to(PROJECT_ROOT) / "best.pth")

    return [
        train_task(f"{tag}_teacher_train", f"{tag} | Teacher full-modality training", "train_teacher_full.py", dataset, teacher_cfg, teacher_dir, device, args.max_train_batches, max_eval_batches=max_eval_batches),
        eval_task(f"{tag}_teacher_eval31", f"{tag} | Teacher 31-combination evaluation", "eval_all_combinations.py", dataset, teacher_cfg, teacher_rel, f"outputs/eval/teacher_{tag}_all_combinations.csv", device, max_eval_batches),
        train_task(f"{tag}_student_vk_train", f"{tag} | Student-VK missing-modality training", "train_student_vk_missing.py", dataset, student_vk_cfg, student_vk_dir, device, args.max_train_batches, teacher_rel=teacher_rel, max_eval_batches=max_eval_batches),
        eval_task(f"{tag}_student_vk_eval31", f"{tag} | Student-VK 31-combination evaluation", "eval_all_combinations.py", dataset, student_vk_cfg, student_vk_rel, f"outputs/eval/student_vk_{tag}_all_combinations.csv", device, max_eval_batches),
        train_task(f"{tag}_student_nv_train", f"{tag} | Student-NV non-visual missing-modality training", "train_student_nv_missing.py", dataset, student_nv_cfg, student_nv_dir, device, args.max_train_batches, teacher_rel=teacher_rel, max_eval_batches=max_eval_batches),
        eval_task(f"{tag}_student_nv_eval15", f"{tag} | Student-NV non-visual combination evaluation", "eval_nonvisual_combinations.py", dataset, student_nv_cfg, student_nv_rel, f"outputs/eval/student_nv_{tag}_nonvisual_combinations.csv", device, max_eval_batches),
        train_task(f"{tag}_baseline_train", f"{tag} | Baseline full-modality training", "train_baseline_full.py", dataset, baseline_cfg, baseline_dir, device, args.max_train_batches, max_eval_batches=max_eval_batches),
        eval_task(f"{tag}_baseline_eval31", f"{tag} | Baseline 31-combination evaluation", "eval_all_combinations.py", dataset, baseline_cfg, baseline_rel, f"outputs/eval/baseline_{tag}_all_combinations.csv", device, max_eval_batches),
    ]


def build_report_tasks() -> list[Task]:
    return [
        report_task("random_teacher_xfi_report", "Random split | Teacher vs official X-Fi report", ["outputs/eval/teacher_random_all_combinations.csv", "outputs/eval/teacher_all_combinations.csv"]),
        report_task("random_student_vk_xfi_report", "Random split | Student-VK vs official X-Fi report", ["outputs/eval/student_vk_random_all_combinations.csv", "outputs/eval/all_combinations.csv"]),
        report_task("random_student_nv_xfi_report", "Random split | Student-NV vs official X-Fi report", ["outputs/eval/student_nv_random_nonvisual_combinations.csv", "outputs/eval/student_nv_nonvisual_combinations.csv"]),
        report_task("random_baseline_xfi_report", "Random split | Baseline vs official X-Fi report", ["outputs/eval/baseline_full_all_combinations.csv"]),
        report_task("ablation_no_distill_xfi_report", "Random split | No-distillation ablation report", ["outputs/eval/ablation_no_distill_all_combinations.csv"]),
        report_task("ablation_uniform_fusion_xfi_report", "Random split | Uniform-fusion ablation report", ["outputs/eval/ablation_uniform_fusion_all_combinations.csv"]),
    ]


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    tasks.extend(build_random_tasks(args))
    tasks.extend(build_protocol_tasks(args, "cross_scene"))
    tasks.extend(build_protocol_tasks(args, "cross_subject"))
    tasks.extend(build_report_tasks())
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
    parser = argparse.ArgumentParser("Run the complete supplemental HPE experiment pipeline aligned with HAR outputs.")
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

    print("HPE supplemental experiment pipeline", flush=True)
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
