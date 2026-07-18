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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = OUTPUT_DIR / "eval"


@dataclass
class Task:
    name: str
    title: str
    command: list[str]
    required: list[Path | None] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    kind: str = "run"


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *args]


def config(name: str) -> Path:
    return PROJECT_ROOT / "configs" / name


def add_resume(command: list[str], output_dir: Path) -> list[str]:
    last = output_dir / "last.pth"
    if last.exists() and "--resume" not in command:
        return [*command, "--resume", str(last)]
    failed = output_dir / "failed.pth"
    if failed.exists() and "--resume" not in command:
        return [*command, "--resume", str(failed)]
    return command


def train_super_task(name: str, title: str, cfg: str, out: str, args: argparse.Namespace, outputs: list[str] | None = None) -> Task:
    output_dir = OUTPUT_DIR / out
    cmd = py("train_super_teacher.py", "--dataset", args.dataset, "--config", str(config(cfg)), "--device", args.device, "--max-train-batches", str(args.max_train_batches))
    if args.max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])
    cmd = add_resume(cmd, output_dir)
    expected = outputs or ["best_joint.pth", "best_hpe.pth", "best_har.pth"]
    return Task(name, title, cmd, [config(cfg)], [output_dir / item for item in expected], "train")


def train_student_task(name: str, title: str, script: str, cfg: str, teacher: str, output: str, variant: str, args: argparse.Namespace) -> Task:
    output_dir = OUTPUT_DIR / output
    cmd = py(script, "--dataset", args.dataset, "--config", str(config(cfg)), "--teacher", str(OUTPUT_DIR / teacher), "--device", args.device, "--max-train-batches", str(args.max_train_batches))
    cmd.extend(["--variant", variant])
    if args.max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])
    cmd = add_resume(cmd, output_dir)
    return Task(name, title, cmd, [config(cfg), OUTPUT_DIR / teacher], [output_dir / "best.pth"], "train")


def eval_task(name: str, title: str, script: str, cfg: str, checkpoint: str, output: str, args: argparse.Namespace, extra: list[str] | None = None) -> Task:
    cmd = py(script, "--dataset", args.dataset, "--config", str(config(cfg)), "--checkpoint", str(OUTPUT_DIR / checkpoint), "--device", args.device, "--output-csv", str(OUTPUT_DIR / output))
    if args.max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if extra:
        cmd.extend(extra)
    return Task(name, title, cmd, [config(cfg), OUTPUT_DIR / checkpoint], [OUTPUT_DIR / output], "eval")


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    protocols = [
        ("random", "super_teacher.yaml", "hpe_student_from_super.yaml", "har_student_from_super.yaml", "super_teacher"),
        ("cross_scene", "super_teacher_cross_scene.yaml", "hpe_student_from_super_cross_scene.yaml", "har_student_from_super_cross_scene.yaml", "super_teacher_cross_scene"),
        ("cross_subject", "super_teacher_cross_subject.yaml", "hpe_student_from_super_cross_subject.yaml", "har_student_from_super_cross_subject.yaml", "super_teacher_cross_subject"),
    ]
    for tag, teacher_cfg, hpe_cfg, har_cfg, teacher_out in protocols:
        teacher_ckpt = f"{teacher_out}/best_joint.pth"
        tasks.append(train_super_task(f"{tag}_super_teacher_train", f"{tag} | Super Teacher training", teacher_cfg, teacher_out, args))
        tasks.append(eval_task(f"{tag}_super_teacher_hpe_eval31", f"{tag} | Super Teacher HPE combinations", "eval_super_teacher_hpe.py", teacher_cfg, teacher_ckpt, f"eval/super_teacher_{tag}_hpe_all_combinations.csv", args))
        tasks.append(eval_task(f"{tag}_super_teacher_har_eval", f"{tag} | Super Teacher HAR combinations", "eval_super_teacher_har.py", teacher_cfg, teacher_ckpt, f"eval/super_teacher_{tag}_har_all_combinations.csv", args))
        hpe_prefix = "super_hpe_student" if tag == "random" else f"super_hpe_student_{tag}"
        har_prefix = "super_har_student" if tag == "random" else f"super_har_student_{tag}"
        tasks.append(train_student_task(f"{tag}_hpe_student_vk_train", f"{tag} | HPE Student-VK training from Super Teacher", "train_hpe_student_from_super.py", hpe_cfg, teacher_ckpt, f"{hpe_prefix}_vk", "vk", args))
        tasks.append(eval_task(f"{tag}_hpe_student_vk_eval31", f"{tag} | HPE Student-VK combinations", "eval_hpe_student.py", hpe_cfg, f"{hpe_prefix}_vk/best.pth", f"eval/super_hpe_student_{tag}_vk_all_combinations.csv", args, ["--variant", "vk"]))
        tasks.append(train_student_task(f"{tag}_hpe_student_nv_train", f"{tag} | HPE Student-NV training from Super Teacher", "train_hpe_student_from_super.py", hpe_cfg, teacher_ckpt, f"{hpe_prefix}_nv", "nv", args))
        tasks.append(eval_task(f"{tag}_hpe_student_nv_eval", f"{tag} | HPE Student-NV nonvisual combinations", "eval_hpe_student.py", hpe_cfg, f"{hpe_prefix}_nv/best.pth", f"eval/super_hpe_student_{tag}_nv_nonvisual_combinations.csv", args, ["--variant", "nv", "--nonvisual"]))
        tasks.append(train_student_task(f"{tag}_har_student_vk_train", f"{tag} | HAR Student-VK training from Super Teacher", "train_har_student_from_super.py", har_cfg, teacher_ckpt, f"{har_prefix}_vk", "vk", args))
        tasks.append(eval_task(f"{tag}_har_student_vk_eval", f"{tag} | HAR Student-VK combinations", "eval_har_student.py", har_cfg, f"{har_prefix}_vk/best.pth", f"eval/super_har_student_{tag}_vk_all_combinations.csv", args, ["--variant", "vk"]))
        tasks.append(train_student_task(f"{tag}_har_student_nv_train", f"{tag} | HAR Student-NV training from Super Teacher", "train_har_student_from_super.py", har_cfg, teacher_ckpt, f"{har_prefix}_nv", "nv", args))
        tasks.append(eval_task(f"{tag}_har_student_nv_eval", f"{tag} | HAR Student-NV nonvisual combinations", "eval_har_student.py", har_cfg, f"{har_prefix}_nv/best.pth", f"eval/super_har_student_{tag}_nv_nonvisual_combinations.csv", args, ["--variant", "nv", "--nonvisual"]))

    tasks.append(train_super_task("ablation_hpe_only_teacher_train", "ablation | HPE-only Super Teacher", "super_teacher_hpe_only.yaml", "super_teacher_hpe_only", args, ["best_hpe.pth"]))
    tasks.append(eval_task("ablation_hpe_only_hpe_eval31", "ablation | HPE-only teacher HPE combinations", "eval_super_teacher_hpe.py", "super_teacher_hpe_only.yaml", "super_teacher_hpe_only/best_hpe.pth", "eval/super_teacher_hpe_only_hpe_all_combinations.csv", args))
    tasks.append(train_super_task("ablation_har_only_teacher_train", "ablation | HAR-only Super Teacher", "super_teacher_har_only.yaml", "super_teacher_har_only", args, ["best_har.pth"]))
    tasks.append(eval_task("ablation_har_only_har_eval", "ablation | HAR-only teacher HAR combinations", "eval_super_teacher_har.py", "super_teacher_har_only.yaml", "super_teacher_har_only/best_har.pth", "eval/super_teacher_har_only_har_all_combinations.csv", args))
    return tasks


def should_skip(task: Task, force: bool) -> tuple[bool, str]:
    missing = [path for path in task.required if path is None or not path.exists()]
    if missing:
        return True, "missing_required=" + ",".join("None" if path is None else str(path) for path in missing)
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
    return ("done" if code == 0 else "failed", time.time() - start, code)


def main() -> None:
    parser = argparse.ArgumentParser("Run Super-Teacher experiments.")
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
    run_dir = OUTPUT_DIR / "super_runs" / run_id
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    (run_dir / "manifest.json").write_text(json.dumps({
        "dataset": args.dataset,
        "gpu": args.gpu,
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "task_count": len(tasks),
        "tasks": [{"name": task.name, "title": task.title, "kind": task.kind, "command": task.command, "outputs": [str(path) for path in task.outputs]} for task in tasks],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("Super-Teacher pipeline")
    print(f"project={PROJECT_ROOT}")
    print(f"dataset={args.dataset}")
    print(f"gpu={args.gpu} | device={args.device}")
    print(f"records={run_dir}")
    rows: list[dict[str, str]] = []
    for index, task in enumerate(tasks, start=1):
        print("\n" + "=" * 88, flush=True)
        print(f"[{index}/{len(tasks)}] {task.title}", flush=True)
        print(f"name={task.name} | kind={task.kind}", flush=True)
        print("command=", " ".join(task.command), flush=True)
        print("outputs=", "; ".join(str(path) for path in task.outputs), flush=True)
        print("=" * 88, flush=True)
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
    print(f"Super pipeline records: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
