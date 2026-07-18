from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_dir(*candidates: str) -> Path:
    for name in candidates:
        path = PROJECT_ROOT / name
        if path.exists():
            return path
    return PROJECT_ROOT / candidates[0]


HPE_ROOT = resolve_project_dir("HPE", "MMFi_HPE")
HAR_ROOT = resolve_project_dir("HAR", "MMFi_HAR")


@dataclass
class Task:
    index: int
    name: str
    title: str
    project: str
    kind: str
    cwd: Path
    command: list[str]
    outputs: list[Path]
    required: list[Path]
    output_dir: Path | None = None


def rel_project(project: str, path: str) -> Path:
    root = HPE_ROOT if project == "HPE" else HAR_ROOT
    return root / path


def py_command(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", script, *args]


def add_resume(command: list[str], output_dir: Path | None) -> list[str]:
    if output_dir is None:
        return command
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    if last_path.exists() and not best_path.exists() and "--resume" not in command:
        return [*command, "--resume", str(last_path)]
    return command


def build_tasks(args: argparse.Namespace) -> list[Task]:
    dataset = args.dataset
    device = args.device
    max_train = str(args.max_train_batches)

    hpe_no_random_dir = rel_project("HPE", "outputs/final_supplement/student_vk_no_random_missing")
    har_no_random_dir = rel_project("HAR", "outputs/final_supplement/student_vk_no_random_missing")
    output_kd_ckpt = existing_checkpoint(
        rel_project("HPE", "outputs/ablation/output_kd/best.pth"),
        rel_project("HPE", "outputs/ablation/output_kd/last.pth"),
    )
    full_structural_kd_ckpt = existing_checkpoint(
        rel_project("HPE", "outputs/ablation/full_structural_kd/best.pth"),
        rel_project("HPE", "outputs/ablation/full_structural_kd/last.pth"),
    )

    tasks = [
        Task(
            1,
            "hpe_no_random_train",
            "HPE | Student-VK training without random missing-modality sampling",
            "HPE",
            "train",
            HPE_ROOT,
            py_command(
                "scripts/train_student_vk_missing.py",
                "--dataset",
                dataset,
                "--config",
                "configs/student_vk_no_random_missing.yaml",
                "--teacher",
                "outputs/teacher_full/best.pth",
                "--device",
                device,
                "--max-train-batches",
                max_train,
            ),
            [hpe_no_random_dir / "best.pth"],
            [rel_project("HPE", "outputs/teacher_full/best.pth")],
            hpe_no_random_dir,
        ),
        Task(
            2,
            "hpe_no_random_eval31",
            "HPE | Student-VK no-random 31-combination evaluation",
            "HPE",
            "eval",
            HPE_ROOT,
            py_command(
                "scripts/eval_all_combinations.py",
                "--dataset",
                dataset,
                "--config",
                "configs/student_vk_no_random_missing.yaml",
                "--checkpoint",
                "outputs/final_supplement/student_vk_no_random_missing/best.pth",
                "--device",
                device,
                "--output-csv",
                "outputs/eval/final_student_vk_no_random_all_combinations.csv",
            ),
            [rel_project("HPE", "outputs/eval/final_student_vk_no_random_all_combinations.csv")],
            [hpe_no_random_dir / "best.pth"],
        ),
        Task(
            3,
            "har_no_random_train",
            "HAR | Student-VK training without random missing-modality sampling",
            "HAR",
            "train",
            HAR_ROOT,
            py_command(
                "scripts/train_student_vk_missing.py",
                "--dataset",
                dataset,
                "--config",
                "configs/student_vk_no_random_missing.yaml",
                "--teacher",
                "outputs/teacher_full/best.pth",
                "--device",
                device,
                "--max-train-batches",
                max_train,
            ),
            [har_no_random_dir / "best.pth"],
            [rel_project("HAR", "outputs/teacher_full/best.pth")],
            har_no_random_dir,
        ),
        Task(
            4,
            "har_no_random_eval15",
            "HAR | Student-VK no-random 15-combination evaluation",
            "HAR",
            "eval",
            HAR_ROOT,
            py_command(
                "scripts/eval_all_combinations.py",
                "--dataset",
                dataset,
                "--config",
                "configs/student_vk_no_random_missing.yaml",
                "--checkpoint",
                "outputs/final_supplement/student_vk_no_random_missing/best.pth",
                "--device",
                device,
                "--output-csv",
                "outputs/eval/final_student_vk_no_random_all_combinations.csv",
                "--output-md",
                "outputs/eval/final_student_vk_no_random_all_combinations.md",
            ),
            [
                rel_project("HAR", "outputs/eval/final_student_vk_no_random_all_combinations.csv"),
                rel_project("HAR", "outputs/eval/final_student_vk_no_random_all_combinations.md"),
            ],
            [har_no_random_dir / "best.pth"],
        ),
        Task(
            5,
            "hpe_output_kd_eval31",
            "HPE | output-KD 31-combination evaluation",
            "HPE",
            "eval",
            HPE_ROOT,
            py_command(
                "scripts/eval_all_combinations.py",
                "--dataset",
                dataset,
                "--config",
                "configs/ablation.yaml",
                "--checkpoint",
                str(output_kd_ckpt.relative_to(HPE_ROOT)) if output_kd_ckpt is not None else "outputs/ablation/output_kd/best.pth",
                "--device",
                device,
                "--output-csv",
                "outputs/eval/final_ablation_output_kd_all_combinations.csv",
            ),
            [rel_project("HPE", "outputs/eval/final_ablation_output_kd_all_combinations.csv")],
            [output_kd_ckpt or rel_project("HPE", "outputs/ablation/output_kd/best.pth")],
        ),
        Task(
            6,
            "hpe_full_structural_kd_eval31",
            "HPE | full-structural-KD 31-combination evaluation",
            "HPE",
            "eval",
            HPE_ROOT,
            py_command(
                "scripts/eval_all_combinations.py",
                "--dataset",
                dataset,
                "--config",
                "configs/ablation.yaml",
                "--checkpoint",
                str(full_structural_kd_ckpt.relative_to(HPE_ROOT)) if full_structural_kd_ckpt is not None else "outputs/ablation/full_structural_kd/best.pth",
                "--device",
                device,
                "--output-csv",
                "outputs/eval/final_ablation_full_structural_kd_all_combinations.csv",
            ),
            [rel_project("HPE", "outputs/eval/final_ablation_full_structural_kd_all_combinations.csv")],
            [full_structural_kd_ckpt or rel_project("HPE", "outputs/ablation/full_structural_kd/best.pth")],
        ),
    ]
    return tasks


def existing_checkpoint(best_path: Path, last_path: Path) -> Path | None:
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    return None


def missing_required(task: Task) -> str | None:
    missing = [path for path in task.required if not path.exists()]
    if missing:
        return "missing_required=" + ";".join(str(path) for path in missing)
    return None


def outputs_exist(task: Task) -> bool:
    return bool(task.outputs) and all(path.exists() for path in task.outputs)


def write_csv_row(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_task(task: Task, run_dirs: dict[str, Path], args: argparse.Namespace) -> dict[str, object]:
    run_dir = run_dirs[task.project]
    log_path = run_dir / "logs" / f"{task.index:02d}_{task.name}.log"
    status_path = run_dir / "status.csv"
    fieldnames = ["index", "name", "title", "project", "kind", "status", "reason", "returncode", "seconds", "outputs", "log"]
    command = add_resume(list(task.command), task.output_dir)

    print("\n" + "=" * 88, flush=True)
    print(f"[{task.index}/6] {task.title}", flush=True)
    print(f"name={task.name} | project={task.project} | kind={task.kind}", flush=True)
    print("command=", " ".join(command), flush=True)
    print("outputs=", "; ".join(str(path) for path in task.outputs), flush=True)
    print("=" * 88, flush=True)

    if outputs_exist(task):
        row = make_status_row(task, "skipped", "outputs_exist", 0, 0.0, log_path)
        write_csv_row(status_path, row, fieldnames)
        print("SKIP | outputs_exist", flush=True)
        return row

    reason = missing_required(task)
    if reason:
        row = make_status_row(task, "skipped", reason, 0, 0.0, log_path)
        write_csv_row(status_path, row, fieldnames)
        print(f"SKIP | {reason}", flush=True)
        return row

    if args.dry_run:
        row = make_status_row(task, "dry_run", "", 0, 0.0, log_path)
        write_csv_row(status_path, row, fieldnames)
        print("DRY-RUN | command not executed", flush=True)
        return row

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(task.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()
    seconds = time.time() - start
    status = "done" if returncode == 0 else "failed"
    row = make_status_row(task, status, "", returncode, seconds, log_path)
    write_csv_row(status_path, row, fieldnames)
    print(f"TASK-END | {status} | seconds={seconds:.1f} | returncode={returncode}", flush=True)
    return row


def make_status_row(task: Task, status: str, reason: str, returncode: int, seconds: float, log_path: Path) -> dict[str, object]:
    return {
        "index": task.index,
        "name": task.name,
        "title": task.title,
        "project": task.project,
        "kind": task.kind,
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "seconds": f"{seconds:.3f}",
        "outputs": ";".join(str(path) for path in task.outputs),
        "log": str(log_path),
    }


def baseline_consistency_report() -> dict[str, object]:
    report: dict[str, object] = {
        "official_hpe_baseline_avg_mpjpe_mm": 103.70,
        "official_hpe_baseline_full_mpjpe_mm": 83.70,
        "checked_files": [],
        "warnings": [],
    }
    check_paths = [
        PROJECT_ROOT / "tables" / "main_hpe_results.csv",
        PROJECT_ROOT / "tables" / "baseline_comparison.csv",
        PROJECT_ROOT / "Paper" / "EAAI.tex",
    ]
    for path in check_paths:
        if not path.exists():
            report["warnings"].append(f"missing_file:{path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        report["checked_files"].append(str(path))
        if "244.91" in text or "244.70" in text:
            report["warnings"].append(f"weak_hpe_reproduction_value_present:{path}")
        if path.name == "EAAI.tex" and "103.70" not in text:
            report["warnings"].append(f"official_hpe_baseline_missing:{path}")
    return report


def write_manifest(tasks: list[Task], run_dirs: dict[str, Path], args: argparse.Namespace) -> None:
    baseline_report = baseline_consistency_report()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "dataset": args.dataset,
        "gpu": args.gpu,
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "dry_run": args.dry_run,
        "tasks": [
            {
                "index": task.index,
                "name": task.name,
                "title": task.title,
                "project": task.project,
                "kind": task.kind,
                "cwd": str(task.cwd),
                "command": task.command,
                "outputs": [str(path) for path in task.outputs],
                "required": [str(path) for path in task.required],
            }
            for task in tasks
        ],
        "baseline_consistency": baseline_report,
    }
    for project, run_dir in run_dirs.items():
        project_manifest = dict(manifest)
        project_manifest["tasks"] = [item for item in manifest["tasks"] if item["project"] == project]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(project_manifest, indent=2), encoding="utf-8")

    print("\nBaseline consistency check:", flush=True)
    if baseline_report["warnings"]:
        for warning in baseline_report["warnings"]:
            print(f"  WARNING | {warning}", flush=True)
    else:
        print("  OK | no weak HPE reproduction values found in checked files", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run final submission supplement experiments for VK-RMD.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true", help="Keep running later tasks after a task fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dirs = {
        "HPE": HPE_ROOT / "outputs" / "final_supplement_runs" / run_id,
        "HAR": HAR_ROOT / "outputs" / "final_supplement_runs" / run_id,
    }
    tasks = build_tasks(args)
    write_manifest(tasks, run_dirs, args)
    print("\nFinal submission supplement pipeline", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"HPE records: {run_dirs['HPE']}", flush=True)
    print(f"HAR records: {run_dirs['HAR']}", flush=True)
    print(f"GPU: {args.gpu} | device: {args.device}", flush=True)
    for task in tasks:
        row = run_task(task, run_dirs, args)
        if row["status"] == "failed" and not args.continue_on_failure:
            print("STOP | task failed; rerun after fixing the issue or pass --continue-on-failure.", flush=True)
            break
    print("\nFinal supplement records:", flush=True)
    print(f"  HPE: {run_dirs['HPE']}", flush=True)
    print(f"  HAR: {run_dirs['HAR']}", flush=True)


if __name__ == "__main__":
    main()
