from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rgb_subset_common import (
    PipelineLogger,
    Task,
    build_manifest,
    load_yaml,
    repo_root_from_script,
    run_task,
    should_resume_training,
    write_yaml,
)


def has_mmwave_layout(path: Path) -> bool:
    return any((path / scene).exists() for scene in ["E01", "E02", "E03", "E04"])


def detect_mmwave_root(repo: Path, mmfi_root: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--mmwave-root does not exist: {override}")
        return override
    candidates = [
        Path("/apps/users/icps_intelligence/data/lyg/code/filtered_mmwave"),
        Path("/apps/users/icps_intelligence/data/lyg/code/NIPS_Data/filtered_mmwave"),
        Path("/apps/users/icps_intelligence/data/lyg/NIPS_Data/filtered_mmwave"),
        Path("/apps/users/icps_intelligence/data/lyg/X-Fi/filtered_mmwave"),
        Path("E:/Deskbook/NIPS_Data/filtered_mmwave"),
        repo / "filtered_mmwave",
        mmfi_root,
    ]
    for path in candidates:
        if path.exists() and has_mmwave_layout(path):
            return path
    raise FileNotFoundError("Could not find an mmWave source root. Pass --mmwave-root explicitly.")


def first_existing(repo: Path, names: list[str]) -> Path:
    for name in names:
        path = repo / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these project directories exist under {repo}: {names}")


def make_config(
    base_config: Path,
    manifest: Path,
    split: str,
    output_dir: str,
    method: str,
    vk_override_root: Path,
    epochs: int,
    max_eval_batches: int | None,
    batch_size: int | None,
) -> Path:
    config: dict[str, Any] = load_yaml(base_config)
    config["method"] = method
    config["split_to_use"] = split
    config["manifest_path"] = str(manifest)
    config["vk_override_root"] = str(vk_override_root)
    config["output_dir"] = output_dir
    config["save_every"] = 5
    config["training_epochs"] = int(epochs)
    if max_eval_batches is not None:
        config["max_eval_batches"] = int(max_eval_batches)
    if batch_size is not None:
        config.setdefault("loader", {})
        config["loader"]["batch_size"] = int(batch_size)
    out_path = base_config.parent / f"mmwave_vk_v2_{method.lower().replace(' ', '_').replace('/', '_')}_{split}.yaml"
    write_yaml(out_path, config)
    return out_path


def train_command(script: str, dataset: Path, config: Path, device: str, max_train_batches: int, teacher: Path | None = None, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        f"scripts/{script}",
        "--dataset",
        str(dataset),
        "--config",
        str(config),
        "--device",
        device,
        "--max-train-batches",
        str(max_train_batches),
    ]
    if teacher is not None:
        command += ["--teacher", str(teacher)]
    if resume is not None and resume.exists():
        command += ["--resume", str(resume)]
    return command


def eval_command(script: str, dataset: Path, config: Path, checkpoint: Path, device: str, output_csv: Path, max_eval_batches: int | None, output_md: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        f"scripts/{script}",
        "--dataset",
        str(dataset),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--output-csv",
        str(output_csv),
    ]
    if output_md is not None:
        command += ["--output-md", str(output_md)]
    if max_eval_batches is not None:
        command += ["--max-eval-batches", str(max_eval_batches)]
    return command


def generator_tasks(repo: Path, manifest: Path, mmwave_root: Path, output_root: Path, args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for protocol in args.protocols:
        run_dir = output_root / "generators_temporal" / protocol
        best = run_dir / "best.pth"
        last = run_dir / "last.pth"
        marker = run_dir / "export_complete.json"
        resume = should_resume_training(last, best, args.epochs, args.resume_completed)
        command = [
            sys.executable,
            "-u",
            str(repo / "RGB-Subset-Fairness" / "mmwave_vk_generator_v2.py"),
            "--manifest",
            str(manifest),
            "--mmwave-root",
            str(mmwave_root),
            "--output-dir",
            str(output_root),
            "--protocol",
            protocol,
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--max-train-batches",
            str(args.max_train_batches),
            "--max-eval-batches",
            str(args.max_eval_batches),
            "--batch-size",
            str(args.generator_batch_size),
            "--num-workers",
            str(args.generator_workers),
            "--num-points",
            str(args.num_points),
            "--window",
            str(args.window),
            "--hidden-dim",
            str(args.hidden_dim),
            "--global-stats-rows",
            str(args.global_stats_rows),
            "--export-all-manifest",
        ]
        if resume:
            command.append("--resume")
        if args.force:
            command += ["--force-train", "--force-export"]
        tasks.append(
            Task(
                name=f"temporal_mmwave_vk_generator_{protocol}",
                cwd=repo,
                command=command,
                outputs=[best, marker],
                required=[manifest],
                completion_checkpoint=last,
                target_epoch=args.epochs - 1,
            )
        )
    return tasks


def downstream_tasks(repo: Path, mmfi_root: Path, manifest: Path, output_root: Path, args: argparse.Namespace) -> list[Task]:
    hpe = first_existing(repo, ["HPE", "MMFi_HPE"])
    har = first_existing(repo, ["HAR", "MMFi_HAR"])
    protocols = [("cross_scene", "cross_scene_split"), ("cross_subject", "cross_subject_split")]
    protocols = [item for item in protocols if item[0] in args.protocols]
    tasks: list[Task] = []
    for label, split in protocols:
        marker = output_root / "generators_temporal" / label / "export_complete.json"
        generated_root = output_root / "generated_vk" / label
        suffix = f"temporal_global_local_{label}"
        if "hpe" in args.tasks:
            teacher_out = f"outputs_mmwave_vk_v2_temporal/teacher_{suffix}"
            student_out = f"outputs_mmwave_vk_v2_temporal/student_{suffix}"
            teacher_cfg = make_config(
                hpe / "configs" / "teacher_full.yaml",
                manifest,
                split,
                teacher_out,
                f"VK-RMD-HPE-Temporal-mmWave-VK-Teacher-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            student_cfg = make_config(
                hpe / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                student_out,
                f"VK-RMD-HPE-Temporal-mmWave-VK-Student-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            teacher_best = hpe / teacher_out / "best.pth"
            teacher_last = hpe / teacher_out / "last.pth"
            student_best = hpe / student_out / "best.pth"
            student_last = hpe / student_out / "last.pth"
            eval_csv = hpe / "outputs_mmwave_vk_v2_temporal" / "eval" / f"student_{suffix}_all_combinations.csv"
            tasks.append(
                Task(
                    name=f"hpe_teacher_{suffix}",
                    cwd=hpe,
                    command=train_command(
                        "train_teacher_full.py",
                        mmfi_root,
                        teacher_cfg,
                        args.device,
                        args.max_train_batches,
                        resume=teacher_last
                        if should_resume_training(teacher_last, teacher_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[teacher_best],
                    required=[teacher_cfg, marker],
                    completion_checkpoint=teacher_last,
                    target_epoch=args.epochs - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"hpe_student_{suffix}",
                    cwd=hpe,
                    command=train_command(
                        "train_student_vk_missing.py",
                        mmfi_root,
                        student_cfg,
                        args.device,
                        args.max_train_batches,
                        teacher=teacher_best,
                        resume=student_last
                        if should_resume_training(student_last, student_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[student_best],
                    required=[teacher_best, student_cfg],
                    completion_checkpoint=student_last,
                    target_epoch=args.epochs - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"hpe_eval_{suffix}",
                    cwd=hpe,
                    command=eval_command("eval_all_combinations.py", mmfi_root, student_cfg, student_best, args.device, eval_csv, args.max_eval_batches),
                    outputs=[eval_csv],
                    required=[student_best],
                )
            )
        if "har" in args.tasks:
            teacher_out = f"outputs_mmwave_vk_v2_temporal/teacher_{suffix}"
            student_out = f"outputs_mmwave_vk_v2_temporal/student_{suffix}"
            teacher_cfg = make_config(
                har / "configs" / "teacher_full.yaml",
                manifest,
                split,
                teacher_out,
                f"VK-RMD-HAR-Temporal-mmWave-VK-Teacher-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            student_cfg = make_config(
                har / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                student_out,
                f"VK-RMD-HAR-Temporal-mmWave-VK-Student-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            teacher_best = har / teacher_out / "best.pth"
            teacher_last = har / teacher_out / "last.pth"
            student_best = har / student_out / "best.pth"
            student_last = har / student_out / "last.pth"
            eval_csv = har / "outputs_mmwave_vk_v2_temporal" / "eval" / f"student_{suffix}_all_combinations.csv"
            eval_md = har / "outputs_mmwave_vk_v2_temporal" / "eval" / f"student_{suffix}_all_combinations.md"
            tasks.append(
                Task(
                    name=f"har_teacher_{suffix}",
                    cwd=har,
                    command=train_command(
                        "train_teacher_full.py",
                        mmfi_root,
                        teacher_cfg,
                        args.device,
                        args.max_train_batches,
                        resume=teacher_last
                        if should_resume_training(teacher_last, teacher_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[teacher_best],
                    required=[teacher_cfg, marker],
                    completion_checkpoint=teacher_last,
                    target_epoch=args.epochs - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"har_student_{suffix}",
                    cwd=har,
                    command=train_command(
                        "train_student_vk_missing.py",
                        mmfi_root,
                        student_cfg,
                        args.device,
                        args.max_train_batches,
                        teacher=teacher_best,
                        resume=student_last
                        if should_resume_training(student_last, student_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[student_best],
                    required=[teacher_best, student_cfg],
                    completion_checkpoint=student_last,
                    target_epoch=args.epochs - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"har_eval_{suffix}",
                    cwd=har,
                    command=eval_command("eval_all_combinations.py", mmfi_root, student_cfg, student_best, args.device, eval_csv, args.max_eval_batches, output_md=eval_md),
                    outputs=[eval_csv],
                    required=[student_best],
                )
            )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run v2 temporal global-local mmWave-generated VK pipeline.")
    parser.add_argument("--rgb-root", required=True, type=Path)
    parser.add_argument("--mmfi-root", required=True, type=Path)
    parser.add_argument("--mmwave-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--generator-batch-size", type=int, default=32)
    parser.add_argument("--generator-workers", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--global-stats-rows", type=int, default=5000)
    parser.add_argument("--tasks", nargs="+", choices=["hpe", "har"], default=["hpe", "har"])
    parser.add_argument("--protocols", nargs="+", choices=["cross_scene", "cross_subject"], default=["cross_scene", "cross_subject"])
    parser.add_argument("--stages", nargs="+", choices=["generate", "downstream"], default=["generate", "downstream"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--manifest-force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = repo_root_from_script()
    mmwave_root = detect_mmwave_root(repo, args.mmfi_root, args.mmwave_root)
    manifest = build_manifest(args.rgb_root, args.mmfi_root, repo / "outputs" / "rgb_subset_manifest", force=args.manifest_force)
    output_root = repo / "RGB-Subset-Fairness" / "outputs_mmwave_vk_v2_temporal"
    run_dir = repo / "outputs" / "mmwave_vk_v2_temporal_runs" / f"mmwave_vk_v2_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = PipelineLogger(
        run_dir,
        {
            "pipeline": "temporal_global_local_mmwave_generated_vk",
            "rgb_root": str(args.rgb_root),
            "mmfi_root": str(args.mmfi_root),
            "mmwave_root": str(mmwave_root),
            "output_root": str(output_root),
            "manifest": str(manifest),
            "device": args.device,
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "protocols": args.protocols,
            "tasks": args.tasks,
            "stages": args.stages,
        },
    )
    tasks: list[Task] = []
    if "generate" in args.stages:
        tasks.extend(generator_tasks(repo, manifest, mmwave_root, output_root, args))
    if "downstream" in args.stages:
        tasks.extend(downstream_tasks(repo, args.mmfi_root, manifest, output_root, args))
    for index, task in enumerate(tasks, start=1):
        run_task(index, task, logger, dry_run=args.dry_run, force=args.force)
    print(f"Temporal mmWave-generated VK pipeline records: {run_dir}")


if __name__ == "__main__":
    main()
