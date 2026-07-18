from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from rgb_subset_common import (
    PipelineLogger,
    Task,
    build_manifest,
    make_vkrmd_config,
    parse_common_args,
    repo_root_from_script,
    run_task,
    should_resume_training,
)


def first_existing(repo: Path, names: list[str]) -> Path:
    for name in names:
        path = repo / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these project directories exist under {repo}: {names}")


def train_command(project: Path, script: str, dataset: Path, config: Path, device: str, max_train_batches: int, checkpoint: Path | None = None, resume: Path | None = None) -> list[str]:
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
    if checkpoint is not None:
        command += ["--teacher", str(checkpoint)]
    if resume is not None and resume.exists():
        command += ["--resume", str(resume)]
    return command


def eval_command(project: Path, script: str, dataset: Path, config: Path, checkpoint: Path, device: str, output_csv: Path, max_eval_batches: int | None, output_md: Path | None = None) -> list[str]:
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


def build_tasks(repo: Path, mmfi_root: Path, manifest: Path, args) -> list[Task]:
    hpe = first_existing(repo, ["HPE", "MMFi_HPE"])
    har = first_existing(repo, ["HAR", "MMFi_HAR"])
    print(f"[VK-RMD] HPE project: {hpe}")
    print(f"[VK-RMD] HAR project: {har}")
    tasks: list[Task] = []
    protocols = [
        ("cross_scene", "cross_scene_split"),
        ("cross_subject", "cross_subject_split"),
    ]
    protocols = [item for item in protocols if item[0] in args.protocols]
    for label, split in protocols:
        if "hpe" in args.tasks:
            hpe_teacher_out = f"outputs_rgb_subset/teacher_vk_{label}"
            hpe_student_out = f"outputs_rgb_subset/student_vk_{label}"
            hpe_teacher_cfg = make_vkrmd_config(
                hpe / "configs" / "teacher_full.yaml",
                manifest,
                split,
                hpe_teacher_out,
                f"VK-RMD-HPE-RGBSubset-Teacher-{label}",
                epochs=args.epochs,
                max_eval_batches=args.max_eval_batches,
                batch_size=args.batch_size,
            )
            hpe_student_cfg = make_vkrmd_config(
                hpe / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                hpe_student_out,
                f"VK-RMD-HPE-RGBSubset-StudentVK-{label}",
                epochs=args.epochs,
                max_eval_batches=args.max_eval_batches,
                batch_size=args.batch_size,
            )
            hpe_teacher_best = hpe / hpe_teacher_out / "best.pth"
            hpe_teacher_last = hpe / hpe_teacher_out / "last.pth"
            hpe_student_best = hpe / hpe_student_out / "best.pth"
            hpe_student_last = hpe / hpe_student_out / "last.pth"
            hpe_eval_csv = hpe / "outputs_rgb_subset" / "eval" / f"student_vk_{label}_all_combinations.csv"
            tasks.append(
                Task(
                    name=f"vkrmd_hpe_teacher_{label}_train",
                    cwd=hpe,
                    command=train_command(hpe, "train_teacher_full.py", mmfi_root, hpe_teacher_cfg, args.device, args.max_train_batches, resume=hpe_teacher_last if should_resume_training(hpe_teacher_last, hpe_teacher_best, args.epochs, args.resume_completed) else None),
                    outputs=[hpe_teacher_best],
                    required=[hpe_teacher_cfg],
                    completion_checkpoint=hpe_teacher_last,
                    target_epoch=(int(args.epochs) - 1) if args.epochs is not None else None,
                )
            )
            tasks.append(
                Task(
                    name=f"vkrmd_hpe_student_{label}_train",
                    cwd=hpe,
                    command=train_command(hpe, "train_student_vk_missing.py", mmfi_root, hpe_student_cfg, args.device, args.max_train_batches, checkpoint=hpe_teacher_best, resume=hpe_student_last if should_resume_training(hpe_student_last, hpe_student_best, args.epochs, args.resume_completed) else None),
                    outputs=[hpe_student_best],
                    required=[hpe_teacher_best, hpe_student_cfg],
                    completion_checkpoint=hpe_student_last,
                    target_epoch=(int(args.epochs) - 1) if args.epochs is not None else None,
                )
            )
            tasks.append(
                Task(
                    name=f"vkrmd_hpe_student_{label}_eval31",
                    cwd=hpe,
                    command=eval_command(hpe, "eval_all_combinations.py", mmfi_root, hpe_student_cfg, hpe_student_best, args.device, hpe_eval_csv, args.max_eval_batches),
                    outputs=[hpe_eval_csv],
                    required=[hpe_student_best],
                )
            )

        if "har" in args.tasks:
            har_teacher_out = f"outputs_rgb_subset/teacher_vk_{label}"
            har_student_out = f"outputs_rgb_subset/student_vk_{label}"
            har_teacher_cfg = make_vkrmd_config(
                har / "configs" / "teacher_full.yaml",
                manifest,
                split,
                har_teacher_out,
                f"VK-RMD-HAR-RGBSubset-Teacher-{label}",
                epochs=args.epochs,
                max_eval_batches=args.max_eval_batches,
                batch_size=args.batch_size,
            )
            har_student_cfg = make_vkrmd_config(
                har / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                har_student_out,
                f"VK-RMD-HAR-RGBSubset-StudentVK-{label}",
                epochs=args.epochs,
                max_eval_batches=args.max_eval_batches,
                batch_size=args.batch_size,
            )
            har_teacher_best = har / har_teacher_out / "best.pth"
            har_teacher_last = har / har_teacher_out / "last.pth"
            har_student_best = har / har_student_out / "best.pth"
            har_student_last = har / har_student_out / "last.pth"
            har_eval_csv = har / "outputs_rgb_subset" / "eval" / f"student_vk_{label}_all_combinations.csv"
            har_eval_md = har / "outputs_rgb_subset" / "eval" / f"student_vk_{label}_all_combinations.md"
            tasks.append(
                Task(
                    name=f"vkrmd_har_teacher_{label}_train",
                    cwd=har,
                    command=train_command(har, "train_teacher_full.py", mmfi_root, har_teacher_cfg, args.device, args.max_train_batches, resume=har_teacher_last if should_resume_training(har_teacher_last, har_teacher_best, args.epochs, args.resume_completed) else None),
                    outputs=[har_teacher_best],
                    required=[har_teacher_cfg],
                    completion_checkpoint=har_teacher_last,
                    target_epoch=(int(args.epochs) - 1) if args.epochs is not None else None,
                )
            )
            tasks.append(
                Task(
                    name=f"vkrmd_har_student_{label}_train",
                    cwd=har,
                    command=train_command(har, "train_student_vk_missing.py", mmfi_root, har_student_cfg, args.device, args.max_train_batches, checkpoint=har_teacher_best, resume=har_student_last if should_resume_training(har_student_last, har_student_best, args.epochs, args.resume_completed) else None),
                    outputs=[har_student_best],
                    required=[har_teacher_best, har_student_cfg],
                    completion_checkpoint=har_student_last,
                    target_epoch=(int(args.epochs) - 1) if args.epochs is not None else None,
                )
            )
            tasks.append(
                Task(
                    name=f"vkrmd_har_student_{label}_eval15",
                    cwd=har,
                    command=eval_command(har, "eval_all_combinations.py", mmfi_root, har_student_cfg, har_student_best, args.device, har_eval_csv, args.max_eval_batches, output_md=har_eval_md),
                    outputs=[har_eval_csv],
                    required=[har_student_best],
                )
            )
    return tasks


def main() -> None:
    args = parse_common_args("Run VK-RMD on the same RGB-available subset.")
    repo = repo_root_from_script()
    manifest_dir = repo / "outputs" / "rgb_subset_manifest"
    manifest = build_manifest(args.rgb_root, args.mmfi_root, manifest_dir, force=args.manifest_force)
    run_dir = repo / "outputs" / "rgb_subset_runs" / f"vkrmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = PipelineLogger(
        run_dir,
        {
            "pipeline": "vkrmd_rgb_subset_matched",
            "rgb_root": str(args.rgb_root),
            "mmfi_root": str(args.mmfi_root),
            "manifest": str(manifest),
            "device": args.device,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        },
    )
    tasks = build_tasks(repo, args.mmfi_root, manifest, args)
    for index, task in enumerate(tasks, start=1):
        run_task(index, task, logger, dry_run=args.dry_run, force=args.force)
    print(f"VK-RMD RGB subset pipeline records: {run_dir}")


if __name__ == "__main__":
    main()
