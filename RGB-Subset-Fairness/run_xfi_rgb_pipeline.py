from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from rgb_subset_common import (
    PipelineLogger,
    Task,
    build_manifest,
    copy_xfi_project,
    ensure_xfi_hpe_eval_limit_support,
    make_xfi_config,
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


def build_tasks(repo: Path, rgb_root: Path, mmfi_root: Path, manifest: Path, args) -> list[Task]:
    src_hpe = first_existing(repo, ["origin-XFI/Ori-HPE", "Ori-HPE"])
    src_har = first_existing(repo, ["origin-XFI/Ori-HAR", "Ori-HAR"])
    rgb_hpe = repo / "Ori-RGB-HPE"
    rgb_har = repo / "Ori-RGB-HAR"
    print(f"[RGB-XFi] Source HPE project: {src_hpe}")
    print(f"[RGB-XFi] Source HAR project: {src_har}")
    print(f"[RGB-XFi] Runtime HPE project: {rgb_hpe}")
    print(f"[RGB-XFi] Runtime HAR project: {rgb_har}")
    copy_xfi_project(src_hpe, rgb_hpe)
    copy_xfi_project(src_har, rgb_har)
    ensure_xfi_hpe_eval_limit_support(rgb_hpe)

    tasks: list[Task] = []
    protocols = [
        ("cross_scene", "cross_scene_split"),
        ("cross_subject", "cross_subject_split"),
    ]
    protocols = [item for item in protocols if item[0] in args.protocols]
    for label, split in protocols:
        if "hpe" in args.tasks:
            hpe_cfg = make_xfi_config(
                rgb_hpe / "config.yaml",
                manifest,
                split,
                rgb_root,
                rgb_hpe / "generated_configs",
            )
            hpe_ckpt_dir = rgb_hpe / "outputs_rgb_subset" / label / "pre-trained_weights"
            hpe_eval_dir = rgb_hpe / "outputs_rgb_subset" / label / "eval"
            hpe_best = hpe_ckpt_dir / "best.pth"
            hpe_last = hpe_ckpt_dir / "last.pth"
            hpe_train_cmd = [
                sys.executable,
                "-u",
                "run.py",
                "--dataset",
                str(mmfi_root),
                "--config",
                str(hpe_cfg),
                "--save-dir",
                str(hpe_ckpt_dir),
                "--max-train-batches",
                str(args.max_train_batches),
            ]
            if args.epochs is not None:
                hpe_train_cmd += ["--epochs", str(args.epochs)]
            if should_resume_training(hpe_last, hpe_best, args.epochs, args.resume_completed):
                hpe_train_cmd += ["--resume"]
            tasks.append(
                Task(
                    name=f"xfi_rgb_hpe_{label}_train",
                    cwd=rgb_hpe,
                    command=hpe_train_cmd,
                    outputs=[hpe_best],
                    required=[hpe_cfg],
                    env={"XFI_FIRST_BRANCH": "rgb"},
                    completion_checkpoint=hpe_last,
                    target_epoch=(int(args.epochs) - 1) if args.epochs is not None else None,
                )
            )
            hpe_eval_cmd = [
                sys.executable,
                "-u",
                str(repo / "RGB-Subset-Fairness" / "eval_xfi_hpe_limited.py"),
                "--project-root",
                str(rgb_hpe),
                "--dataset",
                str(mmfi_root),
                "--config",
                str(hpe_cfg),
                "--pt-weights",
                str(hpe_best),
                "--outputs-dir",
                str(hpe_eval_dir),
                "--device",
                args.device,
            ]
            if args.max_eval_batches is not None:
                hpe_eval_cmd += ["--max-eval-batches", str(args.max_eval_batches)]
            tasks.append(
                Task(
                    name=f"xfi_rgb_hpe_{label}_eval31",
                    cwd=rgb_hpe,
                    command=hpe_eval_cmd,
                    outputs=[hpe_eval_dir / "main_eval_latest.csv"],
                    required=[hpe_best],
                    env={"XFI_FIRST_BRANCH": "rgb"},
                )
            )

        if "har" in args.tasks:
            har_cfg = make_xfi_config(
                rgb_har / "config.yaml",
                manifest,
                split,
                rgb_root,
                rgb_har / "generated_configs",
            )
            har_ckpt_dir = rgb_har / "outputs_rgb_subset" / label / "pre-trained_weights"
            har_eval_dir = rgb_har / "outputs_rgb_subset" / label / "eval"
            har_best = har_ckpt_dir / "best.pth"
            har_last = har_ckpt_dir / "last.pth"
            har_train_cmd = [
                sys.executable,
                "-u",
                "run.py",
                "--dataset",
                str(mmfi_root),
                "--config",
                str(har_cfg),
                "--save-dir",
                str(har_ckpt_dir),
                "--max-train-batches",
                str(args.max_train_batches),
            ]
            if args.epochs is not None:
                har_train_cmd += ["--epochs", str(args.epochs)]
            if should_resume_training(har_last, har_best, args.epochs, args.resume_completed):
                har_train_cmd += ["--resume"]
            tasks.append(
                Task(
                    name=f"xfi_rgb_har_{label}_train",
                    cwd=rgb_har,
                    command=har_train_cmd,
                    outputs=[har_best],
                    required=[har_cfg],
                    env={"XFI_FIRST_BRANCH": "rgb"},
                    completion_checkpoint=har_last,
                    target_epoch=(int(args.epochs) - 1) if args.epochs is not None else None,
                )
            )
            har_eval_cmd = [
                sys.executable,
                "-u",
                "validate_all.py",
                "--dataset",
                str(mmfi_root),
                "--config",
                str(har_cfg),
                "--pt_weights",
                str(har_best),
                "--outputs-dir",
                str(har_eval_dir),
            ]
            if args.max_eval_batches is not None:
                har_eval_cmd += ["--max-eval-batches", str(args.max_eval_batches)]
            tasks.append(
                Task(
                    name=f"xfi_rgb_har_{label}_eval15",
                    cwd=rgb_har,
                    command=har_eval_cmd,
                    outputs=[har_eval_dir / "main_eval_latest.csv"],
                    required=[har_best],
                    env={"XFI_FIRST_BRANCH": "rgb"},
                )
            )
    return tasks


def main() -> None:
    args = parse_common_args("Run X-Fi RGB baseline on the RGB-available subset.")
    repo = repo_root_from_script()
    manifest_dir = repo / "outputs" / "rgb_subset_manifest"
    manifest = build_manifest(args.rgb_root, args.mmfi_root, manifest_dir, force=args.manifest_force)
    run_dir = repo / "outputs" / "rgb_subset_runs" / f"xfi_rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = PipelineLogger(
        run_dir,
        {
            "pipeline": "xfi_rgb_subset_baseline",
            "rgb_root": str(args.rgb_root),
            "mmfi_root": str(args.mmfi_root),
            "manifest": str(manifest),
            "device": args.device,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        },
    )
    tasks = build_tasks(repo, args.rgb_root, args.mmfi_root, manifest, args)
    for index, task in enumerate(tasks, start=1):
        run_task(index, task, logger, dry_run=args.dry_run, force=args.force)
    print(f"X-Fi RGB subset pipeline records: {run_dir}")


if __name__ == "__main__":
    main()
