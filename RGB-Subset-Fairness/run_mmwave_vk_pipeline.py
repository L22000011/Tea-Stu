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


def first_existing(repo: Path, names: list[str]) -> Path:
    for name in names:
        path = repo / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these project directories exist under {repo}: {names}")


def has_mmwave_layout(path: Path) -> bool:
    return (path / "E01").exists() or (path / "E02").exists() or (path / "E03").exists() or (path / "E04").exists()


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
    raise FileNotFoundError(
        "Could not find an mmWave source root. Pass --mmwave-root explicitly. "
        "Supported layouts are <root>/E/S/A/frameXXX.bin and <root>/E/S/A/mmwave/frameXXX.bin. Checked: "
        + "; ".join(str(path) for path in candidates)
    )


def train_command(
    script: str,
    dataset: Path,
    config: Path,
    device: str,
    max_train_batches: int,
    checkpoint: Path | None = None,
    resume: Path | None = None,
) -> list[str]:
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


def eval_command(
    script: str,
    dataset: Path,
    config: Path,
    checkpoint: Path,
    device: str,
    output_csv: Path,
    max_eval_batches: int | None,
    output_md: Path | None = None,
) -> list[str]:
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


def make_mmwave_vk_config(
    base_config: Path,
    manifest: Path,
    split: str,
    output_dir: str,
    method: str,
    vk_override_root: Path,
    epochs: int | None,
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
    if epochs is not None:
        config["training_epochs"] = int(epochs)
    if max_eval_batches is not None:
        config["max_eval_batches"] = int(max_eval_batches)
    if batch_size is not None:
        config.setdefault("loader", {})
        config["loader"]["batch_size"] = int(batch_size)
    out_path = base_config.parent / f"mmwave_vk_{base_config.stem}_{split}.yaml"
    write_yaml(out_path, config)
    return out_path


def generator_command(
    repo: Path,
    manifest: Path,
    mmwave_root: Path,
    output_dir: Path,
    protocol: str,
    args: argparse.Namespace,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(repo / "RGB-Subset-Fairness" / "mmwave_vk_generator.py"),
        "--manifest",
        str(manifest),
        "--mmwave-root",
        str(mmwave_root),
        "--output-dir",
        str(output_dir),
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
        "--export-all-manifest",
    ]
    if resume:
        command.append("--resume")
    if args.force:
        command += ["--force-train", "--force-export"]
    return command


def add_generator_tasks(repo: Path, manifest: Path, mmwave_root: Path, output_root: Path, args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for protocol in args.protocols:
        generator_dir = output_root / "generators" / protocol
        best_path = generator_dir / "best.pth"
        last_path = generator_dir / "last.pth"
        export_marker = generator_dir / "export_complete.json"
        resume = should_resume_training(last_path, best_path, args.epochs, args.resume_completed)
        tasks.append(
            Task(
                name=f"mmwave_vk_generator_{protocol}",
                cwd=repo,
                command=generator_command(repo, manifest, mmwave_root, output_root, protocol, args, resume),
                outputs=[best_path, export_marker],
                required=[manifest],
                completion_checkpoint=last_path,
                target_epoch=int(args.epochs) - 1,
            )
        )
    return tasks


def build_downstream_tasks(repo: Path, mmfi_root: Path, manifest: Path, output_root: Path, args: argparse.Namespace) -> list[Task]:
    hpe = first_existing(repo, ["HPE", "MMFi_HPE"])
    har = first_existing(repo, ["HAR", "MMFi_HAR"])
    print(f"[mmWave-VK] HPE project: {hpe}")
    print(f"[mmWave-VK] HAR project: {har}")
    tasks: list[Task] = []
    protocols = [
        ("cross_scene", "cross_scene_split"),
        ("cross_subject", "cross_subject_split"),
    ]
    protocols = [item for item in protocols if item[0] in args.protocols]
    for label, split in protocols:
        generated_root = output_root / "generated_vk" / label
        export_marker = output_root / "generators" / label / "export_complete.json"
        if "hpe" in args.tasks:
            hpe_teacher_out = f"outputs_mmwave_vk/teacher_vk_{label}"
            hpe_student_out = f"outputs_mmwave_vk/student_vk_{label}"
            hpe_teacher_cfg = make_mmwave_vk_config(
                hpe / "configs" / "teacher_full.yaml",
                manifest,
                split,
                hpe_teacher_out,
                f"VK-RMD-HPE-mmWaveVK-Teacher-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            hpe_student_cfg = make_mmwave_vk_config(
                hpe / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                hpe_student_out,
                f"VK-RMD-HPE-mmWaveVK-Student-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            hpe_teacher_best = hpe / hpe_teacher_out / "best.pth"
            hpe_teacher_last = hpe / hpe_teacher_out / "last.pth"
            hpe_student_best = hpe / hpe_student_out / "best.pth"
            hpe_student_last = hpe / hpe_student_out / "last.pth"
            hpe_eval_csv = hpe / "outputs_mmwave_vk" / "eval" / f"student_vk_{label}_all_combinations.csv"
            tasks.append(
                Task(
                    name=f"mmwave_vk_hpe_teacher_{label}_train",
                    cwd=hpe,
                    command=train_command(
                        "train_teacher_full.py",
                        mmfi_root,
                        hpe_teacher_cfg,
                        args.device,
                        args.max_train_batches,
                        resume=hpe_teacher_last
                        if should_resume_training(hpe_teacher_last, hpe_teacher_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[hpe_teacher_best],
                    required=[hpe_teacher_cfg, export_marker],
                    completion_checkpoint=hpe_teacher_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"mmwave_vk_hpe_student_{label}_train",
                    cwd=hpe,
                    command=train_command(
                        "train_student_vk_missing.py",
                        mmfi_root,
                        hpe_student_cfg,
                        args.device,
                        args.max_train_batches,
                        checkpoint=hpe_teacher_best,
                        resume=hpe_student_last
                        if should_resume_training(hpe_student_last, hpe_student_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[hpe_student_best],
                    required=[hpe_teacher_best, hpe_student_cfg],
                    completion_checkpoint=hpe_student_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"mmwave_vk_hpe_student_{label}_eval31",
                    cwd=hpe,
                    command=eval_command(
                        "eval_all_combinations.py",
                        mmfi_root,
                        hpe_student_cfg,
                        hpe_student_best,
                        args.device,
                        hpe_eval_csv,
                        args.max_eval_batches,
                    ),
                    outputs=[hpe_eval_csv],
                    required=[hpe_student_best],
                )
            )

        if "har" in args.tasks:
            har_teacher_out = f"outputs_mmwave_vk/teacher_vk_{label}"
            har_student_out = f"outputs_mmwave_vk/student_vk_{label}"
            har_teacher_cfg = make_mmwave_vk_config(
                har / "configs" / "teacher_full.yaml",
                manifest,
                split,
                har_teacher_out,
                f"VK-RMD-HAR-mmWaveVK-Teacher-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            har_student_cfg = make_mmwave_vk_config(
                har / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                har_student_out,
                f"VK-RMD-HAR-mmWaveVK-Student-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            har_teacher_best = har / har_teacher_out / "best.pth"
            har_teacher_last = har / har_teacher_out / "last.pth"
            har_student_best = har / har_student_out / "best.pth"
            har_student_last = har / har_student_out / "last.pth"
            har_eval_csv = har / "outputs_mmwave_vk" / "eval" / f"student_vk_{label}_all_combinations.csv"
            har_eval_md = har / "outputs_mmwave_vk" / "eval" / f"student_vk_{label}_all_combinations.md"
            tasks.append(
                Task(
                    name=f"mmwave_vk_har_teacher_{label}_train",
                    cwd=har,
                    command=train_command(
                        "train_teacher_full.py",
                        mmfi_root,
                        har_teacher_cfg,
                        args.device,
                        args.max_train_batches,
                        resume=har_teacher_last
                        if should_resume_training(har_teacher_last, har_teacher_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[har_teacher_best],
                    required=[har_teacher_cfg, export_marker],
                    completion_checkpoint=har_teacher_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"mmwave_vk_har_student_{label}_train",
                    cwd=har,
                    command=train_command(
                        "train_student_vk_missing.py",
                        mmfi_root,
                        har_student_cfg,
                        args.device,
                        args.max_train_batches,
                        checkpoint=har_teacher_best,
                        resume=har_student_last
                        if should_resume_training(har_student_last, har_student_best, args.epochs, args.resume_completed)
                        else None,
                    ),
                    outputs=[har_student_best],
                    required=[har_teacher_best, har_student_cfg],
                    completion_checkpoint=har_student_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"mmwave_vk_har_student_{label}_eval15",
                    cwd=har,
                    command=eval_command(
                        "eval_all_combinations.py",
                        mmfi_root,
                        har_student_cfg,
                        har_student_best,
                        args.device,
                        har_eval_csv,
                        args.max_eval_batches,
                        output_md=har_eval_md,
                    ),
                    outputs=[har_eval_csv],
                    required=[har_student_best],
                )
            )
    return tasks


def add_summary_task(repo: Path) -> Task:
    return Task(
        name="mmwave_vk_feasibility_summary",
        cwd=repo,
        command=[sys.executable, "-u", str(repo / "RGB-Subset-Fairness" / "summarize_mmwave_vk_feasibility.py"), "--repo", str(repo)],
        outputs=[],
        required=[],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run mmWave-generated VK feasibility pipeline.")
    parser.add_argument("--rgb-root", required=True, type=Path)
    parser.add_argument("--mmfi-root", required=True, type=Path)
    parser.add_argument("--mmwave-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override generated HPE/HAR config loader.batch_size.")
    parser.add_argument("--generator-batch-size", type=int, default=64)
    parser.add_argument("--generator-workers", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick feasibility settings: 10 epochs, 300 train batches, 100 eval batches.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["hpe", "har"],
        default=["hpe", "har"],
        help="Downstream tasks to run. Default: both HPE and HAR.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["cross_scene", "cross_subject"],
        default=["cross_scene", "cross_subject"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["generate", "downstream", "summary"],
        default=["generate", "downstream", "summary"],
        help=(
            "Pipeline stages to run. Use 'generate' first to export mmWave-VK, "
            "then run 'downstream' on different GPUs per protocol, and finally 'summary'."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Resume from last.pth even when best.pth exists; useful when extending a completed quick run.",
    )
    parser.add_argument("--manifest-force", action="store_true")
    args = parser.parse_args()
    if args.quick:
        if args.epochs is None:
            args.epochs = 10
        args.max_train_batches = min(int(args.max_train_batches), 300)
        if args.max_eval_batches is None:
            args.max_eval_batches = 100
    if args.epochs is None:
        args.epochs = 10
    return args


def main() -> None:
    args = parse_args()
    repo = repo_root_from_script()
    mmwave_root = detect_mmwave_root(repo, args.mmfi_root, args.mmwave_root)
    manifest_dir = repo / "outputs" / "rgb_subset_manifest"
    manifest = build_manifest(args.rgb_root, args.mmfi_root, manifest_dir, force=args.manifest_force)
    output_root = repo / "RGB-Subset-Fairness" / "outputs_mmwave_vk"
    run_dir = repo / "outputs" / "mmwave_vk_runs" / f"mmwave_vk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = PipelineLogger(
        run_dir,
        {
            "pipeline": "mmwave_generated_vk_feasibility",
            "rgb_root": str(args.rgb_root),
            "mmfi_root": str(args.mmfi_root),
            "mmwave_root": str(mmwave_root),
            "manifest": str(manifest),
            "output_root": str(output_root),
            "device": args.device,
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "tasks": args.tasks,
            "protocols": args.protocols,
        },
    )
    tasks = []
    if "generate" in args.stages:
        tasks.extend(add_generator_tasks(repo, manifest, mmwave_root, output_root, args))
    if "downstream" in args.stages:
        tasks.extend(build_downstream_tasks(repo, args.mmfi_root, manifest, output_root, args))
    if "summary" in args.stages:
        tasks.append(add_summary_task(repo))
    for index, task in enumerate(tasks, start=1):
        run_task(index, task, logger, dry_run=args.dry_run, force=args.force)
    print(f"mmWave-generated VK pipeline records: {run_dir}")


if __name__ == "__main__":
    main()
