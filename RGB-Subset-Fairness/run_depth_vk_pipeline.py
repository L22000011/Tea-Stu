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


def make_depth_vk_config(
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
    out_path = base_config.parent / f"depth_vk_{base_config.stem}_{split}.yaml"
    write_yaml(out_path, config)
    return out_path


def generator_command(repo: Path, manifest: Path, output_dir: Path, protocol: str, args: argparse.Namespace, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(repo / "RGB-Subset-Fairness" / "depth_vk_generator.py"),
        "--manifest",
        str(manifest),
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
        str(args.batch_size),
        "--num-workers",
        str(args.workers),
        "--image-size",
        str(args.image_size),
        "--export-all-manifest",
    ]
    if resume:
        command.append("--resume")
    if args.force:
        command += ["--force-train", "--force-export"]
    return command


def build_generator_tasks(repo: Path, manifest: Path, output_root: Path, args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for protocol in args.protocols:
        generator_dir = output_root / "generators" / protocol
        best_path = generator_dir / "best.pth"
        last_path = generator_dir / "last.pth"
        export_marker = generator_dir / "export_complete.json"
        resume = should_resume_training(last_path, best_path, args.epochs, args.resume_completed)
        tasks.append(
            Task(
                name=f"depth_vk_generator_{protocol}",
                cwd=repo,
                command=generator_command(repo, manifest, output_root, protocol, args, resume),
                outputs=[best_path, export_marker],
                required=[manifest],
                completion_checkpoint=last_path,
                target_epoch=int(args.epochs) - 1,
            )
        )
    return tasks


def protocol_to_split(label: str) -> str:
    if label == "random":
        return "random_split"
    if label == "cross_subject":
        return "cross_subject_split"
    if label == "cross_scene":
        return "cross_scene_split"
    raise ValueError(f"Unsupported protocol label: {label}")


def build_downstream_tasks(repo: Path, mmfi_root: Path, manifest: Path, output_root: Path, args: argparse.Namespace) -> list[Task]:
    hpe = first_existing(repo, ["HPE", "MMFi_HPE"])
    har = first_existing(repo, ["HAR", "MMFi_HAR"])
    print(f"[Depth-VK] HPE project: {hpe}", flush=True)
    print(f"[Depth-VK] HAR project: {har}", flush=True)
    tasks: list[Task] = []
    for label in args.protocols:
        split = protocol_to_split(label)
        generated_root = output_root / "generated_vk" / label
        export_marker = output_root / "generators" / label / "export_complete.json"
        if "hpe" in args.tasks:
            hpe_teacher_out = f"outputs_depth_vk/teacher_vk_{label}"
            hpe_student_out = f"outputs_depth_vk/student_vk_{label}"
            hpe_teacher_cfg = make_depth_vk_config(
                hpe / "configs" / "teacher_full.yaml",
                manifest,
                split,
                hpe_teacher_out,
                f"VK-RMD-HPE-DepthVK-Teacher-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            hpe_student_cfg = make_depth_vk_config(
                hpe / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                hpe_student_out,
                f"VK-RMD-HPE-DepthVK-Student-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            hpe_teacher_best = hpe / hpe_teacher_out / "best.pth"
            hpe_teacher_last = hpe / hpe_teacher_out / "last.pth"
            hpe_student_best = hpe / hpe_student_out / "best.pth"
            hpe_student_last = hpe / hpe_student_out / "last.pth"
            hpe_eval_csv = hpe / "outputs_depth_vk" / "eval" / f"student_vk_{label}_all_combinations.csv"
            tasks.append(
                Task(
                    name=f"depth_vk_hpe_teacher_{label}_train",
                    cwd=hpe,
                    command=train_command(
                        "train_teacher_full.py",
                        mmfi_root,
                        hpe_teacher_cfg,
                        args.device,
                        args.max_train_batches,
                        resume=hpe_teacher_last if should_resume_training(hpe_teacher_last, hpe_teacher_best, args.epochs, args.resume_completed) else None,
                    ),
                    outputs=[hpe_teacher_best],
                    required=[hpe_teacher_cfg, export_marker],
                    completion_checkpoint=hpe_teacher_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"depth_vk_hpe_student_{label}_train",
                    cwd=hpe,
                    command=train_command(
                        "train_student_vk_missing.py",
                        mmfi_root,
                        hpe_student_cfg,
                        args.device,
                        args.max_train_batches,
                        checkpoint=hpe_teacher_best,
                        resume=hpe_student_last if should_resume_training(hpe_student_last, hpe_student_best, args.epochs, args.resume_completed) else None,
                    ),
                    outputs=[hpe_student_best],
                    required=[hpe_teacher_best, hpe_student_cfg],
                    completion_checkpoint=hpe_student_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"depth_vk_hpe_student_{label}_eval31",
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
            har_teacher_out = f"outputs_depth_vk/teacher_vk_{label}"
            har_student_out = f"outputs_depth_vk/student_vk_{label}"
            har_teacher_cfg = make_depth_vk_config(
                har / "configs" / "teacher_full.yaml",
                manifest,
                split,
                har_teacher_out,
                f"VK-RMD-HAR-DepthVK-Teacher-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            har_student_cfg = make_depth_vk_config(
                har / "configs" / "student_vk_missing.yaml",
                manifest,
                split,
                har_student_out,
                f"VK-RMD-HAR-DepthVK-Student-{label}",
                generated_root,
                args.epochs,
                args.max_eval_batches,
                args.batch_size,
            )
            har_teacher_best = har / har_teacher_out / "best.pth"
            har_teacher_last = har / har_teacher_out / "last.pth"
            har_student_best = har / har_student_out / "best.pth"
            har_student_last = har / har_student_out / "last.pth"
            har_eval_csv = har / "outputs_depth_vk" / "eval" / f"student_vk_{label}_all_combinations.csv"
            har_eval_md = har / "outputs_depth_vk" / "eval" / f"student_vk_{label}_all_combinations.md"
            tasks.append(
                Task(
                    name=f"depth_vk_har_teacher_{label}_train",
                    cwd=har,
                    command=train_command(
                        "train_teacher_full.py",
                        mmfi_root,
                        har_teacher_cfg,
                        args.device,
                        args.max_train_batches,
                        resume=har_teacher_last if should_resume_training(har_teacher_last, har_teacher_best, args.epochs, args.resume_completed) else None,
                    ),
                    outputs=[har_teacher_best],
                    required=[har_teacher_cfg, export_marker],
                    completion_checkpoint=har_teacher_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"depth_vk_har_student_{label}_train",
                    cwd=har,
                    command=train_command(
                        "train_student_vk_missing.py",
                        mmfi_root,
                        har_student_cfg,
                        args.device,
                        args.max_train_batches,
                        checkpoint=har_teacher_best,
                        resume=har_student_last if should_resume_training(har_student_last, har_student_best, args.epochs, args.resume_completed) else None,
                    ),
                    outputs=[har_student_best],
                    required=[har_teacher_best, har_student_cfg],
                    completion_checkpoint=har_student_last,
                    target_epoch=int(args.epochs) - 1,
                )
            )
            tasks.append(
                Task(
                    name=f"depth_vk_har_student_{label}_eval15",
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


def summarize(output_root: Path, tables_dir: Path) -> None:
    rows = []
    for path in sorted((output_root / "generators").glob("*/final_eval.csv")):
        text = path.read_text(encoding="utf-8").strip().splitlines()
        if len(text) < 2:
            continue
        header = text[0].split(",")
        values = text[1].split(",")
        rows.append(dict(zip(header, values)))
    if not rows:
        print("[Depth-VK] No final_eval.csv files to summarize.", flush=True)
        return
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "depth_vk_feasibility.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Depth-VK] Summary saved: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run the lightweight Depth-to-VK feasibility pipeline.")
    parser.add_argument("--rgb-root", type=Path, default=None, help="RGB subset root used only to build/reuse the same manifest.")
    parser.add_argument("--mmfi-root", type=Path, default=None, help="MMFi root containing depth and VK files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional existing manifest.csv. If set, --rgb-root/--mmfi-root are not used to build it.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--protocols", nargs="+", choices=["random", "cross_scene", "cross_subject"], default=["random", "cross_subject"])
    parser.add_argument("--stages", nargs="+", choices=["generate", "downstream", "all"], default=["generate"])
    parser.add_argument("--tasks", nargs="+", choices=["hpe", "har"], default=["hpe", "har"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use 10 epochs, 300 train batches, and 100 eval batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 10
        args.max_train_batches = 300
        args.max_eval_batches = 100
    stages = set(args.stages)
    if "all" in stages:
        stages = {"generate", "downstream"}
    repo = repo_root_from_script()
    output_root = repo / "RGB-Subset-Fairness" / "outputs_depth_vk"
    manifest_dir = repo / "RGB-Subset-Fairness" / "outputs" / "rgb_subset_manifest"
    if args.manifest is not None:
        manifest = args.manifest
        if not manifest.exists():
            raise FileNotFoundError(f"--manifest does not exist: {manifest}")
    else:
        if args.rgb_root is None or args.mmfi_root is None:
            raise ValueError("Either pass --manifest, or pass both --rgb-root and --mmfi-root.")
        manifest = build_manifest(args.rgb_root, args.mmfi_root, manifest_dir, force=args.force_manifest)
    timestamp = datetime.now().strftime("depth_vk_%Y%m%d_%H%M%S")
    logger = PipelineLogger(
        repo / "outputs" / "depth_vk_runs" / timestamp,
        {
            "pipeline": "depth_vk_feasibility",
            "rgb_root": str(args.rgb_root) if args.rgb_root is not None else "",
            "mmfi_root": str(args.mmfi_root) if args.mmfi_root is not None else "",
            "manifest": str(manifest),
            "output_root": str(output_root),
            "protocols": args.protocols,
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "device": args.device,
        },
    )
    tasks: list[Task] = []
    if "generate" in stages:
        tasks.extend(build_generator_tasks(repo, manifest, output_root, args))
    if "downstream" in stages:
        if args.mmfi_root is None:
            raise ValueError("--mmfi-root is required for downstream stage.")
        tasks.extend(build_downstream_tasks(repo, args.mmfi_root, manifest, output_root, args))
    print("=" * 88, flush=True)
    print("[Depth-VK] Feasibility pipeline", flush=True)
    print(f"repo={repo}", flush=True)
    print(f"manifest={manifest}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"protocols={args.protocols}", flush=True)
    print(f"stages={sorted(stages)}", flush=True)
    print(f"downstream_tasks={args.tasks}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"tasks={len(tasks)}", flush=True)
    print("=" * 88, flush=True)
    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {task.name}", flush=True)
        run_task(index, task, logger, dry_run=args.dry_run, force=args.force)
    if not args.dry_run:
        summarize(output_root, repo / "tables")
    print(f"[Depth-VK] Pipeline records: {logger.run_dir}", flush=True)


if __name__ == "__main__":
    main()
