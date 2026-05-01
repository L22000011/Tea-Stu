from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    name: str
    command: List[str]
    outputs: tuple[Path, ...]
    required_inputs: tuple[Path, ...] = ()


def rel(path: str) -> Path:
    return PROJECT_ROOT / path


def checkpoint(path: str) -> Path:
    return rel(path) / "best.pth"


def add_common_train_args(command: List[str], args: argparse.Namespace, config: str, output_dir: str) -> List[str]:
    command.extend([
        "--dataset", args.dataset,
        "--config", config,
        "--device", args.device,
        "--output-dir", output_dir,
    ])
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    return command


def add_common_eval_args(command: List[str], args: argparse.Namespace, config: str, checkpoint_path: str) -> List[str]:
    command.extend([
        "--dataset", args.dataset,
        "--config", config,
        "--checkpoint", checkpoint_path,
        "--device", args.device,
    ])
    if args.max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    return command


def train_teacher(args: argparse.Namespace, name: str, config: str, output_dir: str) -> Step:
    cmd = [sys.executable, "-u", str(rel("scripts/train_teacher_full.py"))]
    add_common_train_args(cmd, args, config, output_dir)
    return Step(name=name, command=cmd, outputs=(checkpoint(output_dir),))


def train_student(args: argparse.Namespace, name: str, script: str, config: str, teacher: str, output_dir: str) -> Step:
    cmd = [sys.executable, "-u", str(rel(script))]
    add_common_train_args(cmd, args, config, output_dir)
    cmd.extend(["--teacher", teacher])
    return Step(
        name=name,
        command=cmd,
        outputs=(checkpoint(output_dir),),
        required_inputs=(rel(teacher),),
    )


def train_simple(args: argparse.Namespace, name: str, script: str, config: str, output_dir: str) -> Step:
    cmd = [sys.executable, "-u", str(rel(script))]
    add_common_train_args(cmd, args, config, output_dir)
    return Step(name=name, command=cmd, outputs=(checkpoint(output_dir),))


def eval_all(args: argparse.Namespace, name: str, config: str, ckpt: str, stem: str) -> Step:
    csv_path = rel(f"outputs/eval/{stem}.csv")
    md_path = rel(f"outputs/eval/{stem}.md")
    cmd = [sys.executable, "-u", str(rel("scripts/eval_all_combinations.py"))]
    add_common_eval_args(cmd, args, config, ckpt)
    cmd.extend(["--output-csv", str(csv_path), "--output-md", str(md_path)])
    return Step(name=name, command=cmd, outputs=(csv_path, md_path), required_inputs=(rel(ckpt),))


def eval_nonvisual(args: argparse.Namespace, name: str, config: str, ckpt: str, stem: str) -> Step:
    csv_path = rel(f"outputs/eval/{stem}.csv")
    md_path = rel(f"outputs/eval/{stem}.md")
    cmd = [sys.executable, "-u", str(rel("scripts/eval_nonvisual_combinations.py"))]
    add_common_eval_args(cmd, args, config, ckpt)
    cmd.extend(["--output-csv", str(csv_path), "--output-md", str(md_path)])
    return Step(name=name, command=cmd, outputs=(csv_path, md_path), required_inputs=(rel(ckpt),))


def random_steps(args: argparse.Namespace) -> list[Step]:
    return [
        train_teacher(args, "random.teacher", "configs/teacher_full.yaml", "outputs/teacher_full"),
        eval_all(args, "random.teacher.eval15", "configs/teacher_full.yaml", "outputs/teacher_full/best.pth", "teacher_random_all_combinations"),
        train_student(args, "random.student_vk", "scripts/train_student_vk_missing.py", "configs/student_vk_missing.yaml", "outputs/teacher_full/best.pth", "outputs/student_vk_missing"),
        eval_all(args, "random.student_vk.eval15", "configs/student_vk_missing.yaml", "outputs/student_vk_missing/best.pth", "student_vk_random_all_combinations"),
        train_student(args, "random.student_nv", "scripts/train_student_nv_missing.py", "configs/student_nv_missing.yaml", "outputs/teacher_full/best.pth", "outputs/student_nv_missing"),
        eval_nonvisual(args, "random.student_nv.eval7", "configs/student_nv_missing.yaml", "outputs/student_nv_missing/best.pth", "student_nv_random_nonvisual_combinations"),
    ]


def ablation_steps(args: argparse.Namespace) -> list[Step]:
    steps = [
        train_simple(args, "ablation.no_distill", "scripts/ablate_no_distillation.py", "configs/ablation_no_distill.yaml", "outputs/ablation_no_distill"),
        eval_all(args, "ablation.no_distill.eval15", "configs/ablation_no_distill.yaml", "outputs/ablation_no_distill/best.pth", "ablation_no_distill_all_combinations"),
        train_simple(args, "ablation.uniform_fusion", "scripts/ablate_uniform_fusion.py", "configs/ablation_uniform_fusion.yaml", "outputs/ablation_uniform_fusion"),
        eval_all(args, "ablation.uniform_fusion.eval15", "configs/ablation_uniform_fusion.yaml", "outputs/ablation_uniform_fusion/best.pth", "ablation_uniform_fusion_all_combinations"),
    ]
    if args.include_full_baseline:
        steps.extend([
            train_simple(args, "ablation.full_baseline", "scripts/train_baseline_full.py", "configs/baseline_full.yaml", "outputs/baseline_full"),
            eval_all(args, "ablation.full_baseline.eval15", "configs/baseline_full.yaml", "outputs/baseline_full/best.pth", "baseline_full_all_combinations"),
        ])
    return steps


def cross_scene_steps(args: argparse.Namespace) -> list[Step]:
    teacher = "outputs/teacher_full_cross_scene/best.pth"
    return [
        train_teacher(args, "cross_scene.teacher", "configs/teacher_full_cross_scene.yaml", "outputs/teacher_full_cross_scene"),
        eval_all(args, "cross_scene.teacher.eval15", "configs/teacher_full_cross_scene.yaml", teacher, "teacher_cross_scene_all_combinations"),
        train_student(args, "cross_scene.student_vk", "scripts/train_student_vk_missing.py", "configs/student_vk_missing_cross_scene.yaml", teacher, "outputs/student_vk_missing_cross_scene"),
        eval_all(args, "cross_scene.student_vk.eval15", "configs/student_vk_missing_cross_scene.yaml", "outputs/student_vk_missing_cross_scene/best.pth", "student_vk_cross_scene_all_combinations"),
        train_student(args, "cross_scene.student_nv", "scripts/train_student_nv_missing.py", "configs/student_nv_missing_cross_scene.yaml", teacher, "outputs/student_nv_missing_cross_scene"),
        eval_nonvisual(args, "cross_scene.student_nv.eval7", "configs/student_nv_missing_cross_scene.yaml", "outputs/student_nv_missing_cross_scene/best.pth", "student_nv_cross_scene_nonvisual_combinations"),
    ]


def cross_subject_steps(args: argparse.Namespace) -> list[Step]:
    teacher = "outputs/teacher_full_cross_subject/best.pth"
    return [
        train_teacher(args, "cross_subject.teacher", "configs/teacher_full_cross_subject.yaml", "outputs/teacher_full_cross_subject"),
        eval_all(args, "cross_subject.teacher.eval15", "configs/teacher_full_cross_subject.yaml", teacher, "teacher_cross_subject_all_combinations"),
        train_student(args, "cross_subject.student_vk", "scripts/train_student_vk_missing.py", "configs/student_vk_missing_cross_subject.yaml", teacher, "outputs/student_vk_missing_cross_subject"),
        eval_all(args, "cross_subject.student_vk.eval15", "configs/student_vk_missing_cross_subject.yaml", "outputs/student_vk_missing_cross_subject/best.pth", "student_vk_cross_subject_all_combinations"),
        train_student(args, "cross_subject.student_nv", "scripts/train_student_nv_missing.py", "configs/student_nv_missing_cross_subject.yaml", teacher, "outputs/student_nv_missing_cross_subject"),
        eval_nonvisual(args, "cross_subject.student_nv.eval7", "configs/student_nv_missing_cross_subject.yaml", "outputs/student_nv_missing_cross_subject/best.pth", "student_nv_cross_subject_nonvisual_combinations"),
    ]


def selected_steps(args: argparse.Namespace) -> list[Step]:
    groups = {
        "random": random_steps,
        "ablation": ablation_steps,
        "cross_scene": cross_scene_steps,
        "cross_subject": cross_subject_steps,
    }
    if args.stage == "all":
        order = ["random", "ablation", "cross_scene", "cross_subject"]
    else:
        order = [args.stage]
    steps: list[Step] = []
    for key in order:
        steps.extend(groups[key](args))
    return steps


def outputs_exist(outputs: Iterable[Path]) -> bool:
    return all(path.exists() for path in outputs)


def run_step(step: Step, args: argparse.Namespace, log_dir: Path) -> int:
    print("=" * 80)
    label = "DryRun" if args.dry_run else "Run"
    print(f"[{label}] {step.name}")
    print(" ".join(str(part) for part in step.command))
    print("=" * 80)
    if args.dry_run:
        return 0
    missing_inputs = [path for path in step.required_inputs if not path.exists()]
    if missing_inputs:
        print(f"[Missing] {step.name}: required input not found")
        for path in missing_inputs:
            print(f"  - {path}")
        return 2
    if outputs_exist(step.outputs) and not args.force:
        print(f"[Skip] {step.name}: outputs already exist")
        for path in step.outputs:
            print(f"  - {path}")
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.name.replace('.', '_')}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            step.command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
    print(f"[Done] {step.name}: return_code={process.returncode} | log={log_path}")
    return process.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run the remaining HAR experiments in the HPE-style pipeline.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stage", choices=["all", "random", "ablation", "cross_scene", "cross_subject"], default="all")
    parser.add_argument("--gpu", type=str, default="0", help="Physical GPU id passed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device inside the visible GPU set.")
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun a step even when its outputs already exist.")
    parser.add_argument("--include-full-baseline", action="store_true", help="Also run the full-modality no-KD baseline ablation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = rel(f"outputs/pipeline_runs/{stamp}")
    steps = selected_steps(args)
    print(f"[Pipeline] stage={args.stage} | steps={len(steps)} | dataset={args.dataset}")
    print(f"[Pipeline] gpu={args.gpu} | device={args.device} | max_train_batches={args.max_train_batches}")
    if args.max_eval_batches is None:
        print("[Pipeline] eval=full validation set")
    else:
        print(f"[Pipeline] max_eval_batches={args.max_eval_batches}")
    failed: list[tuple[str, int]] = []
    for step in steps:
        code = run_step(step, args, log_dir)
        if code != 0:
            failed.append((step.name, code))
            if not args.continue_on_error:
                break
    if failed:
        print("[Pipeline] Failed steps:")
        for name, code in failed:
            print(f"  - {name}: return_code={code}")
        raise SystemExit(1)
    print("[Pipeline] All selected steps finished or were skipped.")


if __name__ == "__main__":
    main()



