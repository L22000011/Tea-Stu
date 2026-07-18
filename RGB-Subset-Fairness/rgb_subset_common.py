from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - the project env already uses yaml.
    yaml = None


SCENES = ["E01", "E02", "E03", "E04"]
ALL_ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
HPE_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
HAR_MODALITIES = ["vk", "depth", "lidar", "mmwave"]
TRAIN_SUBJECTS = [
    "S01", "S02", "S03", "S04", "S06", "S07", "S08", "S09",
    "S11", "S12", "S13", "S14", "S16", "S17", "S18", "S19",
    "S21", "S22", "S23", "S24", "S26", "S27", "S28", "S29",
    "S31", "S32", "S33", "S34", "S36", "S37", "S38", "S39",
]
VAL_SUBJECTS = ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def subject_to_scene(subject: str) -> str:
    idx = int(subject[1:])
    if 1 <= idx <= 10:
        return "E01"
    if 11 <= idx <= 20:
        return "E02"
    if 21 <= idx <= 30:
        return "E03"
    if 31 <= idx <= 40:
        return "E04"
    raise ValueError(f"Unknown subject: {subject}")


def frame_to_index(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(digits) - 1


def find_rgb_dir(action_root: Path) -> Path | None:
    for candidate in [action_root / "rgb", action_root / "RGB", action_root]:
        if candidate.exists() and candidate.is_dir():
            files = list(candidate.glob("frame*.*"))
            if files:
                return candidate
    return None


def find_frame_file(directory: Path, idx: int, preferred_exts: Iterable[str] | None = None) -> Path | None:
    number = idx + 1
    stems = [f"frame{number:03d}", f"frame{number:04d}", f"frame{number}"]
    exts = list(preferred_exts or [".npy", ".png", ".jpg", ".jpeg", ".bmp", ".bin", ".mat"])
    for stem in stems:
        for ext in exts:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path
    for path in sorted(directory.glob("frame*.*")):
        try:
            if frame_to_index(path) == idx:
                return path
        except ValueError:
            continue
    return None


def build_manifest(rgb_root: Path, mmfi_root: Path, output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "split_summary.md"
    if manifest_path.exists() and not force:
        return manifest_path

    rows: list[dict[str, str]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for scene_dir in sorted(p for p in rgb_root.iterdir() if p.is_dir() and p.name in SCENES):
        scene = scene_dir.name
        for subject_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            subject = subject_dir.name
            if not subject.startswith("S"):
                continue
            for action_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                action = action_dir.name
                if action not in ALL_ACTIONS:
                    continue
                rgb_dir = find_rgb_dir(action_dir)
                if rgb_dir is None:
                    skip("missing_rgb_dir")
                    continue
                mmfi_action_root = mmfi_root / scene / subject / action
                if not mmfi_action_root.exists():
                    skip("missing_mmfi_action")
                    continue
                required_dirs = {
                    "vk": mmfi_action_root / "rgb",
                    "depth": mmfi_action_root / "depth",
                    "lidar": mmfi_action_root / "lidar",
                    "mmwave": mmfi_action_root / "mmwave",
                    "wifi": mmfi_action_root / "wifi-csi",
                }
                if any(not p.exists() for p in required_dirs.values()):
                    skip("missing_modality_dir")
                    continue
                gt_path = mmfi_action_root / "ground_truth.npy"
                if not gt_path.exists():
                    skip("missing_gt")
                    continue

                for rgb_file in sorted(rgb_dir.glob("frame*.*")):
                    if rgb_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".npy"]:
                        continue
                    try:
                        idx = frame_to_index(rgb_file)
                    except ValueError:
                        skip("bad_rgb_frame_name")
                        continue
                    vk_path = find_frame_file(required_dirs["vk"], idx, [".npy"])
                    depth_path = find_frame_file(required_dirs["depth"], idx)
                    lidar_path = find_frame_file(required_dirs["lidar"], idx, [".bin"])
                    mmwave_path = find_frame_file(required_dirs["mmwave"], idx, [".bin"])
                    wifi_path = find_frame_file(required_dirs["wifi"], idx, [".mat"])
                    if None in [vk_path, depth_path, lidar_path, mmwave_path, wifi_path]:
                        skip("missing_frame_pair")
                        continue
                    rows.append(
                        {
                            "scene": scene,
                            "subject": subject,
                            "action": action,
                            "idx": str(idx),
                            "frame_name": rgb_file.name,
                            "rgb_path": str(rgb_file),
                            "vk_path": str(vk_path),
                            "depth_path": str(depth_path),
                            "lidar_path": str(lidar_path),
                            "mmwave_path": str(mmwave_path),
                            "wifi_path": str(wifi_path),
                            "gt_path": str(gt_path),
                        }
                    )

    fieldnames = [
        "scene", "subject", "action", "idx", "frame_name",
        "rgb_path", "vk_path", "depth_path", "lidar_path", "mmwave_path", "wifi_path", "gt_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_manifest_summary(rows, skipped, summary_path)
    return manifest_path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest_summary(rows: list[dict[str, str]], skipped: dict[str, int], path: Path) -> None:
    def count_where(fn) -> int:
        return sum(1 for row in rows if fn(row))

    lines = [
        "# RGB-Available Manifest Summary",
        "",
        f"- total_samples: {len(rows)}",
        f"- cross_scene_train_samples: {count_where(lambda r: r['scene'] in ['E01', 'E02', 'E03'])}",
        f"- cross_scene_val_samples: {count_where(lambda r: r['scene'] == 'E04')}",
        f"- cross_subject_train_samples: {count_where(lambda r: r['subject'] in TRAIN_SUBJECTS)}",
        f"- cross_subject_val_samples: {count_where(lambda r: r['subject'] in VAL_SUBJECTS)}",
        "",
        "## Per Scene",
    ]
    for scene in SCENES:
        lines.append(f"- {scene}: {count_where(lambda r, s=scene: r['scene'] == s)}")
    lines.append("")
    lines.append("## Skipped")
    if skipped:
        for key, value in sorted(skipped.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to generate configs.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to generate configs.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def make_vkrmd_config(
    base_config: Path,
    manifest: Path,
    split: str,
    output_dir: str,
    method: str,
    epochs: int | None = None,
    max_eval_batches: int | None = None,
    batch_size: int | None = None,
) -> Path:
    config = load_yaml(base_config)
    config["method"] = method
    config["split_to_use"] = split
    config["manifest_path"] = str(manifest)
    config["output_dir"] = output_dir
    config["save_every"] = 5
    if epochs is not None:
        config["training_epochs"] = int(epochs)
    if max_eval_batches is not None:
        config["max_eval_batches"] = int(max_eval_batches)
    if batch_size is not None:
        config.setdefault("loader", {})
        config["loader"]["batch_size"] = int(batch_size)
    out_path = base_config.parent / f"rgb_subset_{base_config.stem}_{split}.yaml"
    write_yaml(out_path, config)
    return out_path


def make_xfi_config(base_config: Path, manifest: Path, split: str, rgb_root: Path, output_dir: Path) -> Path:
    config = load_yaml(base_config)
    config["split_to_use"] = split
    config["manifest_path"] = str(manifest)
    config["rgb_subset_root"] = str(rgb_root)
    config["visual_source"] = "rgb"
    config["modality"] = ["vk", "depth", "lidar", "mmwave"] + (["wifi-csi"] if "wifi-csi" in config.get("modality", []) else [])
    out_path = output_dir / f"rgb_subset_{split}.yaml"
    write_yaml(out_path, config)
    return out_path


def command_to_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


@dataclass
class Task:
    name: str
    cwd: Path
    command: list[str]
    outputs: list[Path]
    required: list[Path]
    env: dict[str, str] | None = None
    completion_checkpoint: Path | None = None
    target_epoch: int | None = None


def checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[checkpoint-audit] Could not read checkpoint epoch from {path}: {exc}")
        return None
    if isinstance(checkpoint, dict):
        value = checkpoint.get("epoch")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def checkpoint_reached(path: Path, target_epoch: int | None) -> bool:
    if target_epoch is None:
        return path.exists()
    epoch = checkpoint_epoch(path)
    if epoch is None:
        return False
    return epoch >= target_epoch


def task_outputs_complete(task: Task) -> tuple[bool, str]:
    if not task.outputs or not all(path.exists() for path in task.outputs):
        return False, "missing_outputs"
    if task.completion_checkpoint is None:
        return True, "outputs_exist"
    epoch = checkpoint_epoch(task.completion_checkpoint)
    if epoch is None:
        return False, f"completion_epoch_missing:{task.completion_checkpoint}"
    if task.target_epoch is not None and epoch < task.target_epoch:
        return False, f"incomplete_epoch:{epoch}<target:{task.target_epoch}"
    return True, f"outputs_exist_epoch:{epoch}"


class PipelineLogger:
    def __init__(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.status_path = run_dir / "status.csv"
        with self.status_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "name", "status", "reason", "returncode", "seconds", "outputs"])
            writer.writeheader()

    def record(self, index: int, task: Task, status: str, reason: str, returncode: int, seconds: float) -> None:
        with self.status_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "name", "status", "reason", "returncode", "seconds", "outputs"])
            writer.writerow(
                {
                    "index": index,
                    "name": task.name,
                    "status": status,
                    "reason": reason,
                    "returncode": returncode,
                    "seconds": f"{seconds:.3f}",
                    "outputs": ";".join(str(p) for p in task.outputs),
                }
            )


def run_task(index: int, task: Task, logger: PipelineLogger, dry_run: bool = False, force: bool = False) -> None:
    print("=" * 88)
    print(f"[{index}] {task.name}")
    print("cwd=", task.cwd)
    print("command=", command_to_text(task.command))
    print("outputs=", "; ".join(str(p) for p in task.outputs))
    if task.required:
        missing = [str(path) for path in task.required if not path.exists()]
        if missing:
            reason = "missing_required=" + ",".join(missing)
            print("SKIP |", reason)
            logger.record(index, task, "skipped", reason, 0, 0.0)
            return
    if task.outputs and not force:
        complete, reason = task_outputs_complete(task)
        if complete:
            print("SKIP |", reason)
            logger.record(index, task, "skipped", reason, 0, 0.0)
            return
        if reason != "missing_outputs":
            print("CONTINUE |", reason)
    if dry_run:
        print("DRY-RUN | command not executed")
        logger.record(index, task, "dry_run", "", 0, 0.0)
        return
    for output in task.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    log_path = logger.logs_dir / f"{index:02d}_{task.name}.log"
    started = time.time()
    env = os.environ.copy()
    if task.env:
        env.update(task.env)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            task.command,
            cwd=str(task.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        returncode = proc.wait()
    seconds = time.time() - started
    status = "done" if returncode == 0 else "failed"
    logger.record(index, task, status, "", returncode, seconds)
    print(f"TASK-END | {status} | seconds={seconds:.1f} | returncode={returncode}")


def maybe_resume_arg(last_path: Path, best_path: Path) -> list[str]:
    if last_path.exists() and not best_path.exists():
        return ["--resume", str(last_path)]
    return []


def should_resume_training(last_path: Path, best_path: Path, epochs: int | None, resume_completed: bool = False) -> bool:
    if not last_path.exists():
        return False
    if resume_completed:
        return True
    if not best_path.exists():
        return True
    if epochs is None:
        return False
    target_epoch = int(epochs) - 1
    return not checkpoint_reached(last_path, target_epoch)


def copy_xfi_project(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(
            f"X-Fi source project does not exist: {src}. "
            "On cloud, expected source folders are usually /apps/.../X-Fi/Ori-HPE and /apps/.../X-Fi/Ori-HAR. "
            "Please sync the latest RGB-Subset-Fairness/run_xfi_rgb_pipeline.py so it can auto-detect them."
        )
    ignore = shutil.ignore_patterns("__pycache__", "outputs", "outputs_cross_baseline", "generated_configs", "*.pyc")
    if not dst.exists():
        shutil.copytree(src, dst, ignore=ignore, symlinks=True)
        return
    for name in ["run.py", "validate_all.py", "syn_DI_dataset.py", "utils.py", "X_Fi.py", "config.yaml"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def ensure_xfi_hpe_eval_limit_support(project: Path) -> None:
    """Patch a copied Ori-HPE runtime so validate_all supports max eval batches."""
    validate_path = project / "validate_all.py"
    utils_path = project / "utils.py"
    if not validate_path.exists() or not utils_path.exists():
        return

    validate_text = validate_path.read_text(encoding="utf-8")
    changed = False
    if "--max-eval-batches" not in validate_text:
        anchor = "    parser.add_argument('--batch-size', type=int, default=None, help='override validation batch size')\n"
        if anchor in validate_text:
            validate_text = validate_text.replace(
                anchor,
                anchor
                + "    parser.add_argument('--max-eval-batches', type=int, default=None, help='limit evaluation batches for fixed-budget validation')\n",
            )
            changed = True
    old_call = "            multi_test(model, val_loader, train_criterion, test_criterion, device, val_random_seed)\n"
    if old_call in validate_text:
        validate_text = validate_text.replace(
            old_call,
            "            multi_test(model, val_loader, train_criterion, test_criterion, device, val_random_seed, max_batches=args.max_eval_batches)\n",
        )
        changed = True
    if changed:
        validate_path.write_text(validate_text, encoding="utf-8")
        print(f"[RGB-XFi] Patched max-eval-batches support: {validate_path}")

    utils_text = utils_path.read_text(encoding="utf-8")
    utils_text, repaired = repair_xfi_hpe_training_counter_patch(utils_text)
    if repaired:
        print(f"[RGB-XFi] Repaired stale seen_samples patch in hpe_train: {utils_path}")
    changed = False
    if "def multi_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):" in utils_text:
        utils_text = utils_text.replace(
            "def multi_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):",
            "def multi_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed, max_batches=None):",
        )
        changed = True
    old_loop = "        for batch_idx, data in enumerate(tqdm(tensor_loader)):\n"
    if old_loop in utils_text and "batch_idx >= max_batches" not in utils_text:
        utils_text = utils_text.replace(
            old_loop,
            old_loop + "            if max_batches is not None and batch_idx >= max_batches:\n                break\n",
        )
        changed = True
    multi_test_start = utils_text.find("def multi_test(")
    multi_test_text = utils_text[multi_test_start:] if multi_test_start >= 0 else ""
    if "seen_samples = 0" not in multi_test_text:
        insert_at = utils_text.find("    with torch.no_grad():\n", multi_test_start)
        if insert_at >= 0:
            insert_at += len("    with torch.no_grad():\n")
            utils_text = utils_text[:insert_at] + "        seen_samples = 0\n" + utils_text[insert_at:]
            changed = True
    multi_test_start = utils_text.find("def multi_test(")
    multi_test_text = utils_text[multi_test_start:] if multi_test_start >= 0 else ""
    if "seen_samples += batch_size" not in multi_test_text and multi_test_start >= 0:
        insert_at = utils_text.find("            batch_size = vk_data.size(0)\n", multi_test_start)
        if insert_at >= 0:
            insert_at += len("            batch_size = vk_data.size(0)\n")
            utils_text = utils_text[:insert_at] + "            seen_samples += batch_size\n" + utils_text[insert_at:]
            changed = True
    if repaired:
        changed = True
    if "dataset_size = seen_samples if max_batches is not None else len(tensor_loader.dataset)" not in utils_text:
        multi_test_start = utils_text.find("def multi_test(")
        target_at = utils_text.find("    dataset_size = len(tensor_loader.dataset)\n", multi_test_start)
        if target_at >= 0:
            target_end = target_at + len("    dataset_size = len(tensor_loader.dataset)\n")
            utils_text = (
                utils_text[:target_at]
                + "    dataset_size = seen_samples if max_batches is not None else len(tensor_loader.dataset)\n"
                + utils_text[target_end:]
            )
            changed = True
    if changed:
        utils_path.write_text(utils_text, encoding="utf-8")
        print(f"[RGB-XFi] Patched fixed-budget multi_test support: {utils_path}")


def repair_xfi_hpe_training_counter_patch(text: str) -> tuple[str, bool]:
    """Remove stale fixed-budget eval counters that were accidentally inserted into training."""
    start = text.find("def hpe_train(")
    end = text.find("def multi_test(", start)
    if start < 0 or end < 0:
        return text, False
    block = text[start:end]
    cleaned_lines = []
    changed = False
    for line in block.splitlines(keepends=True):
        if line.strip() in {"seen_samples = 0", "seen_samples += batch_size"}:
            changed = True
            continue
        cleaned_lines.append(line)
    if not changed:
        return text, False
    return text[:start] + "".join(cleaned_lines) + text[end:], True


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--rgb-root", required=True, type=Path)
    parser.add_argument("--mmfi-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override generated config loader.batch_size.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the fast HPE diagnostic setting: 10 epochs, 300 train batches, 100 eval batches.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["hpe", "har"],
        default=["hpe"],
        help="Tasks to run. Default is HPE-only for the RGB-subset fairness comparison.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["cross_scene", "cross_subject"],
        default=["cross_scene", "cross_subject"],
        help="Protocols to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help=(
            "Resume from last.pth even when best.pth already exists. "
            "Use together with --force and a larger --epochs value to extend a completed short run."
        ),
    )
    parser.add_argument("--manifest-force", action="store_true")
    args = parser.parse_args()
    if args.quick:
        if args.epochs is None:
            args.epochs = 10
        args.max_train_batches = min(int(args.max_train_batches), 300)
        if args.max_eval_batches is None:
            args.max_eval_batches = 100
    return args
