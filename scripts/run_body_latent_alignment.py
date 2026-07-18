from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_dir(*candidates: str) -> Path:
    for name in candidates:
        path = PROJECT_ROOT / name
        if path.exists():
            return path
    return PROJECT_ROOT / candidates[0]


HPE_ROOT = resolve_project_dir("HPE", "MMFi_HPE")
HAR_ROOT = resolve_project_dir("HAR", "MMFi_HAR")
HPE_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
HAR_MODALITIES = ["vk", "depth", "lidar", "mmwave"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "task",
        "anchor_modality",
        "target_modality",
        "same_sample_cosine",
        "shuffled_cosine",
        "cosine_gap",
        "retrieval_top1",
        "random_top1",
        "num_samples",
        "num_batches",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def existing_checkpoint(project_root: Path) -> Path:
    for relative in ["outputs/student_vk_missing/best.pth", "outputs/student_vk_missing/last.pth"]:
        path = project_root / relative
        if path.exists():
            return path
    return project_root / "outputs/student_vk_missing/best.pth"


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> tuple[str, float, str]:
    print("=" * 88, flush=True)
    print(f"TASK-COMMAND | cwd={cwd}", flush=True)
    print(" ".join(command), flush=True)
    print(f"TASK-LOG | {log_path}", flush=True)
    print("=" * 88, flush=True)
    if dry_run:
        return "dry_run", 0.0, ""
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        process.wait()
    seconds = time.time() - start
    if process.returncode != 0:
        return "failed", seconds, f"returncode={process.returncode}; log={log_path}"
    return "done", seconds, ""


@dataclass
class Record:
    task: str
    status: str
    seconds: float
    output: str
    reason: str = ""


def write_status(path: Path, records: list[Record]) -> None:
    rows = [
        {
            "task": record.task,
            "status": record.status,
            "seconds": f"{record.seconds:.3f}",
            "output": record.output,
            "reason": record.reason,
        }
        for record in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "status", "seconds", "output", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def orchestrate(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "outputs" / "body_latent_alignment_runs" / timestamp
    log_dir = run_dir / "logs"
    records: list[Record] = []
    selected = ["HPE", "HAR"] if args.project == "both" else [args.project.upper()]
    for task in selected:
        project_root = HPE_ROOT if task == "HPE" else HAR_ROOT
        output_csv = PROJECT_ROOT / "tables" / f"body_latent_alignment_{task.lower()}.csv"
        checkpoint = Path(args.checkpoint) if args.checkpoint else existing_checkpoint(project_root)
        if not checkpoint.exists():
            records.append(Record(task, "skipped", 0.0, str(output_csv), f"missing_checkpoint={checkpoint}"))
            write_status(run_dir / "status.csv", records)
            continue
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--internal",
            "--project",
            task,
            "--project-root",
            str(project_root),
            "--dataset",
            args.dataset,
            "--config",
            "configs/student_vk_missing.yaml",
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--max-batches",
            str(args.max_batches),
            "--output-csv",
            str(output_csv),
        ]
        status, seconds, reason = run_command(command, PROJECT_ROOT, log_dir / f"{task.lower()}_body_latent_alignment.log", args.dry_run)
        records.append(Record(task, status, seconds, str(output_csv), reason))
        write_status(run_dir / "status.csv", records)
    print(f"Body-latent alignment records: {run_dir}", flush=True)


def evaluate_internal(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from _shared import build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
    from training.engine import move_batch_to_device

    namespace = argparse.Namespace(
        dataset=args.dataset,
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=None,
        resume=None,
        backbone_root=None,
        max_train_batches=None,
        max_eval_batches=None,
        device=args.device,
    )
    config = prepare_config(namespace)
    device = get_device(args.device)
    print("=" * 88, flush=True)
    print("[BodyLatent] Launch VK-to-nonRGB body-latent alignment diagnostic", flush=True)
    print(f"[BodyLatent] project={args.project}", flush=True)
    print(f"[BodyLatent] project_root={project_root}", flush=True)
    print(f"[BodyLatent] dataset={args.dataset}", flush=True)
    print(f"[BodyLatent] checkpoint={args.checkpoint}", flush=True)
    print(f"[BodyLatent] max_batches={args.max_batches}", flush=True)
    print("=" * 88, flush=True)
    print("[BodyLatent] Building dataloaders...", flush=True)
    _, val_loader = build_dataloaders(args.dataset, config)
    print("[BodyLatent] Building model and loading checkpoint...", flush=True)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    model.eval()

    modalities = HPE_MODALITIES if args.project == "HPE" else HAR_MODALITIES
    targets = [name for name in modalities if name != "vk"]
    stats = {
        target: {
            "same_sum": 0.0,
            "shuffled_sum": 0.0,
            "retrieval_correct": 0,
            "random_sum": 0.0,
            "count": 0,
            "batches": 0,
        }
        for target in targets
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"{args.project.lower()}_body_latent", dynamic_ncols=True)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch["inputs"], modalities)
            projected = output.get("projected_tokens")
            if not isinstance(projected, dict) or "vk" not in projected:
                raise KeyError("Model output must contain projected_tokens with a 'vk' entry.")
            vk_vec = F.normalize(projected["vk"].mean(dim=1).float(), dim=1)
            batch_size = vk_vec.size(0)
            if batch_size < 2:
                continue
            for target in targets:
                if target not in projected:
                    continue
                target_vec = F.normalize(projected[target].mean(dim=1).float(), dim=1)
                same = F.cosine_similarity(vk_vec, target_vec, dim=1)
                shuffled = F.cosine_similarity(vk_vec, torch.roll(target_vec, shifts=1, dims=0), dim=1)
                sim = torch.matmul(vk_vec, target_vec.t())
                pred = sim.argmax(dim=1)
                correct = (pred == torch.arange(batch_size, device=pred.device)).sum().item()
                entry = stats[target]
                entry["same_sum"] += float(same.sum().detach().cpu())
                entry["shuffled_sum"] += float(shuffled.sum().detach().cpu())
                entry["retrieval_correct"] += int(correct)
                entry["random_sum"] += float(batch_size * (1.0 / batch_size))
                entry["count"] += int(batch_size)
                entry["batches"] += 1
    rows = []
    for target in targets:
        entry = stats[target]
        count = max(int(entry["count"]), 1)
        same = float(entry["same_sum"]) / count
        shuffled = float(entry["shuffled_sum"]) / count
        retrieval = float(entry["retrieval_correct"]) / count
        random_top1 = float(entry["random_sum"]) / count
        rows.append(
            {
                "task": args.project,
                "anchor_modality": "vk",
                "target_modality": target,
                "same_sample_cosine": fmt(same),
                "shuffled_cosine": fmt(shuffled),
                "cosine_gap": fmt(same - shuffled),
                "retrieval_top1": fmt(retrieval),
                "random_top1": fmt(random_top1),
                "num_samples": entry["count"],
                "num_batches": entry["batches"],
                "checkpoint": args.checkpoint,
            }
        )
        print(
            "[BodyLatent] "
            f"{args.project} VK->{target} | same={same:.4f} | shuffled={shuffled:.4f} | "
            f"gap={same - shuffled:.4f} | top1={retrieval:.4f} | random={random_top1:.4f}",
            flush=True,
        )
    write_csv(Path(args.output_csv), rows)
    print(f"[BodyLatent] Saved {len(rows)} rows to {args.output_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run VK-to-nonRGB body-latent alignment diagnostic.")
    parser.add_argument("--dataset", type=str, required=False, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--project", choices=["both", "hpe", "har", "HPE", "HAR"], default="both")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint override for single-project runs.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-csv", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.internal:
        evaluate_internal(args)
        return
    if not args.dataset:
        raise SystemExit("--dataset is required.")
    orchestrate(args)


if __name__ == "__main__":
    main()
