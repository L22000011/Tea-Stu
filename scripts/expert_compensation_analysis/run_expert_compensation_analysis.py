from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "supplement" / "expert_compensation_analysis"


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    root: Path
    config: str
    checkpoint: str
    modality_sets: list[list[str]]


@dataclass(frozen=True)
class VKCondition:
    name: str
    perturbation_type: str
    severity: float
    fn: Callable[[dict[str, torch.Tensor], float], dict[str, torch.Tensor]]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def reset_project_imports() -> None:
    prefixes = ("data", "models", "training", "utils", "losses", "_shared")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]


def project_context(project_root: Path) -> None:
    os.chdir(project_root)
    reset_project_imports()
    for item in [str(project_root), str(project_root / "scripts")]:
        if item in sys.path:
            sys.path.remove(item)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))


def scaled_noise(tensor: torch.Tensor, severity: float) -> torch.Tensor:
    output = tensor.clone()
    if severity <= 0:
        return output
    scale = output.detach().float().std()
    if not torch.isfinite(scale) or float(scale) < 1e-8:
        scale = torch.tensor(1.0, device=output.device)
    return output + torch.randn_like(output) * scale * float(severity)


def clean_vk(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    return dict(inputs)


def vk_noise(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    output["vk"] = scaled_noise(output["vk"], severity)
    return output


def vk_joint_shuffle(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    vk = output["vk"]
    if vk.dim() >= 3 and vk.size(1) >= 17:
        permutation = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 16, 15, 14, 13, 12, 11], device=vk.device)
        output["vk"] = vk.index_select(1, permutation)
    elif vk.dim() >= 2:
        output["vk"] = torch.flip(vk, dims=[1])
    return output


def vk_sample_mismatch(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    vk = output["vk"]
    if vk.size(0) > 1:
        output["vk"] = torch.roll(vk, shifts=1, dims=0)
    return output


def build_conditions() -> list[VKCondition]:
    return [
        VKCondition("clean_vk", "none", 0.0, clean_vk),
        VKCondition("vk_noise_0.3", "gaussian_scaled_by_batch_std", 0.3, vk_noise),
        VKCondition("vk_noise_0.6", "gaussian_scaled_by_batch_std", 0.6, vk_noise),
        VKCondition("vk_joint_shuffle", "joint_identity_shuffle", 1.0, vk_joint_shuffle),
        VKCondition("vk_sample_mismatch", "batch_sample_roll_mismatch", 1.0, vk_sample_mismatch),
    ]


def generated_vk_path(root: Path, meta: dict[str, Any]) -> Path:
    idx = int(meta.get("idx", meta.get("frame_index", 0)))
    return root / str(meta["scene"]) / str(meta["subject"]) / str(meta["action"]) / f"frame{idx + 1:03d}.npy"


def replace_with_generated_vk(
    inputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
    generated_vk_root: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
    import numpy as np

    output = dict(inputs)
    vk = output["vk"].clone()
    replaced = 0
    missing = 0
    for index, meta in enumerate(metas):
        path = generated_vk_path(generated_vk_root, meta)
        if not path.exists():
            missing += 1
            continue
        try:
            arr = np.load(path).astype("float32").reshape(17, 2)
            vk[index] = torch.tensor(arr, dtype=vk.dtype, device=device)
            replaced += 1
        except Exception:
            missing += 1
    output["vk"] = vk
    return output, replaced, missing


def build_specs(project: str) -> list[ProjectSpec]:
    hpe_root = ROOT / "MMFi_HPE" if (ROOT / "MMFi_HPE").exists() else ROOT / "HPE"
    har_root = ROOT / "MMFi_HAR" if (ROOT / "MMFi_HAR").exists() else ROOT / "HAR"
    specs = {
        "HPE": ProjectSpec(
            name="HPE",
            root=hpe_root,
            config="configs/student_vk_missing.yaml",
            checkpoint="outputs/student_vk_missing/best.pth",
            modality_sets=[
                ["vk", "depth"],
                ["vk", "mmwave"],
                ["vk", "depth", "mmwave"],
                ["vk", "depth", "lidar", "mmwave", "wifi-csi"],
            ],
        ),
        "HAR": ProjectSpec(
            name="HAR",
            root=har_root,
            config="configs/student_vk_missing.yaml",
            checkpoint="outputs/student_vk_missing/best.pth",
            modality_sets=[
                ["vk", "depth"],
                ["vk", "mmwave"],
                ["vk", "depth", "mmwave"],
                ["vk", "depth", "lidar", "mmwave"],
            ],
        ),
    }
    return [specs["HPE"], specs["HAR"]] if project == "both" else [specs[project]]


def namespace_for(args: argparse.Namespace, spec: ProjectSpec) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=args.dataset,
        config=spec.config,
        checkpoint=spec.checkpoint,
        output_dir=None,
        resume=None,
        backbone_root=None,
        max_train_batches=None,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
    )


def planned_batches(loader: Any, max_batches: int | None) -> int | None:
    try:
        length = len(loader)
    except TypeError:
        return max_batches
    return min(length, max_batches) if max_batches else length


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


def mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else float("nan")


def hpe_sample_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1).mean(dim=-1)


def summarize_common(
    task: str,
    modality_set: list[str],
    condition: VKCondition,
    num_batches: int,
    num_samples: int,
    generated_replaced: int,
    generated_missing: int,
    checkpoint: Path,
    values: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "modality_set": "+".join(modality_set),
        "vk_condition": condition.name,
        "perturbation_type": condition.perturbation_type,
        "severity": condition.severity,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "generated_replaced": generated_replaced,
        "generated_missing": generated_missing,
        "checkpoint": str(checkpoint),
    }
    row.update(values)
    return row


def run_hpe_condition(
    model: Any,
    loader: Any,
    device: torch.device,
    modality_set: list[str],
    condition: VKCondition,
    max_batches: int | None,
    generated_vk_root: Path | None,
) -> tuple[dict[str, float], tuple[int, int, int, int]]:
    model.eval()
    final_errors: list[float] = []
    vk_errors: list[float] = []
    best_non_vk_errors: list[float] = []
    mean_non_vk_errors: list[float] = []
    alpha_vk_values: list[float] = []
    alpha_best_non_vk_values: list[float] = []
    alpha_error_x: list[float] = []
    alpha_quality_y: list[float] = []
    fusion_better_than_vk = 0
    fusion_better_than_best_non_vk = 0
    generated_replaced = 0
    generated_missing = 0
    batches = 0
    samples = 0

    from training.engine import move_batch_to_device

    total = planned_batches(loader, max_batches)
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, total=total, desc=f"HPE:{'+'.join(modality_set)}:{condition.name}")):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            inputs = condition.fn(batch["inputs"], condition.severity)
            if condition.name == "generated_vk":
                if generated_vk_root is None:
                    raise RuntimeError("generated_vk condition requires --generated-vk-root.")
                inputs, replaced, missing = replace_with_generated_vk(inputs, batch.get("meta", []), generated_vk_root, device)
                generated_replaced += replaced
                generated_missing += missing

            output = model(inputs, modality_set)
            target = batch["target"]
            alphas = output["alphas"].detach()
            per_expert = torch.stack([hpe_sample_mpjpe(pose, target) for pose in output["modality_poses"]], dim=1)
            final = hpe_sample_mpjpe(output["pose"], target)
            vk_index = modality_set.index("vk")
            non_vk_indices = [i for i, name in enumerate(modality_set) if name != "vk"]
            vk_error = per_expert[:, vk_index]
            non_vk = per_expert[:, non_vk_indices]
            best_non_vk, best_local = torch.min(non_vk, dim=1)
            mean_non_vk = non_vk.mean(dim=1)
            best_indices = torch.tensor([non_vk_indices[int(i)] for i in best_local.detach().cpu().tolist()], device=alphas.device)
            best_alpha = alphas.gather(1, best_indices[:, None]).squeeze(1)

            final_errors.extend(final.detach().cpu().tolist())
            vk_errors.extend(vk_error.detach().cpu().tolist())
            best_non_vk_errors.extend(best_non_vk.detach().cpu().tolist())
            mean_non_vk_errors.extend(mean_non_vk.detach().cpu().tolist())
            alpha_vk_values.extend(alphas[:, vk_index].detach().cpu().tolist())
            alpha_best_non_vk_values.extend(best_alpha.detach().cpu().tolist())
            fusion_better_than_vk += int((final < vk_error).sum().item())
            fusion_better_than_best_non_vk += int((final < best_non_vk).sum().item())

            for expert_index in range(len(modality_set)):
                alpha_error_x.extend(alphas[:, expert_index].detach().cpu().tolist())
                alpha_quality_y.extend((-per_expert[:, expert_index]).detach().cpu().tolist())

            batches += 1
            samples += int(target.size(0))

    values = {
        "metric_name": "MPJPE",
        "final_metric": mean(final_errors),
        "vk_expert_metric": mean(vk_errors),
        "best_non_vk_expert_metric": mean(best_non_vk_errors),
        "mean_non_vk_expert_metric": mean(mean_non_vk_errors),
        "fusion_gain_vs_vk": mean(vk_errors) - mean(final_errors),
        "fusion_gain_vs_best_non_vk": mean(best_non_vk_errors) - mean(final_errors),
        "compensation_rate_vs_vk": fusion_better_than_vk / samples if samples else float("nan"),
        "fusion_better_than_best_non_vk_rate": fusion_better_than_best_non_vk / samples if samples else float("nan"),
        "alpha_vk": mean(alpha_vk_values),
        "alpha_best_non_vk": mean(alpha_best_non_vk_values),
        "alpha_non_vk_sum": 1.0 - mean(alpha_vk_values),
        "alpha_quality_pearson": pearson(alpha_error_x, alpha_quality_y),
    }
    return values, (batches, samples, generated_replaced, generated_missing)


def run_har_condition(
    model: Any,
    loader: Any,
    device: torch.device,
    modality_set: list[str],
    condition: VKCondition,
    max_batches: int | None,
    generated_vk_root: Path | None,
) -> tuple[dict[str, float], tuple[int, int, int, int]]:
    model.eval()
    final_correct_values: list[float] = []
    vk_correct_values: list[float] = []
    best_non_vk_correct_values: list[float] = []
    mean_non_vk_correct_values: list[float] = []
    alpha_vk_values: list[float] = []
    alpha_best_non_vk_values: list[float] = []
    alpha_quality_x: list[float] = []
    alpha_quality_y: list[float] = []
    final_correct_when_vk_wrong = 0
    vk_wrong = 0
    final_better_than_best_non_vk = 0
    generated_replaced = 0
    generated_missing = 0
    batches = 0
    samples = 0

    from training.engine import move_batch_to_device

    total = planned_batches(loader, max_batches)
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, total=total, desc=f"HAR:{'+'.join(modality_set)}:{condition.name}")):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            inputs = condition.fn(batch["inputs"], condition.severity)
            if condition.name == "generated_vk":
                if generated_vk_root is None:
                    raise RuntimeError("generated_vk condition requires --generated-vk-root.")
                inputs, replaced, missing = replace_with_generated_vk(inputs, batch.get("meta", []), generated_vk_root, device)
                generated_replaced += replaced
                generated_missing += missing

            output = model(inputs, modality_set)
            target = batch["target"]
            alphas = output["alphas"].detach()
            expert_logits = output["modality_logits"]
            expert_correct = torch.stack([(logits.argmax(dim=1) == target).float() for logits in expert_logits], dim=1)
            final_correct = (output["logits"].argmax(dim=1) == target).float()
            vk_index = modality_set.index("vk")
            non_vk_indices = [i for i, name in enumerate(modality_set) if name != "vk"]
            vk_correct = expert_correct[:, vk_index]
            non_vk = expert_correct[:, non_vk_indices]
            best_non_vk, best_local = torch.max(non_vk, dim=1)
            mean_non_vk = non_vk.mean(dim=1)
            best_indices = torch.tensor([non_vk_indices[int(i)] for i in best_local.detach().cpu().tolist()], device=alphas.device)
            best_alpha = alphas.gather(1, best_indices[:, None]).squeeze(1)

            final_correct_values.extend(final_correct.detach().cpu().tolist())
            vk_correct_values.extend(vk_correct.detach().cpu().tolist())
            best_non_vk_correct_values.extend(best_non_vk.detach().cpu().tolist())
            mean_non_vk_correct_values.extend(mean_non_vk.detach().cpu().tolist())
            alpha_vk_values.extend(alphas[:, vk_index].detach().cpu().tolist())
            alpha_best_non_vk_values.extend(best_alpha.detach().cpu().tolist())
            final_correct_when_vk_wrong += int(((vk_correct == 0) & (final_correct == 1)).sum().item())
            vk_wrong += int((vk_correct == 0).sum().item())
            final_better_than_best_non_vk += int((final_correct > best_non_vk).sum().item())

            for expert_index in range(len(modality_set)):
                probs = F.softmax(expert_logits[expert_index], dim=1)
                true_prob = probs.gather(1, target[:, None]).squeeze(1)
                alpha_quality_x.extend(alphas[:, expert_index].detach().cpu().tolist())
                alpha_quality_y.extend(true_prob.detach().cpu().tolist())

            batches += 1
            samples += int(target.size(0))

    final_acc = mean(final_correct_values)
    vk_acc = mean(vk_correct_values)
    best_non_vk_acc = mean(best_non_vk_correct_values)
    values = {
        "metric_name": "Accuracy",
        "final_metric": final_acc,
        "vk_expert_metric": vk_acc,
        "best_non_vk_expert_metric": best_non_vk_acc,
        "mean_non_vk_expert_metric": mean(mean_non_vk_correct_values),
        "fusion_gain_vs_vk": final_acc - vk_acc,
        "fusion_gain_vs_best_non_vk": final_acc - best_non_vk_acc,
        "compensation_rate_vs_vk": final_correct_when_vk_wrong / vk_wrong if vk_wrong else float("nan"),
        "fusion_better_than_best_non_vk_rate": final_better_than_best_non_vk / samples if samples else float("nan"),
        "alpha_vk": mean(alpha_vk_values),
        "alpha_best_non_vk": mean(alpha_best_non_vk_values),
        "alpha_non_vk_sum": 1.0 - mean(alpha_vk_values),
        "alpha_quality_pearson": pearson(alpha_quality_x, alpha_quality_y),
    }
    return values, (batches, samples, generated_replaced, generated_missing)


def make_report(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    lines = [
        "# Expert Compensation Analysis",
        "",
        "This diagnostic evaluates whether degraded VK is compensated by non-VK modality experts.",
        "It does not claim calibrated physical sensor reliability.",
        "",
        "## How to read",
        "",
        "- `fusion_gain_vs_vk > 0` means the final fused output is better than the VK expert.",
        "- For HPE, lower MPJPE is better; gains are `VK expert MPJPE - fused MPJPE`.",
        "- For HAR, higher accuracy is better; gains are `fused Acc - VK expert Acc`.",
        "- `alpha_quality_pearson` measures whether larger weights tend to align with better expert quality.",
        "",
        "## Key rows",
        "",
    ]
    for row in rows:
        if row["vk_condition"] not in {"vk_noise_0.6", "vk_joint_shuffle", "generated_vk"}:
            continue
        lines.append(
            f"- {row['task']} `{row['modality_set']}` under `{row['vk_condition']}`: "
            f"final={float(row['final_metric']):.6f}, "
            f"vk_expert={float(row['vk_expert_metric']):.6f}, "
            f"best_non_vk={float(row['best_non_vk_expert_metric']):.6f}, "
            f"fusion_gain_vs_vk={float(row['fusion_gain_vs_vk']):.6f}, "
            f"alpha_vk={float(row['alpha_vk']):.4f}, "
            f"alpha_quality_pearson={float(row['alpha_quality_pearson']):.4f}."
        )
    lines.extend(
        [
            "",
            "## Safe paper statement",
            "",
            "When VK is perturbed, the framework should be interpreted as compensating degraded structural cues through complementary physical modalities if the fused output remains better than the VK expert. The learned weights are task-level contribution proxies, not calibrated sensor-quality measurements.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_project(args: argparse.Namespace, spec: ProjectSpec) -> list[dict[str, Any]]:
    if not spec.root.exists():
        print(f"[{spec.name}] SKIP | missing_project_root={spec.root}", flush=True)
        return []
    checkpoint = spec.root / spec.checkpoint
    config = spec.root / spec.config
    if not checkpoint.exists():
        print(f"[{spec.name}] SKIP | missing_checkpoint={checkpoint}", flush=True)
        return []
    if not config.exists():
        print(f"[{spec.name}] SKIP | missing_config={config}", flush=True)
        return []

    print(f"[{spec.name}] project={spec.root}", flush=True)
    print(f"[{spec.name}] checkpoint={checkpoint}", flush=True)
    print(f"[{spec.name}] modality_sets={['+'.join(item) for item in spec.modality_sets]}", flush=True)

    project_context(spec.root)
    shared = importlib.import_module("_shared")
    ns = namespace_for(args, spec)
    config_dict = shared.prepare_config(ns)
    device = shared.get_device(args.device)
    _, val_loader = shared.build_dataloaders(args.dataset, config_dict)
    model = shared.build_model_from_config(config_dict, device)
    shared.load_optional_checkpoint(model, str(checkpoint), device)

    conditions = build_conditions()
    if args.generated_vk_root is not None:
        conditions.append(VKCondition("generated_vk", "replace_with_generated_vk_root", 1.0, clean_vk))

    rows: list[dict[str, Any]] = []
    total_tasks = len(spec.modality_sets) * len(conditions)
    task_index = 0
    for modality_set in spec.modality_sets:
        for condition in conditions:
            task_index += 1
            print(f"[{spec.name}] task {task_index}/{total_tasks} | set={'+'.join(modality_set)} | condition={condition.name}", flush=True)
            if args.dry_run:
                continue
            if spec.name == "HPE":
                values, counts = run_hpe_condition(model, val_loader, device, modality_set, condition, args.max_eval_batches, args.generated_vk_root)
            else:
                values, counts = run_har_condition(model, val_loader, device, modality_set, condition, args.max_eval_batches, args.generated_vk_root)
            row = summarize_common(spec.name, modality_set, condition, *counts, checkpoint, values)
            rows.append(row)
            append_csv(OUT_DIR / "expert_compensation_summary.csv", row)
            print(
                f"[{spec.name}] done | set={row['modality_set']} | condition={condition.name} | "
                f"final={float(row['final_metric']):.6f} | vk_expert={float(row['vk_expert_metric']):.6f} | "
                f"gain_vs_vk={float(row['fusion_gain_vs_vk']):.6f}",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser("Expert compensation analysis for degraded VK.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--project", choices=["HPE", "HAR", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--generated-vk-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUT_DIR / "expert_compensation_summary.csv"
    if output_csv.exists() and args.force:
        output_csv.unlink()

    print("=" * 88, flush=True)
    print("Expert Compensation Analysis", flush=True)
    print(f"dataset={args.dataset}", flush=True)
    print(f"project={args.project}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"max_eval_batches={args.max_eval_batches}", flush=True)
    print(f"output_dir={OUT_DIR}", flush=True)
    if args.generated_vk_root:
        print(f"generated_vk_root={args.generated_vk_root}", flush=True)
    print("=" * 88, flush=True)

    all_rows: list[dict[str, Any]] = []
    for spec in build_specs(args.project):
        all_rows.extend(run_project(args, spec))

    if not args.dry_run:
        write_csv(output_csv, all_rows)
        make_report(all_rows, OUT_DIR / "expert_compensation_conclusion.md")
        print(f"[Summary] saved: {output_csv}", flush=True)
        print(f"[Summary] saved: {OUT_DIR / 'expert_compensation_conclusion.md'}", flush=True)


if __name__ == "__main__":
    main()
