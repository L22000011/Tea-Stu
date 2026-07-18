from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    root: Path
    config: str
    checkpoint: str
    modality_sets: list[list[str]]
    metric_name: str
    lower_is_better: bool


@dataclass(frozen=True)
class VKCondition:
    name: str
    perturbation_type: str
    severity: float
    fn: Callable[[dict[str, torch.Tensor], float], dict[str, torch.Tensor]]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict[str, Any]) -> None:
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


def add_generated_condition(conditions: list[VKCondition], generated_vk_root: Path | None) -> list[VKCondition]:
    if generated_vk_root is None:
        return conditions
    return conditions + [VKCondition("generated_vk", "replace_with_generated_vk_root", 1.0, clean_vk)]


def generated_vk_path(root: Path, meta: dict[str, Any]) -> Path:
    idx = int(meta.get("idx", meta.get("frame_index", 0)))
    return root / str(meta["scene"]) / str(meta["subject"]) / str(meta["action"]) / f"frame{idx + 1:03d}.npy"


def replace_with_generated_vk(
    inputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
    generated_vk_root: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
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
            import numpy as np

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
            metric_name="MPJPE",
            lower_is_better=True,
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
            metric_name="Accuracy",
            lower_is_better=False,
        ),
    }
    if project == "both":
        return [specs["HPE"], specs["HAR"]]
    return [specs[project]]


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


def planned_batches(loader, max_batches: int | None) -> int | None:
    try:
        length = len(loader)
    except TypeError:
        return max_batches
    if max_batches is None:
        return length
    return min(length, int(max_batches))


def init_weight_accumulator(modality_set: list[str]) -> dict[str, float]:
    return {f"weight_{modality.replace('-', '_')}": 0.0 for modality in modality_set}


def extract_alphas(output: dict[str, Any]) -> torch.Tensor | None:
    for key in ("alphas", "weights", "reliability", "fusion_weights"):
        value = output.get(key)
        if torch.is_tensor(value):
            return value
    fusion = output.get("fusion")
    if isinstance(fusion, dict):
        for key in ("alphas", "weights", "reliability", "fusion_weights"):
            value = fusion.get(key)
            if torch.is_tensor(value):
                return value
    return None


def evaluate_hpe(
    model,
    val_loader,
    device,
    move_batch_to_device,
    compute_metrics,
    modality_set: list[str],
    condition: VKCondition,
    max_batches: int | None,
    generated_vk_root: Path | None,
) -> dict[str, Any]:
    meters = {"mse": 0.0, "mpjpe": 0.0, "pa_mpjpe": 0.0}
    weight_sums = init_weight_accumulator(modality_set)
    count = 0
    weight_count = 0
    generated_replaced = 0
    generated_missing = 0
    total_batches = planned_batches(val_loader, max_batches)
    desc = f"HPE:{'+'.join(modality_set)}:{condition.name}"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=desc, total=total_batches, dynamic_ncols=True)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            if condition.name == "generated_vk":
                if generated_vk_root is None:
                    raise RuntimeError("generated_vk condition requires --generated-vk-root.")
                inputs, replaced, missing = replace_with_generated_vk(batch["inputs"], batch["meta"], generated_vk_root, device)
                generated_replaced += replaced
                generated_missing += missing
            else:
                inputs = condition.fn(batch["inputs"], condition.severity)
            output = model(inputs, modality_set)
            metrics = compute_metrics(output["pose"], batch["target"])
            batch_size = batch["target"].size(0)
            count += batch_size
            for key in meters:
                meters[key] += float(metrics[key]) * batch_size
            alphas = extract_alphas(output)
            if alphas is not None:
                alpha_mean = alphas.detach().float().mean(dim=0).cpu()
                for idx, modality in enumerate(modality_set):
                    weight_sums[f"weight_{modality.replace('-', '_')}"] += float(alpha_mean[idx]) * batch_size
                weight_count += batch_size
    result = {key: value / max(count, 1) for key, value in meters.items()}
    result["num_samples"] = count
    result["generated_replaced"] = generated_replaced
    result["generated_missing"] = generated_missing
    for key, value in weight_sums.items():
        result[key] = value / max(weight_count, 1) if weight_count else float("nan")
    return result


def evaluate_har(
    model,
    val_loader,
    device,
    move_batch_to_device,
    classification_metrics,
    modality_set: list[str],
    condition: VKCondition,
    max_batches: int | None,
    generated_vk_root: Path | None,
) -> dict[str, Any]:
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    weight_sums = init_weight_accumulator(modality_set)
    weight_count = 0
    generated_replaced = 0
    generated_missing = 0
    total_batches = planned_batches(val_loader, max_batches)
    desc = f"HAR:{'+'.join(modality_set)}:{condition.name}"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=desc, total=total_batches, dynamic_ncols=True)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            if condition.name == "generated_vk":
                if generated_vk_root is None:
                    raise RuntimeError("generated_vk condition requires --generated-vk-root.")
                inputs, replaced, missing = replace_with_generated_vk(batch["inputs"], batch["meta"], generated_vk_root, device)
                generated_replaced += replaced
                generated_missing += missing
            else:
                inputs = condition.fn(batch["inputs"], condition.severity)
            output = model(inputs, modality_set)
            logits = output["logits"]
            target = batch["target"]
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == target).sum().item())
            total += int(target.numel())
            all_logits.append(logits.detach().cpu())
            all_targets.append(target.detach().cpu())
            alphas = extract_alphas(output)
            if alphas is not None:
                alpha_mean = alphas.detach().float().mean(dim=0).cpu()
                for idx, modality in enumerate(modality_set):
                    weight_sums[f"weight_{modality.replace('-', '_')}"] += float(alpha_mean[idx]) * target.size(0)
                weight_count += target.size(0)
    num_classes = int(getattr(model, "num_classes", 27)) if hasattr(model, "num_classes") else 27
    metrics = classification_metrics(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0), num_classes) if total else {"macro_f1": 0.0}
    result = {"acc": correct / max(total, 1), "macro_f1": float(metrics["macro_f1"]), "num_samples": total}
    result["generated_replaced"] = generated_replaced
    result["generated_missing"] = generated_missing
    for key, value in weight_sums.items():
        result[key] = value / max(weight_count, 1) if weight_count else float("nan")
    return result


def metric_value(spec: ProjectSpec, metrics: dict[str, Any]) -> float:
    return float(metrics["mpjpe"] if spec.name == "HPE" else metrics["acc"])


def relative_drop(spec: ProjectSpec, clean: float, value: float) -> float:
    if spec.lower_is_better:
        return (value - clean) / max(abs(clean), 1e-8)
    return (clean - value) / max(abs(clean), 1e-8)


def make_row(spec: ProjectSpec, modality_set: list[str], condition: VKCondition, metrics: dict[str, Any], clean_metrics: dict[str, Any], checkpoint: Path, max_batches: int | None) -> dict[str, Any]:
    value = metric_value(spec, metrics)
    clean_value = metric_value(spec, clean_metrics)
    row: dict[str, Any] = {
        "task": spec.name,
        "modality_set": "+".join(modality_set),
        "vk_condition": condition.name,
        "perturbation_type": condition.perturbation_type,
        "severity": condition.severity,
        "metric_name": spec.metric_name,
        "metric_value": value,
        "clean_metric": clean_value,
        "delta_metric": value - clean_value,
        "relative_drop": relative_drop(spec, clean_value, value),
        "num_batches": max_batches if max_batches is not None else "all",
        "num_samples": metrics.get("num_samples", 0),
        "generated_replaced": metrics.get("generated_replaced", 0),
        "generated_missing": metrics.get("generated_missing", 0),
        "checkpoint": str(checkpoint),
    }
    if spec.name == "HPE":
        row["pa_mpjpe"] = metrics.get("pa_mpjpe", "")
        row["clean_pa_mpjpe"] = clean_metrics.get("pa_mpjpe", "")
    else:
        row["macro_f1"] = metrics.get("macro_f1", "")
        row["clean_macro_f1"] = clean_metrics.get("macro_f1", "")
    for modality in modality_set:
        key = f"weight_{modality.replace('-', '_')}"
        clean_weight = clean_metrics.get(key, float("nan"))
        weight = metrics.get(key, float("nan"))
        row[key] = weight
        row[f"clean_{key}"] = clean_weight
        row[f"delta_{key}"] = weight - clean_weight if isinstance(weight, float) and isinstance(clean_weight, float) else ""
    weight_items = [(m, metrics.get(f"weight_{m.replace('-', '_')}", float("-inf"))) for m in modality_set]
    row["dominant_modality"] = max(weight_items, key=lambda item: item[1])[0] if weight_items else ""
    return row


def run_project(args: argparse.Namespace, spec: ProjectSpec) -> list[dict[str, Any]]:
    output_csv = ROOT / "tables" / "vk_reliability_weight_stress.csv"
    project_output = spec.root / "outputs" / "eval" / "vk_reliability_weight_stress.csv"
    checkpoint = spec.root / spec.checkpoint
    config = spec.root / spec.config
    if not spec.root.exists():
        print(f"[{spec.name}] SKIP | missing_project_root={spec.root}", flush=True)
        return []
    if not checkpoint.exists():
        print(f"[{spec.name}] SKIP | missing_checkpoint={checkpoint}", flush=True)
        return []
    if not config.exists():
        print(f"[{spec.name}] SKIP | missing_config={config}", flush=True)
        return []
    if args.force:
        for path in [output_csv, project_output]:
            if path.exists():
                path.unlink()

    print("=" * 92, flush=True)
    print(f"[{spec.name}] VK reliability weight stress test", flush=True)
    print(f"[{spec.name}] root={spec.root}", flush=True)
    print(f"[{spec.name}] checkpoint={checkpoint}", flush=True)
    print(f"[{spec.name}] modality_sets={['+'.join(s) for s in spec.modality_sets]}", flush=True)
    conditions = add_generated_condition(build_conditions(), args.generated_vk_root)
    print(f"[{spec.name}] conditions={[c.name for c in conditions]}", flush=True)
    if args.generated_vk_root:
        print(f"[{spec.name}] generated_vk_root={args.generated_vk_root}", flush=True)
    print(f"[{spec.name}] max_eval_batches={args.max_eval_batches}", flush=True)
    print("=" * 92, flush=True)

    if args.dry_run:
        rows = []
        for modality_set in spec.modality_sets:
            for condition in conditions:
                rows.append(
                    {
                        "task": spec.name,
                        "modality_set": "+".join(modality_set),
                        "vk_condition": condition.name,
                        "dry_run": True,
                    }
                )
        print(f"[{spec.name}] DRY-RUN | would evaluate {len(rows)} rows", flush=True)
        return rows

    project_context(spec.root)
    shared = importlib.import_module("_shared")
    engine = importlib.import_module("training.engine")
    metrics_mod = importlib.import_module("utils.metrics")
    ns = namespace_for(args, spec)
    config_dict = shared.prepare_config(ns)
    device = shared.get_device(args.device)
    _, val_loader = shared.build_dataloaders(args.dataset, config_dict)
    model = shared.build_model_from_config(config_dict, device)
    shared.load_optional_checkpoint(model, str(checkpoint), device)
    model.eval()

    rows: list[dict[str, Any]] = []
    total_tasks = len(spec.modality_sets) * len(conditions)
    task_index = 0
    for modality_set in spec.modality_sets:
        clean_metrics: dict[str, Any] | None = None
        for condition in conditions:
            task_index += 1
            print(f"[{spec.name}] task {task_index}/{total_tasks} | set={'+'.join(modality_set)} | condition={condition.name}", flush=True)
            if spec.name == "HPE":
                metrics = evaluate_hpe(
                    model,
                    val_loader,
                    device,
                    engine.move_batch_to_device,
                    metrics_mod.compute_metrics,
                    modality_set,
                    condition,
                    args.max_eval_batches,
                    args.generated_vk_root,
                )
            else:
                metrics = evaluate_har(
                    model,
                    val_loader,
                    device,
                    engine.move_batch_to_device,
                    metrics_mod.classification_metrics,
                    modality_set,
                    condition,
                    args.max_eval_batches,
                    args.generated_vk_root,
                )
            if condition.name == "clean_vk":
                clean_metrics = metrics
            if clean_metrics is None:
                raise RuntimeError("clean_vk must be evaluated before degraded conditions.")
            row = make_row(spec, modality_set, condition, metrics, clean_metrics, checkpoint, args.max_eval_batches)
            append_row(project_output, row)
            append_row(output_csv, row)
            rows.append(row)
            vk_weight = row.get("weight_vk", "")
            delta_vk = row.get("delta_weight_vk", "")
            print(
                f"[{spec.name}] done | set={row['modality_set']} | condition={condition.name} | "
                f"{spec.metric_name}={row['metric_value']:.6f} | delta={row['delta_metric']:.6f} | "
                f"weight_vk={vk_weight} | delta_weight_vk={delta_vk}",
                flush=True,
            )
    print(f"[{spec.name}] Saved {len(rows)} rows to {project_output}", flush=True)
    print(f"[Summary] Appended rows to {output_csv}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stress-test VK reliability weights under degraded VK input.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--project", choices=["HPE", "HAR", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--generated-vk-root", type=Path, default=None, help="Optional generated VK root laid out as <root>/E/S/A/frameXXX.npy.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for spec in build_specs(args.project):
        all_rows.extend(run_project(args, spec))
    if args.dry_run:
        print(f"[Summary] DRY-RUN | total planned rows={len(all_rows)}", flush=True)
    else:
        summary_path = ROOT / "reports" / "vk_reliability_weight_stress_summary.md"
        ensure_parent(summary_path)
        summary_path.write_text(
            "# VK Reliability Weight Stress Test\n\n"
            f"- rows: {len(all_rows)}\n"
            "- output_csv: tables/vk_reliability_weight_stress.csv\n"
            "- note: weights are learned contribution proxies, not calibrated physical sensor reliability.\n",
            encoding="utf-8",
        )
        print(f"[Summary] Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
