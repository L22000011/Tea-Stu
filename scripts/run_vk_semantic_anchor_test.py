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
    modalities: list[str]
    metric_name: str
    lower_is_better: bool


@dataclass(frozen=True)
class Setting:
    name: str
    perturbed_modality: str
    perturbation_type: str
    severity: float
    fn: Callable[[dict[str, torch.Tensor], float], dict[str, torch.Tensor]]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def corrupt_vk_noise(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    output["vk"] = scaled_noise(output["vk"], severity)
    return output


def corrupt_sensor_noise(inputs: dict[str, torch.Tensor], modality: str, severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    tensor = scaled_noise(output[modality], severity)
    if severity > 0:
        drop_prob = min(0.15, 0.5 * float(severity))
        tensor = tensor.clone()
        tensor[torch.rand_like(tensor.float()) < drop_prob] = 0.0
    output[modality] = tensor
    return output


def make_sensor_noise(modality: str) -> Callable[[dict[str, torch.Tensor], float], dict[str, torch.Tensor]]:
    return lambda inputs, severity: corrupt_sensor_noise(inputs, modality, severity)


def corrupt_vk_joint_shuffle(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    vk = output["vk"]
    if vk.dim() < 3 or vk.size(1) < 17:
        output["vk"] = torch.flip(vk, dims=[1]) if vk.dim() >= 2 else vk
        return output
    # Fixed COCO17-style semantic permutation. Coordinates remain plausible, but joint identity is broken.
    permutation = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 16, 15, 14, 13, 12, 11], device=vk.device)
    output["vk"] = vk.index_select(1, permutation)
    return output


def corrupt_vk_sample_mismatch(inputs: dict[str, torch.Tensor], severity: float) -> dict[str, torch.Tensor]:
    output = dict(inputs)
    vk = output["vk"]
    if vk.size(0) > 1:
        output["vk"] = torch.roll(vk, shifts=1, dims=0)
    return output


def build_settings(modalities: list[str] | None = None) -> list[Setting]:
    modalities = modalities or []
    settings = [
        Setting("clean", "none", "none", 0.0, lambda inputs, severity: dict(inputs)),
        Setting("vk_noise_0.1", "vk", "gaussian_scaled_by_batch_std", 0.1, corrupt_vk_noise),
        Setting("vk_noise_0.3", "vk", "gaussian_scaled_by_batch_std", 0.3, corrupt_vk_noise),
        Setting("vk_noise_0.5", "vk", "gaussian_scaled_by_batch_std", 0.5, corrupt_vk_noise),
    ]
    for modality in ["depth", "lidar", "mmwave", "wifi-csi"]:
        if modality in modalities:
            settings.append(
                Setting(
                    f"{modality.replace('-', '_')}_noise_0.3",
                    modality,
                    "gaussian_scaled_by_batch_std_plus_light_zeroing",
                    0.3,
                    make_sensor_noise(modality),
                )
            )
    settings.extend(
        [
            Setting("vk_joint_shuffle", "vk", "joint_identity_shuffle", 1.0, corrupt_vk_joint_shuffle),
            Setting("vk_sample_mismatch", "vk", "batch_sample_roll_mismatch", 1.0, corrupt_vk_sample_mismatch),
        ]
    )
    return settings


def build_specs(project: str) -> list[ProjectSpec]:
    hpe_root = ROOT / "MMFi_HPE" if (ROOT / "MMFi_HPE").exists() else ROOT / "HPE"
    har_root = ROOT / "MMFi_HAR" if (ROOT / "MMFi_HAR").exists() else ROOT / "HAR"
    specs = {
        "HPE": ProjectSpec(
            name="HPE",
            root=hpe_root,
            config="configs/student_vk_missing.yaml",
            checkpoint="outputs/student_vk_missing/best.pth",
            modalities=["vk", "depth", "lidar", "mmwave", "wifi-csi"],
            metric_name="MPJPE",
            lower_is_better=True,
        ),
        "HAR": ProjectSpec(
            name="HAR",
            root=har_root,
            config="configs/student_vk_missing.yaml",
            checkpoint="outputs/student_vk_missing/best.pth",
            modalities=["vk", "depth", "lidar", "mmwave"],
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


def eval_hpe(model, val_loader, device, move_batch_to_device, compute_metrics, setting: Setting, modalities: list[str], max_batches: int | None) -> dict[str, Any]:
    meters = {"mse": 0.0, "mpjpe": 0.0, "pa_mpjpe": 0.0}
    count = 0
    total_batches = planned_batches(val_loader, max_batches)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"hpe:{setting.name}", total=total_batches, dynamic_ncols=True)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            inputs = setting.fn(batch["inputs"], setting.severity)
            output = model(inputs, modalities)
            metrics = compute_metrics(output["pose"], batch["target"])
            batch_size = batch["target"].size(0)
            count += batch_size
            for key in meters:
                meters[key] += metrics[key] * batch_size
    if count == 0:
        return {"mse": 0.0, "mpjpe": 0.0, "pa_mpjpe": 0.0, "num_samples": 0}
    return {key: value / count for key, value in meters.items()} | {"num_samples": count}


def eval_har(model, val_loader, device, move_batch_to_device, classification_metrics, setting: Setting, modalities: list[str], max_batches: int | None) -> dict[str, Any]:
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    total_batches = planned_batches(val_loader, max_batches)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"har:{setting.name}", total=total_batches, dynamic_ncols=True)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            inputs = setting.fn(batch["inputs"], setting.severity)
            output = model(inputs, modalities)
            logits = output["logits"]
            target = batch["target"]
            pred = torch.argmax(logits, dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
            all_logits.append(logits.detach().cpu())
            all_targets.append(target.detach().cpu())
    if total == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "num_samples": 0}
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    num_classes = int(getattr(model, "num_classes", 27)) if hasattr(model, "num_classes") else 27
    metrics = classification_metrics(logits_cat, targets_cat, num_classes)
    return {"acc": correct / total, "macro_f1": metrics["macro_f1"], "num_samples": total}


def metric_value(spec: ProjectSpec, metrics: dict[str, Any]) -> float:
    if spec.name == "HPE":
        return float(metrics["mpjpe"])
    return float(metrics["acc"])


def relative_change(spec: ProjectSpec, clean: float, value: float) -> float:
    eps = 1e-8
    if spec.lower_is_better:
        return (value - clean) / max(abs(clean), eps)
    return (clean - value) / max(abs(clean), eps)


def planned_batches(val_loader, max_batches: int | None) -> int | None:
    try:
        loader_len = len(val_loader)
    except TypeError:
        return max_batches
    if max_batches is None:
        return loader_len
    return min(loader_len, int(max_batches))


def row_for(spec: ProjectSpec, setting: Setting, metrics: dict[str, Any], clean_metric: float, checkpoint: Path, max_batches: int | None) -> dict[str, Any]:
    value = metric_value(spec, metrics)
    delta = value - clean_metric
    return {
        "task": spec.name,
        "setting": setting.name,
        "perturbed_modality": setting.perturbed_modality,
        "perturbation_type": setting.perturbation_type,
        "severity": setting.severity,
        "metric_name": spec.metric_name,
        "metric_value": value,
        "clean_metric": clean_metric,
        "delta_metric": delta,
        "relative_drop": relative_change(spec, clean_metric, value),
        "num_batches": max_batches if max_batches is not None else "all",
        "num_samples": metrics.get("num_samples", 0),
        "checkpoint": str(checkpoint),
    }


def existing_setting_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("setting", "")) for row in rows if row.get("setting")}


def existing_clean_metric(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if row.get("setting") != "clean":
            continue
        try:
            return float(row["metric_value"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def run_project(args: argparse.Namespace, spec: ProjectSpec) -> list[dict[str, Any]]:
    output_csv = spec.root / "outputs" / "eval" / "vk_semantic_anchor_test.csv"
    checkpoint = spec.root / spec.checkpoint
    config = spec.root / spec.config
    settings = build_settings(spec.modalities)
    expected_setting_names = {setting.name for setting in settings}
    if not spec.root.exists():
        print(f"[{spec.name}] SKIP | missing_project_root={spec.root}", flush=True)
        return []
    if not checkpoint.exists():
        print(f"[{spec.name}] SKIP | missing_checkpoint={checkpoint}", flush=True)
        return []
    if not config.exists():
        print(f"[{spec.name}] SKIP | missing_config={config}", flush=True)
        return []
    existing_rows: list[dict[str, Any]] = []
    existing_names: set[str] = set()
    if output_csv.exists() and not args.force:
        existing_rows = read_existing_rows(output_csv)
        existing_names = existing_setting_names(existing_rows)
        if expected_setting_names.issubset(existing_names):
            print(f"[{spec.name}] SKIP | complete_outputs_exist={output_csv} | use --force to overwrite", flush=True)
            return existing_rows
        missing = sorted(expected_setting_names - existing_names)
        print(f"[{spec.name}] RESUME | partial_output={output_csv} | missing_settings={','.join(missing)}", flush=True)
    if output_csv.exists() and args.force:
        output_csv.unlink()
        existing_rows = []
        existing_names = set()

    print("=" * 88, flush=True)
    print(f"[{spec.name}] VK semantic anchor diagnostic", flush=True)
    print(f"[{spec.name}] root={spec.root}", flush=True)
    print(f"[{spec.name}] config={spec.config}", flush=True)
    print(f"[{spec.name}] checkpoint={checkpoint}", flush=True)
    print(f"[{spec.name}] modalities={'+'.join(spec.modalities)}", flush=True)
    print(f"[{spec.name}] max_eval_batches={args.max_eval_batches}", flush=True)
    print(f"[{spec.name}] settings={len(settings)}", flush=True)
    print("=" * 88, flush=True)

    if args.dry_run:
        dry_rows = [
            row_for(spec, setting, {"mpjpe": 0.0, "acc": 1.0, "num_samples": 0}, 0.0 if spec.lower_is_better else 1.0, checkpoint, args.max_eval_batches)
            for setting in build_settings(spec.modalities)
        ]
        print(f"[{spec.name}] DRY-RUN | would write {output_csv}", flush=True)
        return dry_rows

    project_context(spec.root)
    shared = importlib.import_module("_shared")
    engine = importlib.import_module("training.engine")
    ns = namespace_for(args, spec)
    config_dict = shared.prepare_config(ns)
    device = shared.get_device(args.device)
    _, val_loader = shared.build_dataloaders(args.dataset, config_dict)
    model = shared.build_model_from_config(config_dict, device)
    shared.load_optional_checkpoint(model, str(checkpoint), device)
    model.eval()

    if spec.name == "HPE":
        metrics_mod = importlib.import_module("utils.metrics")

        def evaluate(setting: Setting) -> dict[str, Any]:
            return eval_hpe(model, val_loader, device, engine.move_batch_to_device, metrics_mod.compute_metrics, setting, spec.modalities, args.max_eval_batches)

    else:
        metrics_mod = importlib.import_module("utils.metrics")

        def evaluate(setting: Setting) -> dict[str, Any]:
            return eval_har(model, val_loader, device, engine.move_batch_to_device, metrics_mod.classification_metrics, setting, spec.modalities, args.max_eval_batches)

    rows: list[dict[str, Any]] = []
    clean_metric = existing_clean_metric(existing_rows)
    for index, setting in enumerate(settings, start=1):
        if setting.name in existing_names:
            print(f"[{spec.name}] setting {index}/{len(settings)} | {setting.name} | already done", flush=True)
            continue
        print(f"[{spec.name}] setting {index}/{len(settings)} | {setting.name}", flush=True)
        metrics = evaluate(setting)
        if setting.name == "clean":
            clean_metric = metric_value(spec, metrics)
        if clean_metric is None:
            raise RuntimeError(f"{spec.name} clean setting is required before evaluating {setting.name}.")
        row = row_for(spec, setting, metrics, float(clean_metric), checkpoint, args.max_eval_batches)
        append_rows(output_csv, [row])
        rows.append(row)
        print(
            f"[{spec.name}] done | {setting.name} | {spec.metric_name}={row['metric_value']:.6f} | "
            f"delta={row['delta_metric']:.6f} | relative_drop={row['relative_drop']:.6f}",
            flush=True,
        )
    print(f"[{spec.name}] Saved {len(rows)} rows to {output_csv}", flush=True)
    return existing_rows + rows


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(dict(row) for row in reader)
    return rows


def write_summary(all_rows: list[dict[str, Any]]) -> None:
    if not all_rows:
        print("[Summary] No rows to summarize.", flush=True)
        return
    summary_path = ROOT / "tables" / "vk_semantic_anchor_summary.csv"
    write_rows(summary_path, all_rows)
    print(f"[Summary] Saved {len(all_rows)} rows to {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate whether VK behaves as a structural semantic anchor.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--project", choices=["HPE", "HAR", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for spec in build_specs(args.project):
        all_rows.extend(run_project(args, spec))
    if args.dry_run:
        print(f"[Summary] DRY-RUN | would summarize {len(all_rows)} rows", flush=True)
    else:
        write_summary(all_rows)


if __name__ == "__main__":
    main()
