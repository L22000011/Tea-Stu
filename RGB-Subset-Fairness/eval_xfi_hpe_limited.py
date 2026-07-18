from __future__ import annotations

import argparse
import csv
from itertools import combinations
import os
import sys
from pathlib import Path

import torch
from torch import nn
import yaml
from tqdm import tqdm


DISPLAY_MODALITIES = ["RGB", "Depth", "Lidar", "mmWave", "WiFi-CSI"]

# X-Fi HPE internally orders the mask as visual, depth, mmWave, lidar, WiFi.
MASK_INDEX = {
    "RGB": 0,
    "Depth": 1,
    "mmWave": 2,
    "Lidar": 3,
    "WiFi-CSI": 4,
}


def hpe_combo_specs() -> list[tuple[str, list[bool]]]:
    specs: list[tuple[str, list[bool]]] = []
    for keep_count in range(1, len(DISPLAY_MODALITIES) + 1):
        for combo in combinations(DISPLAY_MODALITIES, keep_count):
            mask = [False, False, False, False, False]
            for name in combo:
                mask[MASK_INDEX[name]] = True
            specs.append(("+".join(combo), mask))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fixed-budget X-Fi HPE evaluation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pt-weights", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-eval-batches", type=int, default=100)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> str:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return "checkpoint_dict"
    model.load_state_dict(checkpoint)
    return "state_dict"


def write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modality", "mse", "mpjpe", "pampjpe"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from evaluate import error
    from syn_DI_dataset import make_dataset, make_dataloader
    from utils import collate_fn_padd, _move_hpe_batch_to_device
    from X_Fi import X_Fi

    device = torch.device(args.device)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    _, val_dataset = make_dataset(str(args.dataset), config)
    generator = torch.manual_seed(config["init_rand_seed"])
    val_loader = make_dataloader(
        val_dataset,
        is_training=False,
        generator=generator,
        **config["loader"],
        collate_fn=collate_fn_padd,
    )

    model = X_Fi().to(device)
    checkpoint_kind = load_checkpoint(model, args.pt_weights, device)
    print(f"[RGB-Subset-XFi-Eval] project_root={project_root}")
    print(f"[RGB-Subset-XFi-Eval] checkpoint={args.pt_weights}")
    print(f"[RGB-Subset-XFi-Eval] checkpoint_format={checkpoint_kind}")
    combo_specs = hpe_combo_specs()
    print(f"[RGB-Subset-XFi-Eval] max_eval_batches_per_combo={args.max_eval_batches}")
    print(f"[RGB-Subset-XFi-Eval] combinations={len(combo_specs)}")
    print("[RGB-Subset-XFi-Eval] mode=all HPE non-empty modality combinations")

    model.eval()
    criterion = nn.MSELoss()
    total_steps = min(args.max_eval_batches, len(val_loader)) if args.max_eval_batches else len(val_loader)
    rows = []

    for combo_index, (modality_name, modality_list) in enumerate(combo_specs, start=1):
        print(
            f"[RGB-Subset-XFi-Eval] combo {combo_index}/{len(combo_specs)} | "
            f"modality={modality_name} | target_batches={total_steps}",
            flush=True,
        )
        seen_samples = 0
        totals = {"mse": 0.0, "mpjpe": 0.0, "pampjpe": 0.0}
        with torch.no_grad():
            for batch_idx, data in enumerate(
                tqdm(val_loader, total=total_steps, desc=f"xfi-eval:{combo_index:02d}/{len(combo_specs)}:{modality_name}")
            ):
                if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
                    break
                vk_data, depth_data, mmwave_data, lidar_data, wifi_data, labels, _ = _move_hpe_batch_to_device(data, device)
                outputs = model(vk_data, depth_data, mmwave_data, lidar_data, wifi_data, modality_list).float()
                batch_size = vk_data.size(0)
                seen_samples += batch_size
                totals["mse"] += criterion(outputs, labels).item() * batch_size
                mpjpe, pampjpe = error(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                totals["mpjpe"] += float(mpjpe) * batch_size
                totals["pampjpe"] += float(pampjpe) * batch_size

        if seen_samples == 0:
            raise RuntimeError(f"No validation samples were evaluated for {modality_name}.")

        row = {
            "modality": modality_name,
            "mse": totals["mse"] / seen_samples,
            "mpjpe": totals["mpjpe"] / seen_samples,
            "pampjpe": totals["pampjpe"] / seen_samples,
        }
        rows.append(row)
        print(
            f"modality: {row['modality']}, mse: {row['mse']:.8f}, "
            f"mpjpe: {row['mpjpe']:.8f}, pampjpe: {row['pampjpe']:.8f}",
            flush=True,
        )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.outputs_dir / "main_eval_latest.csv", rows)
    write_rows(args.outputs_dir / "main_eval_all_combinations.csv", rows)
    text = "".join(
        f"modality: {row['modality']}, mse: {row['mse']:.8f}, "
        f"mpjpe: {row['mpjpe']:.8f}, pampjpe: {row['pampjpe']:.8f}\n"
        for row in rows
    )
    (args.outputs_dir / "main_eval_latest.txt").write_text(text, encoding="utf-8")
    print(f"[RGB-Subset-XFi-Eval] Saved: {args.outputs_dir / 'main_eval_latest.csv'}")


if __name__ == "__main__":
    main()
