from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import move_batch_to_device
from utils.modality import ALL_MODALITIES, all_nonempty_combinations, canonicalize_modalities


def parse_modality_sets(values: Iterable[str] | None) -> List[List[str]]:
    if not values:
        return all_nonempty_combinations(ALL_MODALITIES)
    parsed = []
    for value in values:
        parsed.append(canonicalize_modalities(value.replace(",", "+").split("+")))
    return parsed


def missing_setting(selected: List[str]) -> str:
    missing = [name for name in ALL_MODALITIES if name not in selected]
    return "available={};missing={}".format(
        "+".join(selected),
        "+".join(missing) if missing else "none",
    )


def meta_value(meta: Dict[str, Any], key: str) -> str:
    value = meta.get(key, "NA")
    return str(value)


@torch.no_grad()
def export_weights(model, val_loader, device: torch.device, combinations: List[List[str]], max_batches: int | None) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    sample_id = 0
    for combo in combinations:
        combo_name = "+".join(combo)
        iterator = tqdm(val_loader, desc=f"weights:{combo_name}", leave=False)
        for batch_index, batch in enumerate(iterator):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch["inputs"], combo)
            if "alphas" not in output:
                raise KeyError("Model output does not contain reliability weights under key 'alphas'.")
            alphas = output["alphas"].detach().cpu()
            targets = batch["target"].detach().cpu()
            metas = batch.get("meta", [])
            for local_index in range(alphas.size(0)):
                meta = metas[local_index] if local_index < len(metas) else {}
                label = meta_value(meta, "action")
                if label == "NA" and local_index < targets.numel():
                    label = str(int(targets[local_index].item()))
                for modality_index, modality in enumerate(combo):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "task": "har",
                            "batch_index": batch_index,
                            "sample_index": local_index,
                            "missing_setting": missing_setting(combo),
                            "modality_set": combo_name,
                            "modality": modality,
                            "weight": float(alphas[local_index, modality_index].item()),
                            "label": label,
                            "scene": meta_value(meta, "scene"),
                            "subject": meta_value(meta, "subject"),
                            "frame_index": meta_value(meta, "idx"),
                        }
                    )
                sample_id += 1
    return rows


def write_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "task",
        "batch_index",
        "sample_index",
        "missing_setting",
        "modality_set",
        "modality",
        "weight",
        "label",
        "scene",
        "subject",
        "frame_index",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Reliability] Saved {len(rows)} HAR reliability rows to {output_path}")


def main() -> None:
    parser = base_parser("Export HAR reliability weights from VK-RMD fusion alphas.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/har_reliability_weights.csv")
    parser.add_argument("--max-batches", type=int, default=100, help="Number of validation batches per modality set.")
    parser.add_argument(
        "--modality-set",
        action="append",
        default=None,
        help="Optional modality set such as vk+depth+mmwave. Repeat to export multiple sets.",
    )
    args = parser.parse_args()

    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    combinations = parse_modality_sets(args.modality_set)
    rows = export_weights(model, val_loader, device, combinations, args.max_batches)
    write_rows(args.output_csv, rows)


if __name__ == "__main__":
    main()
