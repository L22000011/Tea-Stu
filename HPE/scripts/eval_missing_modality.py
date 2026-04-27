from __future__ import annotations

from collections import defaultdict

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import evaluate_combinations, write_csv
from utils.modality import ALL_MODALITIES, NONVISUAL_MODALITIES, all_nonempty_combinations


def main() -> None:
    parser = base_parser("Evaluate performance grouped by the number of missing modalities.")
    parser.add_argument("--nonvisual", action="store_true")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/missing_modality_summary.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    reference = NONVISUAL_MODALITIES if args.nonvisual else ALL_MODALITIES
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    rows = evaluate_combinations(
        model,
        val_loader,
        device,
        combinations=all_nonempty_combinations(reference),
        max_batches=config.get("max_eval_batches"),
    )
    groups = defaultdict(list)
    total = len(reference)
    for row in rows:
        available = len(row["modality_set"].split("+"))
        groups[total - available].append(row)
    summary = []
    for missing_count, group_rows in sorted(groups.items()):
        summary.append({
            "missing_count": missing_count,
            "available_count": total - missing_count,
            "mse": sum(float(row["mse"]) for row in group_rows) / len(group_rows),
            "mpjpe": sum(float(row["mpjpe"]) for row in group_rows) / len(group_rows),
            "pa_mpjpe": sum(float(row["pa_mpjpe"]) for row in group_rows) / len(group_rows),
            "num_combinations": len(group_rows),
            "method": config.get("method", "VK-RCD"),
            "split": config.get("split_to_use", "unknown"),
            "protocol": config.get("protocol", "protocol3"),
        })
    write_csv(args.output_csv, summary)


if __name__ == "__main__":
    main()

