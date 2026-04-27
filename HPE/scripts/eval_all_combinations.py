from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import evaluate_combinations, write_csv
from utils.modality import ALL_MODALITIES, all_nonempty_combinations


def main() -> None:
    parser = base_parser("Evaluate all 31 VK/D/L/R/W modality combinations.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/all_combinations.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    rows = evaluate_combinations(
        model,
        val_loader,
        device,
        combinations=all_nonempty_combinations(ALL_MODALITIES),
        max_batches=config.get("max_eval_batches"),
    )
    for row in rows:
        row["method"] = config.get("method", "VK-RCD")
        row["split"] = config.get("split_to_use", "unknown")
        row["protocol"] = config.get("protocol", "protocol3")
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()

