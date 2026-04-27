from __future__ import annotations

from _shared import base_parser, build_dataloaders, get_device, prepare_config
from models import build_model
from training.engine import evaluate_combinations, load_model_checkpoint, write_csv
from utils.modality import NONVISUAL_MODALITIES, all_nonempty_combinations


def main() -> None:
    parser = base_parser("Evaluate all 15 non-visual D/L/R/W modality combinations.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/nonvisual_combinations.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    config.setdefault("model", {})
    config["model"] = dict(config["model"])
    config["model"]["modalities"] = NONVISUAL_MODALITIES
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model(config).to(device)
    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint, device, strict=False)
    rows = evaluate_combinations(
        model,
        val_loader,
        device,
        combinations=all_nonempty_combinations(NONVISUAL_MODALITIES),
        max_batches=config.get("max_eval_batches"),
    )
    for row in rows:
        row["method"] = config.get("method", "VK-RCD-Student-NV")
        row["split"] = config.get("split_to_use", "unknown")
        row["protocol"] = config.get("protocol", "protocol3")
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()

