from __future__ import annotations

from _shared import base_parser, build_dataloaders, get_device, load_optional_checkpoint, prepare_config
from models import build_model
from training.engine import evaluate_combinations
from utils.modality import NONVISUAL_MODALITIES, all_nonempty_combinations
from utils.reporting import write_csv, write_markdown_table


def main() -> None:
    parser = base_parser("Evaluate all 7 non-visual HAR modality combinations.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/nonvisual_combinations.csv")
    parser.add_argument("--output-md", type=str, default="outputs/eval/nonvisual_combinations.md")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_dataloaders(args.dataset, config)
    model = build_model(config, modalities=NONVISUAL_MODALITIES)
    model.to(device)
    load_optional_checkpoint(model, args.checkpoint, device)
    rows = evaluate_combinations(
        model,
        val_loader,
        device,
        combinations=all_nonempty_combinations(NONVISUAL_MODALITIES),
        max_batches=config.get("max_eval_batches"),
    )
    for row in rows:
        row["method"] = config.get("method", "VK-RCD-HAR-NV")
        row["split"] = config.get("split_to_use", "unknown")
        row["protocol"] = config.get("protocol", "protocol3")
    write_csv(args.output_csv, rows)
    write_markdown_table(args.output_md, rows, "HAR non-visual modality combinations")


if __name__ == "__main__":
    main()
