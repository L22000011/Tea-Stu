from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import evaluate_combinations
from utils.modality import ALL_MODALITIES, all_nonempty_combinations
from utils.reporting import write_csv, write_markdown_table


def main() -> None:
    parser = base_parser("Evaluate all seven XRF55 modality combinations.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/all_combinations.csv")
    parser.add_argument("--output-md", type=str, default="outputs/eval/all_combinations.md")
    args = parser.parse_args()
    print("[Launch] eval_all_combinations.py")
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
        row["method"] = config.get("method", "XRF-RCD")
        row["scene"] = config.get("scene", "dml")
    write_csv(args.output_csv, rows)
    write_markdown_table(args.output_md, rows, "XRF55 all modality combinations")


if __name__ == "__main__":
    main()
