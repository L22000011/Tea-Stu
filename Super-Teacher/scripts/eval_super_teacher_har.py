from __future__ import annotations

from _shared import base_parser, get_device, prepare_config, task_dataset_config
from data import build_har_dataloaders
from models import build_super_teacher
from training.engine_common import combinations_for, evaluate_har_combinations, load_model_checkpoint, write_csv


def main() -> None:
    parser = base_parser("Evaluate Super Teacher on HAR modality combinations.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/super_teacher_har_all_combinations.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_har_dataloaders(args.dataset, task_dataset_config(config, "har"))
    model = build_super_teacher(config).to(device)
    load_model_checkpoint(model, args.checkpoint, device, strict=False)
    rows = evaluate_har_combinations(model, val_loader, device, combinations_for(config.get("har_modalities")), config.get("max_eval_batches"), task="har")
    for row in rows:
        row["method"] = config.get("method", "Super-Teacher")
        row["split"] = config.get("split_to_use", "unknown")
        row["task"] = "har"
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
