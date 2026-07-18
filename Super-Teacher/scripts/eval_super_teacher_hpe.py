from __future__ import annotations

from _shared import base_parser, get_device, prepare_config, task_dataset_config
from data import build_hpe_dataloaders
from models import build_super_teacher
from training.engine_common import combinations_for, evaluate_hpe_combinations, load_model_checkpoint, write_csv


def main() -> None:
    parser = base_parser("Evaluate Super Teacher on HPE modality combinations.")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/super_teacher_hpe_all_combinations.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    _, val_loader = build_hpe_dataloaders(args.dataset, task_dataset_config(config, "hpe"))
    model = build_super_teacher(config).to(device)
    load_model_checkpoint(model, args.checkpoint, device, strict=False)
    rows = evaluate_hpe_combinations(model, val_loader, device, combinations_for(config.get("hpe_modalities")), config.get("max_eval_batches"), task="hpe")
    for row in rows:
        row["method"] = config.get("method", "Super-Teacher")
        row["split"] = config.get("split_to_use", "unknown")
        row["task"] = "hpe"
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
