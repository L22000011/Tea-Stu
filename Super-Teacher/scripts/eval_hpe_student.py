from __future__ import annotations

from _shared import base_parser, get_device, prepare_config, task_dataset_config, variant_config
from data import build_hpe_dataloaders
from models import build_hpe_student
from training.engine_common import combinations_for, evaluate_hpe_combinations, load_model_checkpoint, write_csv
from utils.modality import NONVISUAL_HPE_MODALITIES


def main() -> None:
    parser = base_parser("Evaluate HPE student from Super Teacher.")
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--nonvisual", action="store_true")
    parser.add_argument("--output-csv", type=str, default="outputs/eval/super_hpe_student_combinations.csv")
    args = parser.parse_args()
    config = prepare_config(args)
    cfg = variant_config(config, args.variant)
    cfg["dataset_root"] = args.dataset
    device = get_device(args.device)
    _, val_loader = build_hpe_dataloaders(args.dataset, task_dataset_config(cfg, "hpe"))
    model = build_hpe_student(cfg).to(device)
    load_model_checkpoint(model, args.checkpoint, device, strict=False)
    reference = NONVISUAL_HPE_MODALITIES if args.nonvisual else cfg.get("student_modalities")
    rows = evaluate_hpe_combinations(model, val_loader, device, combinations_for(reference), cfg.get("max_eval_batches"))
    for row in rows:
        row["method"] = cfg.get("method", "Super-HPE-Student")
        row["split"] = cfg.get("split_to_use", "unknown")
        row["variant"] = args.variant
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
