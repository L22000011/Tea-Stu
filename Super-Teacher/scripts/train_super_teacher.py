from __future__ import annotations

from _shared import base_parser, get_device, prepare_config, task_dataset_config
from data import build_har_dataloaders, build_hpe_dataloaders
from models import build_super_teacher
from training.engine_common import load_model_checkpoint
from training.engine_super import train_super_teacher


def main() -> None:
    parser = base_parser("Train the cross-task Super Teacher.")
    args = parser.parse_args()
    print("[Launch] train_super_teacher.py")
    config = prepare_config(args)
    device = get_device(args.device)
    weights = config.get("loss", {})
    use_hpe = float(weights.get("lambda_hpe", 1.0)) > 0
    use_har = float(weights.get("lambda_har", 1.0)) > 0
    hpe_train, hpe_val = build_hpe_dataloaders(args.dataset, task_dataset_config(config, "hpe")) if use_hpe else (None, None)
    har_train, har_val = build_har_dataloaders(args.dataset, task_dataset_config(config, "har")) if use_har else (None, None)
    model = build_super_teacher(config).to(device)
    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint, device, strict=False)
    train_super_teacher(model, hpe_train, hpe_val, har_train, har_val, config, device)


if __name__ == "__main__":
    main()
