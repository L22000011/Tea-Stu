from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from training.engine import train_student


def main() -> None:
    parser = base_parser("Train the no-distillation ablation for XRF55.")
    args = parser.parse_args()
    print("[Launch] ablate_no_distillation.py")
    print(f"[Launch] Device: {args.device}")
    print(f"[Launch] Dataset: {args.dataset}")
    print(f"[Launch] Config: {args.config}")
    config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    train_student(model, None, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
