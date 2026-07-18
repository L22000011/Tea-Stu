from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from training.engine import train_teacher


def main() -> None:
    parser = base_parser("Train the full-modality XRF55 baseline.")
    args = parser.parse_args()
    print("[Launch] train_baseline_full.py")
    print(f"[Launch] Device: {args.device}")
    print(f"[Launch] Dataset: {args.dataset}")
    print(f"[Launch] Config: {args.config}")
    config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    print("[Launch] Building baseline model...")
    model = build_model_from_config(config, device)
    train_teacher(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
