from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import train_teacher


def main() -> None:
    parser = base_parser("Train the VK full-modality structural teacher.")
    args = parser.parse_args()
    print("[Launch] train_teacher_full.py")
    config = prepare_config(args)
    config["train_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    device = get_device(args.device)
    print(f"[Launch] Device: {device}")
    print(f"[Launch] Dataset: {args.dataset}")
    print(f"[Launch] Config: {args.config}")
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    print("[Launch] Building teacher model...")
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    print("[Launch] Start teacher training.")
    train_teacher(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

