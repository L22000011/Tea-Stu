from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_optional_checkpoint, prepare_config
from training.engine import train_teacher


def main() -> None:
    parser = base_parser("Train the full-modality VK baseline.")
    args = parser.parse_args()
    config = prepare_config(args)
    config["train_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    load_optional_checkpoint(model, args.checkpoint, device)
    train_teacher(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

