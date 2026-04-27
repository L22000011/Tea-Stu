from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from training.engine import train_teacher


def main() -> None:
    parser = base_parser("Ablation: train HAR with uniform modality fusion weights.")
    args = parser.parse_args()
    print("[Launch] ablate_uniform_fusion.py")
    config = prepare_config(args)
    config.setdefault("model", {})
    config["model"]["fusion_mode"] = "uniform"
    config["method"] = config.get("method", "VK-RCD-HAR-UniformFusion")
    config["train_modalities"] = ["vk", "depth", "lidar", "mmwave"]
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    model = build_model_from_config(config, device)
    train_teacher(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

