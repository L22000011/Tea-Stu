from __future__ import annotations

from pathlib import Path

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from training.engine import train_teacher


def main() -> None:
    parser = base_parser("Run VK skeleton prompt encoder ablations.")
    args = parser.parse_args()
    base_config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, base_config)
    root_output = Path(base_config.get("output_dir", "outputs/ablation_encoder"))

    for name, graph_layers in {"mlp_vk": 0, "skeleton_prompt": 2}.items():
        config = dict(base_config)
        config["model"] = dict(base_config.get("model", {}))
        config["model"]["graph_layers"] = graph_layers
        config["output_dir"] = str(root_output / name)
        config["train_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
        model = build_model_from_config(config, device)
        train_teacher(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

