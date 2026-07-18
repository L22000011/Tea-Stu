from __future__ import annotations

from pathlib import Path

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from training.engine import train_teacher


def main() -> None:
    parser = base_parser("Run VK skeleton prompt encoder ablations.")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated encoder variants or all.")
    args = parser.parse_args()
    base_config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, base_config)
    root_output = Path(base_config.get("output_dir", "outputs/ablation_encoder"))

    variants = {"mlp_vk": 0, "skeleton_prompt": 2}
    selected = list(variants.keys()) if args.variants == "all" else [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in selected if name not in variants]
    if unknown:
        raise ValueError(f"Unknown encoder ablation variants: {unknown}")

    for name in selected:
        graph_layers = variants[name]
        variant_output = root_output / name
        best_path = variant_output / "best.pth"
        last_path = variant_output / "last.pth"
        if best_path.exists():
            print(f"[Ablation] Skip {name}: existing best checkpoint at {best_path}")
            continue
        config = dict(base_config)
        config["model"] = dict(base_config.get("model", {}))
        config["model"]["graph_layers"] = graph_layers
        config["output_dir"] = str(variant_output)
        if last_path.exists():
            config["resume"] = str(last_path)
            print(f"[Ablation] Resume {name} from {last_path}")
        config["train_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
        model = build_model_from_config(config, device)
        train_teacher(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

