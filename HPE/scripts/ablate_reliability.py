from __future__ import annotations

from pathlib import Path

from _shared import base_parser, build_dataloaders, get_device, prepare_config
from models import build_model
from training.engine import load_model_checkpoint, train_student


def main() -> None:
    parser = base_parser("Run reliability fusion ablations.")
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--nonvisual", action="store_true")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated fusion modes or all.")
    args = parser.parse_args()
    base_config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, base_config)
    teacher = build_model(base_config).to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    root_output = Path(base_config.get("output_dir", "outputs/ablation_reliability"))

    all_modes = ["uniform", "attention", "uncertainty"]
    selected = all_modes if args.variants == "all" else [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in selected if name not in all_modes]
    if unknown:
        raise ValueError(f"Unknown reliability ablation variants: {unknown}")

    for mode in selected:
        variant_output = root_output / mode
        best_path = variant_output / "best.pth"
        last_path = variant_output / "last.pth"
        if best_path.exists():
            print(f"[Ablation] Skip {mode}: existing best checkpoint at {best_path}")
            continue
        config = dict(base_config)
        config["model"] = dict(base_config.get("model", {}))
        config["model"]["fusion_mode"] = mode
        config["output_dir"] = str(variant_output)
        if last_path.exists():
            config["resume"] = str(last_path)
            print(f"[Ablation] Resume {mode} from {last_path}")
        if args.nonvisual:
            config["student_modalities"] = ["depth", "lidar", "mmwave", "wifi-csi"]
            config["model"]["modalities"] = ["depth", "lidar", "mmwave", "wifi-csi"]
        else:
            config["student_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
        config["teacher_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
        student = build_model(config).to(device)
        train_student(student, teacher, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

