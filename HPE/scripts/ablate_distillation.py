from __future__ import annotations

from pathlib import Path

from _shared import base_parser, build_dataloaders, get_device, prepare_config
from models import build_model
from training.engine import load_model_checkpoint, train_student


VARIANTS = {
    "no_kd": {"lambda_out": 0.0, "lambda_token": 0.0, "lambda_bone": 0.0, "lambda_rel": 0.0},
    "output_kd": {"lambda_out": 0.5, "lambda_token": 0.0, "lambda_bone": 0.0, "lambda_rel": 0.0},
    "output_token_kd": {"lambda_out": 0.5, "lambda_token": 0.2, "lambda_bone": 0.0, "lambda_rel": 0.0},
    "output_bone_kd": {"lambda_out": 0.5, "lambda_token": 0.0, "lambda_bone": 0.1, "lambda_rel": 0.0},
    "full_structural_kd": {"lambda_out": 0.5, "lambda_token": 0.2, "lambda_bone": 0.1, "lambda_rel": 0.05},
}


def main() -> None:
    parser = base_parser("Run structural distillation ablations.")
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--nonvisual", action="store_true")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated variant names or all.")
    args = parser.parse_args()
    base_config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, base_config)
    teacher = build_model(base_config).to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    root_output = Path(base_config.get("output_dir", "outputs/ablation_distillation"))

    selected = list(VARIANTS.keys()) if args.variants == "all" else [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in selected if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown distillation ablation variants: {unknown}")

    for name in selected:
        weights = VARIANTS[name]
        variant_output = root_output / name
        best_path = variant_output / "best.pth"
        last_path = variant_output / "last.pth"
        if best_path.exists():
            print(f"[Ablation] Skip {name}: existing best checkpoint at {best_path}")
            continue
        config = dict(base_config)
        config["loss"] = dict(base_config.get("loss", {}))
        config["loss"].update(weights)
        config["output_dir"] = str(variant_output)
        if last_path.exists():
            config["resume"] = str(last_path)
            print(f"[Ablation] Resume {name} from {last_path}")
        if args.nonvisual:
            config["student_modalities"] = ["depth", "lidar", "mmwave", "wifi-csi"]
            student_config = dict(config)
            student_config["model"] = dict(config.get("model", {}))
            student_config["model"]["modalities"] = ["depth", "lidar", "mmwave", "wifi-csi"]
        else:
            config["student_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
            student_config = config
        config["teacher_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
        student = build_model(student_config).to(device)
        train_student(student, teacher, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

