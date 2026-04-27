from __future__ import annotations

from pathlib import Path

from _shared import base_parser, build_dataloaders, get_device, prepare_config
from models import build_model
from training.engine import load_model_checkpoint, train_student


def main() -> None:
    parser = base_parser("Run reliability fusion ablations.")
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--nonvisual", action="store_true")
    args = parser.parse_args()
    base_config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, base_config)
    teacher = build_model(base_config).to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    root_output = Path(base_config.get("output_dir", "outputs/ablation_reliability"))

    for mode in ["uniform", "attention", "uncertainty"]:
        config = dict(base_config)
        config["model"] = dict(base_config.get("model", {}))
        config["model"]["fusion_mode"] = mode
        config["output_dir"] = str(root_output / mode)
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

