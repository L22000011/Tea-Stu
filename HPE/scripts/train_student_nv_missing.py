from __future__ import annotations

from _shared import base_parser, build_dataloaders, get_device, prepare_config
from models import build_model
from training.engine import load_model_checkpoint, train_student


def main() -> None:
    parser = base_parser("Train the non-visual arbitrary-modality student.")
    parser.add_argument("--teacher", type=str, required=True)
    args = parser.parse_args()
    config = prepare_config(args)
    config["student_modalities"] = ["depth", "lidar", "mmwave", "wifi-csi"]
    config["teacher_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    config.setdefault("model", {})
    student_config = dict(config)
    student_config["model"] = dict(config.get("model", {}))
    student_config["model"]["modalities"] = ["depth", "lidar", "mmwave", "wifi-csi"]
    teacher_config = dict(config)
    teacher_config["model"] = dict(config.get("model", {}))
    teacher_config["model"]["modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    student = build_model(student_config).to(device)
    teacher = build_model(teacher_config).to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    train_student(student, teacher, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

