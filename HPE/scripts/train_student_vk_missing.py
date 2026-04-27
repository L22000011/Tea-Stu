from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from models import build_model
from training.engine import load_model_checkpoint, train_student


def main() -> None:
    parser = base_parser("Train the arbitrary-modality VK student.")
    parser.add_argument("--teacher", type=str, required=True)
    args = parser.parse_args()
    config = prepare_config(args)
    config["student_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    config["teacher_modalities"] = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    student = build_model_from_config(config, device)
    teacher = build_model(config).to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    train_student(student, teacher, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

