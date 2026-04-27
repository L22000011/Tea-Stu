from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_model_checkpoint, prepare_config
from models import build_model
from training.engine import train_student


def main() -> None:
    parser = base_parser("Train the VK HAR student with arbitrary missing modalities.")
    parser.add_argument("--teacher", type=str, required=True)
    args = parser.parse_args()
    print("[Launch] train_student_vk_missing.py")
    config = prepare_config(args)
    config["student_modalities"] = ["vk", "depth", "lidar", "mmwave"]
    config["teacher_modalities"] = ["vk", "depth", "lidar", "mmwave"]
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    print("[Launch] Building student model...")
    student = build_model_from_config(config, device)
    print("[Launch] Building teacher model...")
    teacher = build_model(config)
    teacher.to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    train_student(student, teacher, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

