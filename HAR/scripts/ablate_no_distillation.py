from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, prepare_config
from training.engine import train_student


def main() -> None:
    parser = base_parser("Ablation: random missing modality training without teacher distillation.")
    args = parser.parse_args()
    print("[Launch] ablate_no_distillation.py")
    config = prepare_config(args)
    config["method"] = config.get("method", "VK-RCD-HAR-NoKD")
    config["student_modalities"] = ["vk", "depth", "lidar", "mmwave"]
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    student = build_model_from_config(config, device)
    train_student(student, None, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()

