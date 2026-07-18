from __future__ import annotations

from _shared import base_parser, build_dataloaders, build_model_from_config, get_device, load_model_checkpoint, prepare_config
from models import build_model
from training.engine import train_student


def main() -> None:
    parser = base_parser("Train the no-semantic ablation for XRF55.")
    parser.add_argument("--teacher", type=str, required=True)
    args = parser.parse_args()
    print("[Launch] ablate_no_semantic.py")
    print(f"[Launch] Device: {args.device}")
    print(f"[Launch] Dataset: {args.dataset}")
    print(f"[Launch] Config: {args.config}")
    print(f"[Launch] Teacher: {args.teacher}")
    config = prepare_config(args)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.dataset, config)
    student = build_model_from_config(config, device)
    teacher = build_model(config)
    teacher.to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    train_student(student, teacher, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
