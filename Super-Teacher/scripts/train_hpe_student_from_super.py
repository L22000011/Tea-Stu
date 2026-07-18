from __future__ import annotations

from pathlib import Path

from _shared import base_parser, get_device, prepare_config, task_dataset_config, variant_config
from data import build_hpe_dataloaders
from models import build_hpe_student, build_super_teacher
from training.engine_common import load_model_checkpoint
from training.engine_students import train_hpe_student


def main() -> None:
    parser = base_parser("Train HPE students from the Super Teacher.")
    parser.add_argument("--variant", type=str, default="all", help="all, vk, or nv")
    args = parser.parse_args()
    config = prepare_config(args)
    device = get_device(args.device)
    teacher = build_super_teacher(config).to(device)
    load_model_checkpoint(teacher, args.teacher, device, strict=False)
    teacher.eval()
    variants = [item["name"] for item in config.get("students", [])] if args.variant == "all" else [args.variant]
    for name in variants:
        cfg = variant_config(config, name)
        cfg["dataset_root"] = args.dataset
        output_dir = Path(cfg["output_dir"])
        if output_dir.joinpath("best.pth").exists():
            print(f"[Skip] HPE student variant {name} already has {output_dir / 'best.pth'}")
            continue
        if output_dir.joinpath("last.pth").exists() and not cfg.get("resume"):
            cfg["resume"] = str(output_dir / "last.pth")
            print(f"[Resume] HPE student variant {name} from {cfg['resume']}")
        train_loader, val_loader = build_hpe_dataloaders(args.dataset, task_dataset_config(cfg, "hpe"))
        student = build_hpe_student(cfg).to(device)
        train_hpe_student(student, teacher, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
