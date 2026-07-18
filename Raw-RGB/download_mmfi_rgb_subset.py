from __future__ import annotations

import argparse
import time
from pathlib import Path

from remotezip import RemoteZip


URL = "https://mmfi-dataset.oss-ap-southeast-1.aliyuncs.com/anonymized_rgb_images/all_images.zip"


PROTOCOL_MINI_TARGETS = [
    # E01: ordinary random/cross-subject source scene
    "E01/S02/A01/rgb",
    "E01/S02/A10/rgb",
    "E01/S02/A25/rgb",
    "E01/S05/A01/rgb",
    "E01/S05/A10/rgb",
    "E01/S05/A25/rgb",
    # E02
    "E02/S12/A01/rgb",
    "E02/S12/A10/rgb",
    "E02/S12/A25/rgb",
    "E02/S15/A01/rgb",
    "E02/S15/A10/rgb",
    "E02/S15/A25/rgb",
    # E03
    "E03/S22/A01/rgb",
    "E03/S22/A10/rgb",
    "E03/S22/A25/rgb",
    "E03/S25/A01/rgb",
    "E03/S25/A10/rgb",
    "E03/S25/A25/rgb",
    # E04: held-out scene candidate for cross-scene sanity checks
    "E04/S32/A01/rgb",
    "E04/S32/A10/rgb",
    "E04/S32/A25/rgb",
    "E04/S35/A01/rgb",
    "E04/S35/A10/rgb",
    "E04/S35/A25/rgb",
]


def normalize(path: str) -> str:
    return path.replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small subset from MMFi anonymized RGB all_images.zip without fetching the full zip."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\Deskbook\Tea\Raw-RGB\MMFi_RGB_SUBSET"),
        help="Directory where selected files will be extracted.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Substring used to select files inside the zip. "
            "Can be repeated. If omitted, defaults to one E01/S02/A25/rgb action."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["single-action", "protocol-mini", "global-percent"],
        default="single-action",
        help=(
            "single-action downloads a tiny action clip. protocol-mini covers "
            "E01-E04, multiple subjects, and multiple actions for sanity training. "
            "global-percent samples from all RGB images across the full archive."
        ),
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=20.0,
        help="Percentage of all RGB PNG files to sample when --preset global-percent is used.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=120,
        help=(
            "Global maximum number of matched files to download. "
            "Use 0 for no global limit. Ignored by protocol-mini unless explicitly set."
        ),
    )
    parser.add_argument(
        "--max-files-per-target",
        type=int,
        default=60,
        help="Maximum number of image files per target action for protocol-mini. Use 0 for all matched files.",
    )
    parser.add_argument(
        "--max-new-files",
        type=int,
        default=0,
        help=(
            "Stop after extracting this many new files. Existing files are skipped. "
            "Use this to grow a large subset in resumable chunks."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print extraction progress every N selected files.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry each file extraction this many times before skipping it.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=3.0,
        help="Seconds to wait between retries after a transient network error.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list zip paths and matched files; do not extract.",
    )
    parser.add_argument(
        "--show-first",
        type=int,
        default=80,
        help="Number of first zip paths to print for checking the internal structure.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.target:
        targets = [normalize(t).strip("/") for t in args.target]
    elif args.preset == "protocol-mini":
        targets = list(PROTOCOL_MINI_TARGETS)
    elif args.preset == "global-percent":
        targets = ["MMFi_Defaced_RGB/"]
    else:
        targets = ["E01/S02/A25/rgb", "S02/A25/rgb"]

    print("Remote zip:", URL)
    print("Output:", args.output)
    print("Targets:", targets)
    print("Opening remote zip. This can take a while for the first index read...")

    with RemoteZip(URL) as zf:
        names = zf.namelist()
        print("Total files in zip:", len(names))

        print(f"\nFirst {min(args.show_first, len(names))} paths:")
        for name in names[: args.show_first]:
            print("  ", name)

        selected = []
        if args.preset == "global-percent" and not args.target:
            image_names = [
                name for name in names
                if normalize(name).startswith("MMFi_Defaced_RGB/")
                and normalize(name).endswith(".png")
            ]
            image_names = sorted(image_names)
            total_images = len(image_names)
            if total_images == 0:
                print("No RGB png files found in remote zip.")
                return
            desired = int(round(total_images * (args.percent / 100.0)))
            desired = max(1, min(desired, total_images))
            step = max(1, total_images // desired)
            selected = image_names[::step][:desired]
            print(f"Global RGB png files: {total_images}")
            print(f"Requested percent: {args.percent:.2f}%")
            print(f"Desired sample count: {desired}")
            print(f"Sampling step: every {step} file(s)")
        elif args.preset == "protocol-mini" and not args.target:
            for target in targets:
                matched = []
                for name in names:
                    norm = normalize(name)
                    if target in norm and not norm.endswith("/"):
                        matched.append(name)
                if args.max_files_per_target > 0:
                    matched = matched[: args.max_files_per_target]
                print(f"Target {target}: selected {len(matched)} files")
                selected.extend(matched)
        else:
            for name in names:
                norm = normalize(name)
                if norm.endswith("/"):
                    continue
                if any(target in norm for target in targets):
                    selected.append(name)

        print("\nMatched files before limit:", len(selected))
        apply_global_limit = (
            args.max_files > 0
            and not (args.preset == "protocol-mini" and not args.target)
            and not (args.preset == "global-percent" and not args.target)
        )
        if apply_global_limit:
            selected = selected[: args.max_files]
        print("Files selected for download:", len(selected))
        for name in selected[:30]:
            print("  ", name)

        if not selected:
            print("\nNo files matched. Re-run with --target using one of the printed path patterns.")
            return

        if args.list_only:
            print("\nList-only mode enabled; no files extracted.")
            return

        new_files = 0
        skipped_existing = 0
        failed_files = []
        for idx, name in enumerate(selected, 1):
            destination = args.output / name
            if destination.exists():
                skipped_existing += 1
                if args.progress_every > 0 and idx % args.progress_every == 0:
                    print(
                        f"[{idx}/{len(selected)}] skipped_existing={skipped_existing} "
                        f"new_files={new_files}"
                    )
                continue
            if args.max_new_files > 0 and new_files >= args.max_new_files:
                print(f"Reached --max-new-files={args.max_new_files}; stopping chunk.")
                break
            if args.progress_every <= 1 or idx % args.progress_every == 0:
                print(f"[{idx}/{len(selected)}] extracting {name}")
            for attempt in range(1, args.retries + 2):
                try:
                    zf.extract(name, args.output)
                    new_files += 1
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    if attempt > args.retries:
                        failed_files.append(name)
                        print(f"[WARN] failed after {args.retries} retries: {name} | {exc}")
                        break
                    print(
                        f"[WARN] extract failed, retry {attempt}/{args.retries}: "
                        f"{name} | {exc}"
                    )
                    time.sleep(args.retry_sleep)

        print(
            f"\nExtraction summary: new_files={new_files}, "
            f"skipped_existing={skipped_existing}, failed_files={len(failed_files)}"
        )
        if failed_files:
            failed_log = args.output / "failed_downloads.txt"
            with failed_log.open("a", encoding="utf-8") as f:
                for name in failed_files:
                    f.write(name + "\n")
            print("Failed file list appended to:", failed_log)

    print("\nDone.")
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
