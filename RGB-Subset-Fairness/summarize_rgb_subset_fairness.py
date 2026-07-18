from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "method", "visual_input", "protocol", "train_samples", "test_samples",
        "combination_count", "avg_metric", "full_metric", "best_subset", "worst_subset", "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(repo: Path, names: list[str]) -> Path:
    for name in names:
        path = repo / name
        if path.exists():
            return path
    return repo / names[0]


def manifest_counts(manifest_path: Path, protocol: str) -> tuple[int, int]:
    rows = read_rows(manifest_path)
    train_subjects = {
        "S01", "S02", "S03", "S04", "S06", "S07", "S08", "S09",
        "S11", "S12", "S13", "S14", "S16", "S17", "S18", "S19",
        "S21", "S22", "S23", "S24", "S26", "S27", "S28", "S29",
        "S31", "S32", "S33", "S34", "S36", "S37", "S38", "S39",
    }
    val_subjects = {"S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"}
    if protocol == "cross_scene":
        return (
            sum(1 for row in rows if row["scene"] in ["E01", "E02", "E03"]),
            sum(1 for row in rows if row["scene"] == "E04"),
        )
    return (
        sum(1 for row in rows if row["subject"] in train_subjects),
        sum(1 for row in rows if row["subject"] in val_subjects),
    )


def summarize_hpe_rows(rows: list[dict[str, str]], modality_key: str, metric_key: str) -> tuple[str, str, str, str, int]:
    values = []
    for row in rows:
        try:
            values.append((row[modality_key], float(row[metric_key])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return "NA", "NA", "NA", "NA", 0
    avg = sum(v for _, v in values) / len(values)
    best = min(values, key=lambda item: item[1])
    worst = max(values, key=lambda item: item[1])
    full = next((v for name, v in values if name.lower().replace(" ", "") in ["vk+depth+lidar+mmwave+wifi-csi", "rgb+depth+lidar+mmwave+wifi-csi"]), "NA")
    return f"{avg:.6f}", f"{full:.6f}" if isinstance(full, float) else "NA", best[0], worst[0], len(values)


def summarize_har_rows(rows: list[dict[str, str]], modality_key: str, metric_key: str) -> tuple[str, str, str, str, int]:
    values = []
    for row in rows:
        try:
            values.append((row[modality_key], float(row[metric_key])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return "NA", "NA", "NA", "NA", 0
    avg = sum(v for _, v in values) / len(values)
    best = max(values, key=lambda item: item[1])
    worst = min(values, key=lambda item: item[1])
    full = next((v for name, v in values if name.lower().replace(" ", "") in ["vk+depth+lidar+mmwave", "rgb+depth+lidar+mmwave"]), "NA")
    return f"{avg:.6f}", f"{full:.6f}" if isinstance(full, float) else "NA", best[0], worst[0], len(values)


def rename_visual(name: str, visual_name: str) -> str:
    if visual_name == "RGB":
        return name.replace("VK", "RGB").replace("vk", "RGB")
    return name


def main() -> None:
    parser = argparse.ArgumentParser("Summarize RGB-subset fairness results.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo
    hpe_project = first_existing(repo, ["HPE", "MMFi_HPE"])
    har_project = first_existing(repo, ["HAR", "MMFi_HAR"])
    manifest = repo / "outputs" / "rgb_subset_manifest" / "manifest.csv"
    hpe_rows = []
    har_rows = []
    for protocol in ["cross_scene", "cross_subject"]:
        train_count, test_count = manifest_counts(manifest, protocol)
        xfi_hpe = read_rows(repo / "Ori-RGB-HPE" / "outputs_rgb_subset" / protocol / "eval" / "main_eval_latest.csv")
        xfi_hpe = [{**row, "modality": rename_visual(row.get("modality", ""), "RGB")} for row in xfi_hpe]
        avg, full, best, worst, combo_count = summarize_hpe_rows(xfi_hpe, "modality", "mpjpe")
        hpe_rows.append(
            {
                "task": "HPE", "method": "X-Fi", "visual_input": "RGB", "protocol": protocol,
                "train_samples": train_count, "test_samples": test_count, "combination_count": combo_count,
                "avg_metric": avg, "full_metric": full, "best_subset": best, "worst_subset": worst,
                "note": "RGB/defaced-RGB baseline on RGB-available subset; lower MPJPE is better.",
            }
        )
        ours_hpe = read_rows(hpe_project / "outputs_rgb_subset" / "eval" / f"student_vk_{protocol}_all_combinations.csv")
        avg, full, best, worst, combo_count = summarize_hpe_rows(ours_hpe, "modality_set", "mpjpe")
        hpe_rows.append(
            {
                "task": "HPE", "method": "VK-RMD", "visual_input": "VK", "protocol": protocol,
                "train_samples": train_count, "test_samples": test_count, "combination_count": combo_count,
                "avg_metric": avg, "full_metric": full, "best_subset": best, "worst_subset": worst,
                "note": "Same RGB-available samples, RGB-free VK representation; lower MPJPE is better.",
            }
        )

        xfi_har = read_rows(repo / "Ori-RGB-HAR" / "outputs_rgb_subset" / protocol / "eval" / "main_eval_latest.csv")
        xfi_har = [{**row, "modality": rename_visual(row.get("modality", ""), "RGB")} for row in xfi_har]
        avg, full, best, worst, combo_count = summarize_har_rows(xfi_har, "modality", "accuracy")
        har_rows.append(
            {
                "task": "HAR", "method": "X-Fi", "visual_input": "RGB", "protocol": protocol,
                "train_samples": train_count, "test_samples": test_count, "combination_count": combo_count,
                "avg_metric": avg, "full_metric": full, "best_subset": best, "worst_subset": worst,
                "note": "RGB/defaced-RGB baseline on RGB-available subset; higher accuracy is better.",
            }
        )
        ours_har = read_rows(har_project / "outputs_rgb_subset" / "eval" / f"student_vk_{protocol}_all_combinations.csv")
        avg, full, best, worst, combo_count = summarize_har_rows(ours_har, "modality_set", "acc")
        har_rows.append(
            {
                "task": "HAR", "method": "VK-RMD", "visual_input": "VK", "protocol": protocol,
                "train_samples": train_count, "test_samples": test_count, "combination_count": combo_count,
                "avg_metric": avg, "full_metric": full, "best_subset": best, "worst_subset": worst,
                "note": "Same RGB-available samples, RGB-free VK representation; higher accuracy is better.",
            }
        )

    write_rows(repo / "tables" / "rgb_subset_fairness_hpe.csv", hpe_rows)
    write_rows(repo / "tables" / "rgb_subset_fairness_har.csv", har_rows)
    print("Saved:", repo / "tables" / "rgb_subset_fairness_hpe.csv")
    print("Saved:", repo / "tables" / "rgb_subset_fairness_har.csv")


if __name__ == "__main__":
    main()
