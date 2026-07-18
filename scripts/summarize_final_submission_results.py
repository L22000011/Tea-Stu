from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_dir(*candidates: str) -> Path:
    for name in candidates:
        path = PROJECT_ROOT / name
        if path.exists():
            return path
    return PROJECT_ROOT / candidates[0]


HPE_ROOT = resolve_project_dir("HPE", "MMFi_HPE")
HAR_ROOT = resolve_project_dir("HAR", "MMFi_HAR")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def modality_count(row: dict[str, str]) -> int:
    value = row.get("modality_set") or row.get("modality") or ""
    return len([item for item in value.split("+") if item])


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_hpe(path: Path) -> tuple[int, str, str]:
    rows = read_rows(path)
    if not rows:
        return 0, "missing", "missing"
    mpjpe = [float(row["mpjpe"]) * 1000.0 for row in rows if row.get("mpjpe")]
    full_count = max(modality_count(row) for row in rows)
    full_rows = [row for row in rows if modality_count(row) == full_count]
    full = float(full_rows[-1]["mpjpe"]) * 1000.0 if full_rows else None
    avg = mean(mpjpe)
    return len(rows), f"{avg:.2f} mm MPJPE" if avg is not None else "NA", f"{full:.2f} mm MPJPE" if full is not None else "NA"


def summarize_har(path: Path) -> tuple[int, str, str]:
    rows = read_rows(path)
    if not rows:
        return 0, "missing", "missing"
    acc = [float(row["acc"]) * 100.0 for row in rows if row.get("acc")]
    full_count = max(modality_count(row) for row in rows)
    full_rows = [row for row in rows if modality_count(row) == full_count]
    full = float(full_rows[-1]["acc"]) * 100.0 if full_rows else None
    avg = mean(acc)
    return len(rows), f"{avg:.2f}% Acc" if avg is not None else "NA", f"{full:.2f}% Acc" if full is not None else "NA"


def add_row(rows: list[dict[str, str]], task: str, method: str, setting: str, path: Path, interpretation: str, policy: str) -> None:
    if task == "HPE":
        count, avg, full = summarize_hpe(path)
    else:
        count, avg, full = summarize_har(path)
    rows.append(
        {
            "task": task,
            "method": method,
            "setting": setting,
            "combination_count": str(count),
            "avg_metric": avg,
            "full_metric": full,
            "interpretation": interpretation,
            "main_text_policy": policy,
        }
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "method", "setting", "combination_count", "avg_metric", "full_metric", "interpretation", "main_text_policy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Submission Supplement Summary",
        "",
        "This file summarizes the final supplement experiments used to close the VK-RMD submission evidence chain.",
        "",
        "| Task | Method | Setting | N | Average | Full | Policy |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['setting']} | {row['combination_count']} | "
            f"{row['avg_metric']} | {row['full_metric']} | {row['main_text_policy']} |"
        )
    lines.extend(
        [
            "",
            "## Writing Rules",
            "",
            "- If no-random is worse than VK-RMD, use it as direct evidence that random missing-modality training is necessary.",
            "- If no-random is similar to VK-RMD, describe the gain by missing severity rather than overclaiming the average.",
            "- If HPE KD variants do not beat no-KD, write distillation as a training component, not as a universally improving module.",
            "- HPE baseline in the main paper must use the official X-Fi table value: 103.70 mm average MPJPE and 83.70 mm full-modality MPJPE.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, str]] = []
    add_row(
        rows,
        "HPE",
        "VK-RMD Student-VK",
        "random missing training",
        HPE_ROOT / "outputs/eval/student_vk_random_all_combinations.csv",
        "full method",
        "main",
    )
    add_row(
        rows,
        "HPE",
        "Student-VK no-random",
        "full-modality training only",
        HPE_ROOT / "outputs/eval/final_student_vk_no_random_all_combinations.csv",
        "tests need for random missing-modality training",
        "main ablation",
    )
    add_row(
        rows,
        "HPE",
        "No KD",
        "random missing training without KD",
        HPE_ROOT / "outputs/eval/ablation_no_distill_all_combinations.csv",
        "distillation boundary",
        "main ablation with cautious wording",
    )
    add_row(
        rows,
        "HPE",
        "Output KD",
        "KD variant",
        HPE_ROOT / "outputs/eval/final_ablation_output_kd_all_combinations.csv",
        "KD all-combination evidence",
        "main or appendix depending on result",
    )
    add_row(
        rows,
        "HPE",
        "Full structural KD",
        "KD variant",
        HPE_ROOT / "outputs/eval/final_ablation_full_structural_kd_all_combinations.csv",
        "structural KD all-combination evidence",
        "appendix if not stronger",
    )
    add_row(
        rows,
        "HAR",
        "VK-RMD Student-VK",
        "random missing training",
        HAR_ROOT / "outputs/eval/student_vk_random_all_combinations.csv",
        "full method",
        "main",
    )
    add_row(
        rows,
        "HAR",
        "Student-VK no-random",
        "full-modality training only",
        HAR_ROOT / "outputs/eval/final_student_vk_no_random_all_combinations.csv",
        "tests need for random missing-modality training",
        "main ablation",
    )
    add_row(
        rows,
        "HAR",
        "No KD",
        "random missing training without KD",
        HAR_ROOT / "outputs/eval/ablation_no_distill_all_combinations.csv",
        "HAR distillation evidence",
        "main ablation",
    )

    csv_path = PROJECT_ROOT / "tables/final_submission_ablation.csv"
    md_path = PROJECT_ROOT / "reports/final_submission_experiment_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
