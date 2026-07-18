#!/usr/bin/env python3
"""Build protocol-audited comparisons with recent MMFi methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PTA = {
    "D": (84.81, 50.72),
    "L": (68.30, 44.44),
    "W": (182.18, 114.63),
    "D+L": (64.68, 42.63),
    "D+W": (95.96, 59.77),
    "L+W": (74.74, 49.49),
    "D+L+W": (68.86, 45.47),
}

COMPASS = {
    "D": 52.0,
    "L": 80.1,
    "R": 83.2,
    "D+L": 66.3,
    "D+R": 78.1,
    "L+R": 90.4,
    "D+L+R": 84.7,
}

ALIASES = {
    "vk": "V",
    "depth": "D",
    "lidar": "L",
    "mmwave": "R",
    "wifi-csi": "W",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "outputs" / "recent_mmfi_missing_modality_comparison",
    )
    return parser.parse_args()


def canonical(value: str) -> str:
    return "+".join(ALIASES[item.strip().lower()] for item in value.split("+"))


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {canonical(row["modality_set"]): row for row in rows}
    if not result:
        raise ValueError(f"No rows found in {path}")
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_rows() -> list[dict[str, str]]:
    common = {
        "dataset": "MM-Fi",
        "comparison_level": "published_protocol_matched",
        "exact_sample_manifest_verified": "no",
        "eligible": "yes_with_protocol_caveat",
    }
    return [
        {
            **common,
            "task": "HPE",
            "baseline": "PTA (2026)",
            "split": "official X-Fi/MM-Fi random split",
            "shared_modalities": "D,L,W",
            "metric": "MPJPE and PA-MPJPE (mm)",
            "architecture_control": "X-Fi modality encoders; 512-D tokens; batch size 16",
            "excluded_rows": "all RGB/VK-containing rows",
            "source": "https://arxiv.org/pdf/2604.05584",
        },
        {
            **common,
            "task": "HAR",
            "baseline": "COMPASS (2026)",
            "split": "MM-Fi S1 random split",
            "shared_modalities": "D,L,R",
            "metric": "top-1 accuracy (%)",
            "architecture_control": "X-Fi encoder families; 32x512 tokens; batch size 16",
            "excluded_rows": "all RGB/VK-containing rows",
            "source": "https://arxiv.org/pdf/2604.02056",
        },
    ]


def build_hpe(local: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for subset, (base_mpjpe, base_pa) in PTA.items():
        if subset not in local:
            raise KeyError(f"HPE result is missing required subset {subset}")
        ours_mpjpe = 1000.0 * float(local[subset]["mpjpe"])
        ours_pa = 1000.0 * float(local[subset]["pa_mpjpe"])
        rows.append(
            {
                "modality_set": subset,
                "baseline": "PTA",
                "pta_mpjpe_mm": round(base_mpjpe, 2),
                "vkrmd_mpjpe_mm": round(ours_mpjpe, 2),
                "delta_mpjpe_ours_minus_pta": round(ours_mpjpe - base_mpjpe, 2),
                "mpjpe_winner": "VK-RMD" if ours_mpjpe < base_mpjpe else "PTA",
                "pta_pa_mpjpe_mm": round(base_pa, 2),
                "vkrmd_pa_mpjpe_mm": round(ours_pa, 2),
                "delta_pa_ours_minus_pta": round(ours_pa - base_pa, 2),
                "pa_mpjpe_winner": "VK-RMD" if ours_pa < base_pa else "PTA",
            }
        )
    return rows


def build_har(local: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for subset, base_acc in COMPASS.items():
        if subset not in local:
            raise KeyError(f"HAR result is missing required subset {subset}")
        ours_acc = 100.0 * float(local[subset]["acc"])
        rows.append(
            {
                "modality_set": subset,
                "baseline": "COMPASS",
                "compass_acc_pct": round(base_acc, 2),
                "vkrmd_acc_pct": round(ours_acc, 2),
                "delta_acc_ours_minus_compass": round(ours_acc - base_acc, 2),
                "accuracy_winner": "VK-RMD" if ours_acc > base_acc else "COMPASS",
            }
        )
    return rows


def tex_delta(value: float, lower_is_better: bool) -> str:
    gain = value < 0 if lower_is_better else value > 0
    macro = "gain" if gain else "loss"
    arrow = r"\downarrow" if value < 0 else r"\uparrow"
    return rf"\{macro}{{${arrow}$ {abs(value):.2f}}}"


def write_hpe_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{c|rrr|rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{MPJPE$\downarrow$ (mm)} & \multicolumn{3}{c}{PA-MPJPE$\downarrow$ (mm)} \\",
        r"$S$ & PTA & VK-RMD & $\Delta$ & PTA & VK-RMD & $\Delta$ \\",
        r"\midrule",
    ]
    previous_size = 1
    for row in rows:
        size = str(row["modality_set"]).count("+") + 1
        if size != previous_size:
            lines.append(r"\midrule")
            previous_size = size
        lines.append(
            f'{row["modality_set"]} & {row["pta_mpjpe_mm"]:.2f} & {row["vkrmd_mpjpe_mm"]:.2f} & '
            f'{tex_delta(float(row["delta_mpjpe_ours_minus_pta"]), True)} & '
            f'{row["pta_pa_mpjpe_mm"]:.2f} & {row["vkrmd_pa_mpjpe_mm"]:.2f} & '
            f'{tex_delta(float(row["delta_pa_ours_minus_pta"]), True)} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_har_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [r"\begin{tabular}{c|rrr}", r"\toprule", r"$S$ & COMPASS & VK-RMD & $\Delta$ Acc. \\", r"\midrule"]
    previous_size = 1
    for row in rows:
        size = str(row["modality_set"]).count("+") + 1
        if size != previous_size:
            lines.append(r"\midrule")
            previous_size = size
        lines.append(
            f'{row["modality_set"]} & {row["compass_acc_pct"]:.2f} & {row["vkrmd_acc_pct"]:.2f} & '
            f'{tex_delta(float(row["delta_acc_ours_minus_compass"]), False)} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    hpe_path = repo / "HPE" / "outputs" / "eval" / "student_vk_random_all_combinations.csv"
    har_path = repo / "HAR" / "outputs" / "eval" / "student_vk_random_all_combinations.csv"
    hpe_local = read_rows(hpe_path)
    har_local = read_rows(har_path)

    for name, rows in (("HPE", hpe_local), ("HAR", har_local)):
        protocols = {row["protocol"] for row in rows.values()}
        splits = {row["split"] for row in rows.values()}
        if protocols != {"protocol3"} or splits != {"random_split"}:
            raise ValueError(f"{name} local results are not the expected random/protocol3 results: {splits}, {protocols}")

    hpe = build_hpe(hpe_local)
    har = build_har(har_local)
    audit = audit_rows()
    write_csv(output / "comparability_audit.csv", audit)
    write_csv(output / "recent_mmfi_hpe_pta.csv", hpe)
    write_csv(output / "recent_mmfi_har_compass.csv", har)
    write_hpe_tex(output / "recent_mmfi_hpe_pta.tex", hpe)
    write_har_tex(output / "recent_mmfi_har_compass.tex", har)

    hpe_mpjpe_wins = sum(row["mpjpe_winner"] == "VK-RMD" for row in hpe)
    hpe_pa_wins = sum(row["pa_mpjpe_winner"] == "VK-RMD" for row in hpe)
    har_wins = sum(row["accuracy_winner"] == "VK-RMD" for row in har)
    summary = [
        {"task": "HPE", "baseline": "PTA", "metric": "MPJPE", "shared_subsets": 7, "vkrmd_wins": hpe_mpjpe_wins, "baseline_wins": 7 - hpe_mpjpe_wins},
        {"task": "HPE", "baseline": "PTA", "metric": "PA-MPJPE", "shared_subsets": 7, "vkrmd_wins": hpe_pa_wins, "baseline_wins": 7 - hpe_pa_wins},
        {"task": "HAR", "baseline": "COMPASS", "metric": "Accuracy", "shared_subsets": 7, "vkrmd_wins": har_wins, "baseline_wins": 7 - har_wins},
    ]
    write_csv(output / "recent_mmfi_missing_comparison_summary.csv", summary)

    report = f"""# Recent MMFi Missing-Modality Comparison

## Eligibility

- HPE compares VK-RMD with PTA on all seven shared non-visual subsets of Depth, LiDAR and WiFi-CSI.
- HAR compares VK-RMD with COMPASS on all seven shared non-visual subsets of Depth, LiDAR and mmWave.
- Both comparisons use the published MM-Fi/X-Fi random protocol, common tasks and metrics. RGB/VK-containing rows are excluded because the visual inputs are not equivalent.
- The competing papers do not publish sample-level manifests; therefore this is a published-protocol comparison, not a paired identical-manifest experiment.

## Results

- HPE versus PTA: VK-RMD wins {hpe_mpjpe_wins}/7 subsets on MPJPE and {hpe_pa_wins}/7 on PA-MPJPE.
- HAR versus COMPASS: VK-RMD wins {har_wins}/7 subsets on top-1 accuracy.
- These are mixed rather than universal gains. LiDAR-dominant cases remain a visible weakness and must not be hidden.

## Paper-safe conclusion

Under the shared official MM-Fi random-split protocol and matched non-visual modality subsets, VK-RMD is competitive with recent missing-modality methods and achieves stronger results on most geometry-complementary subsets. The comparison does not establish universal superiority or exact sample-paired equivalence.
"""
    (output / "recent_mmfi_missing_comparison.md").write_text(report, encoding="utf-8")
    print(f"Saved audited comparison outputs to: {output}")


if __name__ == "__main__":
    main()
