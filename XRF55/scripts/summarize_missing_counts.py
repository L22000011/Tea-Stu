from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reporting import write_csv, write_markdown_table


def read_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser("Summarize XRF55 combination results by missing-modality count.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    buckets = defaultdict(list)
    for row in rows:
        missing = int(row["missing_modalities"])
        buckets[missing].append(row)

    summary = []
    for missing in sorted(buckets):
        bucket = buckets[missing]
        summary.append(
            {
                "missing_modalities": missing,
                "available_modalities": int(bucket[0]["available_modalities"]),
                "combination_count": len(bucket),
                "mean_loss": sum(float(row["loss"]) for row in bucket) / len(bucket),
                "mean_acc": sum(float(row["acc"]) for row in bucket) / len(bucket),
                "mean_macro_f1": sum(float(row["macro_f1"]) for row in bucket) / len(bucket),
                "combination_list": ";".join(row["modality_set"] for row in bucket),
            }
        )
    write_csv(args.output_csv, summary)
    write_markdown_table(args.output_md, summary, "XRF55 results grouped by missing-modality count")


if __name__ == "__main__":
    main()
