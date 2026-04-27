from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.xfi_report import DEFAULT_XFI_TABLE, generate_xfi_hpe_report


def main() -> None:
    parser = argparse.ArgumentParser("Generate a TeX/PDF comparison report against X-Fi HPE Table 1.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--xfi-csv", type=str, default=str(DEFAULT_XFI_TABLE))
    args = parser.parse_args()
    outputs = generate_xfi_hpe_report(args.csv, args.xfi_csv)
    print("Generated X-Fi comparison report:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
