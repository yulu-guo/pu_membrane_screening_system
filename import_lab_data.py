from __future__ import annotations

import argparse
from pathlib import Path

from pu_v2.lab import import_lab_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import measured lab PU data and build calibration features."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    result = import_lab_rows(
        args.input.resolve(),
        args.output_dir.resolve(),
    )
    print(
        f"Imported {len(result)} lab rows to "
        f"{args.output_dir.resolve() / 'data' / 'lab' / 'lab_features.csv'}"
    )


if __name__ == "__main__":
    main()
