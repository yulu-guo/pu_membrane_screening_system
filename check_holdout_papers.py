from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "data" / "training"
HOLDOUT_PATH = ROOT / "data" / "holdout" / "validation_only_papers.csv"


def normalized(text: str) -> str:
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .lower()
    )


def load_training_text() -> str:
    chunks = []
    for path in TRAINING_DIR.glob("*.csv"):
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return normalized("\n".join(chunks))


def main() -> int:
    if not HOLDOUT_PATH.exists():
        print(f"Missing holdout paper file: {HOLDOUT_PATH}")
        return 1
    training_text = load_training_text()
    errors = []
    with HOLDOUT_PATH.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        paper_id = record["paper_id"]
        if record.get("exclude_from_training", "").lower() != "true":
            errors.append(f"{paper_id}: exclude_from_training is not true")
        checks = {
            "DOI": record["doi"],
            "title": record["title"],
            "key metric": record.get("leak_check_string") or "1020.23",
        }
        for label, value in checks.items():
            if normalized(value) in training_text:
                errors.append(
                    f"{paper_id}: {label} appears in the training data"
                )
        print(f"{paper_id}: checked DOI, title, and key metric")
    if errors:
        print("\nValidation-only paper check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nAll holdout papers are excluded from the current training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
