"""Remove artefacts that are no longer referenced by the system.

Every target is resolved and checked to sit inside the project directory before
anything is deleted.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent

TARGETS = [
    # dataset of the removed polyimide / known-structure feature
    "data/structures/polyimide_components.csv",
    # model snapshots taken before the two retraining steps
    "data/models/metrics_before_supplement.json",
    "data/models/metrics_before_mech_tune.json",
    "data/models/tg_model_before_supplement.joblib",
    "data/models/mechanical_model_before_tune.joblib",
]

EMPTY_DIRS = ["data/structures"]


def main() -> None:
    removed = []
    for relative in TARGETS:
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents and path != ROOT:
            print(f"skipped (outside project): {path}")
            continue
        if not path.exists():
            print(f"already gone: {relative}")
            continue
        size = path.stat().st_size
        path.unlink()
        removed.append((relative, size))
        print(f"deleted {relative} ({size / 1024:.0f} KB)")

    for relative in EMPTY_DIRS:
        path = (ROOT / relative).resolve()
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            print(f"removed empty directory {relative}")

    total = sum(size for _, size in removed) / 1024 / 1024
    print(f"\n{len(removed)} files deleted, {total:.1f} MB freed")


if __name__ == "__main__":
    main()
