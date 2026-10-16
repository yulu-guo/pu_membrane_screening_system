"""Add a nominal molecular weight column to the polyol entries.

The shipped polyol rows are small-molecule stand-ins, so their RDKit molecular
weight is far below the commercial oligomer they represent. The family model
needs the real oligomer chain length, so the nominal Mn is recorded here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data" / "monomers" / "monomers.csv"
COLUMN = "nominal_mn_g_mol"
PATTERN = re.compile(r"^POL-([A-Z]+)(\d+)$")


def nominal_mn(monomer_id: str, role: str) -> object:
    if role != "polyol":
        return pd.NA
    match = PATTERN.match(str(monomer_id))
    if not match:
        return pd.NA
    return int(match.group(2))


def main() -> None:
    frame = pd.read_csv(CATALOG)
    frame[COLUMN] = [
        nominal_mn(monomer_id, role)
        for monomer_id, role in zip(frame["id"], frame["role"])
    ]
    frame.to_csv(CATALOG, index=False)
    labeled = int(frame[COLUMN].notna().sum())
    print(f"rows: {len(frame)}, polyols with a nominal Mn: {labeled}")
    print(
        frame.loc[frame[COLUMN].notna(), ["id", COLUMN]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
