"""Build the supplementary Tg training file from newly sourced public data.

Sources (none of them is the removed polyurethane paper):

1. Polymer Tg prediction dataset, DOI-free GitHub release
   ``chem-data-extraction/polymer-Tg-prediction-dataset``, file
   ``data/processed/dataset.csv``, licensed CC BY 4.0. 7 372 records of
   experimentally measured Tg assembled from open-access articles and a public
   Zenodo deposit.
2. The NeurIPS Open Polymer Prediction supplementary Tg table already
   downloaded in the workspace (46 records).
3. ``Tg_C`` rows of the curated ten-property table that are not yet part of the
   Tg training set.

Only structures whose canonical form is absent from the existing Tg file are
written, so merging cannot double-count a polymer under two SMILES spellings.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
WORK = Path(r"C:\Users\Lenovo\Documents\Codex\2026-10-09\wo\work")
RAW = ROOT / "data" / "raw" / "public"
TRAINING = ROOT / "data" / "training"


def canonical(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def archive(source: Path, name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / name
    if not target.exists():
        shutil.copyfile(source, target)
    return target


def main() -> None:
    base = pd.read_csv(TRAINING / "tg_polymers.csv").dropna(subset=["SMILES", "Tg"])
    seen = {key for key in (canonical(s) for s in base["SMILES"]) if key}
    print(f"existing Tg structures: {len(seen)}")

    rows: list[dict] = []

    ccby = archive(WORK / "ext_tg" / "dataset.csv", "polymer_tg_dataset_ccby4.csv")
    frame = pd.read_csv(ccby, low_memory=False)
    added = 0
    for smiles, value, unit, klass, name in zip(
        frame["repeat_unit_smiles"],
        frame["tg_value"],
        frame["tg_unit"],
        frame["polymer_class"],
        frame["polymer_name"],
    ):
        key = canonical(smiles)
        if key is None or key in seen:
            continue
        seen.add(key)
        tg_c = float(value) - 273.15 if str(unit).strip().upper() == "K" else float(value)
        rows.append(
            {
                "SMILES": str(smiles),
                "Tg": tg_c,
                "PID": f"CCBY-{added + 1:05d}",
                "Polymer Class": str(klass),
                "source_name": str(name),
                "source_dataset": "10.5281/zenodo lamalab Tg deposit via CC BY 4.0 GitHub release",
            }
        )
        added += 1
    print(f"added from the CC BY 4.0 dataset: {added}")

    supp = WORK / "neurips_supp_dataset3.csv"
    if supp.exists():
        supp_path = archive(supp, "neurips_supp_dataset3_tg.csv")
        table = pd.read_csv(supp_path)
        added = 0
        for smiles, tg in zip(table["SMILES"], table["Tg"]):
            key = canonical(smiles)
            if key is None or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "SMILES": str(smiles),
                    "Tg": float(tg),
                    "PID": f"NEURIPS-S3-{added + 1:04d}",
                    "Polymer Class": "unspecified",
                    "source_name": "NeurIPS Open Polymer Prediction 2025 supplement",
                    "source_dataset": "NeurIPS Open Polymer Prediction 2025",
                }
            )
            added += 1
        print(f"added from the NeurIPS supplement: {added}")

    curated = pd.read_csv(TRAINING / "network_polymer_multi_property_439.csv")
    tg_rows = curated.loc[curated["property_name"] == "Tg_C"]
    added = 0
    for smiles, value in zip(
        tg_rows["repeat_unit_smiles_canonical"],
        tg_rows["property_value_standardized"],
    ):
        if pd.isna(smiles) or pd.isna(value):
            continue
        key = canonical(smiles)
        if key is None or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "SMILES": str(smiles),
                "Tg": float(value),
                "PID": f"CURATED439-{added + 1:03d}",
                "Polymer Class": str(
                    curated.loc[
                        curated["repeat_unit_smiles_canonical"] == smiles,
                        "polymer_class",
                    ].iloc[0]
                ),
                "source_name": "curated ten-property table",
                "source_dataset": "open reference dataset of 60 polymers",
            }
        )
        added += 1
    print(f"added from the curated ten-property table: {added}")

    output = pd.DataFrame(rows)
    destination = TRAINING / "tg_polymers_extended.csv"
    output.to_csv(destination, index=False)
    print(f"\nwrote {destination.name}: {len(output)} new structures")
    polyurethane = output["Polymer Class"].str.contains(
        "polyurethane|polyurea", case=False, na=False
    ) | output["SMILES"].str.contains("NC(=O)O", regex=False)
    print(f"  of which polyurethane/polyurea: {int(polyurethane.sum())}")
    print(f"  Tg range: {output['Tg'].min():.1f} to {output['Tg'].max():.1f} C")


if __name__ == "__main__":
    main()
