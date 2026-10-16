from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import MonomerCatalog, build_virtual_pu
from .features import MECHANICAL_FEATURES, compile_mechanical_features
from .model import MembranePropertyModel


LAB_PROPERTY_COLUMNS = {
    "Tg_C": "tg_c",
    "Youngs_modulus_MPa": "youngs_modulus_mpa",
    "Tensile_strength_MPa": "tensile_strength_mpa",
    "Elongation_at_break_percent": "elongation_at_break_percent",
}


def _split_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [
        item.strip()
        for item in str(value).replace(",", "|").split("|")
        if item.strip()
    ]


def import_lab_rows(
    source: Path,
    output_dir: Path,
) -> pd.DataFrame:
    catalog = MonomerCatalog.from_csv(
        output_dir / "data" / "monomers" / "monomers.csv"
    )
    model = MembranePropertyModel(
        output_dir / "data" / "models" / "tg_model.joblib",
        output_dir / "data" / "models" / "mechanical_model.joblib",
        output_dir / "data" / "models" / "lab_calibration.joblib",
    )
    frame = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.read_excel(source)
    required = {
        "sample_id",
        "monomer_ids",
        "counts",
        "arrangement",
        "nco_oh_index",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Lab data is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        monomer_ids = _split_list(row["monomer_ids"])
        counts = [int(value) for value in _split_list(row["counts"])]
        if len(monomer_ids) != len(counts):
            raise ValueError(
                f"Row {row_index}: monomer_ids and counts must have equal lengths."
            )
        selected = []
        instance_index = 0
        for monomer_id, count in zip(monomer_ids, counts):
            monomer = catalog.get(monomer_id)
            for _ in range(count):
                item = dict(monomer)
                item["key"] = f"{monomer_id}_{instance_index + 1}"
                selected.append(item)
                instance_index += 1
        processing = {
            "nco_oh_index": float(row["nco_oh_index"]),
            "synthesis_step": int(row.get("synthesis_step", 2)),
            "forming_method": str(
                row.get("forming_method", "solution_casting")
            ),
            "reaction_temperature_1_c": float(
                row.get("reaction_temperature_1_c", 80.0)
            ),
            "reaction_temperature_2_c": float(
                row.get("reaction_temperature_2_c", 80.0)
            ),
            "cross_section_area_mm2": float(
                row.get("cross_section_area_mm2", 30.0)
            ),
            "strain_rate_min_inv": float(
                row.get("strain_rate_min_inv", 0.1)
            ),
        }
        build = build_virtual_pu(
            selected,
            arrangement=str(row["arrangement"]),
            random_seed=int(row.get("random_seed", 42)),
        )
        compiled = compile_mechanical_features(
            build.smiles,
            selected,
            processing,
        )
        base = model.predict(build.smiles, compiled)
        output_row = {
            "sample_id": row["sample_id"],
            "polymer_smiles": build.smiles,
            "sequence": "|".join(build.sequence),
            "base_tg_c": base.tg_c,
            "base_youngs_modulus_mpa": base.youngs_modulus_mpa,
            "base_tensile_strength_mpa": base.tensile_strength_mpa,
            "base_elongation_at_break_percent": (
                base.elongation_at_break_percent
            ),
        }
        output_row.update(compiled.standardized)
        for source_name, normalized_name in LAB_PROPERTY_COLUMNS.items():
            if source_name in frame.columns and pd.notna(row[source_name]):
                output_row[normalized_name] = float(row[source_name])
        rows.append(output_row)

    result = pd.DataFrame(rows)
    destination = output_dir / "data" / "lab" / "lab_features.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
