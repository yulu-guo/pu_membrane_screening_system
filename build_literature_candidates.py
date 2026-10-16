"""Record the first batch of open-access mechanical tables as candidate rows.

These rows are extracted from open-access full texts. Each row carries its DOI
and states what the paper actually varied, because the current 20-feature model
can only use rows whose variation lives in composition and processing rather
than in fillers, blends or printing parameters.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "reference" / "mechanical_literature_candidates.csv"

COLUMNS = [
    "doi",
    "pmcid",
    "source_table",
    "sample",
    "youngs_modulus_mpa",
    "tensile_strength_mpa",
    "elongation_at_break_percent",
    "varied_factor",
    "chemistry_matches_catalog",
    "usable_with_current_features",
    "note",
]

ROWS = [
    # DLP-printed shape-memory polyurethane, DOI 10.3390/ma19091862
    *[
        {
            "doi": "10.3390/ma19091862",
            "pmcid": "PMC13165368",
            "source_table": "Table 4 (DoE trials)",
            "sample": f"Trial-{index}",
            "youngs_modulus_mpa": modulus,
            "tensile_strength_mpa": strength,
            "elongation_at_break_percent": elongation,
            "varied_factor": "layer thickness, exposure time, post-curing time",
            "chemistry_matches_catalog": "partly",
            "usable_with_current_features": False,
            "note": "Photocured DLP resin; the variation is printing parameters, which the model has no field for.",
        }
        for index, (elongation, modulus, strength) in enumerate(
            [
                (47.9, 10.1, 4.3),
                (42.8, 10.5, 4.1),
                (48.5, 11.1, 4.8),
                (40.3, 11.7, 4.4),
                (45.0, 9.3, 3.5),
                (38.8, 10.3, 4.0),
                (59.0, 9.8, 5.0),
                (38.3, 11.4, 4.2),
                (55.0, 11.5, 5.7),
            ],
            start=1,
        )
    ],
    {
        "doi": "10.3390/ma19091862",
        "pmcid": "PMC13165368",
        "source_table": "Table 5 (individual runs)",
        "sample": "optimum, run 1-4 (mean of 4)",
        "youngs_modulus_mpa": 10.9,
        "tensile_strength_mpa": 6.2,
        "elongation_at_break_percent": 64.0,
        "varied_factor": "printing parameters",
        "chemistry_matches_catalog": "partly",
        "usable_with_current_features": False,
        "note": "Mean of four specimens at the optimum; specimen-to-specimen spread is 60.4-68.2% elongation.",
    },
    # Non-isocyanate polyurethane films from lignin and castor oil, DOI 10.1039/d6ra02212j
    *[
        {
            "doi": "10.1039/d6ra02212j",
            "pmcid": "PMC13566744",
            "source_table": "Table 2 (NIPU film systems)",
            "sample": name,
            "youngs_modulus_mpa": modulus,
            "tensile_strength_mpa": strength,
            "elongation_at_break_percent": elongation,
            "varied_factor": "cyclic carbonate to lignin ratio",
            "chemistry_matches_catalog": "no",
            "usable_with_current_features": False,
            "note": "Non-isocyanate chemistry; the catalog holds isocyanate monomers only.",
        }
        for name, strength, elongation, modulus in [
            ("NIPU-Seb CC", 0.37, 70.0, 0.64),
            ("NIPU-CC-1", 2.71, 51.0, 5.00),
            ("NIPU-CC-2", 5.68, 27.0, 21.96),
            ("NIPU-CC-3", 8.59, 28.0, 24.50),
            ("NIPU-CC-4", 6.18, 14.0, 39.48),
            ("NIPU-NL-1", 1.52, 68.2, 2.27),
            ("NIPU-NL-2", 4.75, 23.7, 15.25),
            ("NIPU-NL-3", 8.68, 17.4, 48.76),
            ("NIPU-NL-4", 8.15, 12.0, 64.50),
        ]
    ],
    # Bromobutyl rubber / TPU thermoplastic vulcanizate, DOI 10.1039/d5ra10018f
    *[
        {
            "doi": "10.1039/d5ra10018f",
            "pmcid": "PMC13531526",
            "source_table": "Table 2 (BIIR/TPU TPVs)",
            "sample": name,
            "youngs_modulus_mpa": None,
            "tensile_strength_mpa": strength,
            "elongation_at_break_percent": elongation,
            "varied_factor": "rubber to TPU blend ratio, filler",
            "chemistry_matches_catalog": "partly",
            "usable_with_current_features": False,
            "note": "Dynamic vulcanizate blend; no modulus column reported.",
        }
        for name, strength, elongation in [
            ("BIIR/TPU-70", 3.2, 726.0),
            ("BIIR/TPU-60", 3.6, 551.0),
            ("BIIR/TPU-50", 4.1, 460.0),
            ("CCNs-BIIR/TPU-70", 3.4, 640.0),
        ]
    ],
    # Bipyridyl polyurethane nanocomposites, DOI 10.3390/polym18182191
    *[
        {
            "doi": "10.3390/polym18182191",
            "pmcid": "PMC13611364",
            "source_table": "Table 2 (BBD/G/ZnO/PUs)",
            "sample": name,
            "youngs_modulus_mpa": modulus,
            "tensile_strength_mpa": strength,
            "elongation_at_break_percent": elongation,
            "varied_factor": "graphene and ZnO filler loading",
            "chemistry_matches_catalog": "partly",
            "usable_with_current_features": False,
            "note": "Filler loading is the only thing that changes; the model has no filler field.",
        }
        for name, strength, elongation, modulus in [
            ("BBD/G/ZnO/PU-01", 2.68, 829.0, 0.65),
            ("BBD/G/ZnO/PU-02", 3.99, 746.0, 1.24),
            ("BBD/G/ZnO/PU-03", 7.99, 708.0, 3.07),
            ("BBD/G/ZnO/PU-04", 11.76, 474.0, 5.20),
        ]
    ],
]


def main() -> None:
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    usable = int(frame["usable_with_current_features"].sum())
    print(f"wrote {OUTPUT.name}: {len(frame)} rows from {frame['doi'].nunique()} papers")
    print(f"usable with the current feature set: {usable}")
    print(frame.groupby("doi").size().to_string())


if __name__ == "__main__":
    main()
