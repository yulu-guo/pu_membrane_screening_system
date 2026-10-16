from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from pu_v2.chemistry import canonical_smiles, monomer_summary


TG_URL = (
    "https://cdn.jsdelivr.net/gh/maxhartml/NeuralIPS-Polymer@main/"
    "neurips-open-polymer-prediction-2025/external_data/"
    "TgSS_enriched_cleaned.csv"
)
MECHANICAL_URL = (
    "https://china.scidb.cn/download?"
    "fileId=75eb9aabdddbd030dec7b981ccc5db2f"
)
FRONTIERS_XLSX_URL = (
    "https://public-pages-files-2025.frontiersin.org/articles/452863/file/"
    "Table_2.XLSX/452863_supplementary-materials_tables_2_xlsx/1"
)


PUBCHEM_MONOMERS = [
    {
        "id": "POL-PTMC500",
        "name": "PTMC-500-like poly(trimethylene carbonate) diol",
        "role": "polyol",
        "family": "polycarbonate",
        "pubchem_name": "poly(trimethylene carbonate)",
        "fallback_smiles": "OCCCOC(=O)OCCCO",
        "bio_based_hint": False,
    },
    {
        "id": "POL-PTMC1000",
        "name": "PTMC-1000-like poly(trimethylene carbonate) diol",
        "role": "polyol",
        "family": "polycarbonate",
        "pubchem_name": "poly(trimethylene carbonate) 1000",
        "fallback_smiles": "OCCCOC(=O)OCCCOC(=O)OCCCO",
        "bio_based_hint": False,
    },
    {
        "id": "POL-PTMC2000",
        "name": "PTMC-2000-like poly(trimethylene carbonate) diol",
        "role": "polyol",
        "family": "polycarbonate",
        "pubchem_name": "poly(trimethylene carbonate) 2000",
        "fallback_smiles": "OCCCOC(=O)OCCCOC(=O)OCCCOC(=O)OCCCO",
        "bio_based_hint": False,
    },
    {
        "id": "CE-EG",
        "name": "ethylene glycol",
        "role": "chain_extender",
        "family": "diol",
        "pubchem_name": "ethylene glycol",
        "fallback_smiles": "OCCO",
        "bio_based_hint": True,
    },
    {
        "id": "CE-13PDO",
        "name": "1,3-propanediol",
        "role": "chain_extender",
        "family": "diol",
        "pubchem_name": "1,3-propanediol",
        "fallback_smiles": "OCCCO",
        "bio_based_hint": True,
    },
    {
        "id": "CE-BDO",
        "name": "1,4-butanediol",
        "role": "chain_extender",
        "family": "diol",
        "pubchem_name": "1,4-butanediol",
        "fallback_smiles": "OCCCCO",
        "bio_based_hint": True,
    },
    {
        "id": "CE-15PDO",
        "name": "1,5-pentanediol",
        "role": "chain_extender",
        "family": "diol",
        "pubchem_name": "1,5-pentanediol",
        "fallback_smiles": "OCCCCCO",
        "bio_based_hint": True,
    },
    {
        "id": "CE-HDO",
        "name": "1,6-hexanediol",
        "role": "chain_extender",
        "family": "diol",
        "pubchem_name": "1,6-hexanediol",
        "fallback_smiles": "OCCCCCCO",
        "bio_based_hint": True,
    },
    {
        "id": "CE-NPG",
        "name": "neopentyl glycol",
        "role": "chain_extender",
        "family": "diol",
        "pubchem_name": "neopentyl glycol",
        "fallback_smiles": "OCC(C)(C)CO",
        "bio_based_hint": False,
    },
    {
        "id": "CE-CHDM",
        "name": "1,4-cyclohexanedimethanol",
        "role": "chain_extender",
        "family": "cyclic diol",
        "pubchem_name": "1,4-cyclohexanedimethanol",
        "fallback_smiles": "OCC1CCC(CO)CC1",
        "bio_based_hint": False,
    },
    {
        "id": "CE-HQEE",
        "name": "hydroquinone bis(2-hydroxyethyl) ether",
        "role": "chain_extender",
        "family": "aromatic diol",
        "pubchem_name": "hydroquinone bis(2-hydroxyethyl) ether",
        "fallback_smiles": "OCCOc1ccc(OCCO)cc1",
        "bio_based_hint": False,
    },
    {
        "id": "CE-DMPA",
        "name": "2,2-bis(hydroxymethyl)propionic acid",
        "role": "chain_extender",
        "family": "carboxylated diol",
        "pubchem_name": "2,2-bis(hydroxymethyl)propionic acid",
        "fallback_smiles": "OCC(C)(CO)C(=O)O",
        "bio_based_hint": False,
    },
    {
        "id": "CE-ISOSORBIDE",
        "name": "isosorbide",
        "role": "chain_extender",
        "family": "bio-based cyclic diol",
        "pubchem_name": "isosorbide",
        "fallback_smiles": "OC1COC2C(O)COC12",
        "bio_based_hint": True,
    },
    {
        "id": "CE-BPH",
        "name": "4,4'-dihydroxybiphenyl",
        "role": "chain_extender",
        "family": "aromatic diol",
        "pubchem_name": "4,4'-dihydroxybiphenyl",
        "fallback_smiles": "Oc1ccc(-c2ccc(O)cc2)cc1",
        "bio_based_hint": False,
    },
    {
        "id": "CE-ED",
        "name": "ethylenediamine",
        "role": "chain_extender",
        "family": "diamine",
        "pubchem_name": "ethylenediamine",
        "fallback_smiles": "NCCN",
        "bio_based_hint": False,
    },
    {
        "id": "CE-PDA",
        "name": "1,3-propanediamine",
        "role": "chain_extender",
        "family": "diamine",
        "pubchem_name": "1,3-propanediamine",
        "fallback_smiles": "NCCCN",
        "bio_based_hint": False,
    },
    {
        "id": "CE-BDA",
        "name": "1,4-butanediamine",
        "role": "chain_extender",
        "family": "diamine",
        "pubchem_name": "1,4-butanediamine",
        "fallback_smiles": "NCCCCN",
        "bio_based_hint": False,
    },
    {
        "id": "CE-HDA",
        "name": "1,6-hexanediamine",
        "role": "chain_extender",
        "family": "diamine",
        "pubchem_name": "hexamethylenediamine",
        "fallback_smiles": "NCCCCCCN",
        "bio_based_hint": True,
    },
    {
        "id": "CE-PIP",
        "name": "piperazine",
        "role": "chain_extender",
        "family": "cyclic diamine",
        "pubchem_name": "piperazine",
        "fallback_smiles": "N1CCNCC1",
        "bio_based_hint": False,
    },
    {
        "id": "CE-MDA",
        "name": "4,4'-methylenedianiline",
        "role": "chain_extender",
        "family": "aromatic diamine",
        "pubchem_name": "4,4'-methylenedianiline",
        "fallback_smiles": "Nc1ccc(Cc2ccc(N)cc2)cc1",
        "bio_based_hint": False,
    },
    {
        "id": "CE-MOCA",
        "name": "4,4'-methylene-bis(2-chloroaniline)",
        "role": "chain_extender",
        "family": "aromatic diamine",
        "pubchem_name": "4,4'-methylene-bis(2-chloroaniline)",
        "fallback_smiles": "Nc1ccc(Cc2ccc(N)c(Cl)c2)cc1Cl",
        "bio_based_hint": False,
    },
    {
        "id": "XL-GLY",
        "name": "glycerol",
        "role": "crosslinker",
        "family": "triol",
        "pubchem_name": "glycerol",
        "fallback_smiles": "OCC(O)CO",
        "bio_based_hint": True,
    },
    {
        "id": "XL-TMP",
        "name": "trimethylolpropane",
        "role": "crosslinker",
        "family": "triol",
        "pubchem_name": "trimethylolpropane",
        "fallback_smiles": "CCC(CO)(CO)CO",
        "bio_based_hint": False,
    },
    {
        "id": "XL-PE",
        "name": "pentaerythritol",
        "role": "crosslinker",
        "family": "tetraol",
        "pubchem_name": "pentaerythritol",
        "fallback_smiles": "OCC(CO)(CO)CO",
        "bio_based_hint": False,
    },
    {
        "id": "XL-SORB",
        "name": "sorbitol",
        "role": "crosslinker",
        "family": "polyol",
        "pubchem_name": "D-sorbitol",
        "fallback_smiles": "OCC(O)C(O)C(O)C(O)CO",
        "bio_based_hint": True,
    },
    {
        "id": "XL-TEA",
        "name": "triethanolamine",
        "role": "crosslinker",
        "family": "amino triol",
        "pubchem_name": "triethanolamine",
        "fallback_smiles": "OCCN(CCO)CCO",
        "bio_based_hint": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the PU membrane training data and monomer catalog."
    )
    parser.add_argument(
        "--source-gas-system",
        type=Path,
        default=Path(
            r"C:\Users\Lenovo\Documents\Codex\2026-10-09\wo\outputs"
            r"\pu_screening_system"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            r"C:\Users\Lenovo\Documents\Codex\2026-10-09\wo\work"
            r"\pu_membrane_sources"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Codex-PU-Property-System/0.2"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def fetch_pubchem_smiles(name: str) -> tuple[str | None, str | None]:
    encoded = urllib.parse.quote(name)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/CanonicalSMILES,MolecularFormula/JSON"
    )
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Codex-PU-Property-System/0.2"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        properties = payload["PropertyTable"]["Properties"][0]
        return properties.get("CanonicalSMILES"), properties.get("MolecularFormula")
    except Exception:
        return None, None


def source_monomer_records(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    diisocyanate_path = (
        source_root / "data" / "monomers" / "diisocyanates.csv"
    )
    polyol_path = source_root / "data" / "monomers" / "polyols.csv"
    if diisocyanate_path.exists():
        frame = pd.read_csv(diisocyanate_path)
        for _, row in frame.iterrows():
            records.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "role": "isocyanate",
                    "family": row.get("family", "isocyanate"),
                    "smiles": row["smiles"],
                    "source": row.get("source", "curated + ML paper"),
                    "bio_based_hint": bool(row.get("bio_based_hint", False)),
                }
            )
    if polyol_path.exists():
        frame = pd.read_csv(polyol_path)
        for _, row in frame.iterrows():
            records.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "role": "polyol",
                    "family": row.get("family", "polyol"),
                    "smiles": row["smiles"],
                    "source": row.get("source", "curated"),
                    "bio_based_hint": bool(row.get("bio_based_hint", False)),
                }
            )
    return records


def build_monomer_catalog(
    source_root: Path,
    cache_dir: Path,
) -> pd.DataFrame:
    records = source_monomer_records(source_root)
    cache_path = cache_dir / "pubchem_monomer_cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    for item in PUBCHEM_MONOMERS:
        name = item["pubchem_name"]
        if name not in cache:
            smiles, formula = fetch_pubchem_smiles(name)
            cache[name] = {"smiles": smiles, "formula": formula}
        pubchem_smiles = cache[name].get("smiles")
        record = dict(item)
        record.pop("pubchem_name", None)
        record.pop("fallback_smiles", None)
        record["smiles"] = pubchem_smiles or item["fallback_smiles"]
        record["source"] = (
            f"PubChem name lookup: {name}"
            if pubchem_smiles
            else "Curated fallback SMILES"
        )
        records.append(record)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_smiles_role: set[tuple[str, str]] = set()
    for record in records:
        try:
            smiles = canonical_smiles(record["smiles"])
        except Exception:
            continue
        key = (smiles, record["role"])
        if record["id"] in seen_ids or key in seen_smiles_role:
            continue
        try:
            summary = monomer_summary(smiles, record["role"])
        except Exception:
            continue
        seen_ids.add(record["id"])
        seen_smiles_role.add(key)
        output.append(
            {
                **record,
                "smiles": smiles,
                "soft_segment": record["role"] == "polyol",
                **summary,
            }
        )

    frame = pd.DataFrame(output)
    frame = frame.drop(
        columns=[
            "canonical_smiles",
            "nco_sites",
            "oh_sites",
            "nh2_sites",
        ],
        errors="ignore",
    )
    frame = frame.sort_values(
        ["role", "soft_segment", "family", "name"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return frame


def extract_frontiers_mechanical(path: Path, output: Path) -> pd.DataFrame:
    samples = pd.read_excel(path, sheet_name="Samples")
    responses = pd.read_excel(path, sheet_name="Top-Middle layer")
    features = pd.read_excel(path, sheet_name="Middle Bottom layer")
    sample_table = pd.concat([samples, responses, features], axis=1)
    sample_table = sample_table.loc[:, ~sample_table.columns.duplicated()]
    sample_table.to_csv(output, index=False)
    return sample_table


def build(
    source_root: Path,
    cache_dir: Path,
    output: Path,
) -> None:
    training_dir = output / "data" / "training"
    monomer_dir = output / "data" / "monomers"
    training_dir.mkdir(parents=True, exist_ok=True)
    monomer_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tg_source = download(TG_URL, cache_dir / "tg_polymers.csv")
    mechanical_source = download(
        MECHANICAL_URL,
        cache_dir / "pue_643_mechanical.csv",
    )
    frontiers_source = download(
        FRONTIERS_XLSX_URL,
        cache_dir / "frontiers_pue_18.xlsx",
    )

    tg = pd.read_csv(tg_source)
    pu_mask = (
        tg["Polymer Class"]
        .fillna("")
        .str.contains("Polyurethane|Polyurea", case=False, regex=True)
        | tg["SMILES"].fillna("").str.contains("NC(=O)O", regex=False)
    )
    pu_tg = tg.loc[pu_mask].copy()
    pu_tg.to_csv(training_dir / "pu_tg_structures.csv", index=False)
    tg.to_csv(training_dir / "tg_polymers.csv", index=False)

    shutil.copy2(
        mechanical_source,
        training_dir / "pue_643_mechanical.csv",
    )
    extract_frontiers_mechanical(
        frontiers_source,
        training_dir / "frontiers_pue_18.csv",
    )

    monomer_frame = build_monomer_catalog(source_root, cache_dir)
    monomer_frame.to_csv(monomer_dir / "monomers.csv", index=False)

    lab_template = pd.DataFrame(
        columns=[
            "sample_id",
            "monomer_sequence",
            "monomer_ids",
            "equivalents",
            "nco_oh_index",
            "synthesis_step",
            "forming_method",
            "reaction_temperature_1_c",
            "reaction_temperature_2_c",
            "cross_section_area_mm2",
            "strain_rate_min_inv",
            "Tg_C",
            "Youngs_modulus_MPa",
            "Tensile_strength_MPa",
            "Elongation_at_break_percent",
        ]
    )
    lab_template.to_csv(output / "data" / "lab_template.csv", index=False)

    print(f"Tg rows: {len(tg)}")
    print(f"PU Tg rows: {len(pu_tg)}")
    print(f"Mechanical rows: {len(pd.read_csv(mechanical_source))}")
    role_counts = monomer_frame["role"].value_counts().to_dict()
    print(f"Monomer roles: {role_counts}")
    print(f"Monomer catalog: {monomer_dir / 'monomers.csv'}")


def main() -> None:
    args = parse_args()
    build(
        args.source_gas_system.resolve(),
        args.cache_dir.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
