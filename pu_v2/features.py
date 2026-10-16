from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .chemistry import (
    NCO_PATTERN,
    UREA_PATTERN,
    URETHANE_PATTERN,
    ChemistryError,
)


MECHANICAL_FEATURES = [
    "ZS_CHS",
    "ZS_R",
    "ZS_log_Tr1K",
    "ZS_log_Tr2K",
    "PMStep",
    "Form_Method",
    "ZS_log_CSArea",
    "ZS_log_StrainRate",
    "ZS_log_PO_MW",
    "ZS_log_FCVm",
    "ZS_FCCED",
    "ZS_log_Fchi",
    "ZS_SS_TPSA_norm",
    "ZS_SS_MolLogP_norm",
    "ZS_HS_BertzCT",
    "ZS_SS_VSA_EState8",
    "ZS_SS_PEOE_VSA8",
    "ZS_log_HS_NumNHCO_norm",
    "ZS_FC_NumHAcceptors_norm",
    "ZS_FC_RingCount_norm",
]

FORMING_METHODS = {
    "solution_casting": 1,
    "melt_casting": 2,
    "spin_casting": 3,
    "hot_pressing": 4,
    "microinjection": 5,
}

# The released 643-row dataset is already standardized. The compiler maps
# physically computed monomer and process values into approximately comparable
# standardized ranges. Exact feature mode bypasses this approximation.
RAW_CENTER_SCALE = {
    "ZS_CHS": (0.35, 0.18),
    "ZS_R": (1.10, 0.25),
    "ZS_log_Tr1K": (2.50, 0.08),
    "ZS_log_Tr2K": (2.50, 0.08),
    "PMStep": (1.5, 0.5),
    "Form_Method": (3.0, 1.5),
    "ZS_log_CSArea": (1.5, 0.7),
    "ZS_log_StrainRate": (-1.0, 1.2),
    "ZS_log_PO_MW": (3.2, 0.4),
    "ZS_log_FCVm": (3.0, 0.6),
    "ZS_FCCED": (300.0, 100.0),
    "ZS_log_Fchi": (0.0, 0.8),
    "ZS_SS_TPSA_norm": (40.0, 25.0),
    "ZS_SS_MolLogP_norm": (2.0, 2.0),
    "ZS_HS_BertzCT": (5.0, 1.5),
    "ZS_SS_VSA_EState8": (6.0, 5.0),
    "ZS_SS_PEOE_VSA8": (50.0, 50.0),
    "ZS_log_HS_NumNHCO_norm": (1.5, 0.7),
    "ZS_FC_NumHAcceptors_norm": (4.0, 3.0),
    "ZS_FC_RingCount_norm": (2.0, 2.0),
}


@dataclass
class CompiledFeatures:
    raw: dict[str, float]
    standardized: dict[str, float]
    approximate: bool
    missing_features: list[str]


def _weighted_value(
    monomers: list[dict[str, Any]],
    field: str,
) -> float:
    total = 0.0
    weights = 0.0
    for monomer in monomers:
        weight = float(monomer.get("molecular_weight", 1.0))
        value = float(monomer.get(field, 0.0))
        if not math.isfinite(value):
            value = 0.0
        total += weight * value
        weights += weight
    return total / weights if weights else 0.0


def _z_score(value: float, feature: str) -> float:
    center, scale = RAW_CENTER_SCALE[feature]
    return (value - center) / scale if scale else 0.0


def compile_mechanical_features(
    model_smiles: str,
    selected_monomers: list[dict[str, Any]],
    processing: dict[str, Any],
) -> CompiledFeatures:
    molecule = Chem.MolFromSmiles(model_smiles)
    if molecule is None:
        raise ChemistryError("Could not parse the generated polymer structure.")

    isocyanates = [
        item for item in selected_monomers if item["role"] == "isocyanate"
    ]
    soft = [
        item for item in selected_monomers if item["role"] == "polyol"
    ]
    hard = [
        item
        for item in selected_monomers
        if item["role"] in {"isocyanate", "chain_extender", "crosslinker"}
    ]
    if not soft:
        # If the user omits a polyol, use every non-isocyanate component as soft.
        soft = [
            item for item in selected_monomers if item["role"] != "isocyanate"
        ]
    if not hard:
        hard = isocyanates or selected_monomers
    if not soft:
        soft = selected_monomers

    total_mass = sum(float(item["molecular_weight"]) for item in selected_monomers)
    hard_mass = sum(float(item["molecular_weight"]) for item in hard)
    chs = hard_mass / total_mass if total_mass else 0.0

    polyol_mass = sum(float(item["molecular_weight"]) for item in soft)
    po_mw = (
        sum(
            float(item["molecular_weight"]) ** 2
            for item in soft
        )
        / polyol_mass
        if polyol_mass
        else _weighted_value(soft, "molecular_weight")
    )
    po_mw = max(po_mw, 1.0)

    ss_tpsa = _weighted_value(soft, "tpsa")
    ss_logp = _weighted_value(soft, "mol_logp")
    ss_mol_volume = _weighted_value(soft, "molar_volume")
    if ss_mol_volume <= 0:
        ss_mol_volume = max(
            sum(float(item["molecular_weight"]) for item in soft),
            1.0,
        )
    ss_vsa = _weighted_value(soft, "vsa_estate8")
    ss_peoe = _weighted_value(soft, "peoe_vsa8")

    hs_bertz = _weighted_value(hard, "bertz_ct")
    hs_nhco = _weighted_value(hard, "nhco_groups")
    hs_logp = _weighted_value(hard, "mol_logp")
    hs_tpsa = _weighted_value(hard, "tpsa")

    hba = float(rdMolDescriptors.CalcNumHBA(molecule))
    ring_count = float(molecule.GetRingInfo().NumRings())
    urethane_count = len(
        molecule.GetSubstructMatches(URETHANE_PATTERN, uniquify=True)
    )
    urea_count = len(molecule.GetSubstructMatches(UREA_PATTERN, uniquify=True))
    unreacted_nco = len(molecule.GetSubstructMatches(NCO_PATTERN, uniquify=True))

    # A coarse cohesive-energy-density proxy derived from polarity, hydrogen
    # bonding and molecular volume. It is explicitly treated as approximate.
    ced_proxy = 80.0 + 4.0 * ss_tpsa + 5.0 * hba + 30.0 * (urethane_count + urea_count)
    ced_proxy = min(max(ced_proxy, 80.0), 800.0)
    log_fchi = (hs_logp - ss_logp) + 0.01 * (hs_tpsa - ss_tpsa)

    t1 = float(processing.get("reaction_temperature_1_c", 80.0)) + 273.15
    t2 = float(processing.get("reaction_temperature_2_c", 80.0)) + 273.15
    step = int(processing.get("synthesis_step", 2))
    form_method_name = str(
        processing.get("forming_method", "solution_casting")
    )
    form_method = FORMING_METHODS.get(form_method_name, 1)
    cross_section = max(
        float(processing.get("cross_section_area_mm2", 30.0)),
        0.001,
    )
    strain_rate = max(
        float(processing.get("strain_rate_min_inv", 0.1)),
        1e-6,
    )

    raw = {
        "ZS_CHS": chs,
        "ZS_R": float(processing.get("nco_oh_index", 1.0)),
        "ZS_log_Tr1K": math.log10(max(t1, 1.0)),
        "ZS_log_Tr2K": math.log10(max(t2, 1.0)),
        "PMStep": float(step),
        "Form_Method": float(form_method),
        "ZS_log_CSArea": math.log10(cross_section + 1.0),
        "ZS_log_StrainRate": math.log10(strain_rate),
        "ZS_log_PO_MW": math.log10(po_mw),
        "ZS_log_FCVm": math.log10(max(ss_mol_volume, 1.0)),
        "ZS_FCCED": ced_proxy,
        "ZS_log_Fchi": log_fchi,
        "ZS_SS_TPSA_norm": ss_tpsa,
        "ZS_SS_MolLogP_norm": ss_logp,
        "ZS_HS_BertzCT": math.log10(max(hs_bertz, 0.0) + 1.0),
        "ZS_SS_VSA_EState8": ss_vsa,
        "ZS_SS_PEOE_VSA8": ss_peoe,
        "ZS_log_HS_NumNHCO_norm": math.log10(max(hs_nhco, 0.0) + 1.0),
        "ZS_FC_NumHAcceptors_norm": hba,
        "ZS_FC_RingCount_norm": ring_count,
    }
    standardized = {
        feature: _z_score(raw[feature], feature)
        for feature in MECHANICAL_FEATURES
    }
    missing = []
    if not soft:
        missing.append("soft_segment")
    if not hard:
        missing.append("hard_segment")
    if unreacted_nco:
        missing.append("unreacted_isocyanate")
    return CompiledFeatures(
        raw=raw,
        standardized=standardized,
        approximate=True,
        missing_features=missing,
    )


