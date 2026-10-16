from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from .chemistry import MonomerCatalog, build_virtual_pu
from .features import compile_mechanical_features
from .model import MembranePropertyModel
from .pla_ptmc import PlaPtmcFamilyModel
from .thermal import GeneralPublicThermalModel


PROPERTY_ORDER = [
    "tg_c",
    "youngs_modulus_mpa",
    "tensile_strength_mpa",
    "elongation_at_break_percent",
    "td5_c",
    "max_decomposition_c",
]


@dataclass(frozen=True)
class PropertyTargets:
    targets: dict[str, float]
    weights: dict[str, float]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PropertyTargets":
        raw_targets = payload.get("targets") or {}
        raw_weights = payload.get("weights") or {}
        targets = {}
        weights = {}
        for name in PROPERTY_ORDER:
            target = raw_targets.get(name)
            weight = raw_weights.get(name, 0.0)
            if target in (None, ""):
                continue
            try:
                target_value = float(target)
                weight_value = max(0.0, float(weight))
            except (TypeError, ValueError):
                continue
            if weight_value > 0:
                targets[name] = target_value
                weights[name] = weight_value
        if not targets:
            raise ValueError("At least one target with positive weight is required.")
        return cls(targets=targets, weights=weights)


class FormulationScreener:
    def __init__(
        self,
        catalog: MonomerCatalog,
        model: MembranePropertyModel,
        pla_ptmc_model: PlaPtmcFamilyModel,
        general_thermal_model: GeneralPublicThermalModel,
    ):
        self.catalog = catalog
        self.model = model
        self.pla_ptmc_model = pla_ptmc_model
        self.general_thermal_model = general_thermal_model
        self.scales = {
            "tg_c": 80.0,
            "youngs_modulus_mpa": 300.0,
            "tensile_strength_mpa": 20.0,
            "elongation_at_break_percent": 300.0,
            "td5_c": 40.0,
            "max_decomposition_c": 40.0,
        }

    def _score(
        self,
        prediction: dict[str, float],
        targets: PropertyTargets,
    ) -> float:
        weighted = 0.0
        total_weight = 0.0
        for property_name, target in targets.targets.items():
            weight = targets.weights[property_name]
            scale = self.scales.get(property_name, 1.0)
            weighted += weight * abs(prediction[property_name] - target) / scale
            total_weight += weight
        return weighted / max(total_weight, 1e-9)

    def screen(
        self,
        targets: PropertyTargets,
        roles: list[str],
        number_of_monomers: int,
        max_candidates: int,
        processing: dict[str, Any],
        arrangement: str,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        pools = [
            self.catalog.by_role(role).to_dict(orient="records")
            for role in roles
            if role in self.catalog.frame["role"].unique()
        ]
        if not pools:
            raise ValueError("No monomer roles are available for screening.")

        rng = random.Random(random_seed)
        candidates: list[dict[str, Any]] = []
        attempts = 0
        for _ in range(max_candidates * 3):
            if len(candidates) >= max_candidates:
                break
            attempts += 1
            selected = []
            for position in range(number_of_monomers):
                pool = pools[position % len(pools)] if len(pools) > 1 else pools[0]
                if not pool:
                    continue
                monomer = dict(rng.choice(pool))
                monomer["key"] = f"{monomer['role']}_{position + 1}"
                selected.append(monomer)
            if not any(item["role"] == "isocyanate" for item in selected):
                continue
            if not any(item["role"] != "isocyanate" for item in selected):
                continue
            try:
                build = build_virtual_pu(
                    selected,
                    arrangement=arrangement,
                    random_seed=random_seed + len(candidates),
                )
                compiled = compile_mechanical_features(
                    build.smiles,
                    selected,
                    processing,
                )
                prediction = self.model.predict(build.smiles, compiled)
                hard_fraction = (
                    sum(
                        float(item["molecular_weight"])
                        for item in selected
                        if item["role"]
                        in {"isocyanate", "chain_extender", "crosslinker"}
                    )
                    / max(
                        sum(
                            float(item["molecular_weight"])
                            for item in selected
                        ),
                        1e-9,
                    )
                )
                general_thermal = self.general_thermal_model.predict(
                    build.smiles,
                    composition=hard_fraction,
                )
                polyol_ids = [
                    item["id"] for item in selected if item["role"] == "polyol"
                ]
                pla_count = sum(
                    1
                    for monomer_id in polyol_ids
                    if monomer_id.startswith("POL-PLA")
                )
                ptmc_count = sum(
                    1
                    for monomer_id in polyol_ids
                    if monomer_id.startswith("POL-PTMC")
                )
                pla_ptmc_soft_segment = bool(polyol_ids) and all(
                    monomer_id.startswith("POL-PLA")
                    or monomer_id.startswith("POL-PTMC")
                    for monomer_id in polyol_ids
                )
                family = None
                if pla_count > 0 and ptmc_count > 0 and pla_ptmc_soft_segment:
                    requested_mn = processing.get("soft_segment_mn_g_mol")
                    if requested_mn in (None, ""):
                        soft_segment_mn = (
                            self.pla_ptmc_model.default_soft_segment_mn(selected)
                        )
                        common_mn = None
                    else:
                        soft_segment_mn = float(requested_mn)
                        common_mn = soft_segment_mn
                    fraction = (
                        self.pla_ptmc_model.ptmc_mass_fraction_from_monomers(
                            selected,
                            common_mn,
                        )
                    )
                    if fraction is not None:
                        family = self.pla_ptmc_model.predict(
                            fraction,
                            soft_segment_mn,
                        )
            except Exception:
                continue
            properties = {
                "tg_c": (
                    family.soft_segment_tg_c if family else prediction.tg_c
                ),
                "youngs_modulus_mpa": prediction.youngs_modulus_mpa,
                "tensile_strength_mpa": prediction.tensile_strength_mpa,
                "elongation_at_break_percent": (
                    prediction.elongation_at_break_percent
                ),
                "td5_c": family.td5_c if family else general_thermal["td5_c"],
                "max_decomposition_c": (
                    family.tdmax_c
                    if family
                    else general_thermal["max_decomposition_c"]
                ),
            }
            if any(
                properties.get(name) is None
                for name in targets.targets
            ):
                continue
            candidates.append(
                {
                    "monomers": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "role": item["role"],
                            "smiles": item["smiles"],
                        }
                        for item in selected
                    ],
                    "sequence": build.sequence,
                    "polymer_smiles": build.smiles,
                    "score": self._score(properties, targets),
                    "properties": properties,
                    "uncertainty": prediction.uncertainty,
                    "warnings": prediction.warnings,
                    "approx_features": prediction.feature_compiler_approximate,
                }
            )

        candidates.sort(key=lambda item: item["score"])
        return {
            "targets": targets.targets,
            "weights": targets.weights,
            "examined_candidates": attempts,
            "returned_candidates": len(candidates),
            "results": candidates[:200],
        }
