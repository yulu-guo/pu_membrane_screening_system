from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .features import MECHANICAL_FEATURES, CompiledFeatures


MECHANICAL_TARGETS = {
    "logYM": "youngs_modulus_mpa",
    "logTS": "tensile_strength_mpa",
    "logEB": "elongation_at_break_percent",
}


class ModelError(RuntimeError):
    """Raised when a model bundle is missing or malformed."""


def morgan_bit_vector(smiles: str, n_bits: int = 2048) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ModelError("RDKit could not parse the polymer SMILES.")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=3,
        nBits=n_bits,
    )
    return np.asarray(fingerprint, dtype=np.uint8)


@dataclass
class FormulationPrediction:
    tg_c: float
    tg_uncertainty_c: float
    youngs_modulus_mpa: float
    tensile_strength_mpa: float
    elongation_at_break_percent: float
    uncertainty: dict[str, float]
    feature_compiler_approximate: bool
    warnings: list[str]


class MembranePropertyModel:
    def __init__(
        self,
        tg_model_path: Path,
        mechanical_model_path: Path,
        calibration_path: Path | None = None,
    ):
        if not tg_model_path.exists():
            raise ModelError(f"Tg model not found: {tg_model_path}")
        if not mechanical_model_path.exists():
            raise ModelError(
                f"Mechanical model not found: {mechanical_model_path}"
            )
        self.tg_bundle = joblib.load(tg_model_path)
        self.mechanical_bundle = joblib.load(mechanical_model_path)
        self.tg_model = self.tg_bundle["model"]
        self.mechanical_models = self.mechanical_bundle["models"]
        self.feature_columns = self.mechanical_bundle["feature_columns"]
        self.calibration_bundle = None
        self.calibration_path = calibration_path
        if calibration_path and calibration_path.exists():
            self.calibration_bundle = joblib.load(calibration_path)

    @staticmethod
    def _tree_std(model, vector: np.ndarray) -> float:
        estimators = getattr(model, "estimators_", None)
        if not estimators:
            return 0.0
        predictions = [
            float(tree.predict(vector.reshape(1, -1))[0])
            for tree in estimators
        ]
        return float(np.std(predictions))

    def predict(
        self,
        polymer_smiles: str,
        compiled_features: CompiledFeatures,
    ) -> FormulationPrediction:
        tg_vector = morgan_bit_vector(
            polymer_smiles,
            int(self.tg_bundle.get("fingerprint_bits", 2048)),
        )
        tg_value = float(self.tg_model.predict(tg_vector.reshape(1, -1))[0])
        tg_uncertainty = self._tree_std(self.tg_model, tg_vector)

        mechanical_vector = np.asarray(
            [
                compiled_features.standardized[column]
                for column in self.feature_columns
            ],
            dtype=float,
        )
        predictions = {}
        uncertainties = {}
        for target, model in self.mechanical_models.items():
            log_value = float(model.predict(mechanical_vector.reshape(1, -1))[0])
            raw_value = float(10.0 ** np.clip(log_value, -12, 12))
            if target == "logEB":
                raw_value *= 100.0
            predictions[target] = raw_value
            uncertainties[target] = self._tree_std(model, mechanical_vector)

        if self.calibration_bundle:
            calibration_vector = np.concatenate(
                [
                    np.asarray(
                        [
                            tg_value,
                            np.log10(max(predictions["logYM"], 1e-12)),
                            np.log10(max(predictions["logTS"], 1e-12)),
                            np.log10(max(predictions["logEB"] / 100.0, 1e-12)),
                        ],
                        dtype=float,
                    ),
                    mechanical_vector,
                ]
            )
            calibration_models = self.calibration_bundle.get("models", {})
            if "tg_c" in calibration_models:
                model = calibration_models["tg_c"]
                tg_value = float(
                    model.predict(calibration_vector.reshape(1, -1))[0]
                )
                tg_uncertainty = self._tree_std(model, calibration_vector)
            for target, property_name in {
                "youngs_modulus_mpa": "logYM",
                "tensile_strength_mpa": "logTS",
                "elongation_at_break_percent": "logEB",
            }.items():
                if target not in calibration_models:
                    continue
                model = calibration_models[target]
                predictions[property_name] = float(
                    model.predict(calibration_vector.reshape(1, -1))[0]
                )
                uncertainties[property_name] = self._tree_std(
                    model,
                    calibration_vector,
                )

        warnings = [
            "The 643-row mechanical dataset uses composition/process features rather "
            "than explicit monomer sequences. Monomer mode uses an approximate "
            "feature compiler, so arrangement-sensitive mechanical predictions are "
            "screening estimates."
        ]
        if compiled_features.missing_features:
            warnings.append(
                "Approximate features were required for: "
                + ", ".join(compiled_features.missing_features)
                + "."
            )
        if tg_uncertainty > 60:
            warnings.append(
                "The Tg ensemble spread is high; the polymer is likely outside the "
                "well-supported training region."
            )

        return FormulationPrediction(
            tg_c=tg_value,
            tg_uncertainty_c=tg_uncertainty,
            youngs_modulus_mpa=predictions["logYM"],
            tensile_strength_mpa=predictions["logTS"],
            elongation_at_break_percent=predictions["logEB"],
            uncertainty={
                "tg_log10_c": tg_uncertainty,
                "youngs_modulus_log10_mpa": uncertainties["logYM"],
                "tensile_strength_log10_mpa": uncertainties["logTS"],
                "elongation_at_break_log10_percent": uncertainties["logEB"],
            },
            feature_compiler_approximate=compiled_features.approximate,
            warnings=warnings,
        )
