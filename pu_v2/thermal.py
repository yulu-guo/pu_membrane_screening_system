from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .model import morgan_bit_vector


class GeneralPublicThermalModel:
    """Broad polymer Td/Td5 models trained from downloaded public datasets."""

    def __init__(self, td_model_path: Path, td5_model_path: Path):
        self.td_bundle = (
            joblib.load(td_model_path) if td_model_path.exists() else None
        )
        self.td5_bundle = (
            joblib.load(td5_model_path) if td5_model_path.exists() else None
        )

    @staticmethod
    def _mixture_vector(smiles: str, composition: float) -> np.ndarray:
        return np.concatenate(
            [
                morgan_bit_vector(smiles).astype(float),
                [min(max(float(composition), 0.0), 1.0)],
            ]
        )

    @staticmethod
    def _direct_vector(smiles: str) -> np.ndarray:
        return np.concatenate(
            [morgan_bit_vector(smiles).astype(float), [0.5]]
        )

    def predict(
        self,
        polymer_smiles: str,
        composition: float = 0.5,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "available": False,
            "td5_c": None,
            "max_decomposition_c": None,
            "sources": [],
        }
        if self.td5_bundle:
            model = self.td5_bundle["model"]
            value = float(
                model.predict(
                    self._direct_vector(polymer_smiles).reshape(1, -1)
                )[0]
            )
            result["td5_c"] = value
            result["available"] = True
            result["sources"].append("public general Td5 model")
        if self.td_bundle:
            model = self.td_bundle["model"]
            value = float(
                model.predict(
                    self._mixture_vector(
                        polymer_smiles,
                        composition,
                    ).reshape(1, -1)
                )[0]
            )
            result["max_decomposition_c"] = value
            result["available"] = True
            result["sources"].append("public general Td model")
        return result
