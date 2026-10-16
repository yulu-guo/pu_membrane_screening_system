from __future__ import annotations

import json
import math
from pathlib import Path

from app import Application, json_safe


ROOT = Path(__file__).resolve().parent


def assert_finite(values: dict[str, float]) -> None:
    for key, value in values.items():
        if not math.isfinite(float(value)):
            raise AssertionError(f"Non-finite value for {key}: {value}")


def main() -> None:
    app = Application(ROOT)
    health = app.health()
    assert health["status"] == "ok"
    assert health["monomers"] >= 3000
    assert set(health["properties"]) == {
        "tg_c",
        "youngs_modulus_mpa",
        "tensile_strength_mpa",
        "elongation_at_break_percent",
        "td5_c",
        "max_decomposition_c",
    }

    payload = {
        "components": [
            {"monomer_id": "ISO-MDI", "count": 2},
            {"monomer_id": "POL-PEG400", "count": 1},
            {"monomer_id": "CE-BDO", "count": 1},
        ],
        "arrangement": "alternating",
        "processing": {
            "nco_oh_index": 1.05,
            "synthesis_step": 2,
            "forming_method": "solution_casting",
            "reaction_temperature_1_c": 80,
            "reaction_temperature_2_c": 80,
            "cross_section_area_mm2": 30,
            "strain_rate_min_inv": 0.1,
        },
    }
    prediction = app.predict(payload)
    assert_finite(prediction["properties"])
    assert prediction["sequence"]
    assert prediction["polymer_smiles"].count("*") >= 2

    random_payload = dict(payload)
    random_payload["arrangement"] = "random"
    random_prediction = app.predict(random_payload)
    assert_finite(random_prediction["properties"])

    screening = app.screen(
        {
            "targets": {
                "tg_c": 50.0,
                "youngs_modulus_mpa": 100.0,
                "tensile_strength_mpa": 20.0,
                "elongation_at_break_percent": 300.0,
            },
            "weights": {
                "tg_c": 1.0,
                "youngs_modulus_mpa": 1.0,
                "tensile_strength_mpa": 1.0,
                "elongation_at_break_percent": 2.0,
            },
            "roles": ["isocyanate", "polyol", "chain_extender"],
            "number_of_monomers": 4,
            "max_candidates": 12,
            "arrangement": "alternating",
            "processing": payload["processing"],
        }
    )
    assert screening["returned_candidates"] > 0

    # Every API response has to survive strict JSON, otherwise the browser
    # cannot parse the catalog and the whole page stops working.
    api_payloads = {
        "health": health,
        "properties": {"properties": app.health()["properties"]},
        "catalog": app.catalog_payload(),
        "prediction": prediction,
        "screening": screening,
    }
    for name, api_payload in api_payloads.items():
        json.dumps(json_safe(api_payload), allow_nan=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "monomers": health["monomers"],
                "strict_json_payloads": sorted(api_payloads),
                "prediction": prediction["properties"],
                "random_arrangement_tg": random_prediction["properties"]["tg_c"],
                "screening_candidates": screening["returned_candidates"],
                "metrics": health["metrics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
