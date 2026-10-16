from __future__ import annotations

import argparse
import json
import math
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from rdkit import RDLogger

from pu_v2.chemistry import (
    ChemistryError,
    MonomerCatalog,
    build_virtual_pu,
    polymer_summary,
)
from pu_v2.features import compile_mechanical_features
from pu_v2.model import MembranePropertyModel, ModelError
from pu_v2.pla_ptmc import PlaPtmcFamilyModel
from pu_v2.screening import FormulationScreener, PropertyTargets
from pu_v2.thermal import GeneralPublicThermalModel


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
RDLogger.DisableLog("rdApp.*")


def json_safe(value: Any) -> Any:
    """Replace NaN/Infinity with null so the browser gets valid JSON.

    ``json.dumps`` happily writes bare ``NaN``, which is not valid JSON and
    makes ``JSON.parse`` throw in the browser. Catalog rows legitimately carry
    missing values (for example a polyol without a nominal molecular weight),
    so they are mapped to ``None`` here.
    """

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


PROPERTY_SPECS = [
    {
        "key": "tg_c",
        "label": "玻璃化转变温度 Tg",
        "unit": "deg C",
        "status": "model",
        "training_rows": 7174,
        "source": "Open Polymer Prediction Tg dataset",
    },
    {
        "key": "youngs_modulus_mpa",
        "label": "杨氏模量",
        "unit": "MPa",
        "status": "model",
        "training_rows": 643,
        "source": "Science Data Bank DOI 10.57760/sciencedb.14957",
    },
    {
        "key": "tensile_strength_mpa",
        "label": "拉伸强度",
        "unit": "MPa",
        "status": "model",
        "training_rows": 643,
        "source": "Science Data Bank DOI 10.57760/sciencedb.14957",
    },
    {
        "key": "elongation_at_break_percent",
        "label": "断裂伸长率",
        "unit": "%",
        "status": "model",
        "training_rows": 643,
        "source": "Science Data Bank DOI 10.57760/sciencedb.14957",
    },
    {
        "key": "td5_c",
        "label": "初始分解温度 Td5%",
        "unit": "deg C",
        "status": "model",
        "training_rows": 58,
        "source": "Open reference dataset of 60 polymers across ten thermal properties",
    },
    {
        "key": "max_decomposition_c",
        "label": "最大分解温度",
        "unit": "deg C",
        "status": "model",
        "training_rows": 1023,
        "source": "GitHub ML-DL Td dataset derived from Polymer Genome",
    },
]


DEFAULT_PROCESSING = {
    "nco_oh_index": 1.05,
    "synthesis_step": 2,
    "forming_method": "solution_casting",
    "reaction_temperature_1_c": 80.0,
    "reaction_temperature_2_c": 80.0,
    "cross_section_area_mm2": 30.0,
    "strain_rate_min_inv": 0.1,
    "soft_segment_mn_g_mol": None,
}


class Application:
    def __init__(self, root: Path):
        self.root = root
        self.catalog = MonomerCatalog.from_csv(
            root / "data" / "monomers" / "monomers.csv"
        )
        self.model = MembranePropertyModel(
            root / "data" / "models" / "tg_model.joblib",
            root / "data" / "models" / "mechanical_model.joblib",
            root / "data" / "models" / "lab_calibration.joblib",
        )
        self.general_thermal = GeneralPublicThermalModel(
            root / "data" / "models" / "general_td_model.joblib",
            root / "data" / "models" / "general_td5_model.joblib",
        )
        self.pla_ptmc = PlaPtmcFamilyModel(
            root / "data" / "reference" / "pla_ptmc_family.csv"
        )
        self.screener = FormulationScreener(
            self.catalog,
            self.model,
            self.pla_ptmc,
            self.general_thermal,
        )

    def health(self) -> dict[str, Any]:
        metrics_path = self.root / "data" / "models" / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        public_thermal_path = (
            self.root / "data" / "models" / "public_thermal_metrics.json"
        )
        if public_thermal_path.exists():
            metrics["public_thermal"] = json.loads(
                public_thermal_path.read_text(encoding="utf-8")
            )
        return {
            "status": "ok",
            "monomers": int(len(self.catalog.frame)),
            "roles": self.catalog.frame["role"].value_counts().to_dict(),
            "properties": [item["key"] for item in PROPERTY_SPECS],
            "metrics": metrics,
        }

    def catalog_payload(self) -> dict[str, Any]:
        return {
            "monomers": self.catalog.as_payload(),
            "roles": sorted(self.catalog.frame["role"].unique().tolist()),
        }

    def _expand_components(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_components = payload.get("components") or []
        if not raw_components:
            raise ChemistryError("At least one component is required.")
        selected = []
        index = 0
        for component in raw_components:
            monomer_id = str(component.get("monomer_id", "")).strip()
            if not monomer_id:
                continue
            count = int(component.get("count", 1))
            if count < 1 or count > 12:
                raise ChemistryError("Each component count must be between 1 and 12.")
            monomer = self.catalog.get(monomer_id)
            for _ in range(count):
                item = dict(monomer)
                item["key"] = f"{monomer_id}_{index + 1}"
                selected.append(item)
                index += 1
        if len(selected) > 24:
            raise ChemistryError("At most 24 total monomer instances are supported.")
        return selected

    @staticmethod
    def _processing(payload: dict[str, Any]) -> dict[str, Any]:
        processing = dict(DEFAULT_PROCESSING)
        processing.update(payload.get("processing") or {})
        return processing

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        selected = self._expand_components(payload)
        reaction_family = str(
            payload.get("reaction_family", "polyurethane")
        )
        if reaction_family != "polyurethane":
            raise ChemistryError("Unsupported reaction family.")
        if not any(item["role"] == "isocyanate" for item in selected):
            raise ChemistryError("A formulation must include an isocyanate.")
        if not any(item["role"] != "isocyanate" for item in selected):
            raise ChemistryError("A formulation must include a polyol or chain extender.")

        arrangement = str(payload.get("arrangement", "alternating"))
        if arrangement not in {"alternating", "random", "block", "custom"}:
            raise ChemistryError("Unsupported arrangement.")
        custom_sequence = payload.get("custom_sequence")
        build = build_virtual_pu(
            selected,
            arrangement=arrangement,
            custom_sequence=custom_sequence,
            random_seed=int(payload.get("random_seed", 42)),
        )
        processing = self._processing(payload)
        compiled = compile_mechanical_features(
            build.smiles,
            selected,
            processing,
        )
        prediction = self.model.predict(build.smiles, compiled)
        warnings = list(prediction.warnings)
        if build.skipped_edges:
            warnings.append(
                f"{build.skipped_edges} sequence connection(s) could not be formed "
                "because compatible reactive groups or stoichiometry were missing."
            )
        if build.attachment_points > 4:
            warnings.append(
                "The generated structure has many branch/chain-end attachment points; "
                "this usually indicates a crosslinked or under-specified formulation."
            )
        hard_fraction = (
            sum(
                float(item["molecular_weight"])
                for item in selected
                if item["role"]
                in {"isocyanate", "chain_extender", "crosslinker"}
            )
            / max(
                sum(float(item["molecular_weight"]) for item in selected),
                1e-9,
            )
        )
        general_thermal = self.general_thermal.predict(
            build.smiles,
            composition=hard_fraction,
        )
        polyol_ids = [
            item["id"] for item in selected if item["role"] == "polyol"
        ]
        pla_count = sum(
            1 for monomer_id in polyol_ids if monomer_id.startswith("POL-PLA")
        )
        ptmc_count = sum(
            1 for monomer_id in polyol_ids if monomer_id.startswith("POL-PTMC")
        )
        # The independent family model covers a soft segment built purely from
        # PLA-OH and PTMC-OH. A third polyol family leaves it.
        pla_ptmc_soft_segment = bool(polyol_ids) and all(
            monomer_id.startswith("POL-PLA")
            or monomer_id.startswith("POL-PTMC")
            for monomer_id in polyol_ids
        )
        family = None
        if pla_count > 0 and ptmc_count > 0 and pla_ptmc_soft_segment:
            requested_mn = processing.get("soft_segment_mn_g_mol")
            if requested_mn in (None, ""):
                soft_segment_mn = self.pla_ptmc.default_soft_segment_mn(selected)
                common_mn = None
            else:
                soft_segment_mn = float(requested_mn)
                # One common Mn means the molar ratio of the two polyols is
                # also their mass ratio.
                common_mn = soft_segment_mn
            ptmc_mass_fraction = self.pla_ptmc.ptmc_mass_fraction_from_monomers(
                selected,
                common_mn,
            )
            if ptmc_mass_fraction is not None:
                family = self.pla_ptmc.predict(
                    ptmc_mass_fraction,
                    soft_segment_mn,
                )
                warnings.append(
                    "PLA-OH/PTMC-OH soft segment: Tg and the decomposition "
                    "temperatures come from an independent PLA/PTMC composition "
                    "model (see family_details). The mechanical values still come "
                    "from the general model; no independent mechanical dataset for "
                    "this family is available."
                )
                if family.clipped:
                    warnings.append(
                        "The soft segment is outside the measured 0-50 mass% PTMC "
                        "PLA/PTMC range; the nearest measured composition was used "
                        "for the decomposition temperatures."
                    )
                if requested_mn in (None, "") and (
                    family.soft_segment_mn_g_mol or 0
                ) < 1500:
                    warnings.append(
                        "No soft-segment Mn was given, so the model used the "
                        "catalog polyol molecular weight, which is well below a "
                        "commercial PLA-OH/PTMC-OH diol. Enter the real polyol Mn "
                        "for a representative Tg."
                    )
        td5_c = family.td5_c if family else general_thermal["td5_c"]
        max_decomposition_c = (
            family.tdmax_c if family else general_thermal["max_decomposition_c"]
        )
        return {
            "polymer_smiles": build.smiles,
            "reaction_family": "polyurethane",
            "sequence": build.sequence,
            "structure": polymer_summary(build.smiles),
            "components": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "role": item["role"],
                    "smiles": item["smiles"],
                }
                for item in selected
            ],
            "arrangement": arrangement,
            "processing": processing,
            "description": (
                "Inputs were compiled into a virtual PU structure. Tg uses the "
                "structure model; mechanical properties use the 643-row composition "
                "and process model with an approximate molecule-to-feature compiler."
            ),
            "properties": {
                "tg_c": (
                    family.soft_segment_tg_c if family else prediction.tg_c
                ),
                "youngs_modulus_mpa": prediction.youngs_modulus_mpa,
                "tensile_strength_mpa": prediction.tensile_strength_mpa,
                "elongation_at_break_percent": (
                    prediction.elongation_at_break_percent
                ),
                "td5_c": td5_c,
                "max_decomposition_c": max_decomposition_c,
            },
            "thermal_details": {
                "td5_c": general_thermal["td5_c"],
                "max_decomposition_c": general_thermal[
                    "max_decomposition_c"
                ],
                "source": "Downloaded public polymer thermal datasets",
                "broad_public_models": general_thermal["sources"],
            },
            "family_details": (
                {
                    "family": "PLA-OH/PTMC-OH soft segment",
                    "ptmc_mass_fraction": family.ptmc_mass_fraction_used,
                    "soft_segment_mn_g_mol": family.soft_segment_mn_g_mol,
                    "soft_segment_tg_c": family.soft_segment_tg_c,
                    "lactide_tg_c": family.lactide_tg_c,
                    "carbonate_tg_c": family.carbonate_tg_c,
                    "td5_c": family.td5_c,
                    "tdmax_c": family.tdmax_c,
                    "gordon_taylor_k": family.gordon_taylor_k,
                    "flory_fox_k": family.flory_fox_k,
                    "chain_end_shift_c": family.chain_end_shift_c,
                    "chain_end_applied_to": family.chain_end_applied_to,
                    "clipped_to_measured_range": family.clipped,
                    "structure_model_tg_c": prediction.tg_c,
                    "mechanical_source": "general 643-row model (not family calibrated)",
                    "source_dois": family.sources,
                }
                if family
                else None
            ),
            "uncertainty": prediction.uncertainty,
            "compiled_features": compiled.standardized,
            "raw_features": compiled.raw,
            "feature_compiler_approximate": (
                prediction.feature_compiler_approximate
            ),
            "warnings": warnings,
        }

    def screen(self, payload: dict[str, Any]) -> dict[str, Any]:
        targets = PropertyTargets.from_payload(payload)
        roles = payload.get("roles") or [
            "isocyanate",
            "polyol",
            "chain_extender",
        ]
        result = self.screener.screen(
            targets=targets,
            roles=[str(role) for role in roles],
            number_of_monomers=min(
                max(int(payload.get("number_of_monomers", 4)), 2),
                8,
            ),
            max_candidates=min(
                max(int(payload.get("max_candidates", 100)), 10),
                1000,
            ),
            processing=self._processing(payload),
            arrangement=str(payload.get("arrangement", "alternating")),
            random_seed=int(payload.get("random_seed", 42)),
        )
        return result

class Handler(BaseHTTPRequestHandler):
    server_version = "PUMembraneProperties/0.2"

    @property
    def application(self) -> Application:
        return self.server.application  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            (mime_type or "application/octet-stream")
            + ("; charset=utf-8" if mime_type and mime_type.startswith("text/") else ""),
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self._json(self.application.health())
                return
            if path == "/api/catalog":
                self._json(self.application.catalog_payload())
                return
            if path == "/api/properties":
                self._json({"properties": PROPERTY_SPECS})
                return
            if path in ("/", "/index.html"):
                self._file(UI_DIR / "index.html")
                return
            if path == "/styles.css":
                self._file(UI_DIR / "styles.css")
                return
            if path == "/app.js":
                self._file(UI_DIR / "app.js")
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_500_000:
                raise ValueError("Request body is too large.")
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            if path == "/api/predict":
                self._json(self.application.predict(payload))
                return
            if path == "/api/screen":
                self._json(self.application.screen(payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ChemistryError, ModelError, ValueError, RuntimeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PU membrane property screening system."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = Application(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.application = app  # type: ignore[attr-defined]
    print(f"PU membrane property system: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
