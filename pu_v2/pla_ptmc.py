"""PLA-OH / PTMC-OH mixed-soft-segment family model.

Rebuilt from independent, open-access literature. Nothing from the removed
polyurethane paper (DOI 10.1002/app.57238) is used here. Sources:

* Decomposition-temperature composition series (PLA, PLA/PTMC blends and
  P(LA-TMC) statistical copolymers): Int. J. Mol. Sci. 15(2), 2608 (2014),
  DOI ``10.3390/ijms15022608``, Table 4.
* Glass-transition anchors: Int. J. Mol. Sci. 18(10), 2041 (2017),
  DOI ``10.3390/ijms18102041``, DSC section (neat PLA about 58 C; PLA/PTMC
  7:3 blend 38.8 C; PTMC glass transition about -20 C).
* Chain-length series for a lactide-based polyester: Pharmaceutics 14(5), 993
  (2022), DOI ``10.3390/polym14050993``, Section 3, which quotes Lee et al.:
  a PLGA of 8000 g/mol melts its glass transition at 42.17 C and one of
  110000 g/mol at 52.62 C. These two points are used only to fit the
  Flory-Fox free-volume parameter ``K``.

Two fitted quantities come out of that data:

* ``gordon_taylor_k`` - mixing parameter between PLA and PTMC, fitted to the
  three Tg anchors.
* ``flory_fox_k`` - chain-end free-volume parameter, fitted to the published
  lactide-based molecular-weight series. It is applied to the lactide
  component only, because that is the only side the data covers: no published
  chain-end parameter for PTMC was found. Real PLA-OH/PTMC-OH soft segments are
  oligomers of a few thousand g/mol, so leaving the chain-end term out made the
  family Tg far too high.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class PlaPtmcFamilyError(RuntimeError):
    """Raised when the PLA/PTMC reference table cannot be used."""


@dataclass(frozen=True)
class PlaPtmcEstimate:
    ptmc_mass_fraction_input: float
    ptmc_mass_fraction_used: float
    soft_segment_mn_g_mol: float | None
    soft_segment_tg_c: float
    lactide_tg_c: float
    carbonate_tg_c: float
    td5_c: float
    tdmax_c: float
    gordon_taylor_k: float
    flory_fox_k: float | None
    chain_end_shift_c: float
    chain_end_applied_to: str
    clipped: bool
    sources: list[str] = field(default_factory=list)


def _gordon_taylor(
    ptmc_mass_fraction: float,
    tg_lactide_k: float,
    tg_carbonate_k: float,
    k: float,
) -> float:
    w_lactide = 1.0 - ptmc_mass_fraction
    w_carbonate = ptmc_mass_fraction
    denominator = w_lactide + k * w_carbonate
    if denominator <= 0:
        return tg_lactide_k
    return (w_lactide * tg_lactide_k + k * w_carbonate * tg_carbonate_k) / denominator


def _positive_or_none(value: object) -> float | None:
    """Return a usable positive number, treating NaN and blanks as missing."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


class PlaPtmcFamilyModel:
    """Composition- and chain-length-driven estimates for PLA-OH/PTMC-OH."""

    def __init__(self, table_path: Path):
        if not table_path.exists():
            raise PlaPtmcFamilyError(
                f"PLA/PTMC reference table not found: {table_path}"
            )
        frame = pd.read_csv(table_path)
        required = {"kind", "material_type", "ptmc_mass_fraction"}
        missing = required.difference(frame.columns)
        if missing:
            raise PlaPtmcFamilyError(
                f"PLA/PTMC reference table is missing columns: {sorted(missing)}"
            )

        decomposition = frame.loc[frame["kind"] == "td"].copy()
        self.blend_td = (
            decomposition.loc[decomposition["material_type"] == "blend"]
            .sort_values("ptmc_mass_fraction")
            .reset_index(drop=True)
        )
        self.copolymer_td = (
            decomposition.loc[decomposition["material_type"] == "copolymer"]
            .sort_values("ptmc_mass_fraction")
            .reset_index(drop=True)
        )
        if len(self.blend_td) < 2:
            raise PlaPtmcFamilyError(
                "At least two PLA/PTMC blend decomposition points are required."
            )

        anchors = (
            frame.loc[frame["kind"] == "tg"]
            .dropna(subset=["tg_c"])
            .sort_values("ptmc_mass_fraction")
        )
        if len(anchors) < 2:
            raise PlaPtmcFamilyError(
                "At least two Tg anchors are required to fit the Gordon-Taylor term."
            )
        self.tg_anchors = anchors.reset_index(drop=True)
        flags = frame.loc[frame["kind"] == "tg"].copy()
        flagged = flags.loc[
            flags["flory_fox_applies"].astype(str).str.upper() == "TRUE"
        ]
        self.flory_fox_component = str(flagged["material"].iloc[0]) if len(
            flagged
        ) else "PLA"
        self.flory_fox_targets_lactide = (
            self.flory_fox_component.upper().startswith("PLA")
        )
        weights = self.tg_anchors["ptmc_mass_fraction"].to_numpy(dtype=float)
        self.lactide_tg_k = (
            float(self.tg_anchors.loc[weights.argmin(), "tg_c"]) + 273.15
        )
        self.carbonate_tg_k = (
            float(self.tg_anchors.loc[weights.argmax(), "tg_c"]) + 273.15
        )
        self.gordon_taylor_k = self._fit_gordon_taylor_k(self.tg_anchors)

        length_series = frame.loc[frame["kind"] == "tg_mn"].dropna(
            subset=["mn_g_mol", "tg_c"]
        )
        self.flory_fox_k: float | None = None
        if len(length_series) >= 2:
            self.flory_fox_k = self._fit_flory_fox_k(length_series)

        self.sources = sorted(
            {
                str(value)
                for value in frame["source_doi"].dropna().unique().tolist()
            }
        )

    @staticmethod
    def _least_squares(
        candidates: np.ndarray,
        residuals,
    ) -> float:
        errors = np.empty(candidates.size, dtype=float)
        for index, candidate in enumerate(candidates):
            errors[index] = float(np.sum(np.square(residuals(candidate))))
        return float(candidates[int(errors.argmin())])

    def _fit_gordon_taylor_k(self, anchors: pd.DataFrame) -> float:
        weights = anchors["ptmc_mass_fraction"].to_numpy(dtype=float)
        measured_k = anchors["tg_c"].to_numpy(dtype=float) + 273.15

        def residuals(candidate: float) -> np.ndarray:
            return np.array(
                [
                    _gordon_taylor(
                        float(weight),
                        self.lactide_tg_k,
                        self.carbonate_tg_k,
                        candidate,
                    )
                    for weight in weights
                ]
            ) - measured_k

        coarse = self._least_squares(np.logspace(-2.0, 2.0, 4000), residuals)
        span = coarse * 0.02
        fine = np.linspace(coarse - span, coarse + span, 4001)
        return self._least_squares(fine[fine > 0], residuals)

    @staticmethod
    def _fit_flory_fox_k(series: pd.DataFrame) -> float:
        """Fit ``Tg = Tg_inf - K / Mn`` on a published molecular-weight series."""

        mn = series["mn_g_mol"].to_numpy(dtype=float)
        measured_k = series["tg_c"].to_numpy(dtype=float) + 273.15
        inverse_mn = 1.0 / mn
        slope, intercept = np.polyfit(inverse_mn, measured_k, 1)
        # slope is -K, intercept is Tg at infinite chain length.
        return float(abs(slope))

    def _interpolate_td(self, ptmc_mass_fraction: float) -> tuple[float, float, bool]:
        x = self.blend_td["ptmc_mass_fraction"].to_numpy(dtype=float)
        td5 = self.blend_td["td5_c"].to_numpy(dtype=float)
        tdmax = self.blend_td["tdmax_c"].to_numpy(dtype=float)
        clipped = bool(
            ptmc_mass_fraction < float(x[0]) or ptmc_mass_fraction > float(x[-1])
        )
        used = min(max(ptmc_mass_fraction, float(x[0])), float(x[-1]))
        return (
            float(np.interp(used, x, td5)),
            float(np.interp(used, x, tdmax)),
            clipped,
        )

    def predict(
        self,
        ptmc_mass_fraction: float,
        soft_segment_mn_g_mol: float | None = None,
    ) -> PlaPtmcEstimate:
        fraction = float(min(max(ptmc_mass_fraction, 0.0), 1.0))
        td5_c, tdmax_c, clipped = self._interpolate_td(fraction)

        lactide_tg_k = self.lactide_tg_k
        carbonate_tg_k = self.carbonate_tg_k
        chain_end_shift = 0.0
        if (
            self.flory_fox_k is not None
            and soft_segment_mn_g_mol is not None
            and soft_segment_mn_g_mol > 0
        ):
            shift = self.flory_fox_k / float(soft_segment_mn_g_mol)
            # Only the component the published chain-length data covers is
            # shifted; the other keeps its measured anchor.
            if self.flory_fox_targets_lactide:
                lactide_tg_k -= shift
            else:
                carbonate_tg_k -= shift
            chain_end_shift = -shift

        tg_k = _gordon_taylor(
            fraction,
            lactide_tg_k,
            carbonate_tg_k,
            self.gordon_taylor_k,
        )
        return PlaPtmcEstimate(
            ptmc_mass_fraction_input=float(ptmc_mass_fraction),
            ptmc_mass_fraction_used=fraction,
            soft_segment_mn_g_mol=(
                float(soft_segment_mn_g_mol)
                if soft_segment_mn_g_mol
                else None
            ),
            soft_segment_tg_c=tg_k - 273.15,
            lactide_tg_c=lactide_tg_k - 273.15,
            carbonate_tg_c=carbonate_tg_k - 273.15,
            td5_c=td5_c,
            tdmax_c=tdmax_c,
            gordon_taylor_k=self.gordon_taylor_k,
            flory_fox_k=self.flory_fox_k,
            chain_end_shift_c=chain_end_shift,
            chain_end_applied_to=(
                self.flory_fox_component if chain_end_shift else "none"
            ),
            clipped=clipped,
            sources=list(self.sources),
        )

    @staticmethod
    def ptmc_mass_fraction_from_monomers(
        monomers: list[dict],
        common_mn_g_mol: float | None = None,
    ) -> float | None:
        """PTMC mass fraction of a soft segment made only of PLA and PTMC polyols."""

        pla_mass = 0.0
        ptmc_mass = 0.0
        for item in monomers:
            if item.get("role") != "polyol":
                continue
            monomer_id = str(item.get("id", ""))
            if common_mn_g_mol:
                mass = float(common_mn_g_mol)
            else:
                mass = _positive_or_none(item.get("nominal_mn_g_mol"))
                if mass is None:
                    mass = _positive_or_none(item.get("molecular_weight"))
                if mass is None:
                    continue
            if monomer_id.startswith("POL-PLA"):
                pla_mass += mass
            elif monomer_id.startswith("POL-PTMC"):
                ptmc_mass += mass
        total = pla_mass + ptmc_mass
        if total <= 0:
            return None
        return ptmc_mass / total

    @staticmethod
    def default_soft_segment_mn(monomers: list[dict]) -> float | None:
        """Mass-weighted nominal molecular weight of the PLA/PTMC polyols."""

        total_mass = 0.0
        weighted = 0.0
        for item in monomers:
            if item.get("role") != "polyol":
                continue
            monomer_id = str(item.get("id", ""))
            if not (
                monomer_id.startswith("POL-PLA")
                or monomer_id.startswith("POL-PTMC")
            ):
                continue
            mn = item.get("nominal_mn_g_mol")
            mn = _positive_or_none(mn)
            if mn is None:
                continue
            total_mass += mn
            weighted += mn * mn
        if total_mass <= 0:
            return None
        return weighted / total_mass
