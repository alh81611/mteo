"""Target parameters as weighted averages of MUO/MTO/MTE over p(U) intervals.

Every estimand carries an identification status; nothing extrapolates
silently. Kowalski's reweighting: E[.| p_L < U <= p_H] integrates the
curve with uniform weight 1/(p_H - p_L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mteo.curves import Curve, bound_interval_average
from mteo.exceptions import IdentificationError
from mteo.groups import GroupSummary

_REMEDY = "set extrapolation='linear' or 'bounds', or provide strata for more propensity points"


@dataclass(frozen=True)
class Estimand:
    name: str
    status: str
    _value: float | None = None
    bounds: tuple[float, float] | None = None
    reason: str = ""

    @property
    def value(self) -> float:
        if self.status == "not_identified":
            raise IdentificationError(f"{self.name} is not identified: {self.reason}")
        if self.status == "bounds":
            raise IdentificationError(
                f"{self.name} is only partially identified; use .bounds, not .value"
            )
        if self._value is None:
            raise RuntimeError(f"{self.name}: status {self.status!r} but no value set (internal invariant violated)")
        return self._value


def _bound_cells(g: GroupSummary, treated: bool):
    return g.mto_cells if treated else g.muo_cells


def _interval_estimand(name, g, mto, muo, lo, hi, extrapolation, directions, outcome_bounds):
    if extrapolation == "none":
        return Estimand(name, "not_identified",
                        reason=f"requires MTE over ({lo:.2f}, {hi:.2f}], beyond point identification; {_REMEDY}")
    if extrapolation == "linear":
        if mto is None or muo is None:
            return Estimand(name, "not_identified",
                            reason=f"linear extrapolation unavailable (a curve could not be fit); {_REMEDY}")
        value = mto.average(lo, hi) - muo.average(lo, hi)
        extrap = mto.is_extrapolated(lo, hi) or muo.is_extrapolated(lo, hi)
        return Estimand(name, "extrapolated" if extrap else "point", _value=float(value))
    if extrapolation == "bounds":
        if directions is None or "mto" not in directions or "muo" not in directions:
            raise IdentificationError(
                "bounds mode requires monotone_direction={'mto': ..., 'muo': ...}"
            )
        t_lo, t_hi = bound_interval_average(list((mto.cells if mto else ())) or list(_bound_cells(g, True)),
                                            lo, hi, directions["mto"], outcome_bounds)
        u_lo, u_hi = bound_interval_average(list((muo.cells if muo else ())) or list(_bound_cells(g, False)),
                                            lo, hi, directions["muo"], outcome_bounds)
        return Estimand(name, "bounds", bounds=(float(t_lo - u_hi), float(t_hi - u_lo)))
    raise ValueError(f"unknown extrapolation mode {extrapolation!r}")


def compute_estimands(
    g: GroupSummary,
    mto: Curve | None,
    muo: Curve | None,
    extrapolation: str,
    monotone_direction: dict | None = None,
    outcome_bounds: tuple[float, float] = (-np.inf, np.inf),
) -> dict[str, Estimand]:
    out = {"late": Estimand("late", "point", _value=float(g.late))}
    if g.one_sided:
        out["att"] = Estimand("att", "point", _value=float(g.late))  # treated == compliers
    else:
        out["att"] = _interval_estimand("att", g, mto, muo, 0.0, g.p_i,
                                        extrapolation, monotone_direction, outcome_bounds)
    out["atu"] = _interval_estimand("atu", g, mto, muo, g.p_c, 1.0,
                                    extrapolation, monotone_direction, outcome_bounds)
    out["ate"] = _interval_estimand("ate", g, mto, muo, 0.0, 1.0,
                                    extrapolation, monotone_direction, outcome_bounds)
    return out


def predict_late(mto: Curve, muo: Curve, p_lo: float, p_hi: float) -> Estimand:
    """LATE for the subpopulation with U_D in (p_lo, p_hi] (uniform weights)."""
    value = mto.average(p_lo, p_hi) - muo.average(p_lo, p_hi)
    extrap = mto.is_extrapolated(p_lo, p_hi) or muo.is_extrapolated(p_lo, p_hi)
    return Estimand(f"late({p_lo:.2f},{p_hi:.2f}]", "extrapolated" if extrap else "point",
                    _value=float(value))
