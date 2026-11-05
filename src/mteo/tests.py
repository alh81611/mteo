"""Test statistics for selection and treatment-effect heterogeneity.

Point statistics only; standard errors and p-values come from the
bootstrap engine, which re-computes these on every replication.
"""
from __future__ import annotations

import numpy as np

from mteo.curves import Curve
from mteo.exceptions import IdentificationError
from mteo.groups import GroupSummary


def _cell_mean(g: GroupSummary, cells: str, group: str) -> float:
    for c in getattr(g, cells):
        if c.group == group:
            return c.mean
    raise IdentificationError(f"cell {group!r} not present in {cells}")


def selection_stat(g: GroupSummary) -> float:
    """Kowalski's untreated-outcome selection test statistic.

    E[Y_U | complier] - E[Y_U | never taker]; nonzero means MUO varies in
    p(U), i.e. selection heterogeneity, under LATE assumptions alone.
    """
    return _cell_mean(g, "muo_cells", "c") - _cell_mean(g, "muo_cells", "nt")


def treated_selection_stat(g: GroupSummary) -> float:
    """Treated-outcome analogue: E[Y_T | always taker] - E[Y_T | complier]."""
    if g.one_sided:
        raise IdentificationError(
            "treated-outcome selection test needs always takers; design is one-sided"
        )
    return _cell_mean(g, "mto_cells", "at") - _cell_mean(g, "mto_cells", "c")


def bmw_stat(y, d, z, w=None) -> float:
    """BMW (2017) external-validity statistic Delta_1 - Delta_0.

    Delta_j = E[Y | D=j, Z=1] - E[Y | D=j, Z=0]. Under a linear MTE model,
    Delta_1 = Delta_0 iff the MTE is constant (LATE externally valid).
    """
    y = np.asarray(y, float); d = np.asarray(d, int); z = np.asarray(z, int)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    deltas = {}
    for j in (0, 1):
        cells = {}
        for zz in (0, 1):
            m = (d == j) & (z == zz)
            if not m.any():
                raise IdentificationError(
                    f"BMW test needs nonempty (D={j}, Z={zz}) cell; "
                    "one-sided designs cannot run it"
                )
            cells[zz] = np.average(y[m], weights=w[m])
        deltas[j] = cells[1] - cells[0]
    return float(deltas[1] - deltas[0])


def mte_slope_stat(mto: Curve, muo: Curve) -> float:
    """Slope of the linear MTE: beta_T - beta_U. Nonzero = effect heterogeneity (under AA.1)."""
    if len(mto.coef) < 2 or len(muo.coef) < 2:
        raise IdentificationError("mte_slope_stat needs curves of order >= 1")
    return float(mto.coef[1] - muo.coef[1])


def overid_stat(curve: Curve) -> float:
    """Weighted residual sum of the curve fit vs identified cell means.

    Zero when exactly identified; with strata, a large value rejects the
    maintained functional form (BMW covariate-variation check).
    """
    return float(sum(
        c.n * (c.mean - f) ** 2 for c, f in zip(curve.cells, curve.fitted_cell_means)
    ))
