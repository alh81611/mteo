"""Assumption-free identification arithmetic for the binary-instrument design.

Groups map to p(U) intervals (Kowalski 2023): always takers (0, p_C],
compliers (p_C, p_I], never takers (p_I, 1]. Cell means for unobserved
complier cells are backed out by the LATE weighting formulas; nothing here
imposes shape restrictions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mteo.exceptions import IdentificationError, WeakInstrumentWarning

Interval = tuple[float, float]


@dataclass(frozen=True)
class Cell:
    group: str          # "at" | "c" | "nt"
    treated: bool
    interval: Interval  # (lo, hi] on the unit p(U) interval
    mean: float
    n: float            # effective observations backing the mean (WLS weight)


@dataclass(frozen=True)
class GroupSummary:
    p_c: float
    p_i: float
    one_sided: bool
    shares: dict[str, float]
    arm_means: dict[tuple[int, int], float]
    arm_ns: dict[tuple[int, int], int]
    mto_cells: tuple[Cell, ...]
    muo_cells: tuple[Cell, ...]
    late: float
    n: int


def _wmean(y: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        raise IdentificationError("empty (d, z) cell; cannot compute its mean")
    return float(np.average(y[mask], weights=w[mask]))


def summarize_groups(y, d, z, w=None, min_take_up_gap: float = 0.02) -> GroupSummary:
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=int)
    z = np.asarray(z, dtype=int)
    if not (set(np.unique(d)) <= {0, 1} and set(np.unique(z)) <= {0, 1}):
        raise IdentificationError("treatment and instrument must be binary 0/1")
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if np.isnan(y).any():
        raise IdentificationError("outcome contains NaN; drop or impute explicitly upstream")

    p_c = float(np.average(d[z == 0], weights=w[z == 0]))
    p_i = float(np.average(d[z == 1], weights=w[z == 1]))
    if p_i <= p_c:
        raise IdentificationError(
            f"take-up must rise with the instrument (p_C={p_c:.4f} >= p_I={p_i:.4f}); "
            "data are inconsistent with no-defiers monotonicity as coded"
        )
    if p_i - p_c < min_take_up_gap:
        warnings.warn(
            f"take-up gap p_I - p_C = {p_i - p_c:.4f} is thin; complier quantities and any "
            "extrapolation will be fragile",
            WeakInstrumentWarning,
            stacklevel=2,
        )

    one_sided = not bool(((d == 1) & (z == 0)).any())
    if one_sided:
        p_c = 0.0

    arm_means: dict[tuple[int, int], float] = {}
    arm_ns: dict[tuple[int, int], int] = {}
    for dd in (0, 1):
        for zz in (0, 1):
            m = (d == dd) & (z == zz)
            if dd == 1 and zz == 0 and one_sided:
                continue
            arm_means[(dd, zz)] = _wmean(y, w, m)
            arm_ns[(dd, zz)] = int(m.sum())

    share_c = p_i - p_c
    shares = {"at": p_c, "c": share_c, "nt": 1.0 - p_i}
    n = int(len(y))

    if one_sided:
        complier_t = arm_means[(1, 1)]
        complier_u = (arm_means[(0, 0)] - (1 - p_i) * arm_means[(0, 1)]) / share_c
    else:
        complier_t = (p_i * arm_means[(1, 1)] - p_c * arm_means[(1, 0)]) / share_c
        complier_u = ((1 - p_c) * arm_means[(0, 0)] - (1 - p_i) * arm_means[(0, 1)]) / share_c

    n_c_eff = share_c * n
    mto_cells: list[Cell] = []
    if not one_sided:
        mto_cells.append(Cell("at", True, (0.0, p_c), arm_means[(1, 0)], arm_ns[(1, 0)]))
    mto_cells.append(Cell("c", True, (p_c, p_i), complier_t, n_c_eff))
    muo_cells = [
        Cell("c", False, (p_c, p_i), complier_u, n_c_eff),
        Cell("nt", False, (p_i, 1.0), arm_means[(0, 1)], arm_ns[(0, 1)]),
    ]

    return GroupSummary(
        p_c=p_c, p_i=p_i, one_sided=one_sided, shares=shares,
        arm_means=arm_means, arm_ns=arm_ns,
        mto_cells=tuple(mto_cells), muo_cells=tuple(muo_cells),
        late=complier_t - complier_u, n=n,
    )


@dataclass(frozen=True)
class StrataSummary:
    per_stratum: dict[object, GroupSummary]
    pooled: GroupSummary
    mto_cells: tuple[Cell, ...]
    muo_cells: tuple[Cell, ...]
    n_propensity_points: int


def summarize_strata(y, d, z, s, w=None, min_take_up_gap: float = 0.02) -> StrataSummary:
    """Per-stratum group summaries plus the pooled cell union for curve fitting.

    Pooling assumes MUO/MTO are common across strata in p(U) (BMW additive
    separability); the linearity overidentification test in `mteo.tests`
    is the check on that assumption.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=int)
    z = np.asarray(z, dtype=int)
    s = np.asarray(s)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)

    per: dict[object, GroupSummary] = {}
    for level in np.unique(s):
        m = s == level
        per[level] = summarize_groups(y[m], d[m], z[m], w[m], min_take_up_gap=min_take_up_gap)

    pooled = summarize_groups(y, d, z, w, min_take_up_gap=min_take_up_gap)
    mto_cells = tuple(c for g in per.values() for c in g.mto_cells)
    muo_cells = tuple(c for g in per.values() for c in g.muo_cells)
    edges = {round(e, 12) for g in per.values() for e in (g.p_c, g.p_i)} - {0.0, 1.0}
    return StrataSummary(
        per_stratum=per, pooled=pooled,
        mto_cells=mto_cells, muo_cells=muo_cells,
        n_propensity_points=len(edges),
    )
