"""Separate-approach curve fitting (Brinch-Mogstad-Wiswall 2017).

MUO/MTO are fit per side from (group x stratum) cell means, which are
interval AVERAGES of the curve over p(U) intervals -- the design matrix
integrates the polynomial over each interval rather than evaluating it
at a point. With N cells a polynomial of order N-1 is exactly identified
(BMW Proposition 1); more cells than coefficients gives a WLS fit whose
residuals feed the linearity overidentification test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mteo.exceptions import IdentificationError
from mteo.groups import Cell


def interval_design(intervals: Sequence[tuple[float, float]], order: int) -> np.ndarray:
    X = np.empty((len(intervals), order + 1))
    for j, (a, b) in enumerate(intervals):
        for k in range(order + 1):
            X[j, k] = (b ** (k + 1) - a ** (k + 1)) / ((k + 1) * (b - a))
    return X


@dataclass(frozen=True)
class Curve:
    coef: tuple[float, ...]
    cells: tuple[Cell, ...]
    identified_span: tuple[float, float]
    fitted_cell_means: tuple[float, ...]

    def at(self, p):
        return np.polynomial.polynomial.polyval(p, np.asarray(self.coef))

    def average(self, lo: float, hi: float) -> float:
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"invalid p(U) interval ({lo}, {hi}]")
        return float(sum(
            c * (hi ** (k + 1) - lo ** (k + 1)) / ((k + 1) * (hi - lo))
            for k, c in enumerate(self.coef)
        ))

    def is_extrapolated(self, lo: float, hi: float) -> bool:
        s_lo, s_hi = self.identified_span
        return lo < s_lo - 1e-12 or hi > s_hi + 1e-12


def fit_curve(cells: Sequence[Cell], order: int) -> Curve:
    if order + 1 > len(cells):
        raise IdentificationError(
            f"polynomial order {order} needs {order + 1} cells but only {len(cells)} are "
            "identified; lower the order or provide strata for more propensity points"
        )
    intervals = [c.interval for c in cells]
    X = interval_design(intervals, order)
    yv = np.array([c.mean for c in cells])
    sw = np.sqrt(np.array([c.n for c in cells], dtype=float))
    coef, *_ = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)
    fitted = X @ coef
    return Curve(
        coef=tuple(float(c) for c in coef),
        cells=tuple(cells),
        identified_span=(min(a for a, _ in intervals), max(b for _, b in intervals)),
        fitted_cell_means=tuple(float(v) for v in fitted),
    )


def bound_interval_average(
    cells: Sequence[Cell],
    lo: float,
    hi: float,
    direction: str,
    outcome_bounds: tuple[float, float] = (-np.inf, np.inf),
) -> tuple[float, float]:
    """Bounds on the average of a weakly monotone curve over (lo, hi].

    Uses only adjacent identified cell averages plus user outcome bounds.
    Caveat (document to users): Kowalski (2023) found monotonicity bounds
    uninformative in her application and chose linearity (AA.1) instead --
    expect wide bounds unless outcome_bounds are tight.
    """
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    y_lo, y_hi = outcome_bounds
    means_sorted = sorted(cells, key=lambda c: c.interval[0])
    left_neighbors = [c for c in means_sorted if c.interval[1] <= lo + 1e-12]
    right_neighbors = [c for c in means_sorted if c.interval[0] >= hi - 1e-12]

    lower, upper = y_lo, y_hi
    if direction == "decreasing":
        if left_neighbors:                      # target lies right of an identified cell
            upper = min(upper, left_neighbors[-1].mean)
        if right_neighbors:                     # target lies left of an identified cell
            lower = max(lower, right_neighbors[0].mean)
    else:                                       # increasing
        if left_neighbors:
            lower = max(lower, left_neighbors[-1].mean)
        if right_neighbors:
            upper = min(upper, right_neighbors[0].mean)
    return float(lower), float(upper)
