import numpy as np
import pytest

from mteo.curves import fit_curve
from mteo.exceptions import IdentificationError
from mteo.groups import Cell, summarize_groups
from mteo.tests import (
    bmw_stat, mte_slope_stat, overid_stat, selection_stat, treated_selection_stat,
)


def _degenerate(mu_by_cell, p_c=0.15, p_i=0.41, n_arm=10_000):
    z = np.repeat([0, 1], n_arm)
    d = np.zeros_like(z)
    d[:n_arm][: int(p_c * n_arm)] = 1
    d[n_arm:][: int(p_i * n_arm)] = 1
    y = np.zeros(2 * n_arm)
    for (dd, zz), mu in mu_by_cell.items():
        y[(d == dd) & (z == zz)] = mu
    return y, d, z


def test_selection_stats_match_kowalski_cells():
    y, d, z = _degenerate({(1, 0): 1.89, (0, 0): 1.05, (1, 1): 1.62, (0, 1): 0.85})
    g = summarize_groups(y, d, z)
    c_u = next(c for c in g.muo_cells if c.group == "c").mean
    assert selection_stat(g) == pytest.approx(c_u - 0.85)
    c_t = next(c for c in g.mto_cells if c.group == "c").mean
    assert treated_selection_stat(g) == pytest.approx(1.89 - c_t)


def test_bmw_stat_null_when_effects_constant():
    # Constant treatment effect tau: Y = 1 + tau*D exactly -> Delta1 = Delta0 = 0.
    y, d, z = _degenerate({(1, 0): 3.0, (0, 0): 1.0, (1, 1): 3.0, (0, 1): 1.0})
    assert bmw_stat(y, d, z) == pytest.approx(0.0, abs=1e-12)


def test_mte_slope_and_overid():
    mto = fit_curve([Cell("at", True, (0.0, 0.15), 2.0, 10.0), Cell("c", True, (0.15, 0.41), 1.5, 10.0)], 1)
    muo = fit_curve([Cell("c", False, (0.15, 0.41), 1.2, 10.0), Cell("nt", False, (0.41, 1.0), 0.9, 10.0)], 1)
    assert mte_slope_stat(mto, muo) == pytest.approx(mto.coef[1] - muo.coef[1])
    assert overid_stat(mto) == pytest.approx(0.0, abs=1e-18)


def test_one_sided_treated_stat_raises():
    y, d, z = _degenerate({(1, 0): 0.0, (0, 0): 1.05, (1, 1): 1.62, (0, 1): 0.85}, p_c=0.0)
    g = summarize_groups(y, d, z)
    with pytest.raises(IdentificationError):
        treated_selection_stat(g)


def test_bmw_stat_raises_on_empty_cell():
    # One-sided design: no treated controls -> (D=1, Z=0) cell empty -> must raise,
    # not silently produce NaN. This raise path is hit under bootstrap resampling.
    y, d, z = _degenerate({(0, 0): 1.05, (1, 1): 1.62, (0, 1): 0.85}, p_c=0.0)
    with pytest.raises(IdentificationError):
        bmw_stat(y, d, z)


def test_bmw_stat_honors_weights():
    # (D=1,Z=1) has y=[2,4] with weights [1,3] -> weighted mean 3.5 vs unweighted 3.
    d = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    z = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y = np.array([2.0, 4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    w = np.array([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert bmw_stat(y, d, z, w) == pytest.approx(2.5)   # (3.5 - 1) - (0 - 0)
    assert bmw_stat(y, d, z) == pytest.approx(2.0)       # (3.0 - 1) - (0 - 0)


def test_mte_slope_raises_on_constant_curve():
    flat = fit_curve([Cell("c", True, (0.15, 0.41), 1.5, 10.0)], 0)  # order 0 -> len(coef)==1
    linear = fit_curve(
        [Cell("c", False, (0.15, 0.41), 1.2, 10.0), Cell("nt", False, (0.41, 1.0), 0.9, 10.0)], 1
    )
    with pytest.raises(IdentificationError):
        mte_slope_stat(flat, linear)


def test_overid_stat_positive_when_fit_imperfect():
    # V-shaped means cannot lie on a line -> linear WLS leaves nonzero residuals.
    v = fit_curve(
        [
            Cell("c", True, (0.0, 0.3), 1.0, 10.0),
            Cell("c", True, (0.3, 0.6), 0.0, 10.0),
            Cell("c", True, (0.6, 1.0), 1.0, 10.0),
        ],
        1,
    )
    assert overid_stat(v) > 0.1
