import numpy as np
import pytest

from mteo.curves import fit_curve
from mteo.estimands import Estimand, compute_estimands, predict_late
from mteo.exceptions import IdentificationError
from mteo.groups import Cell, summarize_groups


def _linear_world(p_c=0.15, p_i=0.41):
    # Truth: MTO(p) = 2.0 - 1.5p, MUO(p) = 1.4 - 0.8p  =>  MTE(p) = 0.6 - 0.7p
    mto_avg = lambda a, b: 2.0 - 1.5 * (a + b) / 2
    muo_avg = lambda a, b: 1.4 - 0.8 * (a + b) / 2
    mto_cells = [Cell("at", True, (0.0, p_c), mto_avg(0, p_c), 100.0),
                 Cell("c", True, (p_c, p_i), mto_avg(p_c, p_i), 100.0)]
    muo_cells = [Cell("c", False, (p_c, p_i), muo_avg(p_c, p_i), 100.0),
                 Cell("nt", False, (p_i, 1.0), muo_avg(p_i, 1.0), 100.0)]
    return fit_curve(mto_cells, 1), fit_curve(muo_cells, 1)


def _two_sided_groups():
    n_arm = 10_000
    z = np.repeat([0, 1], n_arm)
    d = np.zeros_like(z)
    d[:n_arm][:1500] = 1
    d[n_arm:][:4100] = 1
    y = np.zeros(2 * n_arm)
    for (dd, zz), mu in {(1, 0): 1.89, (0, 0): 1.05, (1, 1): 1.62, (0, 1): 0.85}.items():
        y[(d == dd) & (z == zz)] = mu
    return summarize_groups(y, d, z)


def test_none_mode_only_late():
    g = _two_sided_groups()
    est = compute_estimands(g, None, None, extrapolation="none")
    assert est["late"].status == "point"
    assert est["late"].value == pytest.approx(g.late)
    for k in ("att", "atu", "ate"):
        assert est[k].status == "not_identified"
        with pytest.raises(IdentificationError):
            _ = est[k].value


def test_linear_mode_matches_truth():
    g = _two_sided_groups()
    mto, muo = _linear_world()
    est = compute_estimands(g, mto, muo, extrapolation="linear")
    # ATE under truth: mean of 0.6 - 0.7p over (0,1] = 0.6 - 0.35
    assert est["ate"].value == pytest.approx(0.25, abs=1e-10)
    assert est["ate"].status == "extrapolated"
    # ATT over (0, 0.41]: 0.6 - 0.7*0.205
    assert est["att"].value == pytest.approx(0.6 - 0.7 * 0.205, abs=1e-10)


def test_one_sided_att_is_late():
    n_arm = 10_000
    z = np.repeat([0, 1], n_arm)
    d = np.zeros_like(z); d[n_arm:][:4100] = 1
    y = np.zeros(2 * n_arm)
    for (dd, zz), mu in {(0, 0): 1.0, (1, 1): 1.5, (0, 1): 0.8}.items():
        y[(d == dd) & (z == zz)] = mu
    g = summarize_groups(y, d, z)
    est = compute_estimands(g, None, None, extrapolation="none")
    assert est["att"].status == "point" and est["att"].value == pytest.approx(g.late)


def _one_sided_groups():
    n_arm = 10_000
    z = np.repeat([0, 1], n_arm)
    d = np.zeros_like(z)
    d[n_arm:][:4100] = 1
    y = np.zeros(2 * n_arm)
    for (dd, zz), mu in {(0, 0): 1.0, (1, 1): 1.5, (0, 1): 0.8}.items():
        y[(d == dd) & (z == zz)] = mu
    return summarize_groups(y, d, z)


def test_bounds_mode_combines_outcome_bounds():
    # Target (0,1] overlaps every identified cell, so bound_interval_average returns
    # the outcome bounds on both sides (Kowalski's "uninformative" case). MTE bounds
    # are then (MTO_lo - MUO_hi, MTO_hi - MUO_lo) = (1-10, 10-1). Asymmetric outcome
    # bounds make a swapped lo/hi pairing detectable.
    g = _one_sided_groups()
    est = compute_estimands(
        g, None, None, extrapolation="bounds",
        monotone_direction={"mto": "decreasing", "muo": "decreasing"},
        outcome_bounds=(1.0, 10.0),
    )
    assert est["ate"].status == "bounds"
    assert est["ate"].bounds == pytest.approx((-9.0, 9.0))
    with pytest.raises(IdentificationError):
        _ = est["ate"].value


def test_bounds_mode_requires_direction():
    g = _one_sided_groups()
    with pytest.raises(IdentificationError, match="monotone_direction"):
        compute_estimands(g, None, None, extrapolation="bounds", outcome_bounds=(0.0, 10.0))


def test_predict_late_flags_extrapolation():
    mto, muo = _linear_world()
    inside = predict_late(mto, muo, 0.2, 0.4)
    outside = predict_late(mto, muo, 0.9, 0.95)
    assert inside.status == "point"
    assert outside.status == "extrapolated"
    assert outside.value == pytest.approx(0.6 - 0.7 * 0.925, abs=1e-10)
