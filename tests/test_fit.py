import numpy as np
import pytest

import mteo
from mteo.dgp import OneSidedDGP, RoyDGP
from mteo.exceptions import IdentificationError


def test_fit_recovers_roy_truth_linear():
    dgp = RoyDGP()
    df = dgp.sample(400_000, seed=9)
    t = dgp.truth()
    r = mteo.fit(df, "y", "d", "z", extrapolation="linear", bootstrap_reps=100, seed=1)
    assert r.estimands["late"].value == pytest.approx(t.late, abs=0.04)
    assert r.estimands["ate"].value == pytest.approx(t.ate, abs=0.08)
    assert r.estimands["ate"].status == "extrapolated"
    assert r.tests.selection.stat == pytest.approx(t.selection_stat, abs=0.04)
    assert r.tests.selection.se is not None and r.tests.selection.se > 0
    assert r.boot.n_boot == 100


def test_one_sided_without_strata_raises_on_extrapolation():
    df = OneSidedDGP().sample(100_000, seed=10)
    with pytest.raises(IdentificationError, match="strata"):
        mteo.fit(df, "y", "d", "z", extrapolation="linear", bootstrap_reps=0)


def test_one_sided_with_strata_extrapolates():
    dgp = OneSidedDGP()
    df = dgp.sample(400_000, seed=11)
    t = dgp.truth()
    r = mteo.fit(df, "y", "d", "z", strata="stratum", extrapolation="linear", bootstrap_reps=0)
    assert r.estimands["att"].status == "point"
    assert r.estimands["ate"].value == pytest.approx(t.ate, rel=0.25)


def test_none_mode_never_extrapolates_and_summary_prints():
    df = RoyDGP().sample(50_000, seed=12)
    r = mteo.fit(df, "y", "d", "z", bootstrap_reps=0)
    assert r.estimands["ate"].status == "not_identified"
    s = r.summary()
    assert "LATE" in s and "not identified" in s.lower()


def test_defier_data_raises():
    df = RoyDGP().sample(20_000, seed=13)
    df["z"] = 1 - df["z"]
    with pytest.raises(IdentificationError):
        mteo.fit(df, "y", "d", "z", bootstrap_reps=0)


def test_explicit_poly_order_too_high_raises():
    df = RoyDGP().sample(20_000, seed=30)
    with pytest.raises(IdentificationError):
        mteo.fit(df, "y", "d", "z", poly_order=5, extrapolation="none", bootstrap_reps=0)


def test_overid_gets_bootstrap_se():
    df = OneSidedDGP().sample(120_000, seed=31)
    r = mteo.fit(df, "y", "d", "z", strata="stratum", extrapolation="linear", bootstrap_reps=25, seed=2)
    assert r.tests.overid is not None
    assert r.tests.overid.se is not None and r.tests.overid.se >= 0


def test_predict_late_raises_one_sided_no_strata():
    from mteo.dgp import OneSidedDGP
    from mteo.exceptions import IdentificationError
    df = OneSidedDGP().sample(80_000, seed=40)
    r = mteo.fit(df, "y", "d", "z", extrapolation="none", bootstrap_reps=0)
    assert r.mto is None  # single MTO point -> no curve identified
    with pytest.raises(IdentificationError):
        r.predict_late(0.6, 0.9)


def test_predict_late_works_with_strata():
    from mteo.dgp import OneSidedDGP
    df = OneSidedDGP().sample(200_000, seed=41)
    r = mteo.fit(df, "y", "d", "z", strata="stratum", extrapolation="linear", bootstrap_reps=0)
    assert r.mto is not None and r.muo is not None
    est = r.predict_late(0.5, 0.7)
    assert est.status in ("point", "extrapolated")
    assert isinstance(est.value, float)
