"""Monte Carlo spot checks: recovery, misspecification honesty, size, coverage.

NOT covered: full power curves across effect sizes, Stata mtemore
cross-validation (needs a license), non-i.i.d. designs beyond the cluster
bootstrap test in test_inference.py.
"""
import pytest

import mteo
from mteo.dgp import RoyDGP


pytestmark = pytest.mark.slow


def test_nonlinear_truth_linear_fit_wrong_ate_right_late():
    # Quadratic MUO/MTO truth; linear AA.1 fit. Complier-range LATE must stay
    # ~unbiased; extrapolated ATE must be wrong by a detectable margin.
    dgp = RoyDGP(muo=(1.0, 1.0, -2.0), mto=(2.0, -0.5, -1.0), p_c=0.2, p_i=0.5)
    t = dgp.truth()
    df = dgp.sample(400_000, seed=21)
    r = mteo.fit(df, "y", "d", "z", extrapolation="linear", bootstrap_reps=0)
    assert r.estimands["late"].value == pytest.approx(t.late, abs=0.05)
    assert abs(r.estimands["ate"].value - t.ate) > 0.05  # documented failure mode


def test_selection_test_size_under_null():
    # Flat MUO -> selection stat should reject ~5% at nominal 5% (normal approx).
    dgp = RoyDGP(muo=(1.0,), mto=(1.5, -0.4))
    rejections = 0
    n_sims = 40
    for s in range(n_sims):
        df = dgp.sample(20_000, seed=100 + s)
        r = mteo.fit(df, "y", "d", "z", bootstrap_reps=200, seed=s)
        zstat = r.tests.selection.stat / r.tests.selection.se
        rejections += abs(zstat) > 1.96
    assert rejections <= 8  # loose binomial bound for 40 sims at 5%


def test_bootstrap_ci_covers_late():
    dgp = RoyDGP()
    t = dgp.truth()
    covered = 0
    n_sims = 30
    for s in range(n_sims):
        df = dgp.sample(20_000, seed=200 + s)
        r = mteo.fit(df, "y", "d", "z", bootstrap_reps=300, seed=s)
        lo, hi = r.boot.ci["late"]
        covered += lo <= t.late <= hi
    assert covered >= 24  # >= 80% observed coverage in 30 sims at nominal 95%
