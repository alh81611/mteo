import numpy as np
import pytest

from mteo.dgp import OneSidedDGP
from mteo.groups import summarize_strata


def test_one_sided_and_strata_take_up_gradient():
    dgp = OneSidedDGP(n_strata=4)
    df = dgp.sample(200_000, seed=6)
    assert ((df.loc[df["z"] == 0, "d"]) == 0).all()
    st = summarize_strata(df["y"], df["d"], df["z"], df["stratum"])
    p_is = [st.per_stratum[k].p_i for k in sorted(st.per_stratum)]
    assert all(b > a for a, b in zip(p_is, p_is[1:]))  # strata order by construction


def test_truth_selection_is_adverse_by_default():
    t = OneSidedDGP().truth()
    assert t.selection_stat > 0  # compliers have higher untreated outcome than never-takers
    assert t.late == pytest.approx(-500.0, abs=1e-9)


def test_outcomes_positive_and_skewed():
    df = OneSidedDGP().sample(50_000, seed=8)
    y = df["y"].to_numpy()
    assert (y > 0).all()
    assert np.mean(y) < np.percentile(y, 60)  # right skew: mean above median region


def test_score_on_u_truth_raises():
    with pytest.raises(ValueError, match="score_on_u"):
        OneSidedDGP(score_on_u=0.5).truth()


def test_nonpositive_mean_rejected_at_construction():
    # Gamma needs a positive mean; a MUO that dips below 0 on [0,1] must raise.
    with pytest.raises(ValueError, match="muo"):
        OneSidedDGP(muo=(1000.0, -5000.0))  # 1000 - 5000u < 0 for u > 0.2


def test_truth_late_is_stratum_mixture_not_naive_avg():
    # Non-constant MTE(u) = 2000u (mto-muo). The pooled complier LATE is the
    # equal-weight stratum mixture sum_s 1000*t_s^2 / sum_s t_s, which the strata
    # estimator targets -- NOT _avg(mte, 0, mean_t) = 1000*mean_t. This test would
    # fail against the old single-p_i formula, guarding the mixture fix.
    dgp = OneSidedDGP(muo=(9000.0, -4000.0), mto=(9000.0, -2000.0), n_strata=5)
    t = dgp.truth()
    thr = np.linspace(0.15, 0.55, 5)
    mixture = sum(1000.0 * x**2 for x in thr) / thr.sum()
    naive = 1000.0 * thr.mean()
    assert t.late == pytest.approx(mixture, abs=1e-9)
    assert abs(t.late - naive) > 1.0  # the fix genuinely changed the value

    # End-to-end: the pooled estimator's LATE lands closer to the mixture truth
    # than to the naive value, confirming truth() matches the estimand.
    df = dgp.sample(800_000, seed=21)
    g = summarize_strata(df["y"], df["d"], df["z"], df["stratum"]).pooled
    assert abs(g.late - mixture) < abs(g.late - naive)
