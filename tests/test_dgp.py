import numpy as np
import pytest

import mteo
from mteo.dgp import RoyDGP


def test_truth_closed_forms():
    dgp = RoyDGP(muo=(1.4, -0.8), mto=(2.0, -1.5), p_c=0.15, p_i=0.41)
    t = dgp.truth()
    # MTE(p) = 0.6 - 0.7p
    assert t.ate == pytest.approx(0.6 - 0.35)
    assert t.late == pytest.approx(0.6 - 0.7 * (0.15 + 0.41) / 2)
    assert t.selection_stat == pytest.approx((-0.8) * ((0.15 + 0.41) / 2 - (0.41 + 1.0) / 2))


def test_sample_take_up_and_recovery():
    dgp = RoyDGP()
    df = dgp.sample(400_000, seed=42)
    t = dgp.truth()
    from mteo.groups import summarize_groups
    g = summarize_groups(df["y"], df["d"], df["z"])
    assert g.p_c == pytest.approx(t.p_c, abs=0.01)
    assert g.p_i == pytest.approx(t.p_i, abs=0.01)
    assert g.late == pytest.approx(t.late, abs=0.03)


def test_sample_is_deterministic_given_seed():
    a = RoyDGP().sample(100, seed=1)
    b = RoyDGP().sample(100, seed=1)
    assert (a == b).all().all()
