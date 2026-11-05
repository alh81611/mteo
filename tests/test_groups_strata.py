import numpy as np
import pytest

from mteo.exceptions import IdentificationError
from mteo.groups import summarize_strata


def _two_strata_data():
    # Stratum "lo": one-sided with p_i=0.2; stratum "hi": one-sided with p_i=0.5.
    n = 50_000
    rng = np.random.default_rng(2)
    s = np.repeat(["lo", "hi"], n)
    z = np.tile(np.repeat([0, 1], n // 2), 2)
    u = rng.uniform(size=2 * n)
    p_i = np.where(s == "lo", 0.2, 0.5)
    d = ((z == 1) & (u <= p_i)).astype(int)
    y = 1.0 + 2.0 * u + rng.normal(0, 0.1, size=2 * n)  # MUO=MTO shape irrelevant here
    return y, d, z, s


def test_per_stratum_and_pooled():
    y, d, z, s = _two_strata_data()
    st = summarize_strata(y, d, z, s)
    assert set(st.per_stratum) == {"lo", "hi"}
    assert st.per_stratum["lo"].p_i == pytest.approx(0.2, abs=0.01)
    assert st.per_stratum["hi"].p_i == pytest.approx(0.5, abs=0.01)
    assert st.pooled.n == len(y)
    # union of cells: each one-sided stratum contributes 1 MTO cell and 2 MUO cells
    assert len(st.mto_cells) == 2
    assert len(st.muo_cells) == 4
    assert st.n_propensity_points == 2


def test_stratum_with_defiers_raises():
    y, d, z, s = _two_strata_data()
    bad = s == "lo"
    d2 = d.copy()
    d2[bad] = ((z[bad] == 0) & (np.random.default_rng(3).uniform(size=bad.sum()) < 0.3)).astype(int)
    with pytest.raises(IdentificationError):
        summarize_strata(y, d2, z, s)
