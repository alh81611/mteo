import numpy as np
import pytest

from mteo.exceptions import IdentificationError, WeakInstrumentWarning
from mteo.groups import summarize_groups


def _oregon_cells():
    # Kowalski (2023) Table 1 / Figure 2 arm-cell values, ER visits:
    # p_C=0.15, p_I=0.41, E[Y|D=1,Z=1]=1.62, E[Y|D=1,Z=0]=1.89,
    # E[Y|D=0,Z=1]=0.85, E[Y|D=0,Z=0]=1.05.
    # Implied complier treated mean: (0.41*1.62 - 0.15*1.89)/(0.41-0.15) = 1.4642 from these
    # rounded display inputs. Kowalski reports ~1.45 from unrounded microdata (benign rounding gap).
    n_arm = 100_000
    rng = np.random.default_rng(0)
    z = np.repeat([0, 1], n_arm)
    d = np.zeros_like(z)
    y = np.zeros(2 * n_arm, float)
    # Control arm: 15% treated at mean 1.89; untreated at mean 1.05.
    d[:n_arm][: int(0.15 * n_arm)] = 1
    # Intervention arm: 41% treated at mean 1.62; untreated at mean 0.85.
    d[n_arm:][: int(0.41 * n_arm)] = 1
    for dd, zz, mu in [(1, 0, 1.89), (0, 0, 1.05), (1, 1, 1.62), (0, 1, 0.85)]:
        m = (d == dd) & (z == zz)
        y[m] = mu  # degenerate outcomes: means are exact
    return y, d, z


def test_oregon_group_arithmetic():
    y, d, z = _oregon_cells()
    g = summarize_groups(y, d, z)
    assert g.p_c == pytest.approx(0.15, abs=1e-12)
    assert g.p_i == pytest.approx(0.41, abs=1e-12)
    assert not g.one_sided
    assert g.shares == pytest.approx({"at": 0.15, "c": 0.26, "nt": 0.59}, abs=1e-12)
    (at,), (c_t,) = [x for x in g.mto_cells if x.group == "at"], [x for x in g.mto_cells if x.group == "c"]
    assert at.mean == pytest.approx(1.89)
    assert c_t.mean == pytest.approx(1.4642307692, abs=1e-9)  # (0.41*1.62 - 0.15*1.89)/0.26, hard-pinned
    assert c_t.interval == (0.15, 0.41)
    c_u = next(x for x in g.muo_cells if x.group == "c")
    nt = next(x for x in g.muo_cells if x.group == "nt")
    assert c_u.mean == pytest.approx((0.85 * 1.05 - 0.59 * 0.85) / 0.26)
    assert nt.mean == pytest.approx(0.85)
    assert g.late == pytest.approx(c_t.mean - c_u.mean)


def test_one_sided_detection_and_cells():
    y, d, z = _oregon_cells()
    d = d.copy()
    d[z == 0] = 0  # control arm cannot take treatment
    g = summarize_groups(y, d, z)
    assert g.one_sided and g.p_c == 0.0
    assert [c.group for c in g.mto_cells] == ["c"]          # single MTO point
    assert [c.group for c in g.muo_cells] == ["c", "nt"]    # MUO keeps two points


def test_defiers_raise():
    y, d, z = _oregon_cells()
    with pytest.raises(IdentificationError):
        summarize_groups(y, d, 1 - z)  # flipped instrument implies p_i < p_c


def test_thin_gap_warns():
    rng = np.random.default_rng(1)
    n = 20_000
    z = rng.integers(0, 2, n)
    d = (rng.uniform(size=n) < np.where(z == 1, 0.31, 0.30)).astype(int)
    y = rng.normal(size=n)
    with pytest.warns(WeakInstrumentWarning):
        summarize_groups(y, d, z)
