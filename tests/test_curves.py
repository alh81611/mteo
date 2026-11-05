import numpy as np
import pytest

from mteo.curves import Curve, fit_curve, interval_design
from mteo.exceptions import IdentificationError
from mteo.groups import Cell


def _cells_from_poly(coef, intervals, n=1000.0):
    def avg(a, b):
        return sum(c * (b ** (k + 1) - a ** (k + 1)) / ((k + 1) * (b - a)) for k, c in enumerate(coef))
    return [Cell("c", True, (a, b), avg(a, b), n) for a, b in intervals]


def test_linear_recovery_exact():
    coef = (1.4, -0.8)
    cells = _cells_from_poly(coef, [(0.15, 0.41), (0.41, 1.0)])
    curve = fit_curve(cells, order=1)
    assert curve.coef == pytest.approx(coef, abs=1e-10)
    assert curve.at(0.5) == pytest.approx(1.4 - 0.8 * 0.5)
    assert curve.average(0.0, 1.0) == pytest.approx(1.4 - 0.4)
    assert curve.identified_span == (0.15, 1.0)
    assert curve.is_extrapolated(0.0, 0.15) and not curve.is_extrapolated(0.2, 0.9)


def test_quadratic_recovery_exact():
    coef = (0.5, 1.0, -2.0)
    cells = _cells_from_poly(coef, [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)])
    curve = fit_curve(cells, order=2)
    assert curve.coef == pytest.approx(coef, abs=1e-10)


def test_order_cap_raises():
    cells = _cells_from_poly((1.0, 1.0), [(0.0, 0.5)])
    with pytest.raises(IdentificationError, match="order"):
        fit_curve(cells, order=1)


def test_overidentified_wls_uses_weights():
    coef = (1.0, -1.0)
    cells = _cells_from_poly(coef, [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)])
    # perturb the low-weight cell; heavy cells dominate the fit
    cells[2] = Cell("c", True, cells[2].interval, cells[2].mean + 5.0, 1e-6)
    curve = fit_curve(cells, order=1)
    assert curve.coef == pytest.approx(coef, abs=1e-3)
