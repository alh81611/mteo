import numpy as np
import pytest

from mteo.curves import bound_interval_average
from mteo.groups import Cell


CELLS = [
    Cell("c", True, (0.15, 0.41), 1.45, 100.0),   # complier treated mean
    # region above 0.41 unobserved for MTO
]


def test_bound_right_of_support_decreasing():
    lo, hi = bound_interval_average(CELLS, 0.41, 1.0, "decreasing", outcome_bounds=(0.0, 10.0))
    assert hi == pytest.approx(1.45)   # can't exceed the adjacent identified average
    assert lo == pytest.approx(0.0)    # floor from outcome bounds


def test_bound_left_of_support_decreasing():
    lo, hi = bound_interval_average(CELLS, 0.0, 0.15, "decreasing", outcome_bounds=(0.0, 10.0))
    assert lo == pytest.approx(1.45)
    assert hi == pytest.approx(10.0)


def test_direction_required():
    with pytest.raises(TypeError):
        bound_interval_average(CELLS, 0.41, 1.0)  # no default direction


def test_unbounded_outcome_gives_infinite_side():
    lo, hi = bound_interval_average(CELLS, 0.41, 1.0, "decreasing")
    assert np.isneginf(lo) and hi == pytest.approx(1.45)


def test_bound_increasing_flips_bounds():
    # Increasing curve, target right of the identified cell: nearest-left mean is
    # now the FLOOR (not the ceiling), and the ceiling is the outcome bound.
    lo, hi = bound_interval_average(CELLS, 0.41, 1.0, "increasing", outcome_bounds=(0.0, 10.0))
    assert lo == pytest.approx(1.45)
    assert hi == pytest.approx(10.0)
    # Target left of the identified cell: nearest-right mean is the CEILING.
    lo2, hi2 = bound_interval_average(CELLS, 0.0, 0.15, "increasing", outcome_bounds=(0.0, 10.0))
    assert lo2 == pytest.approx(0.0)
    assert hi2 == pytest.approx(1.45)


def test_invalid_direction_raises_valueerror():
    with pytest.raises(ValueError):
        bound_interval_average(CELLS, 0.41, 1.0, "sideways")
