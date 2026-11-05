import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import mteo
from mteo.dgp import RoyDGP
from mteo.plots import plot_result


def _fitted():
    df = RoyDGP().sample(50_000, seed=14)
    return mteo.fit(df, "y", "d", "z", extrapolation="linear", bootstrap_reps=0)


def test_plot_returns_axes_with_expected_elements():
    r = _fitted()
    ax = plot_result(r)
    assert len(ax.lines) >= 4          # 2 fitted segments per curve + vlines
    labels = {ln.get_label() for ln in ax.lines} | {c.get_label() for c in ax.collections}
    assert any("MTO" in str(l) for l in labels)
    assert any("MUO" in str(l) for l in labels)
    assert ax.get_xlim()[0] <= 0.0 and ax.get_xlim()[1] >= 1.0
    plt.close("all")


def test_result_plot_method_delegates():
    r = _fitted()
    ax = r.plot()
    assert ax is not None
    plt.close("all")
