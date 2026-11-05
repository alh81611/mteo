"""The Kowalski MUO/MTO figure: cell means and fitted curves over p(U).

Convention (Kowalski 2023, Fig. 6): filled markers = point-identified cell
means; dashed curve segments = extrapolation beyond the identified span.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _draw_curve(ax, curve, label, color):
    lo, hi = curve.identified_span
    for c in curve.cells:
        a, b = c.interval
        ax.scatter([(a + b) / 2], [c.mean], marker="o", facecolors=color,
                   edgecolors=color, zorder=3,
                   label=label if c is curve.cells[0] else None)
    p_in = np.linspace(lo, hi, 50)
    ax.plot(p_in, curve.at(p_in), "-", color=color)
    if lo > 0:
        p_lo = np.linspace(0, lo, 20)
        ax.plot(p_lo, curve.at(p_lo), "--", color=color)
    if hi < 1:
        p_hi = np.linspace(hi, 1, 20)
        ax.plot(p_hi, curve.at(p_hi), "--", color=color)


def plot_result(result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    g = result.groups
    if result.mto is not None:
        _draw_curve(ax, result.mto, "MTO (treated)", "C3")
    else:
        for c in g.mto_cells:
            a, b = c.interval
            ax.scatter([(a + b) / 2], [c.mean], marker="o", facecolors="C3",
                       edgecolors="C3", zorder=3, label="MTO (treated)")
    _draw_curve(ax, result.muo, "MUO (untreated)", "C0")
    for p, name in ((g.p_c, "$p_C$"), (g.p_i, "$p_I$")):
        if 0 < p < 1:
            ax.axvline(p, color="0.6", lw=0.8)
            ax.text(p, ax.get_ylim()[1], name, ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("p(U): percentile of net cost of treatment $U_D$")
    ax.set_ylabel(result.options.get("outcome_label", "outcome"))
    ax.legend(frameon=False)
    return ax
