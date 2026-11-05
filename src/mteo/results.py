"""fit() orchestration and the MTEOResult container."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mteo import curves as _curves
from mteo import estimands as _estimands
from mteo import tests as _tests
from mteo.exceptions import IdentificationError
from mteo.groups import GroupSummary, StrataSummary, summarize_groups, summarize_strata
from mteo.inference import BootstrapResult, bootstrap


@dataclass
class TestStat:
    stat: float
    se: float | None = None
    ci: tuple[float, float] | None = None


@dataclass
class TestResults:
    selection: TestStat
    treated_selection: TestStat | None = None
    bmw: TestStat | None = None
    mte_slope: TestStat | None = None
    overid: TestStat | None = None
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class MTEOResult:
    groups: GroupSummary
    strata: StrataSummary | None
    muo: _curves.Curve | None
    mto: _curves.Curve | None
    tests: TestResults
    estimands: dict[str, _estimands.Estimand]
    boot: BootstrapResult | None
    options: dict

    def predict_late(self, p_lo: float, p_hi: float) -> _estimands.Estimand:
        if self.mto is None or self.muo is None:
            raise IdentificationError(
                "predict_late needs fitted MTO and MUO curves; "
                "one-sided designs require strata to fit MTO"
            )
        return _estimands.predict_late(self.mto, self.muo, p_lo, p_hi)

    def plot(self, ax=None):
        from mteo.plots import plot_result
        return plot_result(self, ax=ax)

    def summary(self) -> str:
        g = self.groups
        lines = [
            "mteo fit summary",
            "================",
            f"n = {g.n}   take-up: p_C = {g.p_c:.3f}, p_I = {g.p_i:.3f} "
            f"(gap {g.p_i - g.p_c:.3f}){'   [ONE-SIDED]' if g.one_sided else ''}",
            f"shares: always {g.shares['at']:.3f} | compliers {g.shares['c']:.3f} "
            f"| never {g.shares['nt']:.3f}",
            "",
        ]

        def se_of(key):
            if self.boot is None or key not in self.boot.se:
                return ""
            return f" (SE {self.boot.se[key]:.4g})"

        lines.append(f"LATE = {g.late:.4g}{se_of('late')}")
        lines.append(f"selection test (untreated, complier - never) = "
                     f"{self.tests.selection.stat:.4g}{se_of('test_selection')}")
        for name, key, ts in [("treated-outcome test", "test_treated_selection", self.tests.treated_selection),
                              ("BMW Delta1-Delta0", "test_bmw", self.tests.bmw),
                              ("MTE slope", "test_mte_slope", self.tests.mte_slope),
                              ("overid", "test_overid", self.tests.overid)]:
            if ts is not None:
                lines.append(f"{name} = {ts.stat:.4g}{se_of(key)}")
        for r in self.tests.reasons.values():
            lines.append(f"note: {r}")
        lines.append("")
        for k in ("late", "att", "atu", "ate"):
            e = self.estimands[k]
            if e.status in ("point", "extrapolated"):
                lines.append(f"{k.upper():4s} = {e.value:.4g}{se_of('est_' + k)}  [{e.status}]")
            elif e.status == "bounds":
                lines.append(f"{k.upper():4s} in [{e.bounds[0]:.4g}, {e.bounds[1]:.4g}]  [bounds]")
            else:
                lines.append(f"{k.upper():4s}: not identified ({e.reason})")
        if g.p_i - g.p_c < 0.05:
            lines.append("WARNING: thin take-up gap; extrapolation leans hard on functional form")
        return "\n".join(lines)


def _estimate_once(df, outcome, treatment, instrument, strata, extrapolation,
                   poly_order, weights, monotone_direction, outcome_bounds,
                   min_take_up_gap):
    y = df[outcome].to_numpy()
    d = df[treatment].to_numpy()
    z = df[instrument].to_numpy()
    w = df[weights].to_numpy() if weights is not None else None

    st = None
    if strata is not None:
        st = summarize_strata(y, d, z, df[strata].to_numpy(), w, min_take_up_gap=min_take_up_gap)
        g = st.pooled
        mto_cells, muo_cells = list(st.mto_cells), list(st.muo_cells)
    else:
        g = summarize_groups(y, d, z, w, min_take_up_gap=min_take_up_gap)
        mto_cells, muo_cells = list(g.mto_cells), list(g.muo_cells)

    def try_fit(cells):
        if len(cells) < 2:
            return None  # single identified point: no MTO/MUO curve (slope) is identified
        order = poly_order if poly_order is not None else min(1, len(cells) - 1)
        return _curves.fit_curve(cells, order)

    mto = try_fit(mto_cells)
    muo = try_fit(muo_cells)

    if extrapolation == "linear" and (mto is None or len(mto.coef) < 2):
        raise IdentificationError(
            "linear MTO not identified from a single propensity point; provide strata "
            "or set extrapolation='none'/'bounds'"
        )

    tr = TestResults(selection=TestStat(_tests.selection_stat(g)))
    if g.one_sided:
        tr.reasons["treated_selection"] = "needs always takers (one-sided design)"
        tr.reasons["bmw"] = "needs treated controls (one-sided design)"
    else:
        tr.treated_selection = TestStat(_tests.treated_selection_stat(g))
        tr.bmw = TestStat(_tests.bmw_stat(y, d, z, w))
    if mto is not None and muo is not None and len(mto.coef) >= 2 and len(muo.coef) >= 2:
        tr.mte_slope = TestStat(_tests.mte_slope_stat(mto, muo))
        if len(mto.cells) + len(muo.cells) > len(mto.coef) + len(muo.coef):
            tr.overid = TestStat(_tests.overid_stat(mto) + _tests.overid_stat(muo))

    est = _estimands.compute_estimands(
        g, mto, muo, extrapolation,
        monotone_direction=monotone_direction, outcome_bounds=outcome_bounds,
    )
    return g, st, mto, muo, tr, est


def _scalars(g, tr, est) -> dict[str, float]:
    out = {"late": g.late, "p_c": g.p_c, "p_i": g.p_i,
           "test_selection": tr.selection.stat}
    if tr.treated_selection is not None:
        out["test_treated_selection"] = tr.treated_selection.stat
    if tr.bmw is not None:
        out["test_bmw"] = tr.bmw.stat
    if tr.mte_slope is not None:
        out["test_mte_slope"] = tr.mte_slope.stat
    if tr.overid is not None:
        out["test_overid"] = tr.overid.stat
    for k, e in est.items():
        if e.status in ("point", "extrapolated"):
            out["est_" + k] = e._value
    return out


def fit(data: pd.DataFrame, outcome: str, treatment: str, instrument: str,
        strata: str | None = None, extrapolation: str = "none",
        poly_order: int | None = None, bootstrap_reps: int = 1000,
        cluster: str | None = None, weights: str | None = None,
        seed: int | None = None, monotone_direction: dict | None = None,
        outcome_bounds: tuple = (-np.inf, np.inf),
        min_take_up_gap: float = 0.02) -> MTEOResult:
    if extrapolation not in {"none", "linear", "bounds"}:
        raise ValueError("extrapolation must be 'none', 'linear', or 'bounds'")

    args = (outcome, treatment, instrument, strata, extrapolation, poly_order,
            weights, monotone_direction, outcome_bounds, min_take_up_gap)
    g, st, mto, muo, tr, est = _estimate_once(data, *args)

    boot = None
    if bootstrap_reps > 0:
        def scalar_fn(df_s):
            g2, _, _, _, tr2, est2 = _estimate_once(df_s, *args)
            return _scalars(g2, tr2, est2)
        boot = bootstrap(data, scalar_fn, n_boot=bootstrap_reps, cluster=cluster, seed=seed)
        for key, ts in [("test_selection", tr.selection),
                        ("test_treated_selection", tr.treated_selection),
                        ("test_bmw", tr.bmw), ("test_mte_slope", tr.mte_slope),
                        ("test_overid", tr.overid)]:
            if ts is not None and key in boot.se:
                ts.se, ts.ci = boot.se[key], boot.ci[key]

    return MTEOResult(groups=g, strata=st, muo=muo, mto=mto, tests=tr,
                      estimands=est, boot=boot,
                      options={"extrapolation": extrapolation, "strata": strata,
                               "poly_order": poly_order, "seed": seed})
