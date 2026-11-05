"""Simulation DGPs with exact known truth for recovery and size/power studies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _avg(coef, lo: float, hi: float) -> float:
    if hi <= lo:
        raise ValueError(f"interval average needs hi > lo, got ({lo}, {hi})")
    return float(sum(
        c * (hi ** (k + 1) - lo ** (k + 1)) / ((k + 1) * (hi - lo))
        for k, c in enumerate(coef)
    ))


@dataclass(frozen=True)
class Truth:
    muo_coef: tuple
    mto_coef: tuple
    p_c: float
    p_i: float
    late: float
    att: float
    atu: float
    ate: float
    selection_stat: float

    @staticmethod
    def avg(coef, lo: float, hi: float) -> float:
        return _avg(coef, lo, hi)


class RoyDGP:
    """Two-sided noncompliance Roy model with polynomial MUO/MTO in U ~ Uniform(0,1)."""

    def __init__(self, muo=(1.4, -0.8), mto=(2.0, -1.5), p_c=0.15, p_i=0.41, noise_sd=1.0):
        if not 0.0 <= p_c < p_i <= 1.0:
            raise ValueError("need 0 <= p_c < p_i <= 1")
        self.muo, self.mto = tuple(muo), tuple(mto)
        self.p_c, self.p_i, self.noise_sd = float(p_c), float(p_i), float(noise_sd)

    def truth(self) -> Truth:
        n = max(len(self.muo), len(self.mto))
        muo_padded = self.muo + (0.0,) * (n - len(self.muo))
        mto_padded = self.mto + (0.0,) * (n - len(self.mto))
        mte = tuple(t - u for t, u in zip(mto_padded, muo_padded))
        return Truth(
            muo_coef=self.muo, mto_coef=self.mto, p_c=self.p_c, p_i=self.p_i,
            late=_avg(mte, self.p_c, self.p_i),
            att=_avg(mte, 0.0, self.p_i),
            atu=_avg(mte, self.p_c, 1.0),
            ate=_avg(mte, 0.0, 1.0),
            selection_stat=_avg(self.muo, self.p_c, self.p_i) - _avg(self.muo, self.p_i, 1.0),
        )

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        u = rng.uniform(size=n)
        z = rng.integers(0, 2, size=n)
        p = np.where(z == 1, self.p_i, self.p_c)
        d = (u <= p).astype(int)
        y_t = np.polynomial.polynomial.polyval(u, np.asarray(self.mto))
        y_u = np.polynomial.polynomial.polyval(u, np.asarray(self.muo))
        y = np.where(d == 1, y_t, y_u) + rng.normal(0.0, self.noise_sd, size=n)
        return pd.DataFrame({"y": y, "d": d, "z": z})


class OneSidedDGP:
    """One-sided-noncompliance RCT with covariate strata (the BMW case).

    Units are treated only in the encouraged arm (z=1) and only when their
    resistance U falls below a stratum-specific take-up threshold, so there are
    no always-takers (p_c = 0). U is resistance to take-up; MUO/MTO are
    polynomial means of a positive, right-skewed outcome (gamma noise -- e.g. a
    cost, count, or duration). X (independent of U by default) drives a
    predicted take-up score whose quantile strata shift the threshold -- the
    clean Brinch-Mogstad-Wiswall covariate-variation case that supplies extra
    propensity points. score_on_u > 0 correlates the score with U, which breaks
    within-stratum uniformity of U (and hence cross-strata pooling); truth()
    then raises rather than reporting a wrong truth.
    """

    def __init__(self, muo=(9000.0, -4000.0), mto=(8500.0, -4000.0), n_strata=5,
                 take_up=(0.15, 0.55), score_on_u=0.0, shape=2.0):
        for name, coef in ("muo", muo), ("mto", mto):
            grid = np.polynomial.polynomial.polyval(np.linspace(0, 1, 101), np.asarray(coef))
            if (grid <= 0).any():
                raise ValueError(f"{name} mean must be positive on [0,1]; got min {grid.min():.1f}")
        self.muo, self.mto = tuple(muo), tuple(mto)
        self.n_strata, self.take_up = int(n_strata), tuple(take_up)
        self.score_on_u, self.shape = float(score_on_u), float(shape)

    def _thresholds(self) -> np.ndarray:
        lo, hi = self.take_up
        return np.linspace(lo, hi, self.n_strata)

    def truth(self) -> Truth:
        if self.score_on_u != 0.0:
            raise ValueError(
                "truth undefined: score_on_u breaks within-stratum uniformity of U"
            )
        thr = self._thresholds()
        p_i = float(thr.mean())
        mte = tuple(t - u for t, u in zip(self.mto, self.muo))
        # The pooled complier / never-taker populations are equal-weight mixtures over
        # strata (each stratum has fraction t_s compliers, U ~ Uniform within it). The
        # correct averages are these mixtures, NOT _avg(coef, 0, mean_t): those coincide
        # only when the integrand is constant. mix_c/mix_n match what the pooled estimator
        # recovers (E[MTO|treated], backed-out complier means) exactly, for any coef.
        def mix_c(coef):  # E[coef(U) | complier] = sum_s int_0^{t_s} coef / sum_s t_s
            return sum(_avg(coef, 0.0, t) * t for t in thr) / thr.sum()

        def mix_n(coef):  # E[coef(U) | never taker] = sum_s int_{t_s}^1 coef / sum_s (1-t_s)
            return sum(_avg(coef, t, 1.0) * (1.0 - t) for t in thr) / (1.0 - thr).sum()

        late = mix_c(mte)  # one-sided: treated == compliers, so ATT == LATE
        # ATU/ATE are Kowalski's UNIFORM-WEIGHT MTE integrals over the U_D intervals
        # (atu over (p_c,1], ate over (0,1]), matching compute_estimands / Curve.average.
        # These are the MTE-reweighting parameters, NOT the classical conditional effects:
        # e.g. the uniform-weight ATU over (0,1] here is _avg(mte,0,1), which differs from
        # the population E[MTE(U)|D=0] (never-takers are not uniform in U). We report the
        # estimand the estimator targets, so recovery tests stay meaningful.
        return Truth(
            muo_coef=self.muo, mto_coef=self.mto, p_c=0.0, p_i=p_i,
            late=late, att=late,
            atu=_avg(mte, 0.0, 1.0), ate=_avg(mte, 0.0, 1.0),
            selection_stat=mix_c(self.muo) - mix_n(self.muo),
        )

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        u = rng.uniform(size=n)
        x = rng.uniform(size=n)
        score = np.clip(x + self.score_on_u * (0.5 - u), 0.0, 1.0)
        stratum = np.minimum(
            (np.argsort(np.argsort(score)) / n * self.n_strata).astype(int),
            self.n_strata - 1,
        )
        z = rng.integers(0, 2, size=n)
        d = ((z == 1) & (u <= self._thresholds()[stratum])).astype(int)
        mean = np.where(
            d == 1,
            np.polynomial.polynomial.polyval(u, np.asarray(self.mto)),
            np.polynomial.polynomial.polyval(u, np.asarray(self.muo)),
        )
        y = rng.gamma(self.shape, mean / self.shape)
        return pd.DataFrame({"y": y, "d": d, "z": z, "stratum": stratum, "score": score})
