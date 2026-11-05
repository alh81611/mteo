"""Nonparametric bootstrap of the full estimation pipeline (Kowalski's practice).

The estimate_fn re-runs everything -- take-up rates, back-outs, curve fits,
tests, estimands -- on each resample, so first-stage estimation error is
reflected in every SE (unlike analytic SEs that condition on the propensity;
cf. the mtefe help-file caveat).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mteo.exceptions import IdentificationError


@dataclass(frozen=True)
class BootstrapResult:
    estimates: dict[str, float]
    se: dict[str, float]
    ci: dict[str, tuple[float, float]]
    n_boot: int
    n_failed: int


def bootstrap(
    df: pd.DataFrame,
    estimate_fn: Callable[[pd.DataFrame], dict[str, float]],
    n_boot: int = 1000,
    cluster: str | None = None,
    seed: int | None = None,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    base = estimate_fn(df)
    keys = list(base)
    draws = np.full((n_boot, len(keys)), np.nan)
    if cluster is not None:
        groups = {k: v for k, v in df.groupby(cluster)}
        labels = list(groups)
    n_failed = 0
    for b in range(n_boot):
        if cluster is None:
            sample = df.sample(frac=1.0, replace=True, random_state=rng.integers(2**31))
        else:
            picked = rng.choice(len(labels), size=len(labels), replace=True)
            sample = pd.concat([groups[labels[i]] for i in picked], ignore_index=True)
        try:
            rep = estimate_fn(sample)
        except IdentificationError:
            n_failed += 1
            continue
        draws[b] = [rep.get(k, np.nan) for k in keys]
    if n_failed > 0.01 * n_boot:
        warnings.warn(
            f"{n_failed}/{n_boot} bootstrap replication(s) failed identification and were "
            "recorded as NaN; interpret intervals with care",
            UserWarning,
            stacklevel=2,
        )
    se = {k: float(np.nanstd(draws[:, i], ddof=1)) for i, k in enumerate(keys)}
    ci = {
        k: (float(np.nanpercentile(draws[:, i], 2.5)), float(np.nanpercentile(draws[:, i], 97.5)))
        for i, k in enumerate(keys)
    }
    return BootstrapResult(estimates=base, se=se, ci=ci, n_boot=n_boot, n_failed=n_failed)
