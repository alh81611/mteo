import numpy as np
import pandas as pd
import pytest

from mteo.inference import bootstrap


def test_bootstrap_se_of_mean_matches_analytic():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"y": rng.normal(0, 2.0, size=4000)})
    res = bootstrap(df, lambda s: {"mean": float(s["y"].mean())}, n_boot=500, seed=7)
    assert res.se["mean"] == pytest.approx(2.0 / np.sqrt(4000), rel=0.15)
    lo, hi = res.ci["mean"]
    assert lo < res.estimates["mean"] < hi


def test_seed_determinism():
    df = pd.DataFrame({"y": np.arange(100.0)})
    f = lambda s: {"m": float(s["y"].mean())}
    a = bootstrap(df, f, n_boot=50, seed=11)
    b = bootstrap(df, f, n_boot=50, seed=11)
    assert a.se["m"] == b.se["m"] and a.ci["m"] == b.ci["m"]


def test_cluster_bootstrap_resamples_clusters():
    # Outcome constant within cluster: i.i.d. bootstrap understates SE, cluster bootstrap doesn't.
    rng = np.random.default_rng(5)
    k = 40
    cl = np.repeat(np.arange(k), 50)
    yc = rng.normal(0, 1, size=k)
    df = pd.DataFrame({"y": yc[cl], "g": cl})
    iid = bootstrap(df, lambda s: {"m": float(s["y"].mean())}, n_boot=300, seed=1)
    clu = bootstrap(df, lambda s: {"m": float(s["y"].mean())}, n_boot=300, cluster="g", seed=1)
    assert clu.se["m"] > 3 * iid.se["m"]


def test_failed_replications_warn():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    def flaky(s):
        from mteo.exceptions import IdentificationError
        if s["y"].iloc[0] != 1.0:
            raise IdentificationError("boom")
        return {"m": float(s["y"].mean())}

    with pytest.warns(UserWarning, match="replication"):
        res = bootstrap(df, flaky, n_boot=200, seed=3)
    assert res.n_failed > 0
