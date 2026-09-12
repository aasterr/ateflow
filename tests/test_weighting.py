"""IPW and AIPW: agreement with the other estimators, overlap diagnostics, double robustness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import DAG, estimate_ate
from ateflow.estimate import aipw, g_computation, ipw
from examples.make_data import ONBOARDING_TRUE_ATE, TRUE_ATE, make, make_onboarding


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_recovers_corridor_ate(method):
    res = estimate_ate(make(), DAG.from_file(ROOT / "examples/corridor.dag"), "led", "speed",
                       method=method, n_boot=0, refute=False)
    assert abs(res.adjusted.value - TRUE_ATE) < 0.02
    assert res.adjusted.diagnostics["clipped"] == 0


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_recovers_onboarding_ate(method):
    res = estimate_ate(make_onboarding(), DAG.from_file(ROOT / "examples/onboarding.dag"),
                       "onboarding_email", "retained_30d", method=method, n_boot=0, refute=False)
    assert abs(res.adjusted.value - ONBOARDING_TRUE_ATE) < 0.02


@pytest.fixture(scope="module")
def hrisim():
    return pd.read_csv(ROOT / "examples/episodes_100_v1.csv")


@pytest.mark.parametrize("method", ["ipw", "aipw"])
def test_saturated_propensity_matches_thesis_on_O(hrisim, method):
    # With one binary confounder every estimator reduces to the same adjustment formula.
    res = estimate_ate(hrisim, DAG.from_file(ROOT / "examples/hrisim.dag"), "A", "T",
                       method=method, n_boot=0, refute=False)
    assert res.adjusted.value == pytest.approx(0.061, abs=5e-4)


def test_positivity_violation_is_reported_not_smoothed(hrisim):
    # The cell Pi=0, O=0 has no treated episode: an additive propensity model
    # would give it a plausible-looking probability; the saturated one flags all 36 rows.
    res = estimate_ate(hrisim, DAG.from_file(ROOT / "examples/hrisim_pipe.dag"), "A", "T",
                       method="ipw", n_boot=0, refute=False)
    assert res.adjusted.diagnostics["clipped"] == 36


def test_aipw_survives_a_misspecified_outcome_model():
    # Outcome is exp(x), the linear outcome model cannot follow it and
    # g-computation is biased; the propensity model is correct, so AIPW is not.
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-1.5 * x)))
    y = 1.0 * t + np.exp(x) + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "t": t, "y": y})

    assert abs(g_computation(df, "t", "y", ["x"]).value - 1.0) > 0.1
    assert abs(aipw(df, "t", "y", ["x"]).value - 1.0) < 0.06
    assert abs(ipw(df, "t", "y", ["x"]).value - 1.0) < 0.1
