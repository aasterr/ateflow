"""Auto estimator choice, the cross-check between estimators, and the answer in words."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import DAG, estimate_ate
from ateflow.data import read_csv
from examples.make_data import make_onboarding

ONBOARDING = DAG.from_file(ROOT / "examples/onboarding.dag")
CONTINUOUS_DAG = "age -> t\nage -> y\nt -> y"


def example(name):
    df, _ = read_csv((ROOT / "examples" / name).read_bytes())
    return df


@pytest.fixture(scope="module")
def onboarding():
    return make_onboarding()


def continuous_confounder(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(age - 40) / 8))).astype(int)
    y = 2.0 * t + 0.1 * age + rng.normal(0, 1, n)
    return pd.DataFrame({"age": age, "t": t, "y": y})


# ---------- auto method ----------

def test_auto_uses_adjustment_formula_with_discrete_confounders(onboarding):
    res = estimate_ate(onboarding, ONBOARDING, "onboarding_email", "retained_30d",
                       n_boot=0, refute=False)
    assert res.method == "adjustment-formula"


def test_auto_uses_aipw_with_a_continuous_confounder():
    res = estimate_ate(continuous_confounder(), CONTINUOUS_DAG, "t", "y", n_boot=0, refute=False)
    assert res.method == "aipw"
    assert abs(res.adjusted.value - 2.0) < 0.15


def test_auto_front_door_uses_the_formula():
    res = estimate_ate(example("ads.csv"), DAG.from_file(ROOT / "examples/ads.dag"),
                       "saw_ad", "purchased", n_boot=0, refute=False)
    assert res.strategy == "frontdoor"
    assert res.method == "adjustment-formula"


def test_explicit_method_is_kept(onboarding):
    res = estimate_ate(onboarding, ONBOARDING, "onboarding_email", "retained_30d",
                       method="g-computation", n_boot=0, refute=False)
    assert res.method == "g-computation"


# ---------- comparison ----------

def test_comparison_lists_every_estimator_of_the_strategy(onboarding):
    res = estimate_ate(onboarding, ONBOARDING, "onboarding_email", "retained_30d",
                       n_boot=200, refute=False)
    assert [c["method"] for c in res.comparison] == ["adjustment-formula", "g-computation", "ipw", "aipw"]
    assert [c["primary"] for c in res.comparison] == [True, False, False, False]
    assert all(c["applicable"] for c in res.comparison)
    assert res.methods_agree


def test_formula_not_applicable_with_continuous_confounder():
    res = estimate_ate(continuous_confounder(), CONTINUOUS_DAG, "t", "y", n_boot=0, refute=False)
    formula = next(c for c in res.comparison if c["method"] == "adjustment-formula")
    assert not formula["applicable"]
    assert "discrete" in formula["reason"]
    assert formula["value"] is None


def test_front_door_compares_two_estimators():
    res = estimate_ate(example("ads.csv"), DAG.from_file(ROOT / "examples/ads.dag"),
                       "saw_ad", "purchased", n_boot=0, refute=False)
    assert [c["method"] for c in res.comparison] == ["adjustment-formula", "g-computation"]


def test_hrisim_methods_disagree():
    res = estimate_ate(example("episodes_100_v1.csv"), DAG.from_file(ROOT / "examples/hrisim_pipe.dag"),
                       "A", "T", n_boot=200, refute=False)
    assert res.method == "adjustment-formula"
    assert res.adjusted.value == pytest.approx(0.108, abs=5e-4)
    assert not res.methods_agree


def test_outcome_binary_flag(onboarding):
    res = estimate_ate(onboarding, ONBOARDING, "onboarding_email", "retained_30d",
                       n_boot=0, refute=False)
    assert res.data_report["outcome_binary"] is True
    res2 = estimate_ate(continuous_confounder(), CONTINUOUS_DAG, "t", "y", n_boot=0, refute=False)
    assert res2.data_report["outcome_binary"] is False
