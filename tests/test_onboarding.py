"""Product-analytics example: onboarding email -> 30-day retention.

Ground truth +0.09, set by the generator. Two traps are pinned here: the
targeted email makes the naive comparison negative, and adjusting for the
first-week activity mediator would hide half of the effect.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import DAG, estimate_ate
from ateflow.estimate import g_computation
from examples.make_data import ONBOARDING_DIRECT, ONBOARDING_TRUE_ATE, make_onboarding

DAG_ = DAG.from_file(ROOT / "examples/onboarding.dag")
T, Y = "onboarding_email", "retained_30d"


@pytest.fixture(scope="module")
def data():
    return make_onboarding()


@pytest.mark.parametrize("method", ["g-computation", "stratification"])
def test_recovers_true_ate(data, method):
    res = estimate_ate(data, DAG_, T, Y, method=method, n_boot=0, refute=False)
    assert res.adjustment_set == ["channel", "plan"]
    assert abs(res.adjusted.value - ONBOARDING_TRUE_ATE) < 0.02


def test_targeting_makes_naive_look_harmful(data):
    res = estimate_ate(data, DAG_, T, Y, n_boot=0, refute=False)
    assert res.naive.value < 0 < res.adjusted.value
    assert res.sign_flip


def test_mediator_is_refused(data):
    with pytest.raises(ValueError, match="backdoor"):
        estimate_ate(data, DAG_, T, Y, adjustment_set=["channel", "plan", "active_week1"],
                     n_boot=0, refute=False)


def test_adjusting_for_mediator_would_leave_only_direct_effect(data):
    # Why the refusal matters: the forbidden set recovers the direct effect, not the total.
    wrong = g_computation(data, T, Y, ["channel", "plan", "active_week1"])
    assert abs(wrong.value - ONBOARDING_DIRECT) < 0.02
    assert wrong.value < ONBOARDING_TRUE_ATE - 0.03


def test_refutations_pass(data):
    res = estimate_ate(data, DAG_, T, Y, n_boot=0, refute=True)
    assert all(r["passed"] for r in res.refutations)
