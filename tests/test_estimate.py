"""Il test di regressione del progetto: recupero dell'ATE noto."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ateflow import DAG, estimate_ate
from examples.make_data import TRUE_ATE, make

DAG_TEXT = Path(__file__).resolve().parents[1].joinpath("examples/corridor.dag").read_text()


@pytest.fixture(scope="module")
def data():
    return make()


@pytest.mark.parametrize("method", ["g-computation", "stratification"])
def test_recovers_true_ate(data, method):
    res = estimate_ate(data, DAG_TEXT, "led", "speed", method=method, n_boot=0, refute=False)
    assert res.adjustment_set == ["crowding"]
    assert abs(res.adjusted.value - TRUE_ATE) < 0.02


def test_naive_estimate_is_biased_and_flips_sign(data):
    res = estimate_ate(data, DAG_TEXT, "led", "speed", n_boot=0, refute=False)
    assert res.naive.value < 0 < res.adjusted.value
    assert res.sign_flip


def test_bootstrap_interval_covers_truth(data):
    res = estimate_ate(data, DAG_TEXT, "led", "speed", n_boot=200, refute=False)
    lo, hi = res.adjusted.ci
    assert lo < TRUE_ATE < hi


def test_adjusting_for_descendant_is_refused(data):
    with pytest.raises(ValueError, match="backdoor"):
        estimate_ate(data, DAG_TEXT, "led", "speed",
                     adjustment_set=["crowding", "waiting_time"], n_boot=0, refute=False)


def test_missing_column_is_reported(data):
    with pytest.raises(ValueError, match="assenti"):
        estimate_ate(data.drop(columns=["crowding"]), DAG_TEXT, "led", "speed",
                     n_boot=0, refute=False)


def test_refutations_pass_on_correct_model(data):
    res = estimate_ate(data, DAG_TEXT, "led", "speed", n_boot=0, refute=True)
    assert all(r["passed"] for r in res.refutations)
