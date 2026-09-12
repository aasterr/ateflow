"""Regressione sui dati reali: i numeri della tesi (HRISim, 100 episodi).

Riferimenti: F. Baldo, *Causal Effect Estimation of Robot Actions for Human
Aware Navigation*, cap. 5, e il notebook `analysis/hrisim_causal_analysis.ipynb`
del repo PeopleFlow. Il DAG assunto e' Pi -> A -> Pe -> S -> T con il
confonditore O -> A e O -> S; la variante aggiunge Pi -> Pe.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import DAG, estimate_ate

BASE = DAG.from_file(ROOT / "examples/hrisim.dag")
VARIANT = DAG.from_file(ROOT / "examples/hrisim_pipe.dag")


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(ROOT / "examples/episodes_100_v1.csv")


def test_marginals_match_thesis(data):
    expected = {"Pi": 0.48, "A": 0.38, "Pe": 0.25, "S": 0.73, "T": 0.76, "O": 0.29}
    for col, value in expected.items():
        assert abs(data[col].mean() - value) < 5e-3


def test_minimal_backdoor_sets():
    assert BASE.minimal_backdoor_set("A", "T") == {"O"}
    assert VARIANT.minimal_backdoor_set("A", "T") == {"O", "Pi"}


def test_naive_estimate_is_confounded_and_sign_inverted(data):
    res = estimate_ate(data, BASE, "A", "T", method="stratification",
                       n_boot=0, refute=False)
    assert res.naive.value == pytest.approx(-0.207, abs=5e-4)
    assert res.sign_flip


@pytest.mark.parametrize("method", ["stratification", "g-computation"])
def test_backdoor_on_O_recovers_primary_estimate(data, method):
    res = estimate_ate(data, BASE, "A", "T", method=method, n_boot=0, refute=False)
    assert res.adjustment_set == ["O"]
    assert res.adjusted.value == pytest.approx(0.061, abs=5e-4)


def test_backdoor_on_Pi_O_uses_overlap_cells_only(data):
    res = estimate_ate(data, VARIANT, "A", "T", method="stratification",
                       n_boot=0, refute=False)
    assert res.adjustment_set == ["O", "Pi"]
    assert res.adjusted.value == pytest.approx(0.108, abs=5e-4)
    # Lo strato Pi=0, O=0 non contiene episodi con A=1: la policy non segnala
    # mai a corridoio libero. I suoi 36 episodi vanno scartati, non riempiti.
    assert res.adjusted.diagnostics["dropped_strata"] == 36


def test_refutations_run_with_stratification(data):
    # Regressione sul covariato casuale binario: con uno continuo la
    # stratificazione degenera in strati singoli e la refutation esplode.
    res = estimate_ate(data, BASE, "A", "T", method="stratification",
                       n_boot=0, refute=True)
    assert all(r["passed"] for r in res.refutations)


def test_adjusting_for_mediator_is_refused(data):
    with pytest.raises(ValueError, match="backdoor"):
        estimate_ate(data, BASE, "A", "T", adjustment_set=["O", "Pe"],
                     n_boot=0, refute=False)
