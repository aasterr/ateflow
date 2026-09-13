"""E-value: how strong a confounder missing from the DAG would need to be."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import service
from ateflow.estimate import adjustment_formula, aipw, g_computation, ipw
from ateflow.sensitivity import evalue, evalue_of_ci


def test_published_example():
    # VanderWeele & Ding (2017): RR 3.9 with CI lower limit 1.8
    assert evalue(3.9) == pytest.approx(7.26, abs=0.01)
    assert evalue_of_ci(3.9, (1.8, 8.7)) == pytest.approx(3.0, abs=0.01)


def test_protective_effect_uses_the_inverse():
    assert evalue(0.5) == pytest.approx(2 + np.sqrt(2))
    assert evalue_of_ci(0.5, (0.3, 0.8)) == pytest.approx(evalue(1 / 0.8))


def test_ci_including_no_effect_gives_one():
    assert evalue(1.0) == 1.0
    assert evalue_of_ci(1.4, (0.9, 2.1)) == 1.0
    assert evalue_of_ci(0.7, (0.4, 1.2)) == 1.0


def synthetic(n=20000, seed=5):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    t = (rng.random(n) < 0.2 + 0.6 * z).astype(int)
    p = 0.2 + 0.3 * z + 0.15 * t  # E[Y | do(0)] = 0.2 + 0.3 * P(z=1) = 0.35
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"z": z, "t": t, "y": y})


@pytest.mark.parametrize("fn", [adjustment_formula, g_computation, ipw, aipw])
def test_estimators_report_the_mean_under_control(fn):
    est = fn(synthetic(), "t", "y", ["z"])
    assert est.mean_control == pytest.approx(0.35, abs=0.015)
    assert est.value == pytest.approx(0.15, abs=0.02)


def payload_for(name):
    spec = service.EXAMPLES[name]
    dag = (service.EXAMPLES_DIR / spec["dag"]).read_text()
    return service.estimate(service.example_bytes(name), dag, spec["treatment"], spec["outcome"], boot=200)


def test_binary_outcome_uses_the_risk_ratio():
    p = payload_for("onboarding")
    s = p["sensitivity"]
    assert s["approximate"] is False
    assert s["rr"] > 1
    assert s["rr_ci"][0] < s["rr"] < s["rr_ci"][1]
    assert s["evalue"] == pytest.approx(evalue(s["rr"]))
    assert 1 < s["evalue_ci"] < s["evalue"]


def test_numeric_outcome_is_approximate():
    p = payload_for("corridor")
    s = p["sensitivity"]
    assert s["approximate"] is True
    assert s["evalue"] > s["evalue_ci"] > 1
    check = p["answer"]["checks"][-1]
    assert check["title"] == "Hidden confounders (approximate)"
    assert check["text"].startswith(
        f"To explain this effect away, a confounder missing from the DAG would need a risk ratio of roughly "
        f"{s['evalue']:.1f} with both led and speed, beyond crowding.")
    assert "rule of thumb" in check["caveat"] and "not as an exact threshold" in check["caveat"]


def test_front_door_has_no_evalue():
    assert payload_for("ads")["sensitivity"] is None


def test_answer_check_wording():
    p = payload_for("onboarding")
    check = p["answer"]["checks"][-1]
    assert check["title"] == "Hidden confounders"
    assert check["ok"] is None
    assert "caveat" not in check  # binary outcome: the risk ratio is exact
    e, e_ci = p["sensitivity"]["evalue"], p["sensitivity"]["evalue_ci"]
    assert check["text"] == (
        "To explain this effect away, a confounder missing from the DAG would need to make both "
        f"onboarding_email and retained_30d about {e:.1f}× more likely, beyond channel and plan. "
        f"To make the 95% CI reach no effect, {e_ci:.1f}× would do."
    )


def test_answer_check_when_ci_includes_no_effect():
    spec = service.EXAMPLES["hrisim"]
    dag = (service.EXAMPLES_DIR / "hrisim_pipe.dag").read_text()
    p = service.estimate(service.example_bytes("hrisim"), dag, "A", "T", boot=200)
    check = p["answer"]["checks"][-1]
    assert p["sensitivity"]["evalue_ci"] == 1.0
    assert check["text"].startswith(
        f"To explain this effect away, a confounder missing from the DAG would need to make both A and T about "
        f"{p['sensitivity']['evalue']:.1f}× more likely, beyond O and Pi. The 95% CI already includes no effect")


def test_report_shows_the_caveat():
    from ateflow.report import render_report
    p = payload_for("corridor")
    html = render_report({"name": "c", "created_at": "2026-09-13T10:00:00", "source": "example: corridor",
                          "dag": (service.EXAMPLES_DIR / "corridor.dag").read_text(), "treatment": "led",
                          "outcome": "speed", "method": "auto", "result": p})
    assert "Hidden confounders (approximate)" in html and "rule of thumb" in html
