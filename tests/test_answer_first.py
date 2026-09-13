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


# ---------- the answer in words ----------

from ateflow import answer, service  # noqa: E402


def payload_for(name, dag_extra="", **kw):
    spec = service.EXAMPLES[name]
    dag = (service.EXAMPLES_DIR / spec["dag"]).read_text() + dag_extra
    p = service.estimate(service.example_bytes(name), dag, spec["treatment"], spec["outcome"],
                         boot=200, **kw)
    return p, spec["treatment"], spec["outcome"]


@pytest.fixture(scope="module")
def onboarding_answer():
    p, t, o = payload_for("onboarding")
    return answer.build(p, t, o)


def test_headline_binary_outcome(onboarding_answer):
    h = onboarding_answer["headline"]
    assert h.startswith("Setting onboarding_email to 1 for everyone, instead of 0, would raise "
                        "the share with retained_30d = 1 by 8.6 percentage points (95% CI +6.")
    assert "cannot tell" not in h


def test_naive_sentence_names_the_confounders_and_the_flip(onboarding_answer):
    n = onboarding_answer["naive"]
    assert n.startswith("Comparing the two groups as they are gives −6.9 points instead")
    assert "channel and plan pull it down by 15.5 points" in n
    assert "direction wrong" in n


def test_technical_line(onboarding_answer):
    assert onboarding_answer["technical"] == "ATE · adjusting for channel, plan · adjustment formula · 8000 rows"


def test_checks_for_a_clean_case(onboarding_answer):
    checks = onboarding_answer["checks"]
    assert [c["title"] for c in checks] == ["Methods agree", "Overlap", "Placebo test", "Random common cause"]
    assert all(c["ok"] for c in checks)


def test_hrisim_answer_is_about_the_comparable_rows_and_flags_trouble():
    p, t, o = payload_for("hrisim", "\nPi -> Pe")
    a = answer.build(p, t, o)
    assert a["headline"].startswith("Among the 64 of 100 rows where both groups occur, setting A to 1")
    assert "cannot tell whether the effect is positive or negative" in a["headline"]
    by_title = {c["title"]: c for c in a["checks"]}
    assert not by_title["Methods disagree"]["ok"]
    assert not by_title["Overlap"]["ok"]
    assert "36 of 100 rows" in by_title["Overlap"]["text"]


def test_front_door_answer():
    p, t, o = payload_for("ads")
    a = answer.build(p, t, o)
    assert a["technical"].startswith("ATE · through visited_site · adjustment formula")
    assert "an unmeasured confounder pushes it up by 26.9 points" in a["naive"]
    assert [c["title"] for c in a["checks"]] == ["Methods agree", "Overlap", "Placebo test"]


def test_numeric_outcome_uses_the_average():
    p, t, o = payload_for("corridor")
    a = answer.build(p, t, o)
    assert a["headline"].startswith("Setting led to 1 for everyone, instead of 0, would raise "
                                    "the average speed by 0.151 (95% CI +0.14")


def test_labelled_treatment_and_outcome():
    rng = np.random.default_rng(3)
    n = 2000
    z = rng.integers(0, 2, n)
    t = rng.random(n) < 0.3 + 0.4 * z
    y = rng.random(n) < 0.2 + 0.2 * t + 0.3 * z
    df = pd.DataFrame({"z": z, "promo": np.where(t, "yes", "no"), "bought": np.where(y, "bought", "left")})
    p = service.estimate(df.to_csv(index=False).encode(), "z -> promo\nz -> bought\npromo -> bought",
                         "promo", "bought", boot=0, outcome_positive="bought")
    a = answer.build(p, "promo", "bought")
    assert a["headline"].startswith('Setting promo to "yes" for everyone, instead of "no", would raise '
                                    'the share with bought = "bought" by ')


def test_no_confounding_sentence():
    rng = np.random.default_rng(4)
    t = rng.integers(0, 2, 1000)
    df = pd.DataFrame({"t": t, "y": t * 1.0 + rng.normal(0, 1, 1000)})
    p = service.estimate(df.to_csv(index=False).encode(), "t -> y", "t", "y", boot=0)
    a = answer.build(p, "t", "y")
    assert a["naive"] == "Nothing confounds this comparison: the two groups as they are already give the answer."
    assert "no adjustment needed" in a["technical"]


def test_old_saved_result_without_new_keys(onboarding_answer):
    p, t, o = payload_for("onboarding")
    for key in ("comparison", "methods_agree", "method"):
        p.pop(key)
    p["data_report"].pop("outcome_binary")
    a = answer.build(p, t, o)
    assert "Methods agree" not in [c["title"] for c in a["checks"]]
    assert a["headline"].startswith("Setting onboarding_email to 1")


def test_answer_is_in_the_payload_and_the_service_op():
    import json
    p, t, o = payload_for("ads")
    assert p["answer"] == answer.build(p, t, o)
    out = json.loads(service.handle("answer", json.dumps({"result": p, "treatment": t, "outcome": o})))
    assert out["ok"]["headline"] == p["answer"]["headline"]


def test_numbers_keep_three_significant_digits():
    p, t, o = payload_for("corridor")
    a = answer.build(p, t, o)
    assert "crowding pulls it down by 0.281" in a["naive"]
    assert "g-computation +0.150" in a["checks"][0]["text"]


def test_report_opens_with_the_answer():
    from ateflow.report import render_report
    p, t, o = payload_for("ads")
    analysis = {"name": "ads", "created_at": "2026-09-13T10:00:00", "source": "example: ads",
                "dag": (service.EXAMPLES_DIR / "ads.dag").read_text(), "treatment": t, "outcome": o,
                "method": "auto", "result": p}
    html = render_report(analysis)
    assert html.index("<h2>Answer</h2>") < html.index("<h2>Question</h2>")
    assert "would raise the share with purchased = 1 by 15.3 percentage points" in html
    assert "method: adjustment formula" in html
