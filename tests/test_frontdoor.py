"""Unmeasured variables, front-door identification and estimation, search that scales."""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import DAG, estimate_ate
from examples.make_data import ADS_TRUE_ATE, make_ads

ADS_DAG = (ROOT / "examples/ads.dag").read_text(encoding="utf-8")


# ---------- graph ----------


def test_unmeasured_line_is_parsed_and_round_trips():
    dag = DAG.parse("u -> t\nu -> y\nt -> m -> y\nunmeasured: u, other thing")
    assert dag.unmeasured == {"u", "other thing"}
    assert dag.observed == {"t", "m", "y"}
    again = DAG.parse(dag.to_text())
    assert (set(again.edges), again.unmeasured) == (set(dag.edges), dag.unmeasured)


def test_unmeasured_confounder_cannot_be_adjusted_for():
    dag = DAG.parse("u -> t\nu -> y\nt -> y\nunmeasured: u")
    assert not dag.satisfies_backdoor("t", "y", ["u"])
    assert dag.identify("t", "y")["strategy"] is None


def test_classic_frontdoor():
    dag = DAG.parse(ADS_DAG)
    ident = dag.identify("saw_ad", "purchased")
    assert (ident["strategy"], ident["variables"]) == ("frontdoor", ["visited_site"])
    assert any("unmeasured" in line for line in ident["explanation"])


@pytest.mark.parametrize("extra, reason", [
    ("intent -> visited_site", "confounded mediator"),
    ("saw_ad -> purchased", "direct effect bypasses the mediator"),
])
def test_frontdoor_conditions_are_enforced(extra, reason):
    dag = DAG.parse(ADS_DAG + "\n" + extra)
    assert not dag.satisfies_frontdoor("saw_ad", "purchased", ["visited_site"]), reason
    assert dag.identify("saw_ad", "purchased")["strategy"] is None


def test_backdoor_is_preferred_when_available():
    # same shape, but the confounder is measured: plain adjustment, no front-door
    dag = DAG.parse(ADS_DAG.replace("unmeasured: intent", ""))
    ident = dag.identify("saw_ad", "purchased")
    assert (ident["strategy"], ident["variables"]) == ("backdoor", ["intent"])


def test_minimal_set_search_scales_to_wide_datasets():
    # three real confounders among 60 variables that have nothing to do with the question
    edges = [(f"c{i}", "t") for i in range(3)] + [(f"c{i}", "y") for i in range(3)] + [("t", "y")]
    edges += [(f"n{i}", f"n{i + 1}") for i in range(57)]
    dag = DAG(edges)
    start = time.perf_counter()
    assert dag.minimal_backdoor_set("t", "y") == {"c0", "c1", "c2"}
    assert dag.identify("t", "y")["strategy"] == "backdoor"
    assert time.perf_counter() - start < 1.0


def test_explanation_names_the_blocking_node():
    dag = DAG.parse((ROOT / "examples/corridor.dag").read_text(encoding="utf-8"))
    lines = dag.identify("led", "speed")["explanation"]
    assert "led <- crowding -> speed — blocked at crowding" in lines


# ---------- estimation ----------


@pytest.fixture(scope="module")
def ads():
    return make_ads()


@pytest.mark.parametrize("method", ["adjustment-formula", "g-computation"])
def test_frontdoor_recovers_the_effect_the_naive_comparison_inflates(ads, method):
    res = estimate_ate(ads, ADS_DAG, "saw_ad", "purchased", method=method, n_boot=200)
    assert res.strategy == "frontdoor"
    assert res.naive.value > ADS_TRUE_ATE + 0.2
    assert abs(res.adjusted.value - ADS_TRUE_ATE) < 0.02
    lo, hi = res.adjusted.ci
    assert lo < ADS_TRUE_ATE < hi
    assert [r["test"] for r in res.refutations] == ["placebo_treatment"]
    assert all(r["passed"] for r in res.refutations)


def test_weighting_methods_explain_why_they_do_not_apply(ads):
    with pytest.raises(ValueError, match="unmeasured"):
        estimate_ate(ads, ADS_DAG, "saw_ad", "purchased", method="ipw", n_boot=0, refute=False)


def test_not_identifiable_is_an_explained_refusal(ads):
    with pytest.raises(ValueError, match="not identifiable"):
        estimate_ate(ads, ADS_DAG + "\nsaw_ad -> purchased", "saw_ad", "purchased",
                     n_boot=0, refute=False)


def test_missing_column_suggests_declaring_it_unmeasured(ads):
    with pytest.raises(ValueError, match="unmeasured: name"):
        estimate_ate(ads, ADS_DAG.replace("unmeasured: intent", ""), "saw_ad", "purchased",
                     n_boot=0, refute=False)


def test_frontdoor_formula_matches_hand_computation():
    # tiny table where the formula can be checked by hand
    rows = []
    for t, m, y, n in [(0, 0, 0, 30), (0, 0, 1, 10), (0, 1, 0, 5), (0, 1, 1, 5),
                       (1, 0, 0, 10), (1, 0, 1, 10), (1, 1, 0, 5), (1, 1, 1, 25)]:
        rows += [{"t": t, "m": m, "y": y}] * n
    df = pd.DataFrame(rows)
    res = estimate_ate(df, "u -> t\nu -> y\nt -> m -> y\nunmeasured: u", "t", "y",
                       method="adjustment-formula", n_boot=0, refute=False)
    p1 = 50 / 100
    ey = {(m, t): df[(df.m == m) & (df.t == t)].y.mean() for m in (0, 1) for t in (0, 1)}
    g = {m: ey[(m, 1)] * p1 + ey[(m, 0)] * (1 - p1) for m in (0, 1)}
    pm = {t: df[df.t == t].m.mean() for t in (0, 1)}
    expected = (pm[1] - pm[0]) * (g[1] - g[0])
    assert res.adjusted.value == pytest.approx(expected)
