"""Real-world CSVs: dialects, codings, missing values and unusable columns."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ateflow import estimate_ate
from ateflow.data import DataError, column_kind, profile, read_csv

DAG = "z -> t\nz -> y\nt -> y"


@pytest.fixture(scope="module")
def base():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.integers(0, 2, n)
    t = rng.binomial(1, 0.3 + 0.4 * z)
    y = 1 + 0.5 * t + z + rng.normal(0, 1, n)
    return pd.DataFrame({"z": z, "t": t, "y": y})


@pytest.fixture(scope="module")
def reference(base):
    return estimate_ate(base, DAG, "t", "y", method="adjustment-formula", n_boot=0, refute=False)


def _estimate(raw: bytes, dag=DAG, **kw):
    df, _ = read_csv(raw)
    kw.setdefault("method", "adjustment-formula")
    return estimate_ate(df, dag, "t", "y", n_boot=0, refute=False, **kw)


# ---------- reading ----------


@pytest.mark.parametrize("dialect, encode", [
    ("comma", lambda df: df.to_csv(index=False).encode()),
    ("excel-it", lambda df: df.to_csv(index=False, sep=";", decimal=",").encode()),
    ("tab", lambda df: df.to_csv(index=False, sep="\t").encode()),
    ("bom", lambda df: df.to_csv(index=False).encode("utf-8-sig")),
])
def test_dialects_give_the_same_estimate(base, reference, dialect, encode):
    assert _estimate(encode(base)).adjusted.value == pytest.approx(reference.adjusted.value)


def test_excel_italian_export_is_detected(base):
    _, info = read_csv(base.to_csv(index=False, sep=";", decimal=",").encode())
    assert (info["delimiter"], info["decimal"]) == ("semicolon", "comma")


def test_windows_encoding_and_names_with_spaces(base):
    raw = base.rename(columns={"z": "fascia  età"}).to_csv(index=False).encode("cp1252")
    df, info = read_csv(raw)
    assert info["encoding"] == "cp1252"
    assert "fascia età" in df.columns
    res = estimate_ate(df, DAG.replace("z", "fascia età"), "t", "y", n_boot=0, refute=False)
    assert res.adjustment_set == ["fascia età"]


def test_reserved_characters_in_names_are_renamed():
    df, info = read_csv(b"a#b,x->y\n1,2\n3,4\n")
    assert list(df.columns) == ["a_b", "x_y"]
    assert info["renamed"] == {"a#b": "a_b", "x->y": "x_y"}


@pytest.mark.parametrize("raw, message", [
    (b"", "empty"),
    (b"z,t,y\n", "no data rows"),
    (b"just one column\n1\n2\n", "single column"),
])
def test_unreadable_files_say_why(raw, message):
    with pytest.raises(DataError, match=message):
        read_csv(raw)


# ---------- codings ----------


@pytest.mark.parametrize("labels", [("yes", "no"), ("TRUE", "FALSE"), ("treated", "control"), ("Sì", "No")])
def test_known_binary_labels_are_coded_automatically(base, reference, labels):
    b = base.assign(t=np.where(base["t"] == 1, *labels))
    res = _estimate(b.to_csv(index=False).encode())
    assert res.adjusted.value == pytest.approx(reference.adjusted.value)
    # pandas parses TRUE/FALSE as booleans, so the label comes back as 'True'
    assert res.data_report["treatment_coding"]["1"].lower() == labels[0].lower()


def test_ambiguous_treatment_labels_ask_for_a_choice(base, reference):
    raw = base.assign(t=base["t"] + 1).to_csv(index=False).encode()
    with pytest.raises(DataError, match="choose which one counts as treated"):
        _estimate(raw)
    res = _estimate(raw, treated_value="2")
    assert res.adjusted.value == pytest.approx(reference.adjusted.value)


def test_text_outcome_needs_its_positive_value(base):
    b = base.assign(y=np.where(base["y"] > 1.5, "churned", "active"))
    raw = b.to_csv(index=False).encode()
    with pytest.raises(DataError, match="positive outcome"):
        _estimate(raw)
    res = _estimate(raw, outcome_positive="churned")
    assert res.data_report["outcome_coding"] == {"1": "churned", "0": "active"}


def test_treatment_with_many_values_is_refused(base):
    raw = base.assign(t=np.arange(len(base)) % 3).to_csv(index=False).encode()
    with pytest.raises(DataError, match="exactly two values"):
        _estimate(raw)


def test_outcome_with_units_is_refused(base):
    raw = base.assign(y=base["y"].map(lambda v: f"{v:.2f} EUR")).to_csv(index=False).encode()
    with pytest.raises(DataError, match="numeric"):
        _estimate(raw)


# ---------- missing values ----------


@pytest.mark.parametrize("column", ["t", "y", "z"])
@pytest.mark.parametrize("method", ["adjustment-formula", "g-computation", "ipw", "aipw"])
def test_incomplete_rows_are_dropped_and_reported(base, column, method):
    b = base.astype(object)
    b.loc[:9, column] = np.nan
    res = _estimate(b.to_csv(index=False).encode(), method=method)
    rep = res.data_report
    assert (rep["rows_in"], rep["rows_used"]) == (400, 390)
    assert rep["missing_by_column"] == {column: 10}
    assert np.isfinite(res.adjusted.value)
    assert any("10 of 400 rows dropped" in w for w in rep["warnings"])


# ---------- confounders ----------


def test_identifier_as_confounder_is_refused(base):
    b = base.assign(user_id=[f"u{i}" for i in range(len(base))])
    with pytest.raises(DataError, match="identifier"):
        _estimate(b.to_csv(index=False).encode(), dag=DAG + "\nuser_id -> t\nuser_id -> y",
                  method="g-computation")


def test_adjustment_formula_on_continuous_confounder_points_to_other_methods(base):
    raw = base.assign(z=np.random.default_rng(1).normal(size=len(base))).to_csv(index=False).encode()
    with pytest.raises(DataError, match="g-computation, IPW or AIPW"):
        _estimate(raw)
    assert np.isfinite(_estimate(raw, method="aipw").adjusted.value)


def test_empty_arm_is_refused(base):
    with pytest.raises(DataError, match="single value"):
        _estimate(base.assign(t=0).to_csv(index=False).encode())


# ---------- profiling ----------


def test_profile_kinds():
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "code": [f"c{i}" for i in range(n)],
        "flag": ["yes", "no"] * 50,
        "group": list("abcde") * 20,
        "score": np.linspace(0, 1, n),
        "level": [1, 2, 3, 4] * 25,
        "same": 1,
        "blank": np.nan,
    })
    kinds = {p["name"]: p["kind"] for p in profile(df)}
    assert kinds == {
        "id": "identifier", "code": "identifier", "flag": "binary", "group": "categorical",
        "score": "continuous", "level": "discrete", "same": "constant", "blank": "empty",
    }
    assert column_kind(pd.Series([1, 1, 0, None])) == "binary"
