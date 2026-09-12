"""The browser entry point: `service.handle` is what the Web Worker calls under Pyodide."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import service

HRISIM_DAG = (ROOT / "examples/hrisim.dag").read_text(encoding="utf-8")


def call(op, payload=None, raw=None):
    return json.loads(service.handle(op, json.dumps(payload or {}), raw))


def test_examples_and_estimate_by_example_name():
    examples = call("examples")["ok"]
    assert set(examples) == {"corridor", "onboarding", "ads", "hrisim"}
    out = call("estimate", {"example": "hrisim", "dag": HRISIM_DAG, "treatment": "A",
                            "outcome": "T", "method": "adjustment-formula", "boot": 0,
                            "refute": False})["ok"]
    assert out["adjusted"]["value"] == pytest.approx(0.061, abs=5e-4)


def test_estimate_on_bytes_with_nulls_from_javascript():
    raw = (ROOT / "examples/episodes_100_v1.csv").read_bytes()
    out = call("estimate", {"name": "e.csv", "dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
                            "method": "ipw", "boot": 0, "refute": False,
                            "treated_value": None, "outcome_positive": None}, raw)["ok"]
    assert out["data_report"]["rows_used"] == 100


def test_user_errors_are_values_with_status():
    out = call("check", {"dag": "a -> b\nb -> a"})
    assert out["status"] == 422 and "cycle" in out["error"]
    assert call("estimate", {"example": "nope", "dag": "a -> b", "treatment": "a",
                             "outcome": "b"})["status"] == 404
    assert call("inspect", {"name": "empty.csv"}, b"")["status"] == 422


def test_bugs_are_values_too_with_a_trace():
    out = call("check", {})  # missing 'dag' key: a programming error, not a data problem
    assert out["status"] == 500 and "Traceback" in out["trace"]


def test_report_renders_from_a_stored_result():
    result = call("estimate", {"example": "hrisim", "dag": HRISIM_DAG, "treatment": "A",
                               "outcome": "T", "method": "adjustment-formula", "boot": 0,
                               "refute": False})["ok"]
    html = call("report", {"analysis": {
        "name": "run", "created_at": "2026-09-13T10:00:00+00:00", "source": "example: hrisim",
        "dag": HRISIM_DAG, "treatment": "A", "outcome": "T", "method": "adjustment-formula",
        "result": result,
    }})["ok"]["html"]
    assert "<svg" in html and "+0.061" in html
