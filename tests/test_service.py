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


def _apply_edit(dag_text: str, edit: dict) -> str:
    """Mirror of the frontend's guide edits, on the DAG text."""
    src, dst = edit["edge"]
    lines = [line.split("#")[0] for line in dag_text.splitlines()]
    keep = [line for line in lines if line.strip() and "->" not in line]  # unmeasured: ...
    edges = [(s.strip(), d.strip()) for s, d in
             (pair for line in lines
              for chain in [line.split("->")]
              for pair in zip(chain, chain[1:]))]
    if edit["op"] == "add":
        assert (src, dst) not in edges, f"{src} -> {dst} already in the DAG"
        edges.append((src, dst))
    else:
        assert (src, dst) in edges, f"{src} -> {dst} not in the DAG"
        edges.remove((src, dst))
        if edit["op"] == "flip":
            edges.append((dst, src))
    return "\n".join([f"{s} -> {d}" for s, d in edges] + keep)


def test_guide_edits_produce_the_quoted_numbers():
    for name, spec in service.EXAMPLES.items():
        dag = (ROOT / "examples" / spec["dag"]).read_text(encoding="utf-8")
        for tip in spec["guide"]["tries"]:
            out = call("estimate", {"example": name, "dag": _apply_edit(dag, tip["edit"]),
                                    "treatment": spec["treatment"], "outcome": spec["outcome"],
                                    "boot": 0, "refute": False})
            if tip["expect"] is None:  # the guide promises "not identifiable"
                assert out.get("status") == 422, (name, tip["edit"], out)
                assert "not identifiable" in out["error"]
                continue
            assert "ok" in out, (name, tip["edit"], out)
            assert out["ok"]["adjusted"]["value"] == pytest.approx(tip["expect"], abs=5e-3), (name, tip["edit"])


def test_dag_check_identifies_minimal_set_and_missing_columns():
    body = call("check", {"dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
                          "columns": ["Pi", "A", "Pe", "S", "T"]})["ok"]
    assert (body["strategy"], body["variables"]) == ("backdoor", ["O"])
    assert any("blocked at O" in line for line in body["explanation"])
    assert body["missing_columns"] == ["O"]


def test_unknown_variable_is_a_user_error():
    out = call("estimate", {"example": "hrisim", "dag": HRISIM_DAG, "treatment": "A",
                            "outcome": "missing_var", "boot": 0})
    assert out["status"] == 422 and "does not appear in the DAG" in out["error"]


def test_messy_upload_round_trip():
    """Excel-Italian CSV with text labels: inspect, estimate with a chosen coding, report."""
    csv = (
        "ID cliente;piano;email ricevuta;stato 30gg;spesa\n"
        + "".join(
            f"C{i};{'pro' if i % 3 == 0 else 'free'};{'sì' if i % 2 else 'no'};"
            f"{'attivo' if (i * 7) % 5 < 3 else 'perso'};{i % 11},5\n"
            for i in range(200)
        )
    ).encode("cp1252")

    cols = call("inspect", {"name": "clienti.csv"}, csv)["ok"]
    kinds = {p["name"]: p["kind"] for p in cols["profile"]}
    assert kinds["ID cliente"] == "identifier" and kinds["email ricevuta"] == "binary"
    assert (cols["info"]["delimiter"], cols["info"]["decimal"]) == ("semicolon", "comma")

    question = {"name": "clienti.csv", "dag": "piano -> email ricevuta\npiano -> stato 30gg\n"
                "email ricevuta -> stato 30gg", "treatment": "email ricevuta",
                "outcome": "stato 30gg", "boot": 0, "refute": False}
    missing_choice = call("estimate", question, csv)
    assert missing_choice["status"] == 422 and "positive outcome" in missing_choice["error"]

    result = call("estimate", {**question, "outcome_positive": "attivo"}, csv)["ok"]
    assert result["data_report"]["treatment_coding"]["1"] == "sì"
    assert result["data_report"]["outcome_coding"] == {"1": "attivo", "0": "perso"}
    assert 'Setting email ricevuta to "sì" for everyone, instead of "no"' in result["answer"]["headline"]
    html = call("report", {"analysis": {
        "name": "email", "created_at": "2026-09-13T10:00:00+00:00", "source": "clienti.csv",
        "dag": question["dag"], "treatment": "email ricevuta", "outcome": "stato 30gg",
        "method": result["method"], "result": result}})["ok"]["html"]
    assert "treated = sì" in html


def test_oversized_upload_is_refused(monkeypatch):
    monkeypatch.setattr(service, "MAX_UPLOAD_MB", 0.001)
    assert call("inspect", {"name": "big.csv"}, b"a,b\n" + b"1,2\n" * 1000)["status"] == 413


def test_report_mentions_the_paradox_and_dropped_rows():
    variant = (ROOT / "examples/hrisim_pipe.dag").read_text(encoding="utf-8")
    for dag, expected, absent in [(HRISIM_DAG, ["-0.207", "+0.061", "Simpson"], "positivity"),
                                  (variant, ["36 of 100 rows"], None)]:
        result = call("estimate", {"example": "hrisim", "dag": dag, "treatment": "A", "outcome": "T",
                                   "method": "adjustment-formula", "boot": 0, "refute": False})["ok"]
        html = call("report", {"analysis": {
            "name": "run", "created_at": "2026-09-13T10:00:00+00:00", "source": "example: hrisim",
            "dag": dag, "treatment": "A", "outcome": "T", "method": "adjustment-formula",
            "result": result}})["ok"]["html"]
        assert all(text in html for text in expected), dag
        if absent:
            assert absent not in html
