"""Persistence and report: save, reload, delete, render."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

HRISIM_DAG = (ROOT / "examples/hrisim.dag").read_text(encoding="utf-8")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ATEFLOW_DB", str(tmp_path / "test.db"))
    from ateflow.server import app

    return TestClient(app)


def _save(client, name="thesis run"):
    return client.post("/api/analyses", data={
        "name": name, "dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
        "method": "adjustment-formula", "example": "hrisim", "boot": 50,
    })


def test_save_list_reload_delete(client):
    res = _save(client)
    assert res.status_code == 200
    analysis_id = res.json()["id"]

    listing = client.get("/api/analyses").json()
    assert [a["id"] for a in listing] == [analysis_id]
    assert listing[0]["name"] == "thesis run"
    assert listing[0]["adjusted"] == pytest.approx(0.061, abs=5e-4)

    full = client.get(f"/api/analyses/{analysis_id}").json()
    assert full["dag"] == HRISIM_DAG
    assert full["source"] == "example: hrisim"
    assert "A" in full["columns"]
    assert full["result"]["naive"]["value"] == pytest.approx(-0.207, abs=5e-4)
    assert "csv" not in full  # bytes stay server-side

    assert client.delete(f"/api/analyses/{analysis_id}").status_code == 200
    assert client.get("/api/analyses").json() == []
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_estimate_can_reuse_saved_data(client):
    analysis_id = _save(client).json()["id"]
    res = client.post("/api/estimate", data={
        "dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
        "method": "adjustment-formula", "saved": analysis_id, "boot": 0, "refute": False,
    })
    assert res.status_code == 200
    assert res.json()["adjusted"]["value"] == pytest.approx(0.061, abs=5e-4)


def test_report_is_standalone_html(client):
    analysis_id = _save(client, name="corridor signal").json()["id"]
    res = client.get(f"/api/analyses/{analysis_id}/report")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    body = res.text
    assert "corridor signal" in body
    assert "-0.207" in body and "+0.061" in body
    assert "<svg" in body and "Simpson" in body
    assert "positivity" not in body  # {O} adjustment drops no group


def test_report_mentions_dropped_rows_for_pi_o(client):
    variant = (ROOT / "examples/hrisim_pipe.dag").read_text(encoding="utf-8")
    res = client.post("/api/analyses", data={
        "name": "overlap check", "dag": variant, "treatment": "A", "outcome": "T",
        "method": "adjustment-formula", "example": "hrisim", "boot": 0,
    })
    body = client.get(f"/api/analyses/{res.json()['id']}/report").text
    assert "36 of 100 rows" in body


def test_report_of_missing_analysis_is_404(client):
    assert client.get("/api/analyses/999/report").status_code == 404
