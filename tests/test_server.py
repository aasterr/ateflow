"""API tests: the endpoints must mirror the library exactly."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ateflow.server import app

client = TestClient(app)

HRISIM_DAG = (ROOT / "examples/hrisim.dag").read_text(encoding="utf-8")


def test_examples_are_listed():
    res = client.get("/api/examples")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"corridor", "onboarding", "hrisim"}
    assert body["onboarding"]["treatment"] == "onboarding_email"
    assert body["hrisim"]["treatment"] == "A"
    assert "->" in body["hrisim"]["dag"]


def _apply_edit(dag_text: str, edit: dict) -> str:
    """Mirror of the frontend's guide edits, on the DAG text."""
    src, dst = edit["edge"]
    edges = [(s.strip(), d.strip()) for s, d in
             (pair for line in dag_text.splitlines()
              for chain in [line.split("#")[0].split("->")]
              for pair in zip(chain, chain[1:]))]
    if edit["op"] == "add":
        assert (src, dst) not in edges, f"{src} -> {dst} already in the DAG"
        edges.append((src, dst))
    else:
        assert (src, dst) in edges, f"{src} -> {dst} not in the DAG"
        edges.remove((src, dst))
        if edit["op"] == "flip":
            edges.append((dst, src))
    return "\n".join(f"{s} -> {d}" for s, d in edges)


def test_guide_edits_produce_the_quoted_numbers():
    from ateflow.server import EXAMPLES

    for name, spec in EXAMPLES.items():
        dag = (ROOT / "examples" / spec["dag"]).read_text(encoding="utf-8")
        for tip in spec["guide"]["tries"]:
            res = client.post("/api/estimate", data={
                "dag": _apply_edit(dag, tip["edit"]), "treatment": spec["treatment"],
                "outcome": spec["outcome"], "method": "stratification",
                "example": name, "boot": 0, "refute": False,
            })
            assert res.status_code == 200, (name, tip["edit"], res.text)
            assert res.json()["adjusted"]["value"] == pytest.approx(tip["expect"], abs=5e-3), (
                name, tip["edit"])


def test_dag_check_identifies_minimal_set():
    res = client.post("/api/dag/check", json={
        "dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
        "columns": ["Pi", "A", "Pe", "S", "T"],
    })
    assert res.status_code == 200
    body = res.json()
    assert body["minimal_adjustment_set"] == ["O"]
    assert body["missing_columns"] == ["O"]


def test_dag_check_rejects_cycles():
    res = client.post("/api/dag/check", json={"dag": "a -> b\nb -> a"})
    assert res.status_code == 422
    assert "cycle" in res.json()["detail"]


def test_estimate_on_bundled_example_matches_thesis():
    res = client.post("/api/estimate", data={
        "dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
        "method": "stratification", "example": "hrisim", "boot": 0, "refute": False,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["naive"]["value"] == pytest.approx(-0.207, abs=5e-4)
    assert body["adjusted"]["value"] == pytest.approx(0.061, abs=5e-4)
    assert body["adjustment_set"] == ["O"]
    assert body["sign_flip"] is True


def test_estimate_on_uploaded_csv():
    csv = (ROOT / "examples/episodes_100_v1.csv").read_bytes()
    res = client.post(
        "/api/estimate",
        data={"dag": HRISIM_DAG, "treatment": "A", "outcome": "T",
              "method": "stratification", "boot": 0, "refute": False},
        files={"file": ("episodes.csv", csv, "text/csv")},
    )
    assert res.status_code == 200
    assert res.json()["adjusted"]["value"] == pytest.approx(0.061, abs=5e-4)


def test_estimate_maps_domain_errors_to_422():
    res = client.post("/api/estimate", data={
        "dag": HRISIM_DAG, "treatment": "A", "outcome": "missing_var",
        "example": "hrisim", "boot": 0,
    })
    assert res.status_code == 422
    assert "does not appear in the DAG" in res.json()["detail"]


def test_columns_endpoint_reads_header():
    csv = (ROOT / "examples/episodes_100_v1.csv").read_bytes()
    res = client.post("/api/columns", files={"file": ("e.csv", csv, "text/csv")})
    assert res.status_code == 200
    body = res.json()
    assert body["rows"] == 100
    assert "A" in body["columns"]
