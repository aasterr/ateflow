"""HTTP API over the library: upload a CSV, declare a DAG, get the estimate.

Run with:

    uvicorn ateflow.server:app --reload

If `ateflow/static/` exists (the built frontend), it is served at `/`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .api import Result, estimate_ate
from .graph import DAG
from .report import render_report

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = {
    "corridor": {
        "data": "corridor.csv",
        "dag": "corridor.dag",
        "treatment": "led",
        "outcome": "speed",
        "description": "Synthetic corridor scenario with a Simpson's paradox (true ATE +0.15).",
    },
    "onboarding": {
        "data": "onboarding.csv",
        "dag": "onboarding.dag",
        "treatment": "onboarding_email",
        "outcome": "retained_30d",
        "description": "Synthetic product analytics: does a targeted onboarding email "
                       "raise 30-day retention? (true ATE +0.09)",
    },
    "hrisim": {
        "data": "episodes_100_v1.csv",
        "dag": "hrisim.dag",
        "treatment": "A",
        "outcome": "T",
        "description": "100 real HRI episodes from the PeopleFlow dataset (thesis numbers).",
    },
}

app = FastAPI(title="ateflow", description=__doc__)


def _result_payload(result: Result) -> dict:
    def estimate(e) -> dict:
        return {
            "value": e.value,
            "method": e.method,
            "adjustment_set": e.adjustment_set,
            "ci": list(e.ci) if e.ci else None,
            "n": e.n,
            "diagnostics": e.diagnostics,
        }

    return {
        "naive": estimate(result.naive),
        "adjusted": estimate(result.adjusted),
        "adjustment_set": result.adjustment_set,
        "alternatives": [sorted(s) for s in result.alternatives],
        "refutations": result.refutations,
        "confounding_bias": result.confounding_bias,
        "sign_flip": result.sign_flip,
        "report": result.report(),
    }


def _read_csv(raw: bytes, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, f"could not parse {name!r} as CSV: {exc}") from exc


class DagCheckRequest(BaseModel):
    dag: str
    treatment: str | None = None
    outcome: str | None = None
    columns: list[str] | None = None


@app.post("/api/dag/check")
def check_dag(req: DagCheckRequest) -> dict:
    """Parses the DAG and, when treatment and outcome are given, identifies it."""
    try:
        graph = DAG.parse(req.dag)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    out: dict = {
        "nodes": sorted(graph.nodes),
        "edges": graph.edges,
        "missing_columns": sorted(graph.nodes - set(req.columns)) if req.columns else [],
    }
    if req.treatment and req.outcome:
        for name in (req.treatment, req.outcome):
            if name not in graph.nodes:
                raise HTTPException(422, f"{name!r} does not appear in the DAG")
        try:
            minimal = sorted(graph.minimal_backdoor_set(req.treatment, req.outcome))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        out["minimal_adjustment_set"] = minimal
        out["alternatives"] = [
            sorted(s)
            for s in graph.backdoor_sets(req.treatment, req.outcome, max_size=len(minimal) + 1)
            if sorted(s) != minimal
        ]
    return out


async def _resolve_data(
    file: UploadFile | None, example: str | None, saved: int | None = None
) -> tuple[pd.DataFrame, bytes, str]:
    """The dataset behind a request: uploaded CSV, bundled example, or saved analysis."""
    if file is not None:
        raw = await file.read()
        return _read_csv(raw, file.filename or "upload"), raw, file.filename or "upload"
    if example is not None:
        spec = EXAMPLES.get(example)
        if spec is None:
            raise HTTPException(404, f"unknown example {example!r}")
        raw = (EXAMPLES_DIR / spec["data"]).read_bytes()
        return pd.read_csv(io.BytesIO(raw)), raw, f"example: {example}"
    if saved is not None:
        record = store.get_analysis(saved)
        if record is None:
            raise HTTPException(404, f"no analysis with id {saved}")
        raw = record["csv"]
        return _read_csv(raw, record["name"]), raw, record["source"]
    raise HTTPException(422, "provide a CSV file, an example name, or a saved analysis id")


def _run(data: pd.DataFrame, dag: str, treatment: str, outcome: str,
         method: str, boot: int, refute: bool, seed: int) -> dict:
    try:
        result = estimate_ate(
            data, dag, treatment=treatment, outcome=outcome,
            method=method, n_boot=boot, refute=refute, seed=seed,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _result_payload(result)


@app.post("/api/estimate")
async def estimate(
    dag: str = Form(...),
    treatment: str = Form(...),
    outcome: str = Form(...),
    method: str = Form("g-computation"),
    boot: int = Form(500),
    seed: int = Form(0),
    refute: bool = Form(True),
    file: UploadFile | None = File(None),
    example: str | None = Form(None),
    saved: int | None = Form(None),
) -> dict:
    """Runs the full estimate on an uploaded CSV, a bundled example, or saved data."""
    data, _, _ = await _resolve_data(file, example, saved)
    return _run(data, dag, treatment, outcome, method, boot, refute, seed)


@app.post("/api/analyses")
async def save_analysis(
    name: str = Form(...),
    dag: str = Form(...),
    treatment: str = Form(...),
    outcome: str = Form(...),
    method: str = Form("g-computation"),
    boot: int = Form(500),
    seed: int = Form(0),
    file: UploadFile | None = File(None),
    example: str | None = Form(None),
    saved: int | None = Form(None),
) -> dict:
    """Runs the estimate and persists everything: dataset, DAG, question, result."""
    data, raw, source = await _resolve_data(file, example, saved)
    payload = _run(data, dag, treatment, outcome, method, boot, True, seed)
    analysis_id = store.save_analysis(
        name=name.strip() or "untitled",
        source=source,
        csv=raw,
        dag=dag,
        treatment=treatment,
        outcome=outcome,
        method=method,
        result=payload,
    )
    return {"id": analysis_id}


@app.get("/api/analyses")
def analyses() -> list[dict]:
    return store.list_analyses()


@app.get("/api/analyses/{analysis_id}")
def analysis(analysis_id: int) -> dict:
    saved = store.get_analysis(analysis_id)
    if saved is None:
        raise HTTPException(404, f"no analysis with id {analysis_id}")
    data = _read_csv(saved["csv"], saved["name"])
    saved.pop("csv")
    saved["columns"] = list(data.columns)
    return saved


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: int) -> dict:
    if not store.delete_analysis(analysis_id):
        raise HTTPException(404, f"no analysis with id {analysis_id}")
    return {"deleted": analysis_id}


@app.get("/api/analyses/{analysis_id}/report")
def report(analysis_id: int) -> HTMLResponse:
    """Standalone HTML report, ready to share or print to PDF."""
    saved = store.get_analysis(analysis_id)
    if saved is None:
        raise HTTPException(404, f"no analysis with id {analysis_id}")
    return HTMLResponse(render_report(saved))


@app.get("/api/examples")
def list_examples() -> dict:
    """The bundled example datasets, with their DAG and default question."""
    out = {}
    for name, spec in EXAMPLES.items():
        df = pd.read_csv(EXAMPLES_DIR / spec["data"], nrows=0)
        out[name] = {
            "description": spec["description"],
            "columns": list(df.columns),
            "dag": (EXAMPLES_DIR / spec["dag"]).read_text(encoding="utf-8"),
            "treatment": spec["treatment"],
            "outcome": spec["outcome"],
        }
    return out


@app.post("/api/columns")
async def columns(file: UploadFile = File(...)) -> dict:
    """Column names of an uploaded CSV, without keeping the file."""
    data = _read_csv(await file.read(), file.filename or "upload")
    return {"columns": list(data.columns), "rows": len(data)}


STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():  # built frontend, absent in bare checkouts
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
