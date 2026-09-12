"""HTTP API over the library, for local installs: upload a CSV, declare a DAG, get the estimate.

Run with:

    uvicorn ateflow.server:app --reload

The public demo does not use this: it runs the same `service` module in the
browser. Here the request logic is `service` too, plus SQLite persistence.
If `ateflow/static/` exists (the built frontend), it is served at `/`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import service, store
from .data import profile
from .service import EXAMPLES, MAX_UPLOAD_MB, ServiceError  # noqa: F401  (re-exported)

app = FastAPI(title="ateflow", description=__doc__)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ServiceError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


class DagCheckRequest(BaseModel):
    dag: str
    treatment: str | None = None
    outcome: str | None = None
    columns: list[str] | None = None


@app.post("/api/dag/check")
def check_dag(req: DagCheckRequest) -> dict:
    """Parses the DAG and, when treatment and outcome are given, identifies it."""
    return _call(service.check_dag, req.dag, req.treatment, req.outcome, req.columns)


async def _resolve_data(
    file: UploadFile | None, example: str | None, saved: int | None = None
) -> tuple[bytes, str, str]:
    """The dataset behind a request: uploaded CSV, bundled example, or saved analysis.

    Returns the bytes, a name for messages, and the source label stored with analyses.
    """
    if file is not None:
        name = file.filename or "upload"
        return await file.read(), name, name
    if example is not None:
        return _call(service.example_bytes, example), example, f"example: {example}"
    if saved is not None:
        record = store.get_analysis(saved)
        if record is None:
            raise HTTPException(404, f"no analysis with id {saved}")
        return record["csv"], record["name"], record["source"]
    raise HTTPException(422, "provide a CSV file, an example name, or a saved analysis id")


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
    treated_value: str | None = Form(None),
    outcome_positive: str | None = Form(None),
) -> dict:
    """Runs the full estimate on an uploaded CSV, a bundled example, or saved data."""
    raw, name, _ = await _resolve_data(file, example, saved)
    return _call(service.estimate, raw, dag, treatment, outcome, method, boot,
                 refute, seed, treated_value, outcome_positive, name=name)


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
    treated_value: str | None = Form(None),
    outcome_positive: str | None = Form(None),
) -> dict:
    """Runs the estimate and persists everything: dataset, DAG, question, result."""
    raw, data_name, source = await _resolve_data(file, example, saved)
    payload = _call(service.estimate, raw, dag, treatment, outcome, method,
                    boot, True, seed, treated_value, outcome_positive, name=data_name)
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
    data, _ = _call(service.load_csv, saved.pop("csv"), saved["name"])
    saved["columns"] = list(data.columns)
    saved["profile"] = profile(data)
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
    return HTMLResponse(service.report_html(saved))


@app.get("/api/examples")
def list_examples() -> dict:
    """The bundled example datasets, with their DAG and default question."""
    return _call(service.list_examples)


@app.post("/api/columns")
async def columns(file: UploadFile = File(...)) -> dict:
    """Reads an uploaded CSV without keeping it: columns, how each looks, what was detected."""
    return _call(service.inspect_csv, await file.read(), file.filename or "upload")


STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():  # built frontend, absent in bare checkouts
    # after the API routes, so /api/* is matched first
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
