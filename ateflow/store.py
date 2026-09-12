"""Analysis persistence on SQLite, stdlib only.

One table: the dataset travels with the analysis (CSV bytes), so a saved
analysis can always be reloaded and re-run, whatever happened to the original
file. The database path comes from ATEFLOW_DB (default: ateflow.db in the
working directory); tests point it at a temp file.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source     TEXT NOT NULL,
    csv        BLOB NOT NULL,
    dag        TEXT NOT NULL,
    treatment  TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    method     TEXT NOT NULL,
    result     TEXT NOT NULL
)
"""


def _db_path() -> str:
    return os.environ.get("ATEFLOW_DB", "ateflow.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def save_analysis(
    name: str,
    source: str,
    csv: bytes,
    dag: str,
    treatment: str,
    outcome: str,
    method: str,
    result: dict,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO analyses (name, created_at, source, csv, dag, treatment,"
            " outcome, method, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source,
                csv,
                dag,
                treatment,
                outcome,
                method,
                json.dumps(result),
            ),
        )
        return cur.lastrowid


def list_analyses() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, source, treatment, outcome, method, result"
            " FROM analyses ORDER BY id DESC"
        ).fetchall()
    out = []
    for row in rows:
        result = json.loads(row["result"])
        out.append(
            {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "source": row["source"],
                "treatment": row["treatment"],
                "outcome": row["outcome"],
                "method": row["method"],
                "adjusted": result["adjusted"]["value"],
                "naive": result["naive"]["value"],
            }
        )
    return out


def get_analysis(analysis_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "source": row["source"],
        "csv": row["csv"],
        "dag": row["dag"],
        "treatment": row["treatment"],
        "outcome": row["outcome"],
        "method": row["method"],
        "result": json.loads(row["result"]),
    }


def delete_analysis(analysis_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return cur.rowcount > 0
