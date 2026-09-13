"""What the web app asks of the engine, independent of how it is asked.

Both front doors use this module: `server.py` (FastAPI, for local installs)
and the browser build, where the same code runs inside Pyodide in a Web
Worker and the data never leave the page. Functions take plain values and
bytes and return JSON-ready dicts; problems the user can fix raise
ServiceError with an HTTP-like status and a readable message.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import pandas as pd

from . import answer
from .api import Result, estimate_ate
from .data import DataError, profile, read_csv
from .graph import DAG
from .report import render_report

MAX_UPLOAD_MB = 20


class ServiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# Where the example files live: next to the package in a checkout, wherever the
# browser build writes them otherwise.
EXAMPLES_DIR = Path(os.environ.get("ATEFLOW_EXAMPLES",
                                   Path(__file__).resolve().parent.parent / "examples"))

# Each example carries a short guide for the demo: the question, why the naive
# and adjusted answers differ, and DAG edits worth trying. The numbers quoted
# are adjustment-formula estimates on the bundled data (the UI default method);
# tests/test_server.py applies every `edit` and checks the result against `expect`,
# so the texts cannot drift away from what the demo actually shows.
EXAMPLES = {
    "corridor": {
        "data": "corridor.csv",
        "dag": "corridor.dag",
        "treatment": "led",
        "outcome": "speed",
        "description": "Synthetic corridor scenario with a Simpson's paradox (true ATE +0.15).",
        "guide": {
            "title": "Does a robot's LED make people walk faster?",
            "story": "A robot in a corridor can switch on an LED signal. The data are "
                     "synthetic, so the true answer is known: +0.15 in walking speed.",
            "why": "The robot turns the LED on mostly when the corridor is crowded, and "
                   "people walk slowly in a crowd anyway. Compared as they are, LED "
                   "episodes look slower (−0.13). Adjusting for crowding compares "
                   "like with like and recovers +0.15.",
            "tries": [
                {
                    "text": "Remove crowding → led. Nothing seems to confound the LED "
                            "any more, the adjustment set is empty and the estimate "
                            "falls back to the biased −0.13.",
                    "edit": {"op": "remove", "edge": ["crowding", "led"]}, "expect": -0.130,
                },
                {
                    "text": "Flip crowding → led. Crowding becomes a consequence of the "
                            "LED, adjusting for it is no longer allowed, and ateflow "
                            "reports −0.13 again: the DAG is an assumption, and the "
                            "answer is only as good as it.",
                    "edit": {"op": "flip", "edge": ["crowding", "led"]}, "expect": -0.130,
                },
            ],
        },
    },
    "onboarding": {
        "data": "onboarding.csv",
        "dag": "onboarding.dag",
        "treatment": "onboarding_email",
        "outcome": "retained_30d",
        "description": "Synthetic product analytics: does a targeted onboarding email "
                       "raise 30-day retention? (true ATE +0.09)",
        "guide": {
            "title": "Did the onboarding email raise retention?",
            "story": "Synthetic signups of a subscription product. Some received an "
                     "onboarding email; the outcome is whether they are still active "
                     "after 30 days. True effect: +9 retention points.",
            "why": "The growth team emailed mostly free-plan users who arrived from paid "
                   "ads, the ones most likely to churn. The emailed group retains worse "
                   "(−6.9 points) because of who they are, not because of the email. "
                   "Adjusting for plan and channel gives about +9. First-week activity is "
                   "a mediator, so it stays out of the adjustment.",
            "tries": [
                {
                    "text": "Remove plan → onboarding_email. Adjusting for channel alone "
                            "leaves most of the bias in: −0.02.",
                    "edit": {"op": "remove", "edge": ["plan", "onboarding_email"]}, "expect": -0.021,
                },
                {
                    "text": "Remove channel → onboarding_email. Adjusting for plan alone "
                            "gives +0.02, still far from +0.09.",
                    "edit": {"op": "remove", "edge": ["channel", "onboarding_email"]}, "expect": 0.023,
                },
                {
                    "text": "Flip onboarding_email → active_week1. Activity now looks like "
                            "a cause of the email, enters the adjustment set, and the "
                            "estimate drops to +0.035: the part of the effect that works "
                            "by bringing users back is thrown away.",
                    "edit": {"op": "flip", "edge": ["onboarding_email", "active_week1"]}, "expect": 0.035,
                },
            ],
        },
    },
    "ads": {
        "data": "ads.csv",
        "dag": "ads.dag",
        "treatment": "saw_ad",
        "outcome": "purchased",
        "description": "Synthetic front-door case: the confounder is unmeasured, the "
                       "effect runs through a site visit (true ATE +0.15).",
        "guide": {
            "title": "Does the ad make people buy? (front-door)",
            "story": "Synthetic users who saw an ad or not, whether they visited the site, "
                     "whether they bought. The dashed node, purchase intent, is not in the "
                     "data. True effect of the ad: +0.15.",
            "why": "The targeting reaches people who were shopping anyway, so buyers are "
                   "over-represented among those who saw the ad: compared as they are, the "
                   "ad looks worth +0.42. Intent cannot be adjusted for because it was never "
                   "measured. But the ad acts only through the visit, and nothing else "
                   "decides the visit: ateflow rebuilds the effect in two steps, ad → visit "
                   "and visit → purchase holding the ad fixed, and gets +0.15.",
            "tries": [
                {
                    "text": "Remove intent → saw_ad. Nothing seems to confound the ad any "
                            "more, ateflow uses no adjustment and reports the biased +0.42.",
                    "edit": {"op": "remove", "edge": ["intent", "saw_ad"]}, "expect": 0.422,
                },
                {
                    "text": "Add intent → visited_site. Now intent also drives the visit, the "
                            "front-door criterion fails, and ateflow says the effect cannot "
                            "be identified from these data instead of guessing.",
                    "edit": {"op": "add", "edge": ["intent", "visited_site"]}, "expect": None,
                },
                {
                    "text": "Add saw_ad → purchased. The ad now also works directly, so the "
                            "visit no longer carries the whole effect: not identifiable either.",
                    "edit": {"op": "add", "edge": ["saw_ad", "purchased"]}, "expect": None,
                },
            ],
        },
    },
    "hrisim": {
        "data": "episodes_100_v1.csv",
        "dag": "hrisim.dag",
        "treatment": "A",
        "outcome": "T",
        "description": "100 real HRI episodes from the PeopleFlow dataset (thesis numbers).",
        "guide": {
            "title": "Does the robot's signal help it succeed? (real data)",
            "story": "100 real episodes of a robot crossing a corridor with people. A: the "
                     "robot emits an LED signal. T: the task succeeds instead of timing "
                     "out. O: static obstacles. The numbers match the reference thesis.",
            "why": "Obstacles change both whether the robot signals and how likely the task "
                   "is to succeed. Compared as they are, signalling episodes succeed less "
                   "(−0.207). Adjusting for O reverses the sign: +0.061.",
            "tries": [
                {
                    "text": "Add Pi → Pe. Pi becomes a confounder too, the set grows to "
                            "{O, Pi} and the estimate rises to +0.108, but 36 of the 100 "
                            "episodes are dropped: with Pi=0 and O=0 the robot never "
                            "signalled, so there is nothing to compare.",
                    "edit": {"op": "add", "edge": ["Pi", "Pe"]}, "expect": 0.108,
                },
                {
                    "text": "Remove O → A. With no confounder declared, the answer is the "
                            "naive −0.207.",
                    "edit": {"op": "remove", "edge": ["O", "A"]}, "expect": -0.207,
                },
            ],
        },
    },
}


def result_payload(result: Result) -> dict:
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
        "data_report": result.data_report,
        "strategy": result.strategy,
        "explanation": result.explanation,
        "method": result.method,
        "comparison": result.comparison,
        "methods_agree": result.methods_agree,
        "report": result.report(),
    }


def load_csv(raw: bytes, name: str) -> tuple[pd.DataFrame, dict]:
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ServiceError(
            413, f"{name!r} is {len(raw) / 2**20:.0f} MB, the limit is {MAX_UPLOAD_MB} MB")
    try:
        return read_csv(raw)
    except DataError as exc:
        raise ServiceError(422, f"{name}: {exc}") from exc


def inspect_csv(raw: bytes, name: str = "upload") -> dict:
    """Columns, how each looks, and what was detected reading the file."""
    data, info = load_csv(raw, name)
    return {"columns": list(data.columns), "rows": len(data),
            "profile": profile(data), "info": info}


def check_dag(dag: str, treatment: str | None = None, outcome: str | None = None,
              columns: list[str] | None = None) -> dict:
    """Parses the DAG and, when treatment and outcome are given, identifies it."""
    try:
        graph = DAG.parse(dag)
    except ValueError as exc:
        raise ServiceError(422, str(exc)) from exc

    out: dict = {
        "nodes": sorted(graph.nodes),
        "edges": graph.edges,
        "unmeasured": sorted(graph.unmeasured),
        "missing_columns": sorted(graph.observed - set(columns)) if columns else [],
    }
    if treatment and outcome:
        try:
            ident = graph.identify(treatment, outcome)
        except ValueError as exc:
            raise ServiceError(422, str(exc)) from exc
        # not identifiable is an answer, not an error: the explanation says why
        out.update(ident)
    return out


def estimate(raw: bytes, dag: str, treatment: str, outcome: str,
             method: str = "auto", boot: int = 500, refute: bool = True,
             seed: int = 0, treated_value: str | None = None,
             outcome_positive: str | None = None, name: str = "upload") -> dict:
    """The full estimate on a CSV given as bytes."""
    data, _ = load_csv(raw, name)
    try:
        result = estimate_ate(
            data, dag, treatment=treatment, outcome=outcome,
            method=method, n_boot=boot, refute=refute, seed=seed,
            treated_value=treated_value or None, outcome_positive=outcome_positive or None,
        )
    except ValueError as exc:
        raise ServiceError(422, str(exc)) from exc
    payload = result_payload(result)
    payload["answer"] = answer.build(payload, treatment, outcome)
    return payload


def example_bytes(name: str) -> bytes:
    spec = EXAMPLES.get(name)
    if spec is None:
        raise ServiceError(404, f"unknown example {name!r}")
    return (EXAMPLES_DIR / spec["data"]).read_bytes()


def list_examples() -> dict:
    """The bundled example datasets, with their DAG, default question and guide."""
    out = {}
    for name, spec in EXAMPLES.items():
        df, _ = load_csv(example_bytes(name), name)
        out[name] = {
            "description": spec["description"],
            "columns": list(df.columns),
            "profile": profile(df),
            "dag": (EXAMPLES_DIR / spec["dag"]).read_text(encoding="utf-8"),
            "treatment": spec["treatment"],
            "outcome": spec["outcome"],
            "guide": spec.get("guide"),
        }
    return out


def report_html(analysis: dict) -> str:
    """Standalone HTML report for a saved analysis (dataset bytes not needed)."""
    return render_report(analysis)


# ---------- browser entry point ----------

def _estimate_op(payload: dict, raw: bytes | None) -> dict:
    example = payload.pop("example", None)
    if raw is None:
        if example is None:
            raise ServiceError(422, "provide a CSV file or an example name")
        raw = example_bytes(example)
        payload.setdefault("name", example)
    return estimate(raw, **payload)


_OPS = {
    "inspect": lambda p, raw: inspect_csv(raw, p.get("name", "upload")),
    "check": lambda p, raw: check_dag(p["dag"], p.get("treatment"), p.get("outcome"),
                                      p.get("columns")),
    "estimate": _estimate_op,
    "examples": lambda p, raw: list_examples(),
    "report": lambda p, raw: {"html": report_html(p["analysis"])},
    # results saved before the answer existed get their sentences on load
    "answer": lambda p, raw: answer.build(p["result"], p["treatment"], p["outcome"]),
}


def handle(op: str, payload_json: str, raw=None) -> str:
    """One call from the Web Worker: JSON in, JSON out, errors as values.

    `raw` is the CSV as bytes, or as the JS Uint8Array proxy Pyodide hands over.
    """
    try:
        if raw is not None and not isinstance(raw, (bytes, bytearray)):
            # a Uint8Array proxy converts; JS null arrives as a JsNull proxy
            raw = raw.to_bytes() if hasattr(raw, "to_bytes") else None
        payload = json.loads(payload_json or "{}")
        return json.dumps({"ok": _OPS[op](payload, raw)})
    except ServiceError as exc:
        return json.dumps({"error": exc.detail, "status": exc.status})
    except Exception as exc:  # a bug, not a data problem: keep the trace for the console
        return json.dumps({"error": f"internal error: {exc}", "status": 500,
                           "trace": traceback.format_exc()})
