# Answer-first UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ateflow picks the estimator itself, checks it against the others, and presents the result as a plain-language answer in a right-hand Answer panel.

**Architecture:** Python owns every number and every sentence (`api.py` auto method + comparison, new `answer.py` wording), so the CLI, the report and the browser engine say the same thing and tests pin the wording. The React app only lays it out: left = question, center = DAG, right = `AnswerPanel` with Answer / Data / Guide tabs.

**Tech Stack:** Python 3.12+ (numpy, pandas, pytest), Pyodide in a Web Worker, React 19 + @xyflow/react, Vite.

## Global Constraints

- Never write "stratification"/"strata" in code, copy or docs: "adjustment formula", "adjusted", "naive".
- Left sidebar stays uncluttered: no method selector, no results.
- No set braces `{a, b}` in user-facing UI text; lists are "a, b".
- Existing explicit-method calls (`method="g-computation"` etc.) keep their behaviour; all current tests stay green.
- Commit messages via `git commit -F file` (no BOM), ending with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run Python tests from the repo root: `.venv/Scripts/python -m pytest -q`.

---

### Task 1: Auto method, comparison and agreement in `estimate_ate`

**Files:**
- Modify: `ateflow/api.py`
- Modify: `ateflow/data.py` (add `outcome_binary` to the report)
- Test: `tests/test_answer_first.py` (new)

**Interfaces:**
- Produces: `estimate_ate(..., method="auto")`; `Result.method: str` (method actually used); `Result.comparison: list[dict]` with keys `method, value (float|None), applicable (bool), reason (str|None), primary (bool)`; `Result.methods_agree: bool`; `data_report["outcome_binary"]: bool`; `api.choose_method(strategy, data, variables) -> str`.

- [ ] **Step 1: failing tests**

```python
"""Auto method choice, the cross-check between estimators, and answer wording."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ateflow import DAG, estimate_ate
from ateflow.data import read_csv
from examples.make_data import make_onboarding

def example(name):
    df, _ = read_csv((ROOT / "examples" / name).read_bytes())
    return df

@pytest.fixture(scope="module")
def onboarding():
    return make_onboarding()

def continuous_confounder(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(age - 40) / 8))).astype(int)
    y = 2.0 * t + 0.1 * age + rng.normal(0, 1, n)
    return pd.DataFrame({"age": age, "t": t, "y": y})

def test_auto_uses_adjustment_formula_with_discrete_confounders(onboarding):
    res = estimate_ate(onboarding, DAG.from_file(ROOT / "examples/onboarding.dag"),
                       "onboarding_email", "retained_30d", n_boot=0, refute=False)
    assert res.method == "adjustment-formula"

def test_auto_uses_aipw_with_a_continuous_confounder():
    res = estimate_ate(continuous_confounder(), "age -> t\nage -> y\nt -> y", "t", "y",
                       n_boot=0, refute=False)
    assert res.method == "aipw"
    assert abs(res.adjusted.value - 2.0) < 0.15

def test_auto_front_door_uses_the_formula():
    res = estimate_ate(example("ads.csv"), DAG.from_file(ROOT / "examples/ads.dag"),
                       "saw_ad", "purchased", n_boot=0, refute=False)
    assert res.strategy == "frontdoor" and res.method == "adjustment-formula"

def test_comparison_lists_every_estimator_of_the_strategy(onboarding):
    res = estimate_ate(onboarding, DAG.from_file(ROOT / "examples/onboarding.dag"),
                       "onboarding_email", "retained_30d", n_boot=200, refute=False)
    methods = [c["method"] for c in res.comparison]
    assert methods == ["adjustment-formula", "g-computation", "ipw", "aipw"]
    assert [c["primary"] for c in res.comparison] == [True, False, False, False]
    assert res.methods_agree

def test_formula_not_applicable_with_continuous_confounder():
    res = estimate_ate(continuous_confounder(), "age -> t\nage -> y\nt -> y", "t", "y",
                       n_boot=0, refute=False)
    formula = next(c for c in res.comparison if c["method"] == "adjustment-formula")
    assert not formula["applicable"] and "discrete" in formula["reason"]

def test_hrisim_methods_disagree():
    dag = DAG.parse((ROOT / "examples/hrisim.dag").read_text() + "\nPi -> Pe")
    res = estimate_ate(example("episodes_100_v1.csv"), dag, "A", "T", n_boot=200, refute=False)
    assert res.method == "adjustment-formula"
    assert round(res.adjusted.value, 3) == 0.108
    assert not res.methods_agree

def test_outcome_binary_flag(onboarding):
    res = estimate_ate(onboarding, DAG.from_file(ROOT / "examples/onboarding.dag"),
                       "onboarding_email", "retained_30d", n_boot=0, refute=False)
    assert res.data_report["outcome_binary"] is True
    res2 = estimate_ate(continuous_confounder(), "age -> t\nage -> y\nt -> y", "t", "y",
                        n_boot=0, refute=False)
    assert res2.data_report["outcome_binary"] is False
```

- [ ] **Step 2:** run `.venv/Scripts/python -m pytest tests/test_answer_first.py -q` → FAIL (`Result` has no `method`).

- [ ] **Step 3: implement.**
  - `data.prepare`: after the outcome coding, `outcome_binary = outcome_coding is not None or set(pd.unique(work[outcome])) <= {0.0, 1.0}`; add to the returned dict.
  - `api.py`: `ESTIMATORS` module-level dict (the current inline one). `choose_method(strategy, data, variables)`: returns `"adjustment-formula"` when every variable's `column_kind` (from `data.py`) is not `"continuous"`, else `"aipw"` for backdoor, `"g-computation"` for front-door. In `estimate_ate`, when `method == "auto"`, resolve after identification using the raw `data` restricted to complete rows. Default parameter becomes `"auto"`.
  - Comparison: after `adjusted`, for every `name, fn` in `ESTIMATORS[strategy]` in the order formula, g-computation, ipw, aipw: primary reuses `adjusted.value`; others call `prepare(..., method=name)` only when name is `adjustment-formula` (the only method with extra data rules) else reuse `data`, then `fn(frame, treatment, outcome, chosen).value`; `DataError`/`ValueError` → `applicable=False, reason=str(exc)`.
  - `methods_agree`: if `adjusted.ci` is None or fewer than two applicable → True; else all applicable values within `[lo, hi]`.
  - `Result` gains `method: str = ""`, `comparison: list = field(default_factory=list)`, `methods_agree: bool = True`. `result_payload` in `service.py` adds `"method"`, `"comparison"`, `"methods_agree"`.
  - CLI `--method` default `"auto"`, choices include `"auto"`.

- [ ] **Step 4:** run the new tests and the whole suite → all PASS.

- [ ] **Step 5:** commit "Auto estimator choice with a cross-check against the others".

### Task 2: `answer.py` — the answer in words

**Files:**
- Create: `ateflow/answer.py`
- Modify: `ateflow/service.py` (`result_payload` adds `"answer"`; new op `"answer"`), `ateflow/report.py` (headline at top, method from result), `ateflow/api.py` (`Result.report()` first line is the headline)
- Test: `tests/test_answer_first.py`

**Interfaces:**
- Consumes: payload dict from `service.result_payload` (keys above), plus `treatment`, `outcome`.
- Produces: `answer.build(payload: dict, treatment: str, outcome: str) -> dict` with keys `headline: str`, `technical: str`, `naive: str`, `checks: list[{ok: bool, title: str, text: str}]`. Service op `answer` takes `{result, treatment, outcome}` and returns that dict.

Wording rules (all tested):
- magnitude: binary outcome → `f"{abs(v)*100:.1f} percentage points"`; numeric → `f"{abs(v):.3g} units of {outcome}"`.
- direction: `rise by` / `fall by` / `not change` (when |v| < 0.0005).
- treatment phrase: coded labels → `{treatment} = "{label1}"` vs `"{label0}"`, else `{treatment}`: `If everyone had {T} instead of nobody` for 0/1; `If everyone had {treatment} = "a" instead of "b"` for labels.
- population: `dropped_rows > 0` → append `, among the {n-dropped} of {n} rows where both groups occur`.
- CI: `(95% CI {lo} to {hi})` in the same unit, signed; crossing zero → append ` — but the data cannot tell whether the effect is positive or negative.`
- headline example onboarding: `If everyone had onboarding_email instead of nobody, retained_30d would rise by 9.0 percentage points (95% CI +6.6 to +11.4).` (numbers taken from the run, test asserts prefix and unit).
- technical: `ATE · adjusting for channel, plan · adjustment formula · 6000 rows`; empty set `no adjustment needed`; front-door `through visited_site`.
- naive: `Comparing the two groups as they are gives {signed naive}: {why}.` with why = backdoor `the confounders channel, plan shift it by {signed bias}`; front-door `an unmeasured confounder shifts it by {signed bias}`; sign flip adds ` — it even gets the direction wrong`. Empty set: `No confounding to remove: the direct comparison is already the answer.`
- checks, in order:
  1. `Methods agree` / `Methods disagree` — text lists applicable estimates `adjustment formula +0.090 · g-computation +0.091 …`; disagree adds `the answer depends on modelling choices, usually because some groups have few comparable rows`.
  2. `Overlap` — ok when no dropped rows and no clipped weights: `every group has both treated and untreated rows`; else `{k} of {n} rows have no comparable rows in the other group and were left out` / `{k} rows have extreme treatment probabilities (weights clipped)`.
  3. `Placebo test` — ok: `shuffling the treatment makes the effect vanish, as it should`; not ok: `the effect survives a shuffled treatment: something in the data or DAG is off`.
  4. `Random common cause` (backdoor only) — ok: `adding a random variable to the adjustment leaves the answer stable`; not ok: `adding a random variable moves the answer: the estimate is fragile`.
  5. one `Data` check per data warning, ok=False, text = warning.

- [ ] **Step 1: failing tests** — onboarding headline starts with `If everyone had onboarding_email instead of nobody, retained_30d would rise by ` and contains `percentage points`; hrisim headline contains `among the 64 of 100 rows where both groups occur`; ads naive text contains `unmeasured`/`confounding`; numeric outcome headline contains `units of y`; labelled treatment (`channel` yes/no column) headline contains `= "yes" instead of "no"`; checks titles for onboarding `["Methods agree", "Overlap", "Placebo test", "Random common cause"]`; `service.handle("answer", ...)` round-trip.
- [ ] **Step 2:** run → FAIL (no module).
- [ ] **Step 3:** implement `answer.py` as specified; hook into payload, report ("Answer" section first, method label from `result.get("method") or analysis["method"]`), and `Result.report()`.
- [ ] **Step 4:** full suite PASS.
- [ ] **Step 5:** commit "Answer in words: headline, naive explanation, trust checks".

### Task 3: Right-hand Answer panel and simplified sidebar

**Files:**
- Create: `frontend/src/AnswerPanel.jsx` (tabs shell + Answer tab), `frontend/src/DataCheck.jsx`, `frontend/src/ExampleGuide.jsx`
- Modify: `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/engine.js` (add `answer`)

**Interfaces:**
- `AnswerPanel` props: `{ tab, setTab, tabs: string[], check, result, stale, busy, ready, onEstimate, onHide, saveName, setSaveName, saving, onSave, dataCheck: ReactNode|null, guide: ReactNode|null }`.
- `DataCheck` props: `{ source, dataInfo, profile, columnNames, onToggle }` (moved JSX, unchanged).
- `ExampleGuide` props: `{ guide, edges, onApply, onReset }` (moved JSX, unchanged).

- [ ] Steps:
  1. Move the data-check table and guide JSX into their components verbatim.
  2. `AnswerPanel` Answer tab: before result → identification summary + explanation list (from `check`), or "Choose a treatment and an outcome"; stale bar with **Estimate again**; busy line; result → `.headline`, `.technical`, "Compared as they are" block (naive value + `answer.naive`, sign-flip badge), "How much to trust it" checklist (✓/⚠ + title + text, methods table under the first check), "Depends on your DAG" `<details>` with explanation and other valid sets (comma lists), save row.
  3. App: remove method state/select (method sent as `"auto"`; `questionKey` without method); remove sidebar result section; button label **Estimate effect**; sidebar "why?" sets tab to Answer and opens panel; tabs = `["Answer", source.kind==="file" && "Data", guide && "Guide"]`; uploads open on Data tab, examples on Answer; after estimate switch to Answer.
  4. Saved analyses: store `method: result.method`; on load, if `result.answer` missing call `engine.answer({result, treatment, outcome})` and merge; old results without `comparison` render the checklist without the methods row (answer.py must tolerate missing keys).
  5. CSS: panel width 380px, headline 20px/1.35 weight 600, checks list, tabs bar; remove dead `.ates` sidebar styles only if unused.
  6. `npm run build` and `npm run smoke` green; commit "Answer panel: the result in words on the right, question on the left".

### Task 4: End-to-end verification and push

- [ ] Full pytest; `npm run build`; `npm run smoke`.
- [ ] Browser (dev server via launch.json "ateflow" or `vite preview`): each example → headline, checks, stale bar after a guide edit, tabs; upload a CSV → Data tab; save + reload an analysis; time an onboarding estimate vs before (~should stay within a few seconds).
- [ ] Update README screenshots text if it mentions "Estimate ATE" or the method selector.
- [ ] Push to master (allowed after green tests).
