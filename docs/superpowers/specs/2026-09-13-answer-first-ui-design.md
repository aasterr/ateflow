# Answer-first UI — design

Date: 2026-09-13. Status: agreed with Francesco in chat (concept + layout C).

## Concept

ateflow answers "did X really cause Y?" from your CSV and your assumptions
written as a DAG, and shows how much the answer depends on those assumptions.
Everything runs in the browser; the data never leave the page.

Audience: someone with a CSV and basic statistics, not a causal-inference
expert. Every element on screen serves one of three promises:

1. **The answer, in words** — "If everyone received the email instead of
   nobody, retained_30d would rise by 9.0 percentage points (95% CI 6.1 to 11.8)."
2. **Why the direct comparison was wrong** — "Comparing the two groups as they
   are gives −6.9 points: plan and channel pull it down by 15.9."
3. **How much to trust it** — do the estimation methods agree, is there
   overlap, do the refutation tests pass, what the DAG assumes.

The user never picks an estimator or an adjustment set. ateflow derives them
from the DAG and the data, states them in the answer, and shows the other
estimators as a check.

What is estimated is always the total effect (ATE), by backdoor adjustment or
by front-door. When rows without a comparison group are dropped, the answer
says it is the effect on the remaining rows.

## Layout: three zones

- **Left — the question.** Header and privacy line; 1 · Data (example chips,
  open CSV); 2 · Question (treatment + coding, outcome + coding, one-line
  identification status); button **Estimate effect**; saved analyses. No
  method selector, no results.
- **Center — the assumptions.** The DAG canvas, unchanged.
- **Right — the answer panel**, always present once data are loaded, with tabs:
  - **Answer** (default): see below.
  - **Data** (uploaded CSV only): the existing data check table.
  - **Guide** (examples only): the existing story, why, tries.
  The panel can still be hidden to a side tab.

## Answer tab

Before any estimate: "What will be estimated" — the identification summary
and its explanation lines (currently in the "why?" details), or why the
effect is not identifiable. The sidebar "why?" link opens this tab.

After an estimate, top to bottom:

1. **Headline** (large): the sentence from `answer.headline`. If the CI
   crosses zero it adds "— but the data cannot tell whether the effect is
   positive or negative."
2. **Technical line** (small, muted): `ATE · adjusting for plan, channel ·
   adjustment formula · 6000 rows` (front-door: `through visited_site`).
   No set braces anywhere in the UI.
3. **Compared as they are**: naive value in the same unit and the sentence
   `answer.naive`; the sign-flip badge stays.
4. **How much to trust it**: a checklist, each row ✓ or ⚠ with one sentence:
   - methods agree / disagree, with a small table of every applicable
     estimator's point estimate (primary marked);
   - overlap: rows dropped for no comparison group, or weights clipped;
   - placebo treatment test; random common cause test (backdoor only);
   - data warnings from the data report.
5. **Depends on your DAG**: collapsible identification explanation, other
   valid adjustment sets, and for examples a link to the Guide tab.
6. Save row and, for saved analyses, the report link.

When the question or DAG changes after an estimate, the answer greys out and
a bar at the top says "The DAG or the question changed" with an
**Estimate again** button.

## Backend changes (Python, shared by CLI, server and Pyodide)

- `estimate_ate(method="auto")` becomes the default. Auto picks:
  backdoor → `adjustment-formula` if every confounder is discrete (or the set
  is empty), else `aipw`; front-door → `adjustment-formula` if every mediator
  is discrete, else `g-computation`. Explicit methods keep working (CLI,
  saved analyses, tests). `Result.method` records the method used.
- `Result.comparison`: list of `{method, value, applicable, reason}` for every
  estimator of the strategy, point estimates on the same prepared rows, no
  bootstrap. Not applicable = the estimator refuses the data (e.g. adjustment
  formula with a continuous confounder), with its message as `reason`.
- `Result.methods_agree`: true when every applicable comparison value lies
  inside the primary 95% CI (true when there is no CI or one method only).
- `data_report.outcome_binary`: true when the outcome was coded from two
  labels or its values are a subset of {0, 1}.
- New module `ateflow/answer.py`: pure functions from a result payload to
  sentences — `headline`, `technical`, `naive`, `checks` (list of
  `{ok, text}`). Units: binary outcome → "percentage points" (value × 100,
  one decimal); numeric outcome → "units of <outcome>" (3 significant digits).
  Treatment wording uses the coding labels when present
  (`channel = "email"` instead of `1`). Population: when overlap rows were
  dropped, "among the N rows where both groups occur". Included in the
  estimate payload as `answer` so the UI does not rebuild wording.
- `report.py` and the CLI print the same headline.

## Frontend structure

`App.jsx` (1100 lines) keeps state and the canvas; the right panel moves to
`AnswerPanel.jsx` (tabs + Answer tab), `DataCheck.jsx` and `ExampleGuide.jsx`.
The method selector and the sidebar result section are removed. Saved
analyses loaded from before this change (explicit method, no `answer`)
render: missing `answer` → the frontend asks the engine to rebuild it via a
new `answer` op.

## Out of scope

Mobile layout, instrumental variables, sensitivity analysis (E-value),
user-chosen estimators in the UI.

## Testing

- pytest: auto selection on each example (corridor/hrisim/onboarding →
  adjustment formula; a continuous-confounder case → AIPW; ads → front-door
  formula); comparison and agreement on onboarding (agree) and hrisim
  (disagree); answer sentences pinned for onboarding, ads, hrisim (dropped
  rows wording) and a numeric-outcome case; old explicit-method calls
  unchanged (existing tests stay green).
- `npm run build` and `npm run smoke` (engine under Pyodide).
- Browser check on each example: headline, checklist, stale bar, tabs, save
  and reload of a saved analysis; timing of an estimate compared with before.
