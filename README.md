# ateflow

Causal effect estimation from a declarative DAG.

[![ci](https://github.com/aasterr/ateflow/actions/workflows/ci.yml/badge.svg)](https://github.com/aasterr/ateflow/actions/workflows/ci.yml)
[![live demo](https://img.shields.io/badge/demo-live-4459d8)](https://aasterr.github.io/ateflow/)
[![license: MIT](https://img.shields.io/badge/license-MIT-1f7a4d)](LICENSE)

**The question it answers, and only that one:** given a binary treatment and a DAG,
how much does the outcome change net of confounders?

Everything that does not serve this sentence stays out.

**[Try it →](https://aasterr.github.io/ateflow/)** — open a CSV, draw the DAG, read the answer in words.
**Everything runs in your browser**: the engine is the same Python package,
compiled to WebAssembly with [Pyodide](https://pyodide.org), so the file you
open is never uploaded anywhere. The first visit downloads Python, numpy and
pandas (a few seconds, then cached); an estimate with 500 bootstrap resamples
and the refutation tests takes 2–5 seconds.

![The question on the left, the DAG in the middle, the answer in words on the right](docs/screenshot.png)

## Usage

The DAG is a text file, one relation per line:

```
crowding -> led
crowding -> speed
led -> speed
led -> waiting_time
```

From the command line:

```bash
python -m ateflow --data examples/corridor.csv --dag examples/corridor.dag \
    --treatment led --outcome speed
```

```
Setting led to 1 for everyone, instead of 0, would raise the average speed by 0.151 (95% CI +0.143 to +0.159).
Comparing the two groups as they are gives −0.130 instead: crowding pulls it down by 0.281 — it even gets the direction wrong.

rows used        : 4000 of 4000 (1935 treated, 2065 control)

naive              ATE = -0.130   adjusting for: none
adjustment-formula ATE = +0.151  95% CI [+0.143, +0.159]   adjusting for: crowding

confounding bias : -0.281
sign flip        : YES
cross-check      : adjustment-formula +0.151 · g-computation +0.150 · ipw +0.151 · aipw +0.151 (agree)
refutation placebo_treatment      ok        mean=-0.0000, sd=+0.0037
refutation random_common_cause    ok        shift=+0.0000, max_drift=+0.0002
```

You never pick the estimator: ateflow uses the exact adjustment formula when
the confounders are discrete, AIPW (doubly robust) when one is continuous, and
runs every other estimator on the same rows as a cross-check. If they do not
all fall inside the 95% CI, the answer says so.

The true ATE of the synthetic dataset is +0.15. The unadjusted estimate has the
wrong sign: `crowding` turns the LED on more often and lowers speed at the same
time.

From Python:

```python
from ateflow import DAG, estimate_ate

res = estimate_ate(df, DAG.from_file("examples/corridor.dag"),
                   treatment="led", outcome="speed")
print(res.report(), res.sign_flip, res.confounding_bias)
```

## Product analytics example

Did the onboarding email raise 30-day retention? The growth team sent it
mostly to the users it feared would churn — free plan, paid-ads signups — so
the raw comparison says the email *hurts*:

```bash
python -m ateflow --data examples/onboarding.csv --dag examples/onboarding.dag \
    --treatment onboarding_email --outcome retained_30d
```

```
Setting onboarding_email to 1 for everyone, instead of 0, would raise the share with retained_30d = 1 by 8.6 percentage points (95% CI +6.1 to +11.0).
Comparing the two groups as they are gives −6.9 points instead: channel and plan pull it down by 15.5 points — it even gets the direction wrong.
```

The true effect is +9 retention points. The DAG also marks `active_week1` as
a mediator: ateflow refuses to adjust for it, because doing so would recover
only the direct +4 points and hide the half of the effect that works by
bringing users back in their first week.

## Unmeasured confounders: front-door

Did seeing the ad make people buy? Ad targeting reaches people who were
shopping anyway, and that purchase intent is not in the data. No adjustment
can remove it, so ateflow looks for a front-door: the ad works only through a
site visit, and nothing else decides the visit.

```
intent -> saw_ad
intent -> purchased
unmeasured: intent
saw_ad -> visited_site -> purchased
```

```bash
python -m ateflow --data examples/ads.csv --dag examples/ads.dag \
    --treatment saw_ad --outcome purchased
```

```
Setting saw_ad to 1 for everyone, instead of 0, would raise the share with purchased = 1 by 15.3 percentage points (95% CI +13.4 to +17.4).
Comparing the two groups as they are gives +42.2 points instead: an unmeasured confounder pushes it up by 26.9 points.
```

The true effect is +0.15; the naive comparison nearly triples it. Identification
always tries backdoor adjustment first, falls back to front-door, and otherwise
refuses with the path that makes the effect unidentifiable — in the app, the
"why?" link next to the identification line opens that explanation in the Answer panel.

## Validation on real data

`examples/episodes_100_v1.csv` holds the 100 HRI episodes of the
[PeopleFlow](https://github.com/aasterr/PeopleFlow/tree/main/analysis) dataset:
a robot traversing a corridor and deciding whether to emit an LED signal (`A`),
with success/timeout as the outcome (`T`) and static obstacles as the
confounder (`O`).

```bash
python -m ateflow --data examples/episodes_100_v1.csv --dag examples/hrisim.dag \
    --treatment A --outcome T
```

ateflow reproduces the estimates of the reference thesis to the third decimal:
naive −0.207 (sign inverted by confounding), backdoor on `O` +0.061, backdoor
on `Pi, O` +0.108 on the 64 episodes with overlap — the group `Pi=0, O=0`
contains no treated episode and is dropped, not imputed.
`tests/test_hrisim.py` pins these numbers as a regression.

## Web app

A visual DAG editor over the same engine: load an example or open a CSV,
draw the arrows, and the minimal adjustment set updates live as the graph
changes; one click runs the full estimate with bootstrap CI and refutations.

The app is static. `ateflow/service.py` holds the request logic; in the page
it runs inside a Web Worker on Pyodide, and saved analyses (dataset included)
stay in the browser's IndexedDB. The same module backs the optional FastAPI
server, so the two front doors cannot drift apart.

```bash
python examples/make_data.py                  # example datasets bundled into the build
cd frontend && npm install && npm run build   # builds into ateflow/static
npm run smoke                                 # the engine under Pyodide in Node, checked against pinned numbers
```

Serve `ateflow/static` with any static file server, or with the API:
`pip install -e ".[server]" && uvicorn ateflow.server:app`.

### Bringing your own CSV

Open the file as it comes out of your tool. ateflow detects the delimiter
(comma, semicolon, tab, pipe), a decimal comma and Windows encodings, so an
Excel export with Italian locale reads as-is. A data check then shows every
column with what it looks like — two values, categories, numeric, an ID,
free text, all missing — and leaves IDs and empty columns out of the DAG.

When a question is asked:

- the treatment must have exactly two values. `0/1`, `true/false`,
  `yes/no`, `sì/no`, `treated/control` are coded automatically; for any other
  pair (`1/2`, `A/B`) you choose which one is the treatment
- the outcome must be numeric or have two values (you choose which counts as 1)
- rows with a missing value in the variables the question uses are dropped;
  how many, and because of which column, is reported with the result
- an ID or free-text column cannot be adjusted for, and the adjustment formula refuses
  continuous confounders — the error says what to do instead

The same checks run from the CLI (`--treated-value`, `--outcome-positive`)
and from Python, where `Result.data_report` holds the details. Uploads are
limited to 20 MB.

Analyses can be saved (dataset included, so they always reload) and exported
as a standalone HTML report with the DAG drawn inline — ready to print to PDF.

For development, `npm run dev` serves the app with hot reload. The HTTP API
(`uvicorn ateflow.server:app`) exposes `/api/estimate`, `/api/dag/check`,
`/api/columns`, `/api/examples` and SQLite-backed `/api/analyses` — docs at
`/docs`; set `ATEFLOW_DB` to move the database file.

## Deploy

The public demo is published to GitHub Pages by `.github/workflows/pages.yml`
on every push to `master`, after the Pyodide smoke test passes.

To self-host, the Dockerfile builds the same static app and serves it with the
API from one container:

```bash
docker build -t ateflow .
docker run -p 8080:8080 -v ateflow_data:/data ateflow
```

## What it does today

- DAG with acyclicity checking and chain parsing (`a -> b -> c`)
- d-separation by moralization of the ancestral subgraph
- Pearl's backdoor criterion, search for the minimal set and the alternative sets,
  restricted to the ancestors of treatment and outcome so wide datasets stay fast
- unmeasured variables (`unmeasured: name`) and Pearl's front-door criterion,
  estimated with the front-door formula or a linear two-step model
- a plain-language explanation of the identification: which paths are
  blocked and where, or why the effect cannot be identified
- explicit refusal of mediators, colliders, and descendants of the treatment
- estimation by g-computation (with T*Z interactions) and by the adjustment formula (exact within groups of confounder values)
- inverse probability weighting and doubly robust AIPW, with overlap
  diagnostics: clipped propensities and effective sample size. With discrete
  confounders the propensity model is saturated, so a cell with no treated
  units is flagged instead of being smoothed over
- percentile bootstrap intervals
- refutation tests: placebo treatment, random common cause

## What it does not do (by choice)

Non-binary treatments, longitudinal data, instrumental variables,
causal discovery, heterogeneous effects by subgroup. They are extensions, not
requirements.

## Development

```bash
pip install -e ".[dev]"
python examples/make_data.py
pytest -q
```

`tests/test_estimate.py` is the regression test of the whole project: if a
refactoring breaks the recovery of the known ATE, the pipeline is broken.
