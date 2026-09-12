# ateflow

Causal effect estimation from a declarative DAG.

[![ci](https://github.com/aasterr/ateflow/actions/workflows/ci.yml/badge.svg)](https://github.com/aasterr/ateflow/actions/workflows/ci.yml)
[![live demo](https://img.shields.io/badge/demo-live-4459d8)](https://ateflow.onrender.com)
[![license: MIT](https://img.shields.io/badge/license-MIT-1f7a4d)](LICENSE)

**The question it answers, and only that one:** given a binary treatment and a DAG,
how much does the outcome change net of confounders?

Everything that does not serve this sentence stays out.

**[Try it →](https://ateflow.onrender.com)** — draw the DAG, get the estimate.
The demo runs on a free instance, so the first load may take a minute to wake
up and an estimate takes a few seconds.

![The DAG editor, with the naive and adjusted estimates side by side](docs/screenshot.png)

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
naive            ATE = -0.130   adjusting for: none
g-computation    ATE = +0.150  95% CI [+0.143, +0.157]   adjusting for: crowding

confounding bias : -0.280
sign flip        : YES
refutation placebo_treatment      ok        mean=-0.0000, sd=+0.0037
refutation random_common_cause    ok        shift=-0.0000, max_drift=+0.0002
```

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
naive            ATE = -0.069   adjusting for: none
g-computation    ATE = +0.090  95% CI [+0.067, +0.114]   adjusting for: channel, plan

confounding bias : -0.158
sign flip        : YES
```

The true effect is +9 retention points. The DAG also marks `active_week1` as
a mediator: ateflow refuses to adjust for it, because doing so would recover
only the direct +4 points and hide the half of the effect that works by
bringing users back in their first week.

## Validation on real data

`examples/episodes_100_v1.csv` holds the 100 HRI episodes of the
[PeopleFlow](https://github.com/aasterr/PeopleFlow/tree/main/analysis) dataset:
a robot traversing a corridor and deciding whether to emit an LED signal (`A`),
with success/timeout as the outcome (`T`) and static obstacles as the
confounder (`O`).

```bash
python -m ateflow --data examples/episodes_100_v1.csv --dag examples/hrisim.dag \
    --treatment A --outcome T --method stratification
```

ateflow reproduces the estimates of the reference thesis to the third decimal:
naive −0.207 (sign inverted by confounding), backdoor on `{O}` +0.061, backdoor
on `{Pi, O}` +0.108 on the 64 episodes with overlap — the stratum `Pi=0, O=0`
contains no treated episode and is dropped, not imputed.
`tests/test_hrisim.py` pins these numbers as a regression.

## Web app

A visual DAG editor over the same engine: load an example or upload a CSV,
draw the arrows, and the minimal adjustment set updates live as the graph
changes; one click runs the full estimate with bootstrap CI and refutations.

```bash
pip install -e ".[server]"
cd frontend && npm install && npm run build   # builds into ateflow/static
uvicorn ateflow.server:app
```

For development, run the API and `npm run dev` side by side: Vite proxies
`/api` to port 8000. The API alone (no built frontend) exposes
`/api/estimate`, `/api/dag/check`, `/api/columns`, `/api/examples` — docs at
`/docs`.

Analyses can be saved (dataset included, so they always reload) and exported
as a standalone HTML report with the DAG drawn inline — ready to print to
PDF. Storage is SQLite, stdlib only; set `ATEFLOW_DB` to move the file.

## Deploy

The Dockerfile builds the frontend and serves everything from one container:

```bash
docker build -t ateflow .
docker run -p 8080:8080 -v ateflow_data:/data ateflow
```

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aasterr/ateflow)

`render.yaml` deploys the demo to Render's free tier (the instance sleeps
when idle and its disk is ephemeral). `fly.toml` is included for Fly.io,
where a mounted volume makes saved analyses persistent — see the comments
at its top.

## What it does today

- DAG with acyclicity checking and chain parsing (`a -> b -> c`)
- d-separation by moralization of the ancestral subgraph
- Pearl's backdoor criterion, search for the minimal set and the alternative sets
- explicit refusal of mediators, colliders, and descendants of the treatment
- estimation by g-computation (with T*Z interactions) and by stratification
- percentile bootstrap intervals
- refutation tests: placebo treatment, random common cause

## What it does not do (by choice)

Non-binary treatments, longitudinal data, front-door, instrumental variables,
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
