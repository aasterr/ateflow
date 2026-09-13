# E-value — design

Date: 2026-09-13. Serves promise 3 of the answer-first concept ("how much to
trust it"): the DAG may have missed a confounder; how strong would it have to
be to explain the answer away?

## What is computed

E-value (VanderWeele & Ding, 2017) on the risk-ratio scale:
`E(RR) = RR + sqrt(RR·(RR − 1))` for RR ≥ 1, and `E(1/RR)` for RR < 1. It is
the minimum strength of association, as a risk ratio, that a confounder
missing from the DAG would need with both the treatment and the outcome,
beyond the adjusted variables, to move the effect to no effect.

- **Binary outcome:** RR = E[Y | do(1)] / E[Y | do(0)], both from the primary
  estimator (each estimator now reports `mean_control`, so
  E[Y | do(1)] = mean_control + ATE). The CI of RR comes from the same
  bootstrap resamples as the ATE CI.
- **Numeric outcome:** approximate conversion of the standardized mean
  difference, RR ≈ exp(0.91 · ATE / sd(Y)), sd over the rows used; CI by
  transforming the ATE CI. Flagged as approximate.
- **E-value of the CI:** from the CI limit closest to no effect; 1 when the CI
  already includes no effect.
- **Front-door:** not computed. Front-door identification already allows an
  unmeasured treatment–outcome confounder; the E-value's question does not
  apply to it.
- RR undefined (a mean under do(·) ≤ 0 for a binary outcome): not computed.

## Where it shows

`Result.sensitivity` → payload `sensitivity`:
`{rr, rr_ci, evalue, evalue_ci, approximate}` or null. `answer.checks` adds a
final check titled **Hidden confounders** with `ok: null` (information, not
pass/fail: there is no universal threshold), e.g.:

> To explain this effect away, a confounder missing from the DAG would need to
> make both onboarding_email and retained_30d about 1.6× more likely, beyond
> channel and plan. To make the 95% CI reach no effect, 1.4× would do.

When the CI already includes no effect: "The 95% CI already includes no
effect, so even a weak confounder missing from the DAG could explain the
estimate away." The panel renders `ok: null` with a neutral "i" mark; the
HTML report likewise.

## Tests

Published numbers (RR 3.9, CI lower 1.8 → E-values 7.26 and 3.00), protective
RR, CI crossing null; `mean_control` of each estimator on a synthetic case
with known E[Y | do(0)]; payload on onboarding (binary), corridor
(approximate), ads (null); answer wording.
