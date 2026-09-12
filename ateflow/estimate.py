"""ATE estimators for a binary treatment.

Four routes, all with the same signature:
  - `g_computation`: linear outcome model + standardization over the sample
  - `adjustment_formula`: backdoor adjustment formula, exact within groups (discrete confounders only)
  - `ipw`: logistic propensity model + inverse probability weighting
  - `aipw`: doubly robust, outcome model plus propensity-weighted correction
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Estimate:
    value: float
    method: str
    adjustment_set: list[str]
    ci: tuple[float, float] | None = None
    n: int = 0
    diagnostics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        ci = f"  95% CI [{self.ci[0]:+.3f}, {self.ci[1]:+.3f}]" if self.ci else ""
        adj = ", ".join(self.adjustment_set) or "none"
        return f"{self.method:<16} ATE = {self.value:+.3f}{ci}   adjusting for: {adj}"


def _check_binary(series: pd.Series, name: str) -> np.ndarray:
    values = pd.unique(series.dropna())
    if not set(values) <= {0, 1, True, False}:
        raise ValueError(f"{name!r} must be binary 0/1, found values {sorted(values)[:5]}")
    return series.astype(float).to_numpy()


def _design_matrix(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Design matrix: numeric columns as they are, categorical ones as dummies."""
    if not covariates:
        return np.empty((len(df), 0))
    block = pd.get_dummies(df[covariates], drop_first=True, dtype=float)
    return block.to_numpy()


def naive(df: pd.DataFrame, treatment: str, outcome: str) -> Estimate:
    """Difference in means between treated and untreated. No adjustment."""
    t = _check_binary(df[treatment], treatment)
    y = df[outcome].astype(float).to_numpy()
    value = y[t == 1].mean() - y[t == 0].mean()
    return Estimate(value=float(value), method="naive", adjustment_set=[], n=len(df))


def g_computation(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
    interactions: bool = True,
) -> Estimate:
    """ATE by standardization: E[Y|T=1,Z] - E[Y|T=0,Z] averaged over the distribution of Z.

    The outcome model is linear in the covariates. With `interactions=True` it
    includes the T*Z terms, so it allows heterogeneous effects.
    """
    t = _check_binary(df[treatment], treatment)
    y = df[outcome].astype(float).to_numpy()
    z = _design_matrix(df, adjustment_set)
    y1, y0, residual_sd = _outcome_model(t, y, z, interactions)
    return Estimate(
        value=float((y1 - y0).mean()),
        method="g-computation",
        adjustment_set=list(adjustment_set),
        n=len(df),
        diagnostics={"residual_sd": residual_sd},
    )


def _outcome_model(
    t: np.ndarray, y: np.ndarray, z: np.ndarray, interactions: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Linear model of Y on T and Z: predictions under T=1 and T=0, residual sd."""
    n = len(t)

    def build(t_vec: np.ndarray) -> np.ndarray:
        cols = [np.ones(n), t_vec]
        if z.shape[1]:
            cols.append(z)
            if interactions:
                cols.append(z * t_vec[:, None])
        return np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])

    x = build(t)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(n - x.shape[1], 1)
    return (build(np.ones(n)) @ beta, build(np.zeros(n)) @ beta,
            float(np.sqrt((resid**2).sum() / dof)))


PROPENSITY_CLIP = 0.01


def _propensity(t: np.ndarray, z: np.ndarray, ridge: float = 1e-4, iters: int = 50) -> np.ndarray:
    """P(T=1 | Z) by logistic regression, fitted with Newton-Raphson.

    A tiny ridge keeps the fit finite under separation (a group with no
    treated units), where the unpenalized coefficients would diverge.
    """
    x = np.column_stack([np.ones(len(t)), z])
    beta = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        grad = x.T @ (t - p) - penalty @ beta
        hess = (x * (p * (1 - p))[:, None]).T @ x + penalty
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))


def _propensity_scores(
    df: pd.DataFrame, t: np.ndarray, covariates: list[str], max_levels: int = 10
) -> np.ndarray:
    """P(T=1 | Z) for every row.

    When every confounder is discrete the model is saturated: one probability
    per observed combination, which is just the treated share of that cell
    (the exact maximum-likelihood fit, computed in closed form). An additive
    logistic model would smooth over a cell with no treated units and hide the
    positivity violation that the adjustment formula reports. Continuous confounders
    go through the logistic regression.
    """
    if covariates and all(df[c].nunique(dropna=False) <= max_levels for c in covariates):
        cells = df.groupby(covariates, dropna=False, sort=False, observed=True).ngroup().to_numpy()
        n_cells = int(cells.max()) + 1
        if n_cells <= max(len(df) // 5, 2):
            treated = np.bincount(cells, weights=t, minlength=n_cells)
            sizes = np.bincount(cells, minlength=n_cells)
            return (treated / sizes)[cells]
    return _propensity(t, _design_matrix(df, covariates))


def _overlap_diagnostics(e_raw: np.ndarray, t: np.ndarray) -> dict:
    """What the weights look like: clipped propensities and effective sample size."""
    w = np.where(t == 1, 1 / e_raw.clip(PROPENSITY_CLIP, 1 - PROPENSITY_CLIP),
                 1 / (1 - e_raw.clip(PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)))
    return {
        "propensity_min": round(float(e_raw.min()), 4),
        "propensity_max": round(float(e_raw.max()), 4),
        "clipped": int(((e_raw < PROPENSITY_CLIP) | (e_raw > 1 - PROPENSITY_CLIP)).sum()),
        "effective_n": round(float(w.sum() ** 2 / (w**2).sum()), 1),
    }


def ipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> Estimate:
    """Inverse probability weighting: reweights each arm to look like the whole sample.

    Uses normalized (Hajek) weights. Propensities are clipped to
    [0.01, 0.99]; how many were clipped is reported, since clipping means
    the positivity assumption is shaky and the estimate leans on few rows.
    """
    t = _check_binary(df[treatment], treatment)
    y = df[outcome].astype(float).to_numpy()
    e_raw = _propensity_scores(df, t, adjustment_set)
    e = e_raw.clip(PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)
    w1, w0 = t / e, (1 - t) / (1 - e)
    if w1.sum() == 0 or w0.sum() == 0:
        raise ValueError("one treatment arm is empty")
    value = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
    return Estimate(
        value=float(value),
        method="ipw",
        adjustment_set=list(adjustment_set),
        n=len(df),
        diagnostics=_overlap_diagnostics(e_raw, t),
    )


def aipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> Estimate:
    """Augmented IPW (doubly robust): outcome model corrected by weighted residuals.

    Consistent if either the outcome model or the propensity model is right,
    so a misspecified linear outcome model no longer biases the estimate on
    its own.
    """
    t = _check_binary(df[treatment], treatment)
    y = df[outcome].astype(float).to_numpy()
    z = _design_matrix(df, adjustment_set)
    e_raw = _propensity_scores(df, t, adjustment_set)
    e = e_raw.clip(PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)
    m1, m0, _ = _outcome_model(t, y, z)
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return Estimate(
        value=float(psi.mean()),
        method="aipw",
        adjustment_set=list(adjustment_set),
        n=len(df),
        diagnostics=_overlap_diagnostics(e_raw, t),
    )


def adjustment_formula(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> Estimate:
    """Adjustment formula: average of the within-group effects, weighted by P(Z).

    Groups that do not contain both treatment arms are dropped and their rows
    counted in `diagnostics['dropped_rows']`: if there are many, positivity is weak.
    """
    t = _check_binary(df[treatment], treatment)
    work = df.assign(**{treatment: t})
    if not adjustment_set:
        return naive(work, treatment, outcome)

    total, weight_used, dropped = 0.0, 0.0, 0
    for _, group in work.groupby(adjustment_set, dropna=False, observed=True):
        arms = group[treatment].unique()
        if not {0.0, 1.0} <= set(arms):
            dropped += len(group)
            continue
        treated = group.loc[group[treatment] == 1, outcome].mean()
        control = group.loc[group[treatment] == 0, outcome].mean()
        w = len(group) / len(work)
        total += w * (treated - control)
        weight_used += w

    if weight_used == 0:
        raise ValueError("no group of confounder values contains both treatment arms: positivity violated")
    return Estimate(
        value=float(total / weight_used),
        method="adjustment-formula",
        adjustment_set=list(adjustment_set),
        n=len(work),
        diagnostics={
            "dropped_rows": dropped,
            "dropped_fraction": round(dropped / len(work), 4),
        },
    )


def bootstrap_ci(
    estimator,
    df: pd.DataFrame,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval over resampling with replacement."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        try:
            draws.append(estimator(df.iloc[idx]).value)
        except ValueError:
            continue
    if len(draws) < n_boot // 2:
        raise ValueError("too many degenerate resamples: sample or groups too small")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def refute_placebo_treatment(
    estimator, df: pd.DataFrame, treatment: str, n_sim: int = 50, seed: int = 0
) -> dict:
    """Permutes the treatment: a correct estimate must collapse towards zero."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_sim):
        shuffled = df.assign(**{treatment: rng.permutation(df[treatment].to_numpy())})
        try:
            values.append(estimator(shuffled).value)
        except ValueError:
            continue
    arr = np.asarray(values)
    return {
        "test": "placebo_treatment",
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "passed": bool(abs(arr.mean()) < 2 * arr.std(ddof=1) / np.sqrt(len(arr)) + 1e-9)
        or bool(abs(arr.mean()) < 0.02),
    }


def refute_random_common_cause(
    estimator_factory, df: pd.DataFrame, original: float, n_sim: int = 20, seed: int = 0
) -> dict:
    """Adds a random covariate to the adjustment: the estimate must stay stable.

    The covariate is binary, so the test also works for the adjustment formula: a
    continuous one would turn every group into a single case, violating
    positivity.
    """
    rng = np.random.default_rng(seed)
    values = []
    for i in range(n_sim):
        noisy = df.assign(_rcc=rng.integers(0, 2, size=len(df)))
        values.append(estimator_factory(noisy, ["_rcc"]).value)
    arr = np.asarray(values)
    shift = float(arr.mean() - original)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "test": "random_common_cause",
        # The systematic shift is the signal; any single resample may wander
        # with sampling noise without telling us anything.
        "shift": shift,
        "max_drift": float(np.abs(arr - original).max()),
        "passed": bool(abs(shift) < 2 * sd / np.sqrt(len(arr)) + 1e-9)
        or bool(abs(shift) < 0.02),
    }
