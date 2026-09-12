"""Stimatori dell'ATE per trattamento binario.

Due strade, entrambe con la stessa firma:
  - `g_computation`: modello lineare dell'esito + standardizzazione sul campione
  - `stratification`: formula di aggiustamento a strati (solo confonditori discreti)
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
        ci = f"  IC95% [{self.ci[0]:+.3f}, {self.ci[1]:+.3f}]" if self.ci else ""
        adj = ", ".join(self.adjustment_set) or "nessuno"
        return f"{self.method:<16} ATE = {self.value:+.3f}{ci}   aggiusto per: {adj}"


def _check_binary(series: pd.Series, name: str) -> np.ndarray:
    values = pd.unique(series.dropna())
    if not set(values) <= {0, 1, True, False}:
        raise ValueError(f"{name!r} deve essere binaria 0/1, trovati valori {sorted(values)[:5]}")
    return series.astype(float).to_numpy()


def _design_matrix(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Matrice di disegno: numeriche così come sono, categoriche con dummy."""
    if not covariates:
        return np.empty((len(df), 0))
    block = pd.get_dummies(df[covariates], drop_first=True, dtype=float)
    return block.to_numpy()


def naive(df: pd.DataFrame, treatment: str, outcome: str) -> Estimate:
    """Differenza di medie fra trattati e non trattati. Nessun aggiustamento."""
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
    """ATE per standardizzazione: E[Y|T=1,Z] - E[Y|T=0,Z] mediato sulla distribuzione di Z.

    Il modello dell'esito è lineare nei covariati. Con `interactions=True` include
    i termini T*Z, quindi ammette effetti eterogenei.
    """
    t = _check_binary(df[treatment], treatment)
    y = df[outcome].astype(float).to_numpy()
    z = _design_matrix(df, adjustment_set)

    def build(t_vec: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(df)), t_vec]
        if z.shape[1]:
            cols.append(z)
            if interactions:
                cols.append(z * t_vec[:, None])
        return np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])

    x = build(t)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y1 = build(np.ones(len(df))) @ beta
    y0 = build(np.zeros(len(df))) @ beta
    resid = y - x @ beta
    dof = max(len(df) - x.shape[1], 1)
    return Estimate(
        value=float((y1 - y0).mean()),
        method="g-computation",
        adjustment_set=list(adjustment_set),
        n=len(df),
        diagnostics={"residual_sd": float(np.sqrt((resid**2).sum() / dof))},
    )


def stratification(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> Estimate:
    """Formula di aggiustamento: media degli effetti di strato, pesata per P(Z).

    Gli strati senza entrambi i bracci di trattamento vengono scartati e riportati
    in `diagnostics['dropped_strata']`: se sono molti, la positività è debole.
    """
    t = _check_binary(df[treatment], treatment)
    work = df.assign(**{treatment: t})
    if not adjustment_set:
        return naive(work, treatment, outcome)

    total, weight_used, dropped = 0.0, 0.0, 0
    for _, stratum in work.groupby(adjustment_set, dropna=False, observed=True):
        arms = stratum[treatment].unique()
        if not {0.0, 1.0} <= set(arms):
            dropped += len(stratum)
            continue
        treated = stratum.loc[stratum[treatment] == 1, outcome].mean()
        control = stratum.loc[stratum[treatment] == 0, outcome].mean()
        w = len(stratum) / len(work)
        total += w * (treated - control)
        weight_used += w

    if weight_used == 0:
        raise ValueError("nessuno strato contiene entrambi i bracci: positività violata")
    return Estimate(
        value=float(total / weight_used),
        method="stratification",
        adjustment_set=list(adjustment_set),
        n=len(work),
        diagnostics={
            "dropped_strata": dropped,
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
    """Intervallo percentile su ricampionamento con reimmissione."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        try:
            draws.append(estimator(df.iloc[idx]).value)
        except ValueError:
            continue
    if len(draws) < n_boot // 2:
        raise ValueError("troppi ricampionamenti degeneri: campione o strati insufficienti")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def refute_placebo_treatment(
    estimator, df: pd.DataFrame, treatment: str, n_sim: int = 50, seed: int = 0
) -> dict:
    """Permuta il trattamento: una stima corretta deve collassare verso zero."""
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
    """Aggiunge un covariato casuale all'aggiustamento: la stima deve restare stabile.

    Il covariato è binario, così il test vale anche per la stratificazione:
    uno continuo renderebbe ogni strato un singolo caso, violando la positività.
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
        # Lo spostamento sistematico è il segnale; il singolo ricampionamento
        # può oscillare quanto vuole il rumore campionario senza dire nulla.
        "shift": shift,
        "max_drift": float(np.abs(arr - original).max()),
        "passed": bool(abs(shift) < 2 * sd / np.sqrt(len(arr)) + 1e-9)
        or bool(abs(shift) < 0.02),
    }
