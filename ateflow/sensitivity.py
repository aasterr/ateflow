"""Sensitivity to a confounder missing from the DAG: the E-value.

VanderWeele & Ding (2017), "Sensitivity analysis in observational research:
introducing the E-value". The E-value is the minimum strength of association,
on the risk-ratio scale, that an unmeasured confounder would need with both the
treatment and the outcome, beyond the adjusted variables, to move the estimate
to no effect. It asks nothing about the confounder except its strength.

Numeric outcomes use the approximate conversion of a standardized mean
difference d to a risk ratio, RR = exp(0.91 d), from the same paper.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .estimate import Estimate, percentile_ci


def evalue(rr: float) -> float:
    """E-value of a risk ratio; protective ratios are inverted first."""
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))


def evalue_of_ci(rr: float, ci: tuple[float, float]) -> float:
    """E-value of the CI limit closest to no effect; 1 when the CI includes it."""
    lo, hi = ci
    if rr >= 1:
        return 1.0 if lo <= 1 else evalue(lo)
    return 1.0 if hi >= 1 else evalue(hi)


def _risk_ratio(e: Estimate) -> float | None:
    mu0 = e.mean_control
    if mu0 is None or mu0 <= 0 or mu0 + e.value <= 0:
        return None
    return (mu0 + e.value) / mu0


def assess(adjusted: Estimate, draws: list[Estimate], data: pd.DataFrame, outcome: str,
           binary: bool) -> dict | None:
    """E-values of a backdoor estimate, or None when a risk ratio cannot be formed.

    `draws` are the bootstrap estimates behind the CI (empty without bootstrap):
    for a binary outcome the CI of the risk ratio comes from the same resamples.
    """
    if binary:
        rr = _risk_ratio(adjusted)
        if rr is None:
            return None
        ratios = [r for r in map(_risk_ratio, draws) if r is not None]
        rr_ci = percentile_ci(ratios) if draws and len(ratios) >= len(draws) // 2 else None
    else:
        sd = float(data[outcome].std(ddof=1))
        if not sd > 0:
            return None
        rr = float(np.exp(0.91 * adjusted.value / sd))
        rr_ci = (tuple(float(np.exp(0.91 * v / sd)) for v in adjusted.ci)
                 if adjusted.ci else None)
    return {
        "rr": rr,
        "rr_ci": list(rr_ci) if rr_ci else None,
        "evalue": evalue(rr),
        "evalue_ci": evalue_of_ci(rr, rr_ci) if rr_ci else None,
        "approximate": not binary,
    }
