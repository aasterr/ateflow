"""Public surface of the library: one question, one answer.

    given a binary treatment and a DAG,
    how much does the outcome change net of confounders?
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .estimate import (
    Estimate,
    aipw,
    bootstrap_ci,
    g_computation,
    ipw,
    naive,
    refute_placebo_treatment,
    refute_random_common_cause,
    stratification,
)
from .graph import DAG


@dataclass
class Result:
    naive: Estimate
    adjusted: Estimate
    adjustment_set: list[str]
    alternatives: list[list[str]]
    refutations: list[dict]

    @property
    def confounding_bias(self) -> float:
        """How far confounding moves the unadjusted estimate."""
        return self.naive.value - self.adjusted.value

    @property
    def sign_flip(self) -> bool:
        """True if the adjustment flips the sign of the effect (Simpson)."""
        return self.naive.value * self.adjusted.value < 0

    def report(self) -> str:
        lines = [
            f"{self.naive}",
            f"{self.adjusted}",
            "",
            f"confounding bias : {self.confounding_bias:+.3f}",
            f"sign flip        : {'YES' if self.sign_flip else 'no'}",
        ]
        if self.alternatives:
            alts = " | ".join("{" + ", ".join(sorted(s)) + "}" for s in self.alternatives[:4])
            lines.append(f"other valid sets : {alts}")
        for ref in self.refutations:
            flag = "ok" if ref["passed"] else "SUSPECT"
            detail = ", ".join(
                f"{k}={v:+.4f}" for k, v in ref.items() if isinstance(v, float)
            )
            lines.append(f"refutation {ref['test']:<22} {flag:<9} {detail}")
        return "\n".join(lines)


def estimate_ate(
    data: pd.DataFrame,
    dag: DAG | str,
    treatment: str,
    outcome: str,
    method: str = "g-computation",
    adjustment_set: list[str] | None = None,
    n_boot: int = 500,
    refute: bool = True,
    seed: int = 0,
) -> Result:
    """Estimates the ATE, identifying the adjustment set from the DAG.

    `dag` accepts a DAG object or the text in 'A -> B' format directly.
    If `adjustment_set` is given, it is validated against the backdoor
    criterion instead of being searched for.
    """
    graph = dag if isinstance(dag, DAG) else DAG.parse(dag)

    missing = (graph.nodes - set(data.columns)) - {"_rcc"}
    if missing:
        raise ValueError(f"DAG variables missing from the data: {sorted(missing)}")
    for name in (treatment, outcome):
        if name not in graph.nodes:
            raise ValueError(f"{name!r} does not appear in the DAG")

    if adjustment_set is None:
        chosen = sorted(graph.minimal_backdoor_set(treatment, outcome))
    else:
        if not graph.satisfies_backdoor(treatment, outcome, adjustment_set):
            raise ValueError(
                f"{sorted(adjustment_set)} does not satisfy the backdoor criterion for "
                f"{treatment} -> {outcome}"
            )
        chosen = sorted(adjustment_set)

    estimators = {
        "g-computation": g_computation,
        "stratification": stratification,
        "ipw": ipw,
        "aipw": aipw,
    }
    if method not in estimators:
        raise ValueError(f"unknown method {method!r}, use one of {sorted(estimators)}")
    fn = estimators[method]

    def run(frame: pd.DataFrame, extra: list[str] | None = None) -> Estimate:
        return fn(frame, treatment, outcome, chosen + list(extra or []))

    adjusted = run(data)
    if n_boot:
        adjusted.ci = bootstrap_ci(run, data, n_boot=n_boot, seed=seed)

    refutations: list[dict] = []
    if refute:
        refutations.append(refute_placebo_treatment(run, data, treatment, seed=seed))
        refutations.append(
            refute_random_common_cause(
                lambda frame, extra: run(frame, extra), data, adjusted.value, seed=seed
            )
        )

    alternatives = [
        s for s in graph.backdoor_sets(treatment, outcome, max_size=len(chosen) + 1)
        if sorted(s) != chosen
    ]
    return Result(
        naive=naive(data, treatment, outcome),
        adjusted=adjusted,
        adjustment_set=chosen,
        alternatives=alternatives,
        refutations=refutations,
    )
