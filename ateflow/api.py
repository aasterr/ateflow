"""Public surface of the library: one question, one answer.

    given a binary treatment and a DAG,
    how much does the outcome change net of confounders?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import column_kind, prepare
from .estimate import (
    Estimate,
    aipw,
    bootstrap_ci,
    frontdoor_formula,
    frontdoor_g_computation,
    g_computation,
    ipw,
    naive,
    refute_placebo_treatment,
    refute_random_common_cause,
    adjustment_formula,
)
from .graph import DAG

# every estimator of each identification strategy, in the order they are compared
ESTIMATORS = {
    "backdoor": {
        "adjustment-formula": adjustment_formula,
        "g-computation": g_computation,
        "ipw": ipw,
        "aipw": aipw,
    },
    "frontdoor": {
        "adjustment-formula": frontdoor_formula,
        "g-computation": frontdoor_g_computation,
    },
}


def choose_method(strategy: str, data: pd.DataFrame, variables: list[str]) -> str:
    """The estimator ateflow uses when the user does not pick one.

    The exact adjustment formula whenever the variables are discrete, since it
    assumes no model; with a continuous variable, AIPW for backdoor (right if
    either of its two models is) and g-computation for front-door.
    """
    rows = data[variables].dropna()
    if all(column_kind(rows[v]) != "continuous" for v in variables):
        return "adjustment-formula"
    return "aipw" if strategy == "backdoor" else "g-computation"


@dataclass
class Result:
    naive: Estimate
    adjusted: Estimate
    adjustment_set: list[str]
    alternatives: list[list[str]]
    refutations: list[dict]
    data_report: dict = field(default_factory=dict)
    strategy: str = "backdoor"
    explanation: list[str] = field(default_factory=list)
    method: str = ""
    # point estimates of every estimator of the strategy on the same rows
    comparison: list[dict] = field(default_factory=list)
    methods_agree: bool = True

    @property
    def confounding_bias(self) -> float:
        """How far confounding moves the unadjusted estimate."""
        return self.naive.value - self.adjusted.value

    @property
    def sign_flip(self) -> bool:
        """True if the adjustment flips the sign of the effect (Simpson)."""
        return self.naive.value * self.adjusted.value < 0

    def report(self) -> str:
        lines = []
        rep = self.data_report
        if rep:
            lines.append(f"rows used        : {rep['rows_used']} of {rep['rows_in']} "
                         f"({rep['treated']} treated, {rep['control']} control)")
            coding = rep["treatment_coding"]
            if coding["1"] != "1":
                lines.append(f"treated means   : {coding['1']!r} (control: {coding['0']!r})")
            for w in rep["warnings"]:
                lines.append(f"warning          : {w}")
            lines.append("")
        adjusted = f"{self.adjusted}"
        if self.strategy == "frontdoor":
            lines.append(f"identified by    : front-door through {{{', '.join(self.adjustment_set)}}}")
            adjusted = adjusted.replace("adjusting for:", "through:")
        lines += [
            f"{self.naive}",
            adjusted,
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
    method: str = "auto",
    adjustment_set: list[str] | None = None,
    n_boot: int = 500,
    refute: bool = True,
    seed: int = 0,
    treated_value: str | None = None,
    outcome_positive: str | None = None,
) -> Result:
    """Estimates the ATE, identifying from the DAG how it can be computed.

    `dag` accepts a DAG object or the text in 'A -> B' format directly.
    Identification tries backdoor adjustment first and front-door when an
    unmeasured confounder leaves no measured adjustment set; `Result.strategy`
    and `Result.explanation` say which and why. If `adjustment_set` is given,
    it is validated against the backdoor criterion instead of being searched for.

    The data go through `data.prepare` first: incomplete rows are dropped,
    a two-valued treatment is coded 0/1 (`treated_value` says which value is
    the treatment when the labels do not), and unusable variables are
    refused. What happened is in `Result.data_report`.
    """
    graph = dag if isinstance(dag, DAG) else DAG.parse(dag)

    missing = (graph.observed - set(data.columns)) - {"_rcc"}
    if missing:
        raise ValueError(
            f"DAG variables missing from the data: {sorted(missing)}. If they are real "
            "but not measured, declare them with a line 'unmeasured: name'"
        )

    if adjustment_set is None:
        ident = graph.identify(treatment, outcome)
        if ident["strategy"] is None:
            raise ValueError(" ".join(ident["explanation"]))
    else:
        for name in (treatment, outcome):
            if name not in graph.nodes:
                raise ValueError(f"{name!r} does not appear in the DAG")
        if not graph.satisfies_backdoor(treatment, outcome, adjustment_set):
            raise ValueError(
                f"{sorted(adjustment_set)} does not satisfy the backdoor criterion for "
                f"{treatment} -> {outcome}"
            )
        ident = {"strategy": "backdoor", "variables": sorted(adjustment_set), "alternatives": [],
                 "explanation": [f"Adjusting for the given set {{{', '.join(sorted(adjustment_set))}}}, "
                                 "which satisfies the backdoor criterion."]}
    strategy, chosen = ident["strategy"], ident["variables"]

    estimators = ESTIMATORS[strategy]
    if method == "auto":
        method = choose_method(strategy, data, chosen)
    if method not in estimators:
        if strategy == "frontdoor" and method in ("ipw", "aipw"):
            raise ValueError(
                f"{method.upper()} weights by the probability of treatment given confounders, "
                "and here the confounder is unmeasured. For this front-door question use the "
                "adjustment formula or g-computation"
            )
        raise ValueError(f"unknown method {method!r}, use one of {sorted(estimators)}")
    fn = estimators[method]

    raw = data
    data, data_report = prepare(
        data, treatment, outcome, chosen, method=method,
        treated_value=treated_value, outcome_positive=outcome_positive, strategy=strategy,
    )

    def run(frame: pd.DataFrame, extra: list[str] | None = None) -> Estimate:
        return fn(frame, treatment, outcome, chosen + list(extra or []))

    adjusted = run(data)
    if n_boot:
        adjusted.ci = bootstrap_ci(run, data, n_boot=n_boot, seed=seed)

    comparison = []
    for name, other in estimators.items():
        entry = {"method": name, "value": None, "applicable": True, "reason": None,
                 "primary": name == method}
        try:
            if name == method:
                entry["value"] = adjusted.value
            else:
                # the prepared rows are shared; only the adjustment formula adds data rules
                if name == "adjustment-formula":
                    prepare(raw, treatment, outcome, chosen, method=name,
                            treated_value=treated_value, outcome_positive=outcome_positive,
                            strategy=strategy)
                entry["value"] = other(data, treatment, outcome, chosen).value
        except ValueError as exc:
            entry.update(applicable=False, reason=str(exc))
        comparison.append(entry)
    values = [c["value"] for c in comparison if c["applicable"]]
    methods_agree = (adjusted.ci is None or len(values) < 2
                     or all(adjusted.ci[0] <= v <= adjusted.ci[1] for v in values))

    refutations: list[dict] = []
    if refute:
        refutations.append(refute_placebo_treatment(run, data, treatment, seed=seed))
        if strategy == "backdoor":  # an extra "confounder" only means something for adjustment
            refutations.append(
                refute_random_common_cause(
                    lambda frame, extra: run(frame, extra), data, adjusted.value, seed=seed
                )
            )

    return Result(
        naive=naive(data, treatment, outcome),
        adjusted=adjusted,
        adjustment_set=chosen,
        alternatives=[set(a) for a in ident["alternatives"]],
        refutations=refutations,
        data_report=data_report,
        strategy=strategy,
        explanation=ident["explanation"],
        method=method,
        comparison=comparison,
        methods_agree=methods_agree,
    )
