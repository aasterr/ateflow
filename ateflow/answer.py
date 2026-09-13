"""The result of an estimate, in words.

Pure functions of the estimate payload (`service.result_payload`), so the web
app, the HTML report and the CLI say the same sentences. They tolerate
payloads saved by older versions, which lack `comparison`, `method` or
`data_report["outcome_binary"]`.
"""

from __future__ import annotations

METHOD_LABELS = {
    "adjustment-formula": "adjustment formula",
    "g-computation": "g-computation",
    "ipw": "IPW",
    "aipw": "AIPW",
}

MINUS = "−"

APPROXIMATE_EVALUE = (
    "The outcome is a number, not yes/no, so this E-value comes from a rule of thumb: the effect "
    "is converted to a risk ratio from its size in standard deviations. Read it as an order of "
    "magnitude, not as an exact threshold."
)


def _signed(text: str) -> str:
    return text.replace("-", MINUS) if text.startswith("-") else text


class _Scale:
    """How values of this outcome are written: percentage points or plain numbers."""

    def __init__(self, binary: bool):
        self.binary = binary

    def size(self, v: float) -> str:
        return f"{abs(v) * 100:.1f} percentage points" if self.binary else _digits(f"{abs(v):#.3g}")

    def signed(self, v: float, unit: bool = True) -> str:
        if self.binary:
            return _signed(f"{v * 100:+.1f}") + (" points" if unit else "")
        return _signed(_digits(f"{v:+#.3g}"))


def _digits(text: str) -> str:
    """Three significant digits keeping trailing zeros (0.150), without a bare final dot (151.)."""
    return text.rstrip(".") if "e" not in text else text


def _is_binary(payload: dict) -> bool:
    rep = payload.get("data_report") or {}
    if "outcome_binary" in rep:
        return bool(rep["outcome_binary"])
    return rep.get("outcome_coding") is not None


def _labels(coding: dict | None) -> tuple[str, str] | None:
    """The two labels of a coded variable, or None when it was already 0/1."""
    if not coding or {coding["1"], coding["0"]} <= {"1", "0", "1.0", "0.0", "True", "False"}:
        return None
    return coding["1"], coding["0"]


def _names(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def headline(payload: dict, treatment: str, outcome: str) -> str:
    rep = payload.get("data_report") or {}
    adjusted = payload["adjusted"]
    v = adjusted["value"]
    scale = _Scale(_is_binary(payload))

    t_labels = _labels(rep.get("treatment_coding"))
    setting = (f'setting {treatment} to "{t_labels[0]}" for everyone, instead of "{t_labels[1]}",'
               if t_labels else f"setting {treatment} to 1 for everyone, instead of 0,")
    if scale.binary:
        o_labels = _labels(rep.get("outcome_coding"))
        target = f'the share with {outcome} = "{o_labels[0]}"' if o_labels else f"the share with {outcome} = 1"
    else:
        target = f"the average {outcome}"

    if abs(v) < 0.0005:
        change = f"would leave {target} unchanged"
    else:
        change = f"would {'raise' if v > 0 else 'lower'} {target} by {scale.size(v)}"
    ci = adjusted.get("ci")
    if ci:
        change += f" (95% CI {scale.signed(ci[0], unit=False)} to {scale.signed(ci[1], unit=False)})"

    dropped = adjusted.get("diagnostics", {}).get("dropped_rows", 0)
    if dropped:
        n = adjusted["n"]
        text = f"Among the {n - dropped} of {n} rows where both groups occur, {setting} {change}."
    else:
        text = f"{setting[0].upper()}{setting[1:]} {change}."
    if ci and ci[0] < 0 < ci[1]:
        text += " The data cannot tell whether the effect is positive or negative."
    return text


def technical(payload: dict) -> str:
    adjusted = payload["adjusted"]
    variables = payload.get("adjustment_set") or []
    if payload.get("strategy") == "frontdoor":
        how = f"through {', '.join(variables)}"
    elif variables:
        how = f"adjusting for {', '.join(variables)}"
    else:
        how = "no adjustment needed"
    method = payload.get("method") or adjusted.get("method", "")
    rows = (payload.get("data_report") or {}).get("rows_used", adjusted.get("n"))
    return f"ATE · {how} · {METHOD_LABELS.get(method, method)} · {rows} rows"


def naive(payload: dict) -> str:
    variables = payload.get("adjustment_set") or []
    frontdoor = payload.get("strategy") == "frontdoor"
    if not variables and not frontdoor:
        return "Nothing confounds this comparison: the two groups as they are already give the answer."
    scale = _Scale(_is_binary(payload))
    bias = payload["confounding_bias"]
    who = "an unmeasured confounder" if frontdoor else _names(variables)
    singular = frontdoor or len(variables) == 1
    verb = ("pushes it up" if bias > 0 else "pulls it down") if singular else \
           ("push it up" if bias > 0 else "pull it down")
    points = " points" if scale.binary else ""
    text = (f"Comparing the two groups as they are gives {scale.signed(payload['naive']['value'])} "
            f"instead: {who} {verb} by {scale.size(bias).replace(' percentage points', '')}{points}")
    if payload.get("sign_flip"):
        text += " — it even gets the direction wrong"
    return text + "."


def checks(payload: dict, treatment: str = "the treatment", outcome: str = "the outcome") -> list[dict]:
    scale = _Scale(_is_binary(payload))
    adjusted = payload["adjusted"]
    out = []

    comparison = payload.get("comparison") or []
    applicable = [c for c in comparison if c["applicable"]]
    if len(applicable) >= 2:
        agree = payload.get("methods_agree", True)
        listed = " · ".join(
            f"{METHOD_LABELS.get(c['method'], c['method'])} {scale.signed(c['value'], unit=False)}"
            for c in applicable)
        text = f"{listed}{' points' if scale.binary else ''}."
        text += (" All fall inside the 95% CI." if agree else
                 " Not all fall inside the 95% CI: the answer depends on modelling choices, "
                 "usually because some groups have few comparable rows.")
        out.append({"ok": agree, "title": "Methods agree" if agree else "Methods disagree", "text": text})

    diag = adjusted.get("diagnostics", {})
    dropped, clipped, n = diag.get("dropped_rows", 0), diag.get("clipped", 0), adjusted["n"]
    if dropped:
        out.append({"ok": False, "title": "Overlap",
                    "text": f"{dropped} of {n} rows have no comparable rows in the other group "
                            f"and were left out: the answer is about the other {n - dropped}."})
    elif clipped:
        out.append({"ok": False, "title": "Overlap",
                    "text": f"{clipped} of {n} rows are almost certain to be treated or untreated; "
                            "their weights were capped and the estimate leans on few rows."})
    else:
        out.append({"ok": True, "title": "Overlap",
                    "text": "Every group has both treated and untreated rows to compare."})

    wording = {
        "placebo_treatment": ("Placebo test",
                              "Shuffling the treatment at random makes the effect vanish, as it should.",
                              "The effect survives a shuffled treatment: something in the data or "
                              "the DAG is off."),
        "random_common_cause": ("Random common cause",
                                "Adding a random variable to the adjustment leaves the answer stable.",
                                "Adding a random variable to the adjustment moves the answer: "
                                "the estimate is fragile."),
    }
    for r in payload.get("refutations") or []:
        if r["test"] in wording:
            title, good, bad = wording[r["test"]]
            out.append({"ok": bool(r["passed"]), "title": title, "text": good if r["passed"] else bad})

    for w in (payload.get("data_report") or {}).get("warnings", []):
        out.append({"ok": False, "title": "Data", "text": w[0].upper() + w[1:] + "."})

    if sens := payload.get("sensitivity"):
        check = {"ok": None, "title": "Hidden confounders", "text": _hidden(payload, sens, treatment, outcome)}
        if sens.get("approximate"):
            # shown apart from the sentence: a bare number reads as exact
            check["title"] = "Hidden confounders (approximate)"
            check["caveat"] = APPROXIMATE_EVALUE
        out.append(check)
    return out


def _hidden(payload: dict, sens: dict, treatment: str, outcome: str) -> str:
    """The E-values in words. Information, not pass/fail: no strength is safe in general."""
    variables = payload.get("adjustment_set") or []
    beyond = f", beyond {_names(variables)}" if variables else ""
    if sens.get("approximate"):
        text = (f"To explain this effect away, a confounder missing from the DAG would need a risk "
                f"ratio of roughly {sens['evalue']:.1f} with both {treatment} and {outcome}{beyond}.")
    else:
        text = (f"To explain this effect away, a confounder missing from the DAG would need to make "
                f"both {treatment} and {outcome} about {sens['evalue']:.1f}× more likely{beyond}.")
    if sens.get("evalue_ci") is not None:
        if sens["evalue_ci"] <= 1:
            text += (" The 95% CI already includes no effect, so even a weak one could "
                     "explain the estimate away.")
        else:
            unit = "" if sens.get("approximate") else "×"
            text += f" To make the 95% CI reach no effect, {sens['evalue_ci']:.1f}{unit} would do."
    return text


def build(payload: dict, treatment: str, outcome: str) -> dict:
    return {
        "headline": headline(payload, treatment, outcome),
        "technical": technical(payload),
        "naive": naive(payload),
        "checks": checks(payload, treatment, outcome),
    }
