"""Standalone HTML report for a saved analysis: one file, print-friendly.

The DAG is drawn in Python as inline SVG with the same layered layout the
frontend uses (x by longest path from a root, y by order within the layer).
"""

from __future__ import annotations

import html
from datetime import datetime

from . import answer
from .answer import METHOD_LABELS
from .graph import DAG

_NODE_H = 34
_X_GAP = 170
_Y_GAP = 74
_PAD = 30


def _layout(dag: DAG) -> dict[str, tuple[float, float]]:
    depth = {n: 0 for n in dag.nodes}
    for _ in range(len(depth)):
        changed = False
        for src, dst in dag.edges:
            if depth[dst] < depth[src] + 1:
                depth[dst] = depth[src] + 1
                changed = True
        if not changed:
            break
    per_layer: dict[int, int] = {}
    pos = {}
    for name in sorted(dag.nodes, key=lambda n: (depth[n], n)):
        d = depth[name]
        i = per_layer.get(d, 0)
        per_layer[d] = i + 1
        pos[name] = (_PAD + d * _X_GAP, _PAD + i * _Y_GAP + (d % 2) * 26)
    return pos


def _node_w(name: str) -> float:
    return max(46, 18 + 9 * len(name))


def dag_svg(dag: DAG, treatment: str, outcome: str, adjustment: list[str]) -> str:
    pos = _layout(dag)
    width = max(x + _node_w(n) for n, (x, y) in pos.items()) + _PAD
    height = max(y for _, y in pos.values()) + _NODE_H + _PAD

    colors = {"treatment": "#4459d8", "outcome": "#1f7a4d", "adjust": "#b47708"}

    def role(name: str) -> str | None:
        if name == treatment:
            return "treatment"
        if name == outcome:
            return "outcome"
        if name in adjustment:
            return "adjust"
        return None

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}"'
        f' width="{width:.0f}" style="max-width:100%">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#9a9a94"/></marker></defs>',
    ]
    for src, dst in dag.edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        parts.append(
            f'<line x1="{x1 + _node_w(src):.0f}" y1="{y1 + _NODE_H / 2:.0f}"'
            f' x2="{x2 - 3:.0f}" y2="{y2 + _NODE_H / 2:.0f}"'
            ' stroke="#9a9a94" stroke-width="1.4" marker-end="url(#arrow)"/>'
        )
    for name, (x, y) in pos.items():
        w = _node_w(name)
        color = colors.get(role(name), "#c9c9c2")
        fill = {"adjust": "#fdf8ec"}.get(role(name), "#ffffff")
        dash = ' stroke-dasharray="5 4"' if name in dag.unmeasured else ""
        parts.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{_NODE_H}" rx="8"'
            f' fill="{fill}" stroke="{color}" stroke-width="1.6"{dash}/>'
            f'<text x="{x + w / 2:.0f}" y="{y + _NODE_H / 2 + 4.5:.0f}" text-anchor="middle"'
            f' font-family="Segoe UI, sans-serif" font-size="14" font-weight="600"'
            f' fill="{color if role(name) else "#1a1a1a"}">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_report(analysis: dict) -> str:
    result = analysis["result"]
    dag = DAG.parse(analysis["dag"])
    treatment, outcome = analysis["treatment"], analysis["outcome"]
    words = result.get("answer") or answer.build(result, treatment, outcome)
    method = result.get("method") or analysis["method"]
    adjustment = result["adjustment_set"]
    naive, adjusted = result["naive"], result["adjusted"]
    e = html.escape

    created = analysis["created_at"]
    try:
        created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        pass

    strategy = result.get("strategy", "backdoor")  # absent in analyses saved before front-door
    names = e(", ".join(adjustment))
    if strategy == "frontdoor":
        question_how = (f"identified by the <strong>front-door</strong> criterion through "
                        f"<strong>{names}</strong>: the confounder is unmeasured, the effect "
                        "is rebuilt through the mediators.")
        adjusted_label = f"front-door through {names}"
    elif adjustment:
        question_how = (f"adjusting for <strong>{names}</strong> "
                        "(minimal backdoor set identified from the DAG below).")
        adjusted_label = f"adjusted for {names}"
    else:
        question_how = "with no adjustment needed: nothing in the DAG below confounds the comparison."
        adjusted_label = "adjusted (nothing to adjust)"
    explanation = result.get("explanation") or []

    interpretation = []
    if result["sign_flip"]:
        interpretation.append(
            "The unadjusted estimate has the <strong>opposite sign</strong> of the "
            "adjusted one — a Simpson's paradox: reading the raw association would "
            "invert the conclusion."
        )
    interpretation.append(
        f"Confounding moves the naive estimate by "
        f"<strong>{result['confounding_bias']:+.3f}</strong> relative to the adjusted one."
    )
    dropped = adjusted["diagnostics"].get("dropped_rows", 0)
    if dropped:
        interpretation.append(
            f"<strong>{dropped} of {adjusted['n']} rows</strong> lie in groups of confounder values where the "
            "treatment never varies (positivity violation); they were excluded, not imputed."
        )
    data_report = result.get("data_report")  # absent in analyses saved before it existed
    if data_report:
        coding = data_report["treatment_coding"]
        treated = "" if coding["1"] in ("1", "1.0") else f" (treated = {e(coding['1'])!s})"
        interpretation.append(
            f"The estimate uses <strong>{data_report['rows_used']} of "
            f"{data_report['rows_in']} rows</strong>: {data_report['treated']} treated and "
            f"{data_report['control']} control{treated}."
        )
        interpretation.extend(f"Data warning: {e(w)}." for w in data_report["warnings"])
    clipped = adjusted["diagnostics"].get("clipped", 0)
    if clipped:
        interpretation.append(
            f"<strong>{clipped} of {adjusted['n']} rows</strong> have a propensity score "
            "below 1% or above 99% and were clipped (positivity is weak there): the "
            "weighted estimate leans on few comparable units."
        )
    if adjusted.get("ci"):
        lo, hi = adjusted["ci"]
        if lo < 0 < hi:
            interpretation.append(
                "The 95% bootstrap interval crosses zero: the direction of the effect "
                "is plausible but not statistically settled at this sample size."
            )

    refutation_rows = "".join(
        f"<tr><td>{e(r['test'].replace('_', ' '))}</td>"
        f"<td class='{'ok' if r['passed'] else 'bad'}'>{'passed' if r['passed'] else 'SUSPECT'}</td>"
        f"<td>{e(', '.join(f'{k}={v:+.4f}' for k, v in r.items() if isinstance(v, float)))}</td></tr>"
        for r in result["refutations"]
    )
    alternatives = (
        " — or ".join(", ".join(s) for s in result["alternatives"][:4])
        if result["alternatives"]
        else "none within one extra variable"
    )
    ci_text = (
        f"95% CI [{adjusted['ci'][0]:+.3f}, {adjusted['ci'][1]:+.3f}]"
        if adjusted.get("ci")
        else "no bootstrap interval"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light">
<title>{e(analysis['name'])} — ateflow report</title>
<style>
  body {{ font-family: "Segoe UI", system-ui, sans-serif; color: #1a1a1a;
         background: #ffffff; max-width: 760px; margin: 40px auto;
         padding: 0 20px; line-height: 1.5; }}
  h1 {{ margin-bottom: 2px; }} .meta {{ color: #6b6b6b; margin-top: 0; }}
  h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 1px;
        color: #6b6b6b; margin-top: 34px; border-bottom: 1px solid #e3e3de;
        padding-bottom: 4px; }}
  .ates {{ display: flex; gap: 28px; align-items: baseline; }}
  .ate .lbl {{ font-size: 13px; color: #6b6b6b; }}
  .ate .val {{ font-size: 34px; font-weight: 700; }}
  .naive .val {{ color: #6b6b6b; text-decoration: line-through; }}
  .adjusted .val {{ color: #1f7a4d; }}
  .ci {{ font-size: 13px; color: #6b6b6b; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  td {{ border-top: 1px solid #e3e3de; padding: 6px 10px 6px 0; }}
  .ok {{ color: #1f7a4d; }} .bad {{ color: #b3362b; }} .info {{ color: #4459d8; }}
  ul {{ padding-left: 20px; }} li {{ margin: 6px 0; }}
  .headline {{ font-size: 21px; font-weight: 600; line-height: 1.4; margin-bottom: 4px; }}
  .checks {{ list-style: none; padding-left: 0; }}
  footer {{ margin-top: 44px; color: #9a9a94; font-size: 12px;
            border-top: 1px solid #e3e3de; padding-top: 8px; }}
  @media print {{ body {{ margin: 0 auto; }} }}
</style>
</head>
<body>
<h1>{e(analysis['name'])}</h1>
<p class="meta">{created} · data: {e(analysis['source'])} · method: {e(METHOD_LABELS.get(method, method))}</p>

<h2>Answer</h2>
<p class="headline">{e(words['headline'])}</p>
<p class="meta">{e(words['technical'])}</p>
<p>{e(words['naive'])}</p>
<ul class="checks">{''.join(
    f"<li><span class='{'info' if c['ok'] is None else 'ok' if c['ok'] else 'bad'}'>{'ℹ' if c['ok'] is None else '✓' if c['ok'] else '⚠'}</span> <strong>{e(c['title'])}</strong> — {e(c['text'])}{f"<br><em class='bad'>{e(c['caveat'])}</em>" if c.get('caveat') else ""}</li>"
    for c in words['checks'])}</ul>

<h2>Question</h2>
<p>Effect of <strong>{e(treatment)}</strong> on <strong>{e(outcome)}</strong>,
{question_how}</p>

<h2>Identification</h2>
<ul>{''.join(f'<li>{e(line)}</li>' for line in explanation)}</ul>

<h2>Causal model</h2>
{dag_svg(dag, treatment, outcome, adjustment)}

<h2>Estimate</h2>
<div class="ates">
  <div class="ate naive"><div class="lbl">naive</div>
    <div class="val">{naive['value']:+.3f}</div></div>
  <div class="ate adjusted"><div class="lbl">{adjusted_label}</div>
    <div class="val">{adjusted['value']:+.3f}</div>
    <div class="ci">{ci_text}</div></div>
</div>

<h2>Reading</h2>
<ul>{''.join(f'<li>{p}</li>' for p in interpretation)}</ul>

<h2>Refutation tests</h2>
<table>{refutation_rows}</table>

<h2>Alternative sets</h2>
<p>{e(alternatives) if not result['alternatives'] else alternatives}</p>

<footer>Generated by ateflow — causal effect estimation from a declarative DAG.</footer>
</body>
</html>"""
