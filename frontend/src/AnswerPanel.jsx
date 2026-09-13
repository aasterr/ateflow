/* The right-hand panel. Its Answer tab carries the three promises of ateflow:
   the answer in words, why the direct comparison is wrong, how much to trust it.
   Every sentence comes from the engine (ateflow/answer.py); this only lays it out. */

const IDENT_SUMMARY = {
  backdoor: "Backdoor adjustment",
  frontdoor: "Front-door criterion",
  none: "Not identifiable",
};

function Identification({ check }) {
  if (!check) return <p className="hint">Checking the DAG…</p>;
  if (check.error) return <p className="hint bad">{check.error}</p>;
  const kind = check.strategy ?? "none";
  return (
    <div className={`ident ident-${kind}`}>
      <p className="ident-title">{IDENT_SUMMARY[kind]}</p>
      <ul>
        {check.explanation.map((line) => <li key={line}>{line}</li>)}
      </ul>
    </div>
  );
}

function Before({ check, hasQuestion }) {
  if (!hasQuestion) {
    return (
      <div className="answer-empty">
        <h2>What do you want to know?</h2>
        <p>
          Choose on the left the <strong>treatment</strong> (the thing that was done, with two
          values) and the <strong>outcome</strong> (what it should have changed). Draw on the
          canvas which variables cause which.
        </p>
      </div>
    );
  }
  return (
    <div className="answer-empty">
      <h2>What will be estimated</h2>
      <p>
        The average effect of the treatment on the outcome (ATE), cleaned of the confounders
        your DAG declares. This is how the DAG makes it computable:
      </p>
      <Identification check={check} />
      {check?.strategy && <p className="hint">Press <strong>Estimate effect</strong> on the left.</p>}
    </div>
  );
}

/* Zero, the interval, the effect and the direct comparison on one axis. */
function IntervalBar({ bar }) {
  const values = [0, bar.naive, bar.effect, ...(bar.ci ?? [])];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = (hi - lo || 1) * 0.12;
  lo -= pad;
  hi += pad;
  const W = 300;
  const x = (v) => ((v - lo) / (hi - lo)) * W;
  return (
    <svg className="interval" viewBox={`0 0 ${W} 44`} role="img"
      aria-label="effect with its 95% interval, the direct comparison and zero on one axis">
      <line x1="0" x2={W} y1="20" y2="20" className="axis" />
      <line x1={x(0)} x2={x(0)} y1="8" y2="32" className="zero" />
      {Math.abs(x(0) - x(bar.naive)) > 30 && <text x={x(0)} y="42" className="tick">0</text>}
      {bar.ci && <rect x={x(bar.ci[0])} width={Math.max(x(bar.ci[1]) - x(bar.ci[0]), 2)} y="14" height="12" rx="6" className="ci" />}
      <circle cx={x(bar.naive)} cy="20" r="5" className="naive-dot" />
      <text x={x(bar.naive)} y="42" className="tick">direct</text>
      <circle cx={x(bar.effect)} cy="20" r="5.5" className="effect-dot" />
      <text x={x(bar.effect)} y="7" className="tick effect-tick">effect</text>
    </svg>
  );
}

function Summary({ s }) {
  return (
    <div className="summary">
      <div className="sum-row">
        <span className="sum-label">What</span>
        <div>
          <strong>{s.what}</strong>
          <p>{s.what_detail}</p>
        </div>
      </div>
      <div className="sum-row">
        <span className="sum-label">How</span>
        <div>
          <strong>{s.how}</strong>
          <p>{s.how_detail}</p>
        </div>
      </div>
      <div className="sum-row">
        <span className="sum-label">Result</span>
        <div>
          <div className="nums">
            <div className="num naive">
              <span>direct comparison</span>
              <b>{s.naive}</b>
            </div>
            <div className="num-arrow" aria-hidden="true">→</div>
            <div className="num effect">
              <span>effect</span>
              <b>{s.effect}</b>
            </div>
          </div>
          <p>
            {s.unit && <>{s.unit} · </>}
            {s.ci ? <>95% CI {s.ci}</> : "no interval computed"}
          </p>
          <IntervalBar bar={s.bar} />
        </div>
      </div>
    </div>
  );
}

function Result({ result, onGuide }) {
  const a = result.answer;
  const rep = result.data_report;
  const coding = (c) => (c && !["1", "1.0", "True"].includes(c["1"]) ? c["1"] : null);
  return (
    <>
      <Summary s={a.summary} />

      <h3>In words</h3>
      <p className="in-words">{a.headline}</p>

      <h3>Why the direct comparison is off</h3>
      <p>{a.naive}</p>

      <h3>How much to trust it</h3>
      <ul className="checks">
        {a.checks.map((c) => (
          <li key={c.title + c.text} className={`check ${c.ok === null ? "info" : c.ok ? "ok" : "bad"}`}>
            <span className="mark" aria-hidden="true">{c.ok === null ? "i" : c.ok ? "✓" : "!"}</span>
            <div>
              <strong>{c.title}</strong>
              <p>{c.text}</p>
              {c.caveat && <p className="caveat">{c.caveat}</p>}
            </div>
          </li>
        ))}
      </ul>
      {rep && (
        <p className="hint">
          {rep.rows_used} of {rep.rows_in} rows used · {rep.treated} treated, {rep.control} control
          {coding(rep.treatment_coding) && <> · treated = “{coding(rep.treatment_coding)}”</>}
          {coding(rep.outcome_coding) && <> · outcome 1 = “{coding(rep.outcome_coding)}”</>}
        </p>
      )}

      <details className="depends">
        <summary>What this answer assumes</summary>
        <ul>
          {result.explanation.map((line) => <li key={line}>{line}</li>)}
        </ul>
        {result.alternatives.length > 0 && (
          <p className="hint">
            Also valid: adjusting for {result.alternatives.slice(0, 3).map((s) => s.join(", ")).join(" — or ")}.
          </p>
        )}
        <p className="hint">
          Change an arrow on the canvas and estimate again to see how much the answer depends on it.
          {onGuide && <> <button className="link inline" onClick={onGuide}>Ideas in the guide</button></>}
        </p>
      </details>
    </>
  );
}

export default function AnswerPanel({
  tabs, tab, setTab, onHide,
  check, hasQuestion, result, stale, busy, ready, onEstimate,
  saveName, setSaveName, saving, onSave, savePlaceholder,
  dataCheck, guide,
}) {
  return (
    <aside className="panel">
      <div className="panel-head">
        <nav className="tabs">
          {tabs.map((t) => (
            <button key={t} className={t === tab ? "tab active" : "tab"} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
        <button className="link" title="hide the panel" onClick={onHide}>hide</button>
      </div>

      {tab === "Answer" && (
        <div className="answer">
          {busy && (
            <p className="hint engine-status">
              <span className="spinner" /> Estimating: 500 bootstrap resamples, every estimator and
              the refutation tests, in your browser. A few seconds.
            </p>
          )}
          {!busy && !result && <Before check={check} hasQuestion={hasQuestion} />}
          {!busy && result && (
            <>
              {stale && (
                <div className="stale-bar">
                  <span>The DAG or the question changed since this answer.</span>
                  <button className="chip small" disabled={!ready} onClick={onEstimate}>
                    Estimate again
                  </button>
                </div>
              )}
              <div className={stale ? "answer-body stale" : "answer-body"}>
                <Result result={result} onGuide={guide ? () => setTab("Guide") : null} />
              </div>
              {!stale && (
                <div className="save-row">
                  <input
                    placeholder={savePlaceholder}
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                  />
                  <button className="chip" disabled={saving} onClick={onSave}>
                    {saving ? "saving…" : "save"}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
      {tab === "Data" && dataCheck}
      {tab === "Guide" && guide}
    </aside>
  );
}
