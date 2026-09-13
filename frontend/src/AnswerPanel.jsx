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

function Result({ result, onGuide }) {
  const a = result.answer;
  const rep = result.data_report;
  const coding = (c) => (c && !["1", "1.0", "True"].includes(c["1"]) ? c["1"] : null);
  return (
    <>
      <p className="headline">{a.headline}</p>
      <p className="technical">{a.technical}</p>

      <h3>Compared as they are</h3>
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
