import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

/* ---------- DAG helpers ---------- */

function parseDagText(text) {
  const edges = [];
  for (const raw of text.split("\n")) {
    const line = raw.split("#")[0].trim();
    if (!line || !line.includes("->")) continue;
    const chain = line.split("->").map((t) => t.trim());
    for (let i = 0; i + 1 < chain.length; i++) {
      if (chain[i] && chain[i + 1]) edges.push([chain[i], chain[i + 1]]);
    }
  }
  return edges;
}

function edgesToDagText(nodes, edges) {
  const connected = new Set(edges.flatMap((e) => [e.source, e.target]));
  const lines = edges.map((e) => `${e.source} -> ${e.target}`);
  for (const n of nodes) if (!connected.has(n.id)) lines.push(n.id);
  return lines.join("\n");
}

/* Layered layout: x by longest path from a root, y by order within the layer. */
function layout(names, pairs) {
  const depth = Object.fromEntries(names.map((n) => [n, 0]));
  for (let pass = 0; pass < names.length; pass++) {
    let changed = false;
    for (const [src, dst] of pairs) {
      if (depth[src] === undefined || depth[dst] === undefined) continue;
      if (depth[dst] < depth[src] + 1) {
        depth[dst] = depth[src] + 1;
        changed = true;
      }
    }
    if (!changed) break;
  }
  const perLayer = {};
  return Object.fromEntries(
    names.map((n) => {
      const d = depth[n] ?? 0;
      perLayer[d] = (perLayer[d] ?? 0) + 1;
      return [n, { x: 60 + d * 200, y: 40 + (perLayer[d] - 1) * 110 + (d % 2) * 40 }];
    })
  );
}

/* ---------- custom node ---------- */

function VariableNode({ data }) {
  return (
    <div className={`var-node ${data.role}`}>
      <Handle type="target" position={Position.Left} />
      <span>{data.label}</span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { variable: VariableNode };

const EDGE_OPTS = {
  type: "default",
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
};

/* ---------- app ---------- */

export default function App() {
  const [examples, setExamples] = useState({});
  const [source, setSource] = useState(null); // {kind:'example', name} | {kind:'file', file, rows}
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [treatment, setTreatment] = useState("");
  const [outcome, setOutcome] = useState("");
  const [method, setMethod] = useState("stratification");
  const [check, setCheck] = useState(null); // {minimal, error}
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [savedList, setSavedList] = useState([]);
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const fileRef = useRef(null);

  const refreshSaved = useCallback(() => {
    fetch("/api/analyses").then((r) => r.json()).then(setSavedList).catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/examples").then((r) => r.json()).then(setExamples).catch(() => {});
    refreshSaved();
  }, [refreshSaved]);

  const columnNames = useMemo(() => nodes.map((n) => n.id), [nodes]);
  const dagText = useMemo(() => edgesToDagText(nodes, edges), [nodes, edges]);

  const roleOf = useCallback(
    (name) => {
      if (name === treatment) return "treatment";
      if (name === outcome) return "outcome";
      if (check?.minimal?.includes(name)) return "adjust";
      return "plain";
    },
    [treatment, outcome, check]
  );

  useEffect(() => {
    setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, role: roleOf(n.id) } })));
  }, [roleOf]);

  const buildGraph = useCallback((names, pairs) => {
    const pos = layout(names, pairs);
    setNodes(
      names.map((name) => ({
        id: name,
        type: "variable",
        position: pos[name],
        deletable: false,
        data: { label: name, role: "plain" },
      }))
    );
    setEdges(
      pairs
        .filter(([s, d]) => names.includes(s) && names.includes(d))
        .map(([s, d]) => ({ id: `${s}->${d}`, source: s, target: d, ...EDGE_OPTS }))
    );
  }, []);

  const loadExample = (name) => {
    const ex = examples[name];
    if (!ex) return;
    buildGraph(ex.columns.filter((c) => c !== "episode_id"), parseDagText(ex.dag));
    setTreatment(ex.treatment);
    setOutcome(ex.outcome);
    setSource({ kind: "example", name });
    setResult(null);
    setError("");
  };

  const loadFile = async (file) => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/columns", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) {
      setError(body.detail ?? "upload failed");
      return;
    }
    buildGraph(body.columns.filter((c) => c !== "episode_id"), []);
    setTreatment("");
    setOutcome("");
    setSource({ kind: "file", file, rows: body.rows });
    setResult(null);
    setError("");
  };

  /* live identification check, debounced. String deps only: updating node
     roles rebuilds the array identities and must not retrigger the fetch. */
  const columnKey = useMemo(() => [...columnNames].sort().join("|"), [columnNames]);
  useEffect(() => {
    setCheck(null);
    if (!treatment || !outcome || treatment === outcome) return;
    const timer = setTimeout(async () => {
      const res = await fetch("/api/dag/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dag: dagText,
          treatment,
          outcome,
          columns: columnKey ? columnKey.split("|") : [],
        }),
      });
      const body = await res.json();
      setCheck(res.ok ? { minimal: body.minimal_adjustment_set } : { error: body.detail });
    }, 250);
    return () => clearTimeout(timer);
  }, [dagText, treatment, outcome, columnKey]);

  const onConnect = useCallback(
    (conn) =>
      setEdges((es) =>
        conn.source === conn.target ? es : addEdge({ ...conn, ...EDGE_OPTS }, es)
      ),
    []
  );

  const questionForm = () => {
    const form = new FormData();
    form.append("dag", dagText);
    form.append("treatment", treatment);
    form.append("outcome", outcome);
    form.append("method", method);
    form.append("boot", "500");
    if (source.kind === "file") form.append("file", source.file);
    else if (source.kind === "saved") form.append("saved", source.id);
    else form.append("example", source.name);
    return form;
  };

  const run = async () => {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("/api/estimate", { method: "POST", body: questionForm() });
      const body = await res.json();
      if (!res.ok) setError(body.detail ?? "estimation failed");
      else setResult(body);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const saveAnalysis = async () => {
    setSaving(true);
    try {
      const form = questionForm();
      form.append("name", saveName || `${treatment} on ${outcome}`);
      const res = await fetch("/api/analyses", { method: "POST", body: form });
      if (!res.ok) setError((await res.json()).detail ?? "save failed");
      else {
        setSaveName("");
        refreshSaved();
      }
    } finally {
      setSaving(false);
    }
  };

  const loadSaved = async (id) => {
    const res = await fetch(`/api/analyses/${id}`);
    if (!res.ok) return;
    const a = await res.json();
    buildGraph(a.columns.filter((c) => c !== "episode_id"), parseDagText(a.dag));
    setTreatment(a.treatment);
    setOutcome(a.outcome);
    setMethod(a.method);
    setSource({ kind: "saved", id: a.id, name: a.name });
    setResult(a.result);
    setError("");
  };

  const deleteSaved = async (id) => {
    await fetch(`/api/analyses/${id}`, { method: "DELETE" });
    if (source?.kind === "saved" && source.id === id) setSource(null);
    refreshSaved();
  };

  const ready =
    source && treatment && outcome && treatment !== outcome && edges.length > 0 && !check?.error;

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>ateflow</h1>
          <p>Causal effect estimation from a declarative DAG.</p>
        </header>

        <section>
          <h2>1 · Data</h2>
          <div className="row">
            {Object.keys(examples).map((name) => (
              <button
                key={name}
                className={source?.kind === "example" && source.name === name ? "chip active" : "chip"}
                title={examples[name].description}
                onClick={() => loadExample(name)}
              >
                {name}
              </button>
            ))}
            <button className="chip" onClick={() => fileRef.current?.click()}>
              upload CSV…
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              hidden
              onChange={(e) => loadFile(e.target.files?.[0])}
            />
          </div>
          {source?.kind === "file" && (
            <p className="hint">{source.file.name} — {source.rows} rows</p>
          )}
        </section>

        {columnNames.length > 0 && (
          <section>
            <h2>2 · Question</h2>
            <label>
              treatment
              <select value={treatment} onChange={(e) => setTreatment(e.target.value)}>
                <option value="">—</option>
                {columnNames.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label>
              outcome
              <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                <option value="">—</option>
                {columnNames.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label>
              method
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                <option value="stratification">stratification</option>
                <option value="g-computation">g-computation</option>
              </select>
            </label>
            {check?.minimal && (
              <p className="hint ok">
                identified — adjusting for {"{"}{check.minimal.join(", ") || "∅"}{"}"}
              </p>
            )}
            {check?.error && <p className="hint bad">{check.error}</p>}
            <button className="primary" disabled={!ready || busy} onClick={run}>
              {busy ? "estimating…" : "Estimate ATE"}
            </button>
          </section>
        )}

        {error && <p className="hint bad">{error}</p>}

        {result && (
          <section className="results">
            <h2>3 · Result</h2>
            <div className="ates">
              <div className="ate naive">
                <span className="label">naive</span>
                <span className="value">{result.naive.value.toFixed(3)}</span>
              </div>
              <div className="arrow">→</div>
              <div className="ate adjusted">
                <span className="label">adjusted {"{"}{result.adjustment_set.join(", ")}{"}"}</span>
                <span className="value">{result.adjusted.value.toFixed(3)}</span>
                {result.adjusted.ci && (
                  <span className="ci">
                    95% CI [{result.adjusted.ci[0].toFixed(3)}, {result.adjusted.ci[1].toFixed(3)}]
                  </span>
                )}
              </div>
            </div>
            {result.sign_flip && (
              <p className="badge">Simpson's paradox: adjustment flips the sign</p>
            )}
            <dl>
              <dt>confounding bias</dt>
              <dd>{result.confounding_bias.toFixed(3)}</dd>
              {result.adjusted.diagnostics.dropped_strata > 0 && (
                <>
                  <dt>dropped (no overlap)</dt>
                  <dd>
                    {result.adjusted.diagnostics.dropped_strata} of {result.adjusted.n} rows
                  </dd>
                </>
              )}
              {result.refutations.map((r) => (
                <div key={r.test} className="refutation">
                  <dt>{r.test.replaceAll("_", " ")}</dt>
                  <dd className={r.passed ? "ok" : "bad"}>{r.passed ? "ok" : "suspect"}</dd>
                </div>
              ))}
            </dl>
            {result.alternatives.length > 0 && (
              <p className="hint">
                other valid sets: {result.alternatives.slice(0, 3).map((s) => `{${s.join(", ")}}`).join("  ")}
              </p>
            )}
            <div className="save-row">
              <input
                placeholder={`${treatment} on ${outcome}`}
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
              />
              <button className="chip" disabled={saving} onClick={saveAnalysis}>
                {saving ? "saving…" : "save"}
              </button>
            </div>
          </section>
        )}

        {savedList.length > 0 && (
          <section>
            <h2>Saved analyses</h2>
            <ul className="saved">
              {savedList.map((a) => (
                <li key={a.id}>
                  <button className="link" onClick={() => loadSaved(a.id)} title="load">
                    {a.name}
                  </button>
                  <span className="val">{a.adjusted >= 0 ? "+" : ""}{a.adjusted.toFixed(3)}</span>
                  <a href={`/api/analyses/${a.id}/report`} target="_blank" rel="noreferrer">
                    report
                  </a>
                  <button className="link danger" onClick={() => deleteSaved(a.id)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <footer>
          drag between handles to add an edge · select an edge and press Delete to remove it
        </footer>
      </aside>

      <main className="canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={(ch) => setNodes((ns) => applyNodeChanges(ch, ns))}
          onEdgesChange={(ch) => setEdges((es) => applyEdgeChanges(ch, es))}
          onConnect={onConnect}
          deleteKeyCode={["Backspace", "Delete"]}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} />
          <Controls showInteractive={false} />
        </ReactFlow>
        {nodes.length === 0 && (
          <div className="empty">Load an example or upload a CSV to start drawing the DAG.</div>
        )}
      </main>
    </div>
  );
}
