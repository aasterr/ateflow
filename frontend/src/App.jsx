import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  Handle,
  MarkerType,
  Position,
  applyEdgeChanges,
  applyNodeChanges,
  getStraightPath,
  useConnection,
  useInternalNode,
  useReactFlow,
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

/* Layered layout. Straight edges make two things matter: nodes of a layer are
   ordered by their parents' height (fewer crossings), and an edge that skips
   layers must not run through the nodes it skips, so those get pushed aside. */
const LAYER_GAP = 230;
const ROW_GAP = 90;
const CLEARANCE = 42;

function layout(names, pairs) {
  pairs = pairs.filter(([s, d]) => names.includes(s) && names.includes(d));
  const depth = Object.fromEntries(names.map((n) => [n, 0]));
  for (let pass = 0; pass < names.length; pass++) {
    let changed = false;
    for (const [src, dst] of pairs) {
      if (depth[dst] < depth[src] + 1) {
        depth[dst] = depth[src] + 1;
        changed = true;
      }
    }
    if (!changed) break;
  }

  const layers = [];
  for (const n of names) (layers[depth[n]] ??= []).push(n);
  const y = {};
  layers.forEach((layer, d) => {
    if (d > 0) {
      const bary = (n) => {
        const ys = pairs.filter(([, dst]) => dst === n).map(([s]) => y[s]);
        return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
      };
      layer.sort((a, b) => bary(a) - bary(b));
    }
    layer.forEach((n, i) => (y[n] = (i - (layer.length - 1) / 2) * ROW_GAP * 1.4));
  });

  for (let iter = 0; iter < 30; iter++) {
    let moved = false;
    for (const [s, d] of pairs) {
      for (let k = depth[s] + 1; k < depth[d]; k++) {
        for (const n of layers[k]) {
          const lineY = y[s] + ((y[d] - y[s]) * (k - depth[s])) / (depth[d] - depth[s]);
          const off = y[n] - lineY;
          if (Math.abs(off) < CLEARANCE) {
            y[n] = lineY + (off >= 0 ? CLEARANCE : -CLEARANCE);
            moved = true;
          }
        }
      }
    }
    for (const layer of layers) {
      const sorted = [...layer].sort((a, b) => y[a] - y[b]);
      for (let i = 1; i < sorted.length; i++) {
        if (y[sorted[i]] - y[sorted[i - 1]] < ROW_GAP) {
          y[sorted[i]] = y[sorted[i - 1]] + ROW_GAP;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  // positions are top-left corners: centre each node on its layer column
  const approxWidth = (n) => 48 + n.length * 7.5;
  return Object.fromEntries(
    names.map((n) => [n, { x: depth[n] * LAYER_GAP - approxWidth(n) / 2, y: y[n] - 17 }])
  );
}

/* ---------- geometry: straight edges between node borders ---------- */

function nodeBox(node) {
  const { x, y } = node.internals.positionAbsolute;
  const w = node.measured?.width ?? 0;
  const h = node.measured?.height ?? 0;
  return { cx: x + w / 2, cy: y + h / 2, w, h };
}

/* Point where the ray from the box centre towards (tx, ty) leaves the box, plus a gap. */
function borderPoint(box, tx, ty, gap = 0) {
  const dx = tx - box.cx;
  const dy = ty - box.cy;
  if (dx === 0 && dy === 0) return { x: box.cx, y: box.cy };
  const t = Math.min(
    dx ? box.w / 2 / Math.abs(dx) : Infinity,
    dy ? box.h / 2 / Math.abs(dy) : Infinity
  );
  const len = Math.hypot(dx, dy);
  const k = t + gap / len;
  return { x: box.cx + dx * k, y: box.cy + dy * k };
}

/* ---------- custom node ---------- */

/* The whole body is the connection handle: press on a variable and drag to
   another one. Where the drag starts is the cause, where it ends the effect.
   Nodes move by their grip, so the two gestures never compete. */
function VariableNode({ id, data }) {
  const connection = useConnection();
  const isTarget = connection.inProgress && connection.fromNode.id !== id;
  return (
    <div className={`var-node ${data.role}`}>
      <div className="grip" title="drag to move">⋮⋮</div>
      <div className="var-body">
        {/* Both handles stay mounted (edges need them to render); only the
            one that should receive the pointer right now is live. */}
        <Handle
          className={`body-handle ${connection.inProgress ? "idle" : ""}`}
          type="source"
          position={Position.Right}
        />
        <Handle
          className={`body-handle ${isTarget ? "" : "idle"}`}
          type="target"
          position={Position.Left}
          isConnectableStart={false}
        />
        <span>{data.label}</span>
      </div>
    </div>
  );
}
const nodeTypes = { variable: VariableNode };

function StraightEdge({ id, source, target, markerEnd, selected }) {
  const s = useInternalNode(source);
  const t = useInternalNode(target);
  const { setEdges } = useReactFlow();
  if (!s || !t) return null;

  const sb = nodeBox(s);
  const tb = nodeBox(t);
  const start = borderPoint(sb, tb.cx, tb.cy, 2);
  const end = borderPoint(tb, sb.cx, sb.cy, 4);
  const [path, midX, midY] = getStraightPath({
    sourceX: start.x, sourceY: start.y, targetX: end.x, targetY: end.y,
  });

  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={24}
        className={selected ? "edge-selected" : undefined} />
      {selected && (
        <EdgeLabelRenderer>
          <button
            className="edge-delete nodrag nopan"
            style={{ transform: `translate(-50%, -50%) translate(${midX}px, ${midY}px)` }}
            title="remove this edge"
            onClick={() => setEdges((es) => es.filter((e) => e.id !== id))}
          >
            ×
          </button>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
const edgeTypes = { straight: StraightEdge };

/* The line being drawn: from the border of the start node to the cursor. */
function ConnectionLine({ fromNode, toX, toY }) {
  if (!fromNode) return null;
  const start = borderPoint(nodeBox(fromNode), toX, toY, 2);
  return (
    <g>
      <path className="connection-draft" d={`M${start.x},${start.y} L${toX},${toY}`} />
      <circle cx={toX} cy={toY} r={3.5} className="connection-tip" />
    </g>
  );
}

const EDGE_OPTS = {
  type: "straight",
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18, color: "#6b6b6b" },
};

const edgeId = (s, d) => `${s}->${d}`;

/* Guide edits from /api/examples: {op: 'add' | 'remove' | 'flip', edge: [src, dst]}. */
function applyGuideEdit(edges, { op, edge: [s, d] }) {
  const rest = edges.filter(
    (e) => !(e.source === s && e.target === d) && !(e.source === d && e.target === s)
  );
  if (op === "remove") return rest;
  const [from, to] = op === "flip" ? [d, s] : [s, d];
  return [...rest, { id: edgeId(from, to), source: from, target: to, ...EDGE_OPTS }];
}

function guideEditApplied(edges, { op, edge: [s, d] }) {
  const has = (a, b) => edges.some((e) => e.source === a && e.target === b);
  if (op === "add") return has(s, d);
  if (op === "flip") return has(d, s) && !has(s, d);
  return !has(s, d) && !has(d, s);
}

/* Identity of a question, to tell whether a shown result still matches the canvas. */
const questionKey = (pairs, treatment, outcome, method) =>
  [pairs.map(([s, d]) => `${s}->${d}`).sort().join(","), treatment, outcome, method].join("|");

/* ---------- app ---------- */

export default function App() {
  const [examples, setExamples] = useState({});
  const [source, setSource] = useState(null); // {kind:'example', name} | {kind:'file', file, rows}
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [graphKey, setGraphKey] = useState(0);
  const [treatment, setTreatment] = useState("");
  const [outcome, setOutcome] = useState("");
  const [method, setMethod] = useState("stratification");
  const [check, setCheck] = useState(null); // {minimal, error}
  const [result, setResult] = useState(null);
  const [resultKey, setResultKey] = useState("");
  const [guideOpen, setGuideOpen] = useState(true);
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
  const currentKey = questionKey(
    edges.map((e) => [e.source, e.target]), treatment, outcome, method
  );
  const stale = result && resultKey !== currentKey;
  const guide = source?.kind === "example" ? examples[source.name]?.guide : null;

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
    setGraphKey((k) => k + 1); // remount the canvas so fitView frames the new graph
    setNodes(
      names.map((name) => ({
        id: name,
        type: "variable",
        position: pos[name],
        deletable: false,
        dragHandle: ".grip",
        data: { label: name, role: "plain" },
      }))
    );
    setEdges(
      pairs
        .filter(([s, d]) => names.includes(s) && names.includes(d))
        .map(([s, d]) => ({ id: edgeId(s, d), source: s, target: d, ...EDGE_OPTS }))
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

  /* Drawing b -> a over an existing a -> b flips the arrow instead of
     creating a two-node cycle; drawing an existing edge again is a no-op. */
  const onConnect = useCallback(
    ({ source: s, target: d }) =>
      setEdges((es) => {
        if (s === d || es.some((e) => e.source === s && e.target === d)) return es;
        return [
          ...es.filter((e) => !(e.source === d && e.target === s)),
          { id: edgeId(s, d), source: s, target: d, ...EDGE_OPTS },
        ];
      }),
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
      else {
        setResult(body);
        setResultKey(currentKey);
      }
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
    setResultKey(questionKey(parseDagText(a.dag), a.treatment, a.outcome, a.method));
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
                <option value="ipw">IPW (weighting)</option>
                <option value="aipw">AIPW (doubly robust)</option>
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
            {busy && (
              <p className="hint">
                500 bootstrap resamples and the refutation tests — a few seconds
                on the free demo instance.
              </p>
            )}
          </section>
        )}

        {error && <p className="hint bad">{error}</p>}

        {result && (
          <section className={stale ? "results stale" : "results"}>
            <h2>3 · Result</h2>
            {stale && (
              <p className="hint bad">
                The question changed since this estimate — press Estimate ATE again.
              </p>
            )}
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
              {result.adjusted.diagnostics.effective_n !== undefined && (
                <>
                  <dt>effective sample size</dt>
                  <dd>
                    {Math.round(result.adjusted.diagnostics.effective_n)} of {result.adjusted.n}
                  </dd>
                </>
              )}
              {result.adjusted.diagnostics.clipped > 0 && (
                <>
                  <dt>weak overlap (clipped)</dt>
                  <dd className="bad">
                    {result.adjusted.diagnostics.clipped} of {result.adjusted.n} rows
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
          drag from a variable to another to add a cause → effect edge · move a
          variable by its ⋮⋮ grip · click an edge to remove it
        </footer>
      </aside>

      <main className="canvas">
        <ReactFlow
          key={graphKey}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          connectionLineComponent={ConnectionLine}
          connectionRadius={0}
          onNodesChange={(ch) => setNodes((ns) => applyNodeChanges(ch, ns))}
          onEdgesChange={(ch) => setEdges((es) => applyEdgeChanges(ch, es))}
          onConnect={onConnect}
          deleteKeyCode={["Backspace", "Delete"]}
          fitView
          fitViewOptions={{ padding: 0.25, maxZoom: 1.4 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} />
          <Controls showInteractive={false} />
        </ReactFlow>
        {nodes.length === 0 && (
          <div className="empty">Load an example or upload a CSV to start drawing the DAG.</div>
        )}
      </main>

      {guide && (
        guideOpen ? (
          <aside className="guide">
            <div className="guide-head">
              <span className="eyebrow">About this example</span>
              <button className="link" title="hide" onClick={() => setGuideOpen(false)}>hide</button>
            </div>
            <h2>{guide.title}</h2>
            <p>{guide.story}</p>
            <h3>Why naive and adjusted differ</h3>
            <p>{guide.why}</p>
            <h3>Try changing the DAG</h3>
            <ul>
              {guide.tries.map((tip) => {
                const applied = guideEditApplied(edges, tip.edit);
                return (
                  <li key={tip.edit.op + tip.edit.edge.join()}>
                    <p>{tip.text}</p>
                    <button
                      className={applied ? "chip small applied" : "chip small"}
                      disabled={applied}
                      onClick={() => setEdges((es) => applyGuideEdit(es, tip.edit))}
                    >
                      {applied ? "applied — now estimate" : "apply to the DAG"}
                    </button>
                  </li>
                );
              })}
            </ul>
            <button className="link" onClick={() => loadExample(source.name)}>
              ↺ reset to the original DAG
            </button>
          </aside>
        ) : (
          <button className="guide-tab" onClick={() => setGuideOpen(true)}>
            About this example
          </button>
        )
      )}
    </div>
  );
}
