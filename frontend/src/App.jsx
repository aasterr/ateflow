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
import { engine, onProgress } from "./engine.js";
import { store } from "./store.js";
import AnswerPanel from "./AnswerPanel.jsx";
import DataCheck, { LEFT_OUT } from "./DataCheck.jsx";
import ExampleGuide from "./ExampleGuide.jsx";

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

/* Names listed on an `unmeasured: a, b` line: variables in the causal story with no column. */
function parseUnmeasured(text) {
  const names = [];
  for (const raw of text.split("\n")) {
    const line = raw.split("#")[0].trim();
    if (line.toLowerCase().startsWith("unmeasured:")) {
      names.push(...line.slice("unmeasured:".length).split(",").map((t) => t.trim()).filter(Boolean));
    }
  }
  return names;
}

function edgesToDagText(nodes, edges) {
  const connected = new Set(edges.flatMap((e) => [e.source, e.target]));
  const lines = edges.map((e) => `${e.source} -> ${e.target}`);
  for (const n of nodes) if (!connected.has(n.id) && !n.data.unmeasured) lines.push(n.id);
  const hidden = nodes.filter((n) => n.data.unmeasured).map((n) => n.id);
  if (hidden.length) lines.push(`unmeasured: ${hidden.join(", ")}`);
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
  if (pairs.length === 0) {
    // a fresh upload: a loose grid to draw on, not one tall column
    const cols = Math.ceil(Math.sqrt(names.length));
    return Object.fromEntries(
      names.map((n, i) => [n, { x: (i % cols) * LAYER_GAP, y: Math.floor(i / cols) * ROW_GAP * 1.4 }])
    );
  }
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
function VariableNode({ id, data, selected }) {
  const connection = useConnection();
  const { deleteElements } = useReactFlow();
  const isTarget = connection.inProgress && connection.fromNode.id !== id;
  return (
    <div className={`var-node ${data.role} ${data.unmeasured ? "unmeasured" : ""}`}>
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
        {data.unmeasured && <em className="tag">unmeasured</em>}
      </div>
      {data.unmeasured && selected && (
        <button
          className="node-delete nodrag"
          title="remove this unmeasured variable"
          onClick={() => deleteElements({ nodes: [{ id }] })}
        >
          ×
        </button>
      )}
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

/* Identity of a question, to tell whether a shown result still matches the canvas. */
const questionKey = (pairs, ...rest) =>
  [pairs.map(([s, d]) => `${s}->${d}`).sort().join(","), ...rest].join("|");

/* ---------- data check helpers (mirror ateflow/data.py) ---------- */

const usableColumns = (profile) => profile.filter((p) => !LEFT_OUT.has(p.kind)).map((p) => p.name);

const TRUE_TOKENS = new Set(["1", "true", "yes", "y", "si", "sì", "treated", "treatment", "on"]);
const FALSE_TOKENS = new Set(["0", "false", "no", "n", "control", "untreated", "off"]);
const token = (v) => {
  const s = String(v).trim().toLowerCase();
  return s === "1.0" ? "1" : s === "0.0" ? "0" : s;
};

/* For a two-valued column: which value the server will read as 1 on its own, or null. */
function autoPositive(values) {
  const tokens = values.map(token);
  if (!tokens.every((t) => TRUE_TOKENS.has(t) || FALSE_TOKENS.has(t))) return null;
  const positives = values.filter((v) => TRUE_TOKENS.has(token(v)));
  return positives.length === 1 ? positives[0] : null;
}
const isZeroOne = (values) => values.map(token).sort().join() === "0,1";

/* ---------- app ---------- */

export default function App() {
  const [examples, setExamples] = useState({});
  // {kind:'example', name} | {kind:'file', name, bytes, rows} | {kind:'saved', id, name, bytes|example}
  const [source, setSource] = useState(null);
  const [engineStage, setEngineStage] = useState("starting");
  const [engineError, setEngineError] = useState("");
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [graphKey, setGraphKey] = useState(0);
  const [treatment, setTreatment] = useState("");
  const [outcome, setOutcome] = useState("");
  const [check, setCheck] = useState(null); // {strategy, variables, explanation} | {error}
  const [result, setResult] = useState(null);
  const [resultKey, setResultKey] = useState("");
  const [panelOpen, setPanelOpen] = useState(true);
  const [tab, setTab] = useState("Answer"); // Answer | Data | Guide
  const [profile, setProfile] = useState([]); // one entry per CSV column, from the server
  const [dataInfo, setDataInfo] = useState(null); // what was detected reading an upload
  const [treatedValue, setTreatedValue] = useState("");
  const [outcomePositive, setOutcomePositive] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [savedList, setSavedList] = useState([]);
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const fileRef = useRef(null);

  const refreshSaved = useCallback(() => {
    store.list().then(setSavedList).catch(() => {});
  }, []);

  const bootEngine = useCallback(() => {
    setEngineError("");
    engine
      .init()
      .then(() => engine.examples())
      .then(setExamples)
      .catch((err) => setEngineError(err.message));
  }, []);

  useEffect(() => {
    const off = onProgress(setEngineStage);
    bootEngine();
    refreshSaved();
    return off;
  }, [bootEngine, refreshSaved]);
  const engineReady = engineStage === "ready" && !engineError;

  // measured variables only: unmeasured ones have no column and cannot be treatment or outcome
  const columnNames = useMemo(() => nodes.filter((n) => !n.data.unmeasured).map((n) => n.id), [nodes]);
  const unmeasuredNames = useMemo(() => nodes.filter((n) => n.data.unmeasured).map((n) => n.id), [nodes]);
  const dagText = useMemo(() => edgesToDagText(nodes, edges), [nodes, edges]);
  const currentKey = questionKey(
    edges.map((e) => [e.source, e.target]), treatment, outcome, treatedValue, outcomePositive
  );
  const profileOf = (name) => profile.find((p) => p.name === name);

  /* What the question panel must say about the chosen treatment and outcome
     before anything is sent: wrong kind of column, or a coding to choose. */
  const treatmentInfo = useMemo(() => {
    const p = profile.find((c) => c.name === treatment);
    if (!p) return null;
    if (p.kind !== "binary") {
      return { error: `${treatment} has ${p.unique} distinct values: the treatment must have exactly two` };
    }
    return isZeroOne(p.examples) ? { values: null } : { values: p.examples };
  }, [profile, treatment]);

  const outcomeInfo = useMemo(() => {
    const p = profile.find((c) => c.name === outcome);
    if (!p) return null;
    if (p.kind === "binary" && !isZeroOne(p.examples) && !p.numeric) return { values: p.examples };
    if (!p.numeric && p.kind !== "binary") {
      return { error: `${outcome} is not numeric (e.g. "${p.examples[0]}"): the outcome must be a number or have two values` };
    }
    if (p.kind === "constant" || p.kind === "empty") return { error: `${outcome} never varies` };
    return { values: null };
  }, [profile, outcome]);

  const chooseTreatment = (name) => {
    setTreatment(name);
    const p = profile.find((c) => c.name === name);
    setTreatedValue(p?.kind === "binary" && !isZeroOne(p.examples) ? autoPositive(p.examples) ?? "" : "");
  };
  const chooseOutcome = (name) => {
    setOutcome(name);
    const p = profile.find((c) => c.name === name);
    setOutcomePositive(
      p?.kind === "binary" && !p.numeric && !isZeroOne(p.examples) ? autoPositive(p.examples) ?? "" : ""
    );
  };
  const stale = result && resultKey !== currentKey;
  const guide = source?.kind === "example" ? examples[source.name]?.guide : null;
  const tabs = ["Answer", ...(source?.kind === "file" ? ["Data"] : []), ...(guide ? ["Guide"] : [])];
  useEffect(() => {
    if (!tabs.includes(tab)) setTab("Answer");
  }, [tabs.join(), tab]); // eslint-disable-line react-hooks/exhaustive-deps

  const roleOf = useCallback(
    (name) => {
      if (name === treatment) return "treatment";
      if (name === outcome) return "outcome";
      if (check?.variables?.includes(name)) return "adjust";
      return "plain";
    },
    [treatment, outcome, check]
  );

  useEffect(() => {
    setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, role: roleOf(n.id) } })));
  }, [roleOf]);

  const buildGraph = useCallback((measured, pairs, unmeasured = []) => {
    const names = [...measured, ...unmeasured.filter((u) => !measured.includes(u))];
    const pos = layout(names, pairs);
    setGraphKey((k) => k + 1); // remount the canvas so fitView frames the new graph
    setNodes(
      names.map((name) => ({
        id: name,
        type: "variable",
        position: pos[name],
        deletable: unmeasured.includes(name),
        dragHandle: ".grip",
        data: { label: name, role: "plain", unmeasured: unmeasured.includes(name) },
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
    buildGraph(usableColumns(ex.profile), parseDagText(ex.dag), parseUnmeasured(ex.dag));
    setProfile(ex.profile);
    setDataInfo(null);
    setTreatment(ex.treatment);
    setOutcome(ex.outcome);
    setTreatedValue("");
    setOutcomePositive("");
    setSource({ kind: "example", name });
    setResult(null);
    setError("");
    setTab("Answer");
  };

  const loadFile = async (file) => {
    if (!file) return;
    setError("");
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const body = await engine.inspect(bytes, file.name);
      buildGraph(usableColumns(body.profile), []);
      setProfile(body.profile);
      setDataInfo(body.info);
      setTreatment("");
      setOutcome("");
      setTreatedValue("");
      setOutcomePositive("");
      setSource({ kind: "file", name: file.name, bytes, rows: body.rows });
      setResult(null);
      setPanelOpen(true);
      setTab("Data");
    } catch (err) {
      setError(err.message);
    }
  };

  /* live identification check, debounced. String deps only: updating node
     roles rebuilds the array identities and must not retrigger the check. */
  const columnKey = useMemo(() => [...columnNames].sort().join("|"), [columnNames]);
  useEffect(() => {
    setCheck(null);
    if (!engineReady || !treatment || !outcome || treatment === outcome) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      engine
        .check({ dag: dagText, treatment, outcome, columns: columnKey ? columnKey.split("|") : [] })
        .then((body) => !cancelled && setCheck(body))
        .catch((err) => !cancelled && setCheck({ error: err.message }));
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [engineReady, dagText, treatment, outcome, columnKey]);

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

  const question = () => ({
    dag: dagText,
    treatment,
    outcome,
    method: "auto",
    boot: 500,
    treated_value: treatedValue || null,
    outcome_positive: outcomePositive || null,
  });

  /* The dataset of the current source: bytes for files and saved uploads, a name for examples. */
  const dataOf = (src) =>
    src.bytes ? { payload: { name: src.name }, bytes: src.bytes } : { payload: { example: src.example ?? src.name } };

  const run = async () => {
    setBusy(true);
    setError("");
    setPanelOpen(true);
    setTab("Answer");
    try {
      const { payload, bytes } = dataOf(source);
      const body = await engine.estimate({ ...question(), ...payload }, bytes);
      setResult(body);
      setResultKey(currentKey);
    } catch (err) {
      setResult(null);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  /* Saves the shown result with its question and data; nothing is re-estimated. */
  const saveAnalysis = async () => {
    setSaving(true);
    try {
      await store.save({
        name: saveName || `${treatment} on ${outcome}`,
        source: source.kind === "file" ? source.name : `example: ${source.example ?? source.name}`,
        example: source.bytes ? null : source.example ?? source.name,
        csv: source.bytes ?? null,
        dag: dagText,
        treatment,
        outcome,
        method: result.method,
        result,
      });
      setSaveName("");
      refreshSaved();
    } catch (err) {
      setError(`could not save: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const loadSaved = async (id) => {
    try {
      const a = await store.get(id);
      if (!a) return;
      const data = a.csv ? await engine.inspect(a.csv, a.name) : examples[a.example];
      buildGraph(usableColumns(data.profile), parseDagText(a.dag), parseUnmeasured(a.dag));
      setProfile(data.profile);
      setDataInfo(null);
      setTreatment(a.treatment);
      setOutcome(a.outcome);
      // the coding chosen when it was saved travels in the stored result
      const rep = a.result.data_report;
      const tv = rep && !isZeroOne(Object.values(rep.treatment_coding)) ? rep.treatment_coding["1"] : "";
      const op = rep?.outcome_coding && !isZeroOne(Object.values(rep.outcome_coding))
        ? rep.outcome_coding["1"] : "";
      setTreatedValue(tv);
      setOutcomePositive(op);
      setSource({ kind: "saved", id: a.id, name: a.name, bytes: a.csv, example: a.example });
      // analyses saved before the answer in words existed get their sentences now
      const answer = a.result.answer ??
        (await engine.answer({ result: a.result, treatment: a.treatment, outcome: a.outcome }));
      setResult({ ...a.result, answer });
      setResultKey(questionKey(parseDagText(a.dag), a.treatment, a.outcome, tv, op));
      setError("");
      setPanelOpen(true);
      setTab("Answer");
    } catch (err) {
      setError(`could not load: ${err.message}`);
    }
  };

  /* Opens the standalone report in a new tab; it is rendered here, from the stored result. */
  const openReport = async (id) => {
    const tab = window.open("", "_blank"); // opened synchronously so popup blockers allow it
    try {
      const { csv, ...a } = await store.get(id);
      const html = await engine.report(a);
      const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
      if (tab) tab.location.href = url;
      else window.location.href = url;
    } catch (err) {
      tab?.close();
      setError(`could not render the report: ${err.message}`);
    }
  };

  const deleteSaved = async (id) => {
    await store.remove(id);
    if (source?.kind === "saved" && source.id === id) setSource(null);
    refreshSaved();
  };

  const ready =
    source && treatment && outcome && treatment !== outcome && edges.length > 0 && check?.strategy &&
    !treatmentInfo?.error && !outcomeInfo?.error &&
    !(treatmentInfo?.values && !treatedValue) && !(outcomeInfo?.values && !outcomePositive);

  const showWhy = () => {
    setPanelOpen(true);
    setTab("Answer");
  };

  const addUnmeasured = () => {
    const raw = window.prompt("Name of the unmeasured variable (e.g. motivation)");
    if (raw == null) return;
    const name = raw.trim().replace(/\s+/g, " ");
    if (!name || /[,#:]|->/.test(name)) {
      setError("an unmeasured variable needs a name without commas, colons, '#' or '->'");
      return;
    }
    if (nodes.some((n) => n.id === name)) {
      setError(`${name} is already in the DAG`);
      return;
    }
    setError("");
    const xs = nodes.map((n) => n.position.x);
    const ys = nodes.map((n) => n.position.y);
    const position = nodes.length
      ? { x: Math.min(...xs), y: Math.max(...ys) + 110 }
      : { x: 0, y: 0 };
    setNodes((ns) => [
      ...ns,
      {
        id: name,
        type: "variable",
        position,
        deletable: true,
        dragHandle: ".grip",
        data: { label: name, role: "plain", unmeasured: true },
      },
    ]);
  };

  /* Upload only: add or remove a column from the canvas, keeping the arrows among the rest. */
  const toggleColumn = (name) => {
    const names = columnNames.includes(name)
      ? columnNames.filter((c) => c !== name)
      : profile.map((p) => p.name).filter((c) => c === name || columnNames.includes(c));
    buildGraph(names, edges.map((e) => [e.source, e.target]), unmeasuredNames);
    if (!names.includes(treatment)) setTreatment("");
    if (!names.includes(outcome)) setOutcome("");
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>ateflow</h1>
          <p>Causal effect estimation from a declarative DAG.</p>
        </header>

        <p className="privacy">
          Runs entirely in your browser: files you load are never uploaded anywhere.
        </p>
        {!engineReady && !engineError && (
          <p className="hint engine-status">
            <span className="spinner" /> Starting the engine — {engineStage}… (first visit
            takes a few seconds, then it is cached)
          </p>
        )}
        {engineError && (
          <p className="hint bad">
            The engine could not start: {engineError}.{" "}
            <button className="link" onClick={bootEngine}>retry</button>
          </p>
        )}

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
            <button className="chip" disabled={!engineReady} onClick={() => fileRef.current?.click()}>
              open CSV…
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.tsv,.txt,text/csv"
              hidden
              onChange={(e) => {
                loadFile(e.target.files?.[0]);
                e.target.value = ""; // choosing the same file again must reload it
              }}
            />
          </div>
          {source?.kind === "file" && (
            <p className="hint">
              {source.name} — {source.rows} rows · checked in the panel on the right
            </p>
          )}
        </section>

        {columnNames.length > 0 && (
          <section>
            <h2>2 · Question</h2>
            <label>
              treatment
              <select value={treatment} onChange={(e) => chooseTreatment(e.target.value)}>
                <option value="">—</option>
                {columnNames.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            {treatmentInfo?.error && <p className="hint bad">{treatmentInfo.error}</p>}
            {treatmentInfo?.values && (
              <label className="sub">
                treated means
                <select value={treatedValue} onChange={(e) => setTreatedValue(e.target.value)}>
                  <option value="">choose…</option>
                  {treatmentInfo.values.map((v) => <option key={v}>{v}</option>)}
                </select>
              </label>
            )}
            <label>
              outcome
              <select value={outcome} onChange={(e) => chooseOutcome(e.target.value)}>
                <option value="">—</option>
                {columnNames.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            {outcomeInfo?.error && <p className="hint bad">{outcomeInfo.error}</p>}
            {outcomeInfo?.values && (
              <label className="sub">
                counts as 1
                <select value={outcomePositive} onChange={(e) => setOutcomePositive(e.target.value)}>
                  <option value="">choose…</option>
                  {outcomeInfo.values.map((v) => <option key={v}>{v}</option>)}
                </select>
              </label>
            )}
            {check?.strategy && (
              <p className="hint ok">
                {check.strategy === "frontdoor"
                  ? `identified by front-door, through ${check.variables.join(", ")}`
                  : check.variables.length
                    ? `identified, adjusting for ${check.variables.join(", ")}`
                    : "identified, no adjustment needed"}{" "}
                <button className="link why" onClick={showWhy}>why?</button>
              </p>
            )}
            {check && check.strategy === null && (
              <p className="hint bad">
                not identifiable from this DAG{" "}
                <button className="link why" onClick={showWhy}>why?</button>
              </p>
            )}
            {check?.error && <p className="hint bad">{check.error}</p>}
            <button className="primary" disabled={!ready || busy} onClick={run}>
              {busy ? "estimating…" : "Estimate effect"}
            </button>
          </section>
        )}

        {error && <p className="hint bad">{error}</p>}

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
                  <button className="link report-link" onClick={() => openReport(a.id)}>
                    report
                  </button>
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
          variable by its ⋮⋮ grip · click an edge to remove it · dashed = unmeasured
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
          <div className="empty">Load an example or open a CSV to start drawing the DAG.</div>
        )}
        {nodes.length > 0 && (
          <div className="canvas-tools">
            <button
              className="chip small"
              onClick={addUnmeasured}
              title="a variable that matters causally but has no column in the data"
            >
              + unmeasured variable
            </button>
          </div>
        )}
      </main>

      {source && (
        panelOpen ? (
          <AnswerPanel
            tabs={tabs}
            tab={tab}
            setTab={setTab}
            onHide={() => setPanelOpen(false)}
            check={check}
            hasQuestion={Boolean(treatment && outcome && treatment !== outcome)}
            result={result}
            stale={stale}
            busy={busy}
            ready={ready}
            onEstimate={run}
            saveName={saveName}
            setSaveName={setSaveName}
            saving={saving}
            onSave={saveAnalysis}
            savePlaceholder={`${treatment} on ${outcome}`}
            dataCheck={source.kind === "file" && profile.length > 0 && (
              <DataCheck source={source} dataInfo={dataInfo} profile={profile}
                columnNames={columnNames} onToggle={toggleColumn} />
            )}
            guide={guide && (
              <ExampleGuide guide={guide} edges={edges}
                onApply={(edit) => setEdges((es) => applyGuideEdit(es, edit))}
                onReset={() => loadExample(source.name)} />
            )}
          />
        ) : (
          <button className="guide-tab" onClick={() => setPanelOpen(true)}>
            {result && !stale ? "Show the answer" : "Show the panel"}
          </button>
        )
      )}
    </div>
  );
}
