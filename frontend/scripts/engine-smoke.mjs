// Runs the browser engine (Pyodide + public/py) under Node: the same Python the
// Web Worker runs, checked against the numbers the native test suite pins.
// Usage: node scripts/sync-python.mjs && node scripts/engine-smoke.mjs
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const here = dirname(fileURLToPath(import.meta.url));
const pyDir = resolve(here, "..", "public", "py");
const version = JSON.parse(readFileSync(resolve(here, "..", "node_modules", "pyodide", "package.json"))).version;

let t = performance.now();
const lap = (label) => {
  const now = performance.now();
  console.log(`${label.padEnd(34)} ${((now - t) / 1000).toFixed(2)} s`);
  t = now;
};

const py = await loadPyodide({ packageCacheDir: "node_modules/.pyodide-cache", packageBaseUrl: `https://cdn.jsdelivr.net/pyodide/v${version}/full/` });
await py.loadPackage(["numpy", "pandas"], { messageCallback: () => {} });
lap("boot + numpy/pandas");

const { files } = JSON.parse(readFileSync(join(pyDir, "manifest.json"), "utf8"));
for (const rel of files) {
  const path = `/ateflow_app/${rel}`;
  py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
  py.FS.writeFile(path, readFileSync(join(pyDir, rel)));
}
py.runPython(`import os, sys; sys.path.insert(0, "/ateflow_app"); os.environ["ATEFLOW_EXAMPLES"] = "/ateflow_app/examples"`);
const handle = py.pyimport("ateflow.service").handle;
lap("import ateflow");

const call = (op, payload, raw = null) => {
  const out = JSON.parse(handle(op, JSON.stringify(payload), raw));
  if (out.error) throw new Error(`${op}: ${out.error}\n${out.trace ?? ""}`);
  return out.ok;
};

let failures = 0;
const expect = (label, actual, target, tol = 5e-4) => {
  const ok = Math.abs(actual - target) <= tol;
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${actual.toFixed(4)} (expected ${target})`);
};

const examples = call("examples", {});
lap("examples (read + profile)");

const dag = (name) => examples[name].dag;
const hrisim = call("estimate", { example: "hrisim", dag: dag("hrisim"), treatment: "A", outcome: "T", method: "stratification", boot: 0, refute: false });
expect("hrisim stratification", hrisim.adjusted.value, 0.061);
expect("hrisim naive", hrisim.naive.value, -0.207);

for (const method of ["stratification", "g-computation", "ipw", "aipw"]) {
  t = performance.now();
  const r = call("estimate", { example: "onboarding", dag: dag("onboarding"), treatment: "onboarding_email", outcome: "retained_30d", method, boot: 500, refute: true });
  lap(`onboarding ${method} (500 boot + refute)`);
  expect(`onboarding ${method}`, r.adjusted.value, 0.09, 0.01);
}

// continuous confounder: the logistic propensity path, the slow one in WebAssembly
let seed = 1;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const rows = Array.from({ length: 5000 }, () => {
  const x = rand() * 4 - 2;
  const t = rand() < 1 / (1 + Math.exp(-x)) ? 1 : 0;
  return `${x.toFixed(4)},${t},${(0.5 * t + x + rand()).toFixed(4)}`;
});
const continuous = new TextEncoder().encode("x,t,y\n" + rows.join("\n"));
for (const method of ["g-computation", "ipw", "aipw"]) {
  t = performance.now();
  const r = call("estimate", { name: "cont.csv", dag: "x -> t\nx -> y\nt -> y", treatment: "t", outcome: "y", method, boot: 500, refute: true }, continuous);
  lap(`continuous ${method} (500 boot + refute)`);
  expect(`continuous ${method}`, r.adjusted.value, 0.5, 0.08);
}

// an uploaded file, passed as bytes the way the worker receives them
const messy = new TextEncoder().encode("id;g;t;y\n" + Array.from({ length: 300 }, (_, i) => `${i};${i % 2};${(i * 7) % 3 === 0 ? "sì" : "no"};${(i % 5) + 0.5}`.replace(".", ",")).join("\n"));
const inspected = call("inspect", { name: "messy.csv" }, messy);
console.log(`${inspected.info.delimiter === "semicolon" && inspected.info.decimal === "comma" ? "ok  " : "FAIL"} upload dialect: ${JSON.stringify(inspected.info)}`);
const up = call("estimate", { name: "messy.csv", dag: "g -> t\ng -> y\nt -> y", treatment: "t", outcome: "y", method: "aipw", boot: 50, refute: false }, messy);
console.log(`ok   upload estimate ${up.adjusted.value.toFixed(3)}, treated = ${up.data_report.treatment_coding["1"]}`);

const err = JSON.parse(handle("estimate", JSON.stringify({ example: "hrisim", dag: "a -> b\nb -> a", treatment: "a", outcome: "b" }), null));
console.log(`${err.status === 422 ? "ok  " : "FAIL"} errors come back as values: ${err.error}`);
if (err.status !== 422) failures++;

const html = call("report", { analysis: { name: "smoke", created_at: new Date().toISOString(), source: "example: hrisim", dag: dag("hrisim"), treatment: "A", outcome: "T", method: "stratification", result: hrisim } }).html;
console.log(`${html.includes("<svg") ? "ok  " : "FAIL"} report renders (${html.length} chars)`);

process.exit(failures ? 1 : 0);
