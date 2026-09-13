// The engine: CPython + numpy + pandas compiled to WebAssembly (Pyodide), running
// ateflow/service.py off the main thread. Data given to it never leave the page.

const PYODIDE_VERSION = "314.0.6";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const APP_DIR = "/ateflow_app";

let handle = null;
let booting = null;

const progress = (stage) => self.postMessage({ type: "progress", stage });

async function boot(base) {
  progress("downloading Python");
  const { loadPyodide } = await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`);
  const py = await loadPyodide({ indexURL: INDEX_URL });

  progress("loading numpy and pandas");
  await py.loadPackage(["numpy", "pandas"]);

  progress("loading ateflow");
  // the manifest is always revalidated; the files carry its version, so a new
  // release never runs next to modules cached from the previous one
  const manifest = await fetch(`${base}py/manifest.json`, { cache: "no-cache" }).then((r) => r.json());
  const files = await Promise.all(
    manifest.files.map(async (rel) => {
      const res = await fetch(`${base}py/${rel}?v=${manifest.version}`);
      if (!res.ok) throw new Error(`could not fetch ${rel}: ${res.status}`);
      return [rel, new Uint8Array(await res.arrayBuffer())];
    })
  );
  for (const [rel, bytes] of files) {
    const path = `${APP_DIR}/${rel}`;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, bytes);
  }
  py.runPython(`
import os, sys
sys.path.insert(0, "${APP_DIR}")
os.environ["ATEFLOW_EXAMPLES"] = "${APP_DIR}/examples"
`);
  handle = py.pyimport("ateflow.service").handle;
  progress("ready");
}

self.onmessage = async ({ data }) => {
  const { id, op, payload, raw, base } = data;
  try {
    if (op === "init") {
      booting ??= boot(base).catch((err) => {
        booting = null; // a later init retries, e.g. after the network comes back
        throw err;
      });
      await booting;
      self.postMessage({ id, result: { ok: true } });
      return;
    }
    if (booting) await booting;
    if (!handle) throw new Error("the engine is not loaded yet");
    const out = JSON.parse(handle(op, JSON.stringify(payload ?? {}), raw ?? null));
    if (out.trace) console.error(out.trace);
    self.postMessage({ id, result: out });
  } catch (err) {
    self.postMessage({ id, result: { error: String(err?.message ?? err), status: 500 } });
  }
};
