// Promise API over the engine worker. Every call resolves to the engine's
// payload or rejects with an Error carrying the readable message.

const worker = new Worker(new URL("./engine.worker.js", import.meta.url), { type: "module" });
const pending = new Map();
const listeners = new Set();
let nextId = 0;
// BASE_URL is "./" (relative build): resolve it against the page, not the worker script
const BASE = new URL(import.meta.env.BASE_URL, document.baseURI).href;

worker.onmessage = ({ data }) => {
  if (data.type === "progress") {
    listeners.forEach((fn) => fn(data.stage));
    return;
  }
  const entry = pending.get(data.id);
  if (!entry) return;
  pending.delete(data.id);
  if (data.result.error !== undefined) {
    const err = new Error(data.result.error);
    err.status = data.result.status;
    entry.reject(err);
  } else {
    entry.resolve(data.result.ok);
  }
};

function call(op, payload, raw) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    // the CSV bytes are copied into the worker, the caller keeps its own buffer
    worker.postMessage({ id, op, payload, raw, base: BASE });
  });
}

export const onProgress = (fn) => {
  listeners.add(fn);
  return () => listeners.delete(fn);
};

export const engine = {
  init: () => call("init"),
  examples: () => call("examples"),
  inspect: (bytes, name) => call("inspect", { name }, bytes),
  check: (payload) => call("check", payload),
  estimate: (payload, bytes) => call("estimate", payload, bytes),
  report: (analysis) => call("report", { analysis }).then((r) => r.html),
  answer: (payload) => call("answer", payload),
};
