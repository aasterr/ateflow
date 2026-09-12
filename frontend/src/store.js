// Saved analyses live in this browser (IndexedDB), dataset bytes included, so a
// saved analysis reloads and re-runs without anything ever leaving the machine.

const DB = "ateflow";
const STORE = "analyses";

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(req?.result);
    t.onerror = () => reject(t.error);
  });
}

export const store = {
  save: (analysis) => tx("readwrite", (s) => s.add({ ...analysis, created_at: new Date().toISOString() })),
  get: (id) => tx("readonly", (s) => s.get(id)),
  remove: (id) => tx("readwrite", (s) => s.delete(id)),
  list: async () => {
    const all = (await tx("readonly", (s) => s.getAll())) ?? [];
    return all
      .map(({ id, name, created_at, result }) => ({ id, name, created_at, adjusted: result.adjusted.value }))
      .reverse();
  },
};
