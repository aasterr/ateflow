// Copies the Python engine and the example datasets into public/py, where the
// Web Worker fetches them and writes them into Pyodide's filesystem.
// Runs before `vite` and `vite build`; public/py is generated, not committed.
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const out = resolve(root, "frontend", "public", "py");

// The command line stays out: the browser has no terminal.
const CLI_ONLY = new Set(["cli.py", "__main__.py"]);

const engine = readdirSync(join(root, "ateflow"))
  .filter((f) => f.endsWith(".py") && !CLI_ONLY.has(f))
  .map((f) => `ateflow/${f}`);
const examples = readdirSync(join(root, "examples"))
  .filter((f) => f.endsWith(".csv") || f.endsWith(".dag"))
  .map((f) => `examples/${f}`);

for (const needed of ["examples/corridor.csv", "examples/onboarding.csv", "examples/ads.csv"]) {
  if (!examples.includes(needed)) {
    console.error(`missing ${needed}: run \`python examples/make_data.py\` first`);
    process.exit(1);
  }
}

rmSync(out, { recursive: true, force: true });
// A hash of every file: the worker puts it in each URL, so a browser that cached
// an older release can never mix its modules with the new ones.
const hash = createHash("sha256");
for (const rel of [...engine, ...examples]) {
  const dest = join(out, rel);
  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(join(root, rel), dest);
  hash.update(rel).update(readFileSync(dest));
}
const version = hash.digest("hex").slice(0, 12);
writeFileSync(join(out, "manifest.json"), JSON.stringify({ version, files: [...engine, ...examples] }, null, 2));
console.log(`synced ${engine.length} engine modules and ${examples.length} example files into public/py`);
if (!existsSync(join(out, "ateflow", "service.py"))) process.exit(1);
