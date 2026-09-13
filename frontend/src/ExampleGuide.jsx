/* The Guide tab of a bundled example: its story, and DAG edits to try in one click. */

/* Whether a guide edit {op: 'add' | 'remove' | 'flip', edge: [src, dst]} is already on the canvas. */
function editApplied(edges, { op, edge: [s, d] }) {
  const has = (a, b) => edges.some((e) => e.source === a && e.target === b);
  if (op === "add") return has(s, d);
  if (op === "flip") return has(d, s) && !has(s, d);
  return !has(s, d) && !has(d, s);
}

export default function ExampleGuide({ guide, edges, onApply, onReset }) {
  return (
    <div className="example-guide">
      <h2>{guide.title}</h2>
      <p>{guide.story}</p>
      <h3>Why the direct comparison is wrong</h3>
      <p>{guide.why}</p>
      <h3>Try changing the DAG</h3>
      <ul>
        {guide.tries.map((tip) => {
          const applied = editApplied(edges, tip.edit);
          return (
            <li key={tip.edit.op + tip.edit.edge.join()}>
              <p>{tip.text}</p>
              <button
                className={applied ? "chip small applied" : "chip small"}
                disabled={applied}
                onClick={() => onApply(tip.edit)}
              >
                {applied ? "applied — now estimate" : "apply to the DAG"}
              </button>
            </li>
          );
        })}
      </ul>
      <button className="link" onClick={onReset}>↺ reset to the original DAG</button>
    </div>
  );
}
