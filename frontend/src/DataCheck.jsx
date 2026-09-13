/* The Data tab: how an uploaded CSV was read, and which columns go on the canvas. */

export const KIND_NOTE = {
  binary: "two values",
  discrete: "few numeric values",
  continuous: "numeric",
  categorical: "categories",
  identifier: "looks like an ID",
  text: "free text",
  constant: "never varies",
  empty: "all missing",
};
export const LEFT_OUT = new Set(["identifier", "text", "constant", "empty"]);

export default function DataCheck({ source, dataInfo, profile, columnNames, onToggle }) {
  return (
    <div className="data-check">
      <h2>{source.name}</h2>
      {dataInfo && (
        <p className="hint">
          {dataInfo.rows} rows · {dataInfo.delimiter}-separated · decimal {dataInfo.decimal}
          {dataInfo.encoding !== "utf-8" && <> · {dataInfo.encoding} encoding</>}
        </p>
      )}
      {dataInfo && Object.keys(dataInfo.renamed).length > 0 && (
        <p className="warn">
          Renamed for the DAG:{" "}
          {Object.entries(dataInfo.renamed).map(([a, b]) => `“${a}” → ${b}`).join(", ")}
        </p>
      )}
      <p>
        Tick the variables that belong in the DAG. The treatment needs exactly two
        values; the outcome a number or two values. Rows with a missing value in the
        variables you use are dropped, and counted in the answer.
      </p>
      <table>
        <tbody>
          {profile.map((p) => {
            const on = columnNames.includes(p.name);
            return (
              <tr key={p.name} className={on ? "" : "off"}>
                <td>
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={p.kind === "empty"}
                    onChange={() => onToggle(p.name)}
                    aria-label={`include ${p.name}`}
                  />
                </td>
                <td>
                  <div className="col-name">{p.name}</div>
                  <div className="col-examples" title={p.examples.join(", ")}>
                    {p.examples.join(", ")}
                  </div>
                </td>
                <td>
                  <span className={`kind ${LEFT_OUT.has(p.kind) ? "kind-out" : ""}`}>
                    {KIND_NOTE[p.kind]}
                  </span>
                  {p.missing > 0 && <div className="col-missing">{p.missing} missing</div>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
