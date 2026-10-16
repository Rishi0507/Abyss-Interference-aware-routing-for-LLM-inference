import { useEffect, useState } from "react";
import { api, colorOf, fmt, labelOf, labelSpec, Point, Research, ResearchSet } from "../api";
import LineChart from "../charts/LineChart";
import { resultOrder } from "../data";

const robustLabels: Record<string, string> = {
  "long-prefill-2048": "vLLM long-prefill cap of 2048 tokens",
  "chunk-2048": "Step budget of 2048 tokens instead of 8192",
  "replicas-40pct-slower": "Every replica 40% slower than modeled",
  "replicas-25pct-faster": "Every replica 25% faster than modeled",
  "bias-spread-30pct": "Hidden per-replica speed spread of ±30%",
  "prefill-gemm-slower": "Matrix-multiply efficiency 27% below the model",
  "attention-slower": "Attention efficiency 40% below the model",
  "decode-bandwidth-lower": "Memory-bandwidth efficiency 25% below the model",
  "prefill-and-attention-slower": "Matrix-multiply and attention efficiency both below the model",
  "prefill-faster-decode-slower": "Matrix multiplies 30% faster, memory bandwidth 25% slower",
};

const sensitivityLabels: Record<string, string> = {
  reference: "Reference objectives",
  "tpot-120ms": "TPOT 120 ms",
  "tpot-200ms": "TPOT 200 ms",
  "ttft-2x": "TTFT objective doubled",
  "ttft-2x-tpot-150ms": "TTFT doubled, TPOT 150 ms",
  "tpot-min-16": "TPOT only for responses of 16+ tokens",
};

const baselineFamilies = ["llm-d", "dynamo", "sglang", "chwbl", "sita"];
const heuristics = ["sita", "least-work"];
const notExisting = ["abyss", "abyss-selfish", ...heuristics];

function familyOf(policy: string) {
  return policy.split(":")[0];
}

function cell(pts: Point[], policy: string, scale: number) {
  return pts.find((p) => p.policy === policy && p.scale === scale);
}

function bestBaseline(pts: Point[], scale: number, exclude: string[], only?: string[]) {
  let best: Point | undefined;
  for (const p of pts) {
    if (p.scale !== scale || exclude.includes(familyOf(p.policy))) continue;
    if (only && !only.includes(familyOf(p.policy))) continue;
    if (!best || p.attainment > best.attainment) best = p;
  }
  return best;
}

function Named({ p }: { p?: Point }) {
  return (
    <>
      <Pct p={p} />
      <div className="small muted">{p ? labelSpec(p.policy) : ""}</div>
    </>
  );
}

function Pct({ p }: { p?: Point }) {
  if (!p) return <span className="muted">n/a</span>;
  return (
    <>
      {fmt.pct(p.attainment, 1)}
      <span className="muted"> ± {(100 * p.attainment_sd).toFixed(1)}</span>
    </>
  );
}

function Tuned({ sets: unsorted }: { sets: ResearchSet[] }) {
  const sets = [...unsorted].sort((a, b) => resultOrder.indexOf(a.name) - resultOrder.indexOf(b.name));
  return (
    <section className="section">
      <div className="section-head">
        <h2>Baselines tuned on held-out data</h2>
        <p>
          Each baseline's parameters were grid searched on one trace window or seed, then every policy was evaluated on two windows the search never saw.
          The table lists, per baseline family, the better of its default and tuned settings on the held-out data, which favors the baselines.
        </p>
      </div>
      <div className="grid-2">
        {sets.map((s) => {
          const scales = [...new Set(s.points.map((p) => p.scale))].sort((a, b) => a - b);
          const rows = ["abyss", "abyss-selfish", "llm-d-predicted", "preble", "least-work"];
          return (
            <div className="glass panel table-scroll" key={s.name}>
              <div className="panel-head">
                <h3>{s.title}</h3>
              </div>
              <table className="data">
                <thead>
                  <tr>
                    <th>Policy</th>
                    {scales.map((sc) => (
                      <th key={sc}>{sc}× load</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r} className={r === "abyss" ? "highlight" : undefined}>
                      <td>
                        <span className="policy-name">
                          <i className="swatch" style={{ background: colorOf(r) }} />
                          {labelOf(r)}
                        </span>
                      </td>
                      {scales.map((sc) => (
                        <td key={sc}>
                          <Pct p={cell(s.points, r, sc)} />
                        </td>
                      ))}
                    </tr>
                  ))}
                  {baselineFamilies.map((fam) => {
                    const choice = s.choices?.find((c) => c.family === fam);
                    return (
                      <tr key={fam}>
                        <td>
                          <span className="policy-name">
                            <i className="swatch" style={{ background: colorOf(fam) }} />
                            {labelOf(fam)}
                          </span>
                          <div className="small muted">tuned: {choice ? labelSpec(choice.best).replace(labelOf(fam), "").trim() || "default" : "n/a"}</div>
                        </td>
                        {scales.map((sc) => {
                          const d = cell(s.points, fam, sc);
                          const t = choice ? cell(s.points, choice.best, sc) : undefined;
                          const best = !t || (d && d.attainment >= t.attainment) ? d : t;
                          return (
                            <td key={sc}>
                              <Pct p={best} />
                              <div className="small muted">{best === d ? "default" : "tuned"}</div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Robustness({ sets }: { sets: ResearchSet[] }) {
  const order = Object.keys(robustLabels);
  const sorted = [...sets].sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
  return (
    <section className="section">
      <div className="section-head">
        <h2>Robustness</h2>
        <p>
          The conversation trace rerun with different engine settings, with replicas that run faster or slower than Abyss's model expects, and with model
          errors that change prefill and decode speed by different amounts. Each cell averages three trace windows.
        </p>
      </div>
      <div className="glass panel table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Condition</th>
              <th>Abyss, 0.75×</th>
              <th>Best existing, 0.75×</th>
              <th>Size-interval or least work, 0.75×</th>
              <th>Abyss, 1×</th>
              <th>Best existing, 1×</th>
              <th>Size-interval or least work, 1×</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => {
              const cells = [0.75, 1].map((sc) => ({
                pig: cell(s.points, "abyss", sc),
                other: bestBaseline(s.points, sc, notExisting),
                heur: bestBaseline(s.points, sc, [], heuristics),
              }));
              return (
                <tr key={s.name}>
                  <td>{robustLabels[s.name] ?? s.name}</td>
                  {cells.map((c, i) => [
                    <td key={`p${i}`} style={{ fontWeight: 600 }}>
                      <Pct p={c.pig} />
                    </td>,
                    <td key={`o${i}`}>
                      <Named p={c.other} />
                    </td>,
                    <td key={`h${i}`}>
                      <Named p={c.heur} />
                    </td>,
                  ])}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Sensitivity({ sets }: { sets: ResearchSet[] }) {
  const order = Object.keys(sensitivityLabels);
  const presets = resultOrder.filter((p) => sets.some((s) => s.name.startsWith(p + "-")));
  return (
    <section className="section">
      <div className="section-head">
        <h2>Sensitivity to the latency objectives</h2>
        <p>
          The production traces rerun with looser objectives, and with the TPOT objective applied only to responses of at least 16 tokens. The heuristic
          column is the best of five size-interval and least-work settings in hindsight, which favors it.
        </p>
      </div>
      <div className="grid-2">
        {presets.map((preset) => {
          const rows = sets
            .filter((s) => s.name.startsWith(preset + "-"))
            .map((s) => ({ key: s.name.slice(preset.length + 1), s }))
            .sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
          return (
            <div className="glass panel table-scroll" key={preset}>
              <div className="panel-head">
                <h3>{rows[0]?.s.title}</h3>
              </div>
              <table className="data">
                <thead>
                  <tr>
                    <th>Objectives</th>
                    <th>Abyss, 1×</th>
                    <th>Best existing, 1×</th>
                    <th>Heuristic, 1×</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ key, s }) => (
                    <tr key={key}>
                      <td>{sensitivityLabels[key] ?? key}</td>
                      <td style={{ fontWeight: 600 }}>
                        <Pct p={cell(s.points, "abyss", 1)} />
                      </td>
                      <td>
                        <Named p={bestBaseline(s.points, 1, notExisting)} />
                      </td>
                      <td>
                        <Named p={bestBaseline(s.points, 1, [], heuristics)} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Scaling({ sets, research }: { sets: ResearchSet[]; research: Research }) {
  const fleets = sets
    .map((s) => ({ n: Number(s.name.replace("replicas-", "")), s }))
    .filter((x) => !Number.isNaN(x.n))
    .sort((a, b) => a.n - b.n);
  const policies = ["abyss", "abyss-k8", "llm-d", "dynamo", "preble"];
  const micro = research.microbench;
  return (
    <section className="section">
      <div className="section-head">
        <h2>Fleet size</h2>
        <p>
          Mixed traffic on 8 to 64 Llama 3.1 8B replicas with load proportional to fleet size. The shortlist variant prices only the 8 most promising
          replicas, chosen by a cheap backlog estimate, plus the replica with the longest cached prefix.
        </p>
      </div>
      <div className="grid-2">
        <div className="glass panel table-scroll">
          <div className="panel-head">
            <h3>Requests meeting their targets</h3>
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Replicas</th>
                {policies.map((p) => (
                  <th key={p}>{p === "abyss-k8" ? "Abyss, shortlist of 8" : labelOf(p)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {fleets.map(({ n, s }) => (
                <tr key={n}>
                  <td>{n}</td>
                  {policies.map((p) => {
                    const pt = s.points.find((x) => x.policy === p);
                    return <td key={p}>{pt ? fmt.pct(pt.attainment, 1) : ""}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {micro && (
          <div className="glass panel">
            <div className="panel-head">
              <h3>Decision time</h3>
              <span className="note">{micro.cpu}</span>
            </div>
            <LineChart
              series={["all", "all-parallel", "k8"].map((mode, i) => ({
                key: mode,
                label: mode === "all" ? "Every replica" : mode === "k8" ? "Shortlist of 8" : "Every replica, parallel",
                color: ["#b87a00", "#2a78d6", "#1baf7a"][i],
                points: micro.rows.filter((r) => r.mode === mode).map((r) => ({ x: r.replicas, y: r.micros / 1000 })),
              }))}
              yFormat={(v) => `${v.toFixed(v < 10 ? 1 : 0)} ms`}
              xFormat={(v) => String(v)}
              xLabel="Replicas"
              height={240}
            />
          </div>
        )}
      </div>
    </section>
  );
}

function CalibrationView({ research }: { research: Research }) {
  const c = research.calibration;
  if (!c) return null;
  return (
    <section className="section">
      <div className="section-head">
        <h2>Calibrating the performance model</h2>
        <p>
          <code>abyss calibrate</code> measures prefill and decode timings on any OpenAI-compatible server and fits the step-time model to them. Run
          against an emulated H100 secretly configured 30% slower than its model, the default model is off by {fmt.pct(c.default_median_abs_rel_error)}{" "}
          and the fitted model by {fmt.pct(c.median_abs_rel_error, 1)} (median).
        </p>
      </div>
      <div className="glass panel table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Measurement</th>
              <th>Measured</th>
              <th>Fitted model</th>
              <th>Default model</th>
            </tr>
          </thead>
          <tbody>
            {c.rows.map((r, i) => (
              <tr key={i}>
                <td>{r.sample.kind === "prefill" ? `Prefill, ${fmt.num(r.sample.prompt_tokens)} tokens` : `Decode step, batch of ${r.sample.batch}`}</td>
                <td>{fmt.msFine(r.sample.seconds)}</td>
                <td>{fmt.msFine(r.predicted_s)}</td>
                <td>{fmt.msFine(r.default_predicted_s)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function ResearchPage() {
  const [data, setData] = useState<Research | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.research().then(setData).catch((e: Error) => setError(e.message));
  }, []);
  return (
    <div className="page" style={{ gap: 56 }}>
      <div className="section-head">
        <h2>Research</h2>
        <p>
          Additional experiments that test whether the main results hold up: stronger and tuned baselines, different engine settings, replicas that do
          not match the model, larger fleets, and the calibration path for real hardware. The full write-up is available as a paper draft.
        </p>
        <div style={{ marginTop: 8 }}>
          <a className="btn primary" href="/abyss-paper.pdf" target="_blank" rel="noreferrer">
            Read the paper draft (PDF)
          </a>
        </div>
      </div>
      {error && <div className="error-box">Could not load research results: {error}</div>}
      {!data && !error && <div className="skeleton" style={{ height: 420 }} />}
      {data?.tuned && data.tuned.length > 0 && <Tuned sets={data.tuned} />}
      {data?.robustness && data.robustness.length > 0 && <Robustness sets={data.robustness} />}
      {data?.sensitivity && data.sensitivity.length > 0 && <Sensitivity sets={data.sensitivity} />}
      {data?.scaling && data.scaling.length > 0 && <Scaling sets={data.scaling} research={data} />}
      {data && <CalibrationView research={data} />}
    </div>
  );
}
