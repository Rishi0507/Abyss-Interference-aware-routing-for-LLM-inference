import { useEffect, useMemo, useState } from "react";
import { api, colorOf, DecisionSample, fmt, labelOf, policyOrder, SimResponse, SimRun } from "../api";
import LineChart, { Series } from "../charts/LineChart";
import { HBars, HeatStrip, StackedBars } from "../charts/Bars";
import { useCatalog } from "../data";
import { costSegments } from "./Overview";

const selectable = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "round-robin"];
const initial = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "sita"];

export default function Lab() {
  const catalog = useCatalog();
  const [preset, setPreset] = useState("conversation-70b");
  const [scale, setScale] = useState(1);
  const [duration, setDuration] = useState(600);
  const [seed, setSeed] = useState(0);
  const [policies, setPolicies] = useState<string[]>(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<SimResponse | null>(null);

  const run = () => {
    setBusy(true);
    setError(null);
    api
      .simulate({ preset, policies, scale, duration, seed })
      .then(setRes)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    if (window.location.hash.includes("run=1")) run();
  }, []);

  const presets = catalog.data?.presets ?? [];
  const current = presets.find((p) => p.name === preset);

  return (
    <div className="page">
      <div className="section-head">
        <div style={{ display: "grid", gap: 8 }}>
          <h2>Lab</h2>
          <p className="secondary" style={{ maxWidth: "46em" }}>
            Replay a workload on a simulated fleet with several routing policies side by side, then step through the decisions Abyss made.
          </p>
        </div>
      </div>
      <div className="lab-grid">
        <div className="glass panel" style={{ display: "grid", gap: 18, position: "sticky", top: 96 }}>
          <div className="field">
            <label htmlFor="preset">Workload and fleet</label>
            <select id="preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.title}
                </option>
              ))}
            </select>
            {current && <span className="small muted">{current.description}</span>}
          </div>
          <div className="field">
            <label htmlFor="scale">
              Load: <span className="tabular">{scale.toFixed(2)}×</span> the trace rate
            </label>
            <input id="scale" type="range" min={0.25} max={1.5} step={0.05} value={scale} onChange={(e) => setScale(Number(e.target.value))} />
          </div>
          <div className="grid-2" style={{ gap: 12 }}>
            <div className="field">
              <label htmlFor="duration">Window</label>
              <select id="duration" value={duration} onChange={(e) => setDuration(Number(e.target.value))}>
                {[300, 600, 900, 1200].map((d) => (
                  <option key={d} value={d}>
                    {d / 60} min
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="seed">Seed</label>
              <select id="seed" value={seed} onChange={(e) => setSeed(Number(e.target.value))}>
                {[0, 1, 2, 3, 4].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="field">
            <span className="label">Policies</span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {selectable.map((p) => (
                <button
                  key={p}
                  className="chip"
                  aria-pressed={policies.includes(p)}
                  onClick={() => setPolicies((o) => (o.includes(p) ? o.filter((x) => x !== p) : [...o, p]))}
                >
                  <i className="swatch" style={{ background: colorOf(p) }} />
                  {labelOf(p)}
                </button>
              ))}
            </div>
          </div>
          <button className="btn primary" onClick={run} disabled={busy || policies.length === 0} style={{ justifyContent: "center" }}>
            {busy && <span className="spinner" />}
            {busy ? "Simulating" : "Run simulation"}
          </button>
          {res && !busy && (
            <span className="small muted tabular">
              {fmt.num(res.workload.requests)} requests per policy, simulated in {res.wall_s.toFixed(1)} s.
            </span>
          )}
          {error && <div className="error-box">{error}</div>}
        </div>
        <div style={{ display: "grid", gap: 20 }}>{res ? <Results res={res} /> : <Placeholder busy={busy} />}</div>
      </div>
    </div>
  );
}

function smooth(pts: { x: number; y: number }[]) {
  return pts.map((p, i) => {
    const w = pts.slice(Math.max(0, i - 1), i + 2);
    return { x: p.x, y: w.reduce((a, q) => a + q.y, 0) / w.length };
  });
}

function heatRows(run: SimRun, metric: "prefill" | "decode" | "busy", all: SimRun[]) {
  const pick = (b: SimRun["timeline"][number], i: number) =>
    metric === "busy" ? b.busy[i] ?? 0 : metric === "prefill" ? b.prefill_tokens?.[i] ?? 0 : b.decode_tokens?.[i] ?? 0;
  let max = 1e-9;
  for (const r of all) r.timeline.forEach((b) => r.summary.replicas.forEach((_, i) => (max = Math.max(max, pick(b, i)))));
  if (metric === "busy") max = 1;
  return run.summary.replicas.map((_, i) => ({
    key: String(i),
    label: `Replica ${i}`,
    raw: run.timeline.map((b) => pick(b, i)),
    values: run.timeline.map((b) => pick(b, i) / max),
  }));
}

function Placeholder({ busy }: { busy: boolean }) {
  if (busy) return <div className="skeleton" style={{ height: 520 }} />;
  return (
    <div className="glass empty" style={{ minHeight: 420 }}>
      <h3>Choose a workload and run a simulation</h3>
      <p style={{ maxWidth: "36em" }}>
        Results show how many requests met their targets under each policy, how that changed over time, how load spread across replicas, and a
        sample of Abyss's individual decisions.
      </p>
    </div>
  );
}

function Results({ res }: { res: SimResponse }) {
  const runs = [...res.runs].filter((r) => !r.error).sort((a, b) => policyOrder.indexOf(a.policy) - policyOrder.indexOf(b.policy));
  const bucket = runs[0]?.timeline.length ? res.workload.duration_s / runs[0].timeline.length : 10;
  const [heat, setHeat] = useState(runs.find((r) => r.policy === "abyss")?.policy ?? runs[0]?.policy);
  const [metric, setMetric] = useState<"prefill" | "decode" | "busy">("prefill");
  const series = (key: "attainment" | "ttft_p99"): Series[] =>
    runs.map((r) => ({
      key: r.policy,
      label: labelOf(r.policy),
      color: colorOf(r.policy),
      emphasis: r.policy === "abyss",
      points: smooth(r.timeline.filter((b) => b.arrivals > 0).map((b) => ({ x: Math.round(b.t), y: b[key] }))),
    }));
  const heatRun = runs.find((r) => r.policy === heat) ?? runs[0];
  return (
    <>
      <div className="glass panel">
        <div className="panel-head">
          <h3>Requests meeting their latency targets</h3>
          <span className="note">{res.fleet.length} replicas</span>
        </div>
        <HBars
          max={1}
          rows={runs.map((r) => ({
            key: r.policy,
            label: labelOf(r.policy),
            color: colorOf(r.policy),
            value: r.summary.slo_attainment,
            text: fmt.pct(r.summary.slo_attainment, 1),
            emphasis: r.policy === "abyss",
            detail: `TTFT met ${fmt.pct(r.summary.ttft_attainment)}, TPOT met ${fmt.pct(r.summary.tpot_attainment)}, ${r.summary.timed_out} abandoned`,
          }))}
        />
      </div>
      <div className="glass panel table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Policy</th>
              <th>Goodput</th>
              <th>TTFT p50</th>
              <th>TTFT p99</th>
              <th>TPOT p50</th>
              <th>TPOT p99</th>
              <th>Cache hits</th>
              <th>Abandoned</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.policy} className={r.policy === "abyss" ? "highlight" : undefined}>
                <td>
                  <span className="policy-name">
                    <i className="swatch" style={{ background: colorOf(r.policy) }} />
                    {labelOf(r.policy)}
                  </span>
                </td>
                <td>{r.summary.goodput_rps.toFixed(2)}/s</td>
                <td>{fmt.ms(r.summary.ttft.p50)}</td>
                <td>{fmt.ms(r.summary.ttft.p99)}</td>
                <td>{fmt.ms(r.summary.tpot.p50)}</td>
                <td>{fmt.ms(r.summary.tpot.p99)}</td>
                <td>{fmt.pct(r.summary.hit_rate, 1)}</td>
                <td>{fmt.num(r.summary.timed_out)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid-2">
        <div className="glass panel">
          <div className="panel-head">
            <h3>Attainment over the window</h3>
          </div>
          <LineChart series={series("attainment")} yDomain={[0, 1]} yFormat={(v) => fmt.pct(v)} xFormat={(v) => `${Math.round(v / 60)}m`} markers={false} legend={false} directLabels={false} height={230} />
        </div>
        <div className="glass panel">
          <div className="panel-head">
            <h3>TTFT p99 over the window</h3>
          </div>
          <LineChart series={series("ttft_p99")} logY yFormat={(v) => fmt.ms(v)} xFormat={(v) => `${Math.round(v / 60)}m`} markers={false} legend={false} directLabels={false} height={230} />
        </div>
      </div>
      {heatRun && (
        <div className="glass panel">
          <div className="panel-head" style={{ flexWrap: "wrap" }}>
            <h3>Replica load over the window</h3>
            <div className="segmented" role="group" aria-label="Metric">
              {(
                [
                  ["prefill", "Prefill tokens"],
                  ["decode", "Decode tokens"],
                  ["busy", "Busy time"],
                ] as const
              ).map(([k, l]) => (
                <button key={k} aria-pressed={metric === k} onClick={() => setMetric(k)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div style={{ marginBottom: 14 }}>
            <div className="segmented" role="group" aria-label="Policy">
              {runs.map((r) => (
                <button key={r.policy} aria-pressed={r.policy === heatRun.policy} onClick={() => setHeat(r.policy)}>
                  {labelOf(r.policy)}
                </button>
              ))}
            </div>
          </div>
          <HeatStrip
            buckets={heatRun.timeline.length}
            label={(i) => `${Math.round((i * bucket) / 60)} to ${Math.round(((i + 1) * bucket) / 60)} min`}
            format={(v) => (metric === "busy" ? `${Math.round(100 * Math.min(v, 1))}% busy` : `${fmt.k(v / bucket)} tokens/s`)}
            rows={heatRows(heatRun, metric, runs)}
          />
          <p className="small muted" style={{ marginTop: 10 }}>
            {heatRun.summary.replicas.map((r, i) => `Replica ${i}: ${r.name.replace(" x", " TP")}, ${fmt.num(r.requests)} requests, ${fmt.pct(r.hit_rate)} cache hits`).join(". ")}.
          </p>
        </div>
      )}
      {res.decisions.length > 0 && <Decisions decisions={res.decisions} overrides={res.override_ratio} />}
    </>
  );
}

function Decisions({ decisions, overrides }: { decisions: DecisionSample[]; overrides: number }) {
  const [only, setOnly] = useState(true);
  const list = useMemo(() => decisions.filter((d) => !only || d.chosen !== d.selfish), [decisions, only]);
  const [idx, setIdx] = useState(0);
  const d = list[Math.min(idx, list.length - 1)];
  return (
    <div className="glass panel">
      <div className="panel-head" style={{ flexWrap: "wrap" }}>
        <h3>Abyss decisions</h3>
        <span className="note">Different from the request's own best option in {fmt.pct(overrides, 1)} of placements</span>
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 16 }}>
        <div className="segmented" role="group" aria-label="Filter">
          <button aria-pressed={only} onClick={() => { setOnly(true); setIdx(0); }}>
            Overridden only
          </button>
          <button aria-pressed={!only} onClick={() => { setOnly(false); setIdx(0); }}>
            All sampled
          </button>
        </div>
        <button className="btn" onClick={() => setIdx((i) => Math.max(0, i - 1))} disabled={idx === 0}>
          Previous
        </button>
        <button className="btn" onClick={() => setIdx((i) => Math.min(list.length - 1, i + 1))} disabled={idx >= list.length - 1}>
          Next
        </button>
        <span className="small muted tabular">
          {list.length ? `${Math.min(idx, list.length - 1) + 1} of ${list.length}` : "none"}
        </span>
      </div>
      {d && (
        <>
          <p className="secondary small" style={{ marginBottom: 14 }}>
            At {d.t.toFixed(0)} s: a {fmt.num(d.input)} token prompt with a {fmt.ms(d.ttft_slo)} first-token target.
          </p>
          <StackedBars
            segments={costSegments}
            format={(v) => v.toFixed(2)}
            rows={[...d.costs]
              .sort((a, b) => a.backend - b.backend)
              .map((c) => ({
                key: String(c.backend),
                label: `Replica ${c.backend}`,
                values: { own: c.own, interference: c.interference, cache: c.cache, delay: Math.max(c.total - c.own - c.interference - c.cache, 0) },
                emphasis: c.backend === d.chosen,
                marker: c.backend === d.chosen ? "Chosen" : c.backend === d.selfish ? "Best on its own" : undefined,
                note: `Predicted TTFT ${fmt.ms(c.pred_ttft)}, TPOT ${fmt.msFine(c.pred_tpot)}, ${c.hit_blocks} cached blocks, ${c.evicted_blocks} evictions`,
              }))}
          />
        </>
      )}
    </div>
  );
}
