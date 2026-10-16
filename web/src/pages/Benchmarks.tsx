import { useEffect, useMemo, useState } from "react";
import { colorOf, fmt, labelOf, plotted, policyOrder, ResultSet } from "../api";
import LineChart, { Series } from "../charts/LineChart";
import { fleetText, Loadable, sortResults } from "../data";

const defaultOn = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "preble", "sita"];

type Metric = { key: "attainment" | "goodput" | "ttft_p50" | "tpot_p50"; title: string; format: (v: number) => string; domain?: [number, number]; log?: boolean };

const metrics: Metric[] = [
  { key: "attainment", title: "Requests meeting their latency targets", format: (v) => fmt.pct(v), domain: [0, 1] },
  { key: "goodput", title: "Goodput, requests per second", format: (v) => v.toFixed(1) },
  { key: "ttft_p50", title: "Median time to first token", format: (v) => fmt.ms(v), log: true },
  { key: "tpot_p50", title: "Median time per output token", format: (v) => fmt.ms(v), log: true },
];

export default function Benchmarks({ results }: { results: Loadable<ResultSet[]> }) {
  const sets = useMemo(() => sortResults(results.data ?? []), [results.data]);
  const [name, setName] = useState<string>("");
  const [on, setOn] = useState<string[]>(defaultOn);
  const [scale, setScale] = useState<number | null>(null);
  const rs = sets.find((s) => s.name === name) ?? sets[0];

  useEffect(() => {
    if (rs && (scale === null || !rs.points.some((p) => p.scale === scale))) setScale(rs.detail_scale);
  }, [rs, scale]);

  if (results.loading) return <div className="page"><div className="skeleton" style={{ height: 480 }} /></div>;
  if (!rs) {
    return (
      <div className="page">
        <div className="glass empty">
          <h3>No benchmark results yet</h3>
          <p>Run a sweep to populate this page.</p>
          <div className="code">abyss bench --preset conversation-70b --seeds 3</div>
        </div>
      </div>
    );
  }

  const available = policyOrder.filter((p) => plotted(p) && rs.points.some((pt) => pt.policy === p));
  const scales = [...new Set(rs.points.map((p) => p.scale))].sort((a, b) => a - b);
  const series = (m: Metric): Series[] =>
    available
      .filter((p) => on.includes(p))
      .map((p) => ({
        key: p,
        label: labelOf(p),
        color: colorOf(p),
        emphasis: p === "abyss",
        points: rs.points
          .filter((pt) => pt.policy === p)
          .map((pt) => {
            const y = pt[m.key];
            const sd = m.key === "attainment" ? pt.attainment_sd : 0;
            return { x: Number(pt.rate.toFixed(2)), y, ...(sd ? { lo: Math.max(0, y - sd), hi: Math.min(1, y + sd) } : {}) };
          }),
      }));
  const rows = rs.points.filter((p) => p.scale === scale).sort((a, b) => policyOrder.indexOf(a.policy) - policyOrder.indexOf(b.policy));
  const slo = rs.preset.cluster.slo;
  const lengths = rs.details.filter((d) => ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "preble", "sita"].includes(d.policy)).sort((a, b) => policyOrder.indexOf(a.policy) - policyOrder.indexOf(b.policy));

  return (
    <div className="page">
      <div className="section-head">
        <div style={{ display: "grid", gap: 8 }}>
          <h2>Benchmarks</h2>
          <p className="secondary" style={{ maxWidth: "48em" }}>
            Each point averages {rs.seeds} independent runs. Replicas run slightly faster or slower than their model in ways the router cannot see, no
            policy knows output lengths in advance, and clients give up on requests that wait too long for a first token.
          </p>
        </div>
      </div>
      <div className="segmented" role="group" aria-label="Workload">
        {sets.map((s) => (
          <button key={s.name} aria-pressed={s.name === rs.name} onClick={() => setName(s.name)}>
            {s.title}
          </button>
        ))}
      </div>
      <div className="glass panel" style={{ display: "grid", gap: 6 }}>
        <p>{rs.preset.description}</p>
        <p className="small muted tabular">
          {fmt.num(rs.workload.requests)} requests at {rs.workload.rate_rps.toFixed(2)} req/s reference rate, mean prompt {fmt.num(rs.workload.mean_input)}{" "}
          tokens (p99 {fmt.num(rs.workload.p99_input)}), mean output {fmt.num(rs.workload.mean_output)} tokens, {fmt.pct(rs.workload.reuse_ratio)} of prompt
          tokens reusable. Fleet: {fleetText(rs.fleet)}. Objectives: first token within {slo.ttft_base_s} s plus {slo.ttft_per_ktok_s} s per thousand prompt
          tokens, {Math.round(slo.tpot_s * 1000)} ms per output token.
        </p>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {available.map((p) => (
          <button key={p} className="chip" aria-pressed={on.includes(p)} onClick={() => setOn((o) => (o.includes(p) ? o.filter((x) => x !== p) : [...o, p]))}>
            <i className="swatch" style={{ background: colorOf(p) }} />
            {labelOf(p)}
          </button>
        ))}
      </div>
      <div className="grid-2">
        {metrics.map((m) => (
          <div className="glass panel" key={m.key}>
            <div className="panel-head">
              <h3>{m.title}</h3>
            </div>
            <LineChart
              series={series(m)}
              yDomain={m.domain}
              logY={m.log}
              yFormat={m.format}
              xFormat={(v) => v.toFixed(1)}
              xLabel="Offered load, requests per second"
              legend={false}
              height={260}
            />
          </div>
        ))}
      </div>
      <div className="glass panel">
        <div className="panel-head" style={{ flexWrap: "wrap" }}>
          <h3>All policies at one load level</h3>
          <div className="segmented" role="group" aria-label="Load level">
            {scales.map((s) => (
              <button key={s} aria-pressed={s === scale} onClick={() => setScale(s)}>
                {s}× trace rate
              </button>
            ))}
          </div>
        </div>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Policy</th>
                <th>SLO met</th>
                <th>Goodput</th>
                <th>TTFT p50</th>
                <th>TTFT p99</th>
                <th>TPOT p50</th>
                <th>TPOT p99</th>
                <th>Cache hits</th>
                <th>Abandoned</th>
                <th>$ per M good tokens</th>
                <th>Decision time</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.policy} className={p.policy === "abyss" ? "highlight" : undefined}>
                  <td>
                    <span className="policy-name">
                      <i className="swatch" style={{ background: colorOf(p.policy) }} />
                      {labelOf(p.policy)}
                    </span>
                  </td>
                  <td>
                    {fmt.pct(p.attainment, 1)}
                    <span className="muted"> ± {(100 * p.attainment_sd).toFixed(1)}</span>
                  </td>
                  <td>{p.goodput.toFixed(2)}/s</td>
                  <td>{fmt.ms(p.ttft_p50)}</td>
                  <td>{fmt.ms(p.ttft_p99)}</td>
                  <td>{fmt.ms(p.tpot_p50)}</td>
                  <td>{fmt.ms(p.tpot_p99)}</td>
                  <td>{fmt.pct(p.hit_rate, 1)}</td>
                  <td>{fmt.pct(p.timed_out, 1)}</td>
                  <td>{p.usd_per_m_good_tokens > 0 && p.usd_per_m_good_tokens < 1e5 ? fmt.usd(p.usd_per_m_good_tokens) : ""}</td>
                  <td>{p.decision_us >= 1000 ? `${(p.decision_us / 1000).toFixed(1)} ms` : `${Math.max(1, Math.round(p.decision_us))} µs`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {lengths.length > 0 && (
        <div className="glass panel">
          <div className="panel-head">
            <h3>Attainment by prompt length</h3>
            <span className="note">{rs.detail_scale}× trace rate, first window</span>
          </div>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Policy</th>
                  {lengths[0].length_attainment.map((b) => (
                    <th key={b.label}>
                      {b.label} tokens <span className="muted">({fmt.num(b.requests)})</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {lengths.map((d) => (
                  <tr key={d.policy} className={d.policy === "abyss" ? "highlight" : undefined}>
                    <td>
                      <span className="policy-name">
                        <i className="swatch" style={{ background: colorOf(d.policy) }} />
                        {labelOf(d.policy)}
                      </span>
                    </td>
                    {d.length_attainment.map((b) => (
                      <td key={b.label}>{b.requests ? fmt.pct(b.attainment) : ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
