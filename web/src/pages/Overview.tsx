import { useEffect, useMemo, useState } from "react";
import { api, colorOf, dataColor, DecisionSample, fmt, labelOf, ResultSet } from "../api";
import LineChart, { Series } from "../charts/LineChart";
import { StackedBars } from "../charts/Bars";
import { fleetText, Loadable, sortResults } from "../data";

const heroPolicies = ["abyss", "abyss-selfish", "llm-d-predicted", "dynamo", "preble", "sita"];

export const costSegments = [
  { key: "own", label: "Risk to this request", color: "#2a78d6" },
  { key: "interference", label: "Harm to other requests", color: "#eb6834" },
  { key: "cache", label: "Cache it would evict", color: "#1baf7a" },
  { key: "delay", label: "Latency tie-breaker", color: "#a3acbd" },
];

export function attainmentSeries(rs: ResultSet, policies: string[], x: "rate" | "scale" = "rate"): Series[] {
  return policies
    .filter((p) => rs.points.some((pt) => pt.policy === p))
    .map((p) => ({
      key: p,
      label: labelOf(p),
      color: colorOf(p),
      emphasis: p === "abyss",
      points: rs.points
        .filter((pt) => pt.policy === p)
        .map((pt) => ({ x: x === "rate" ? Number(pt.rate.toFixed(2)) : pt.scale, y: pt.attainment, lo: Math.max(0, pt.attainment - pt.attainment_sd), hi: Math.min(1, pt.attainment + pt.attainment_sd) })),
    }));
}

function Hero({ results }: { results: Loadable<ResultSet[]> }) {
  const main = results.data?.find((r) => r.name === "conversation-70b") ?? results.data?.[0];
  return (
    <section className="hero">
      <div style={{ display: "grid", gap: 22 }}>
        <h1>Request routing for LLM inference fleets</h1>
        <p className="lede">
          Abyss sits in front of a group of model servers and decides which one handles each request. Before placing a request it simulates every server a
          few seconds ahead, then picks the one where the request meets its latency target while slowing down the requests already running the least.
        </p>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <a className="btn primary" href="#/lab">
            Open the lab
          </a>
          <a className="btn" href="#/benchmarks">
            See benchmarks
          </a>
        </div>
      </div>
      <div className="glass panel">
        <div className="panel-head">
          <h3>Requests meeting their latency targets</h3>
        </div>
        {main ? (
          <>
            <LineChart
              series={attainmentSeries(main, heroPolicies)}
              yDomain={[0, 1]}
              yFormat={(v) => fmt.pct(v)}
              xFormat={(v) => v.toFixed(1)}
              xLabel="Offered load, requests per second"
              height={290}
            />
            <p className="small muted" style={{ marginTop: 10 }}>
              Mooncake conversation trace replayed on {fleetText(main.fleet)} with Llama 3.1 70B. Mean of {main.seeds} disjoint trace windows; bands show one
              standard deviation.
            </p>
          </>
        ) : (
          <div className="skeleton" style={{ height: 320 }} />
        )}
      </div>
    </section>
  );
}

function pickDecision(ds: DecisionSample[]): DecisionSample | undefined {
  let best: DecisionSample | undefined;
  let score = -1;
  for (const d of ds) {
    if (d.selfish === d.chosen || d.costs.length < 2 || d.input < 2000) continue;
    const self = d.costs.find((c) => c.backend === d.selfish);
    const chosen = d.costs.find((c) => c.backend === d.chosen);
    if (!self || !chosen || chosen.pred_ttft > d.ttft_slo) continue;
    const s = self.interference + self.cache - chosen.interference - chosen.cache;
    if (s > score) {
      score = s;
      best = d;
    }
  }
  return best ?? ds.find((d) => d.selfish !== d.chosen) ?? ds[0];
}

function Pricing() {
  const [decision, setDecision] = useState<DecisionSample | null>(null);
  const [overrides, setOverrides] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .simulate({ preset: "conversation-70b", policies: ["abyss"], scale: 1, duration: 300 })
      .then((r) => {
        setDecision(pickDecision(r.decisions) ?? null);
        setOverrides(r.override_ratio);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const rows = useMemo(() => {
    if (!decision) return [];
    return [...decision.costs]
      .sort((a, b) => a.backend - b.backend)
      .map((c) => ({
        key: String(c.backend),
        label: `Replica ${c.backend}`,
        values: { own: c.own, interference: c.interference, cache: c.cache, delay: Math.max(c.total - c.own - c.interference - c.cache, 0) },
        emphasis: c.backend === decision.chosen,
        marker: c.backend === decision.chosen ? "Chosen" : c.backend === decision.selfish ? "Best on its own" : undefined,
        note: `Predicted TTFT ${fmt.ms(c.pred_ttft)}, TPOT ${fmt.msFine(c.pred_tpot)}, ${c.hit_blocks} cached blocks, ${c.evicted_blocks} evictions`,
      }));
  }, [decision]);

  const chosen = decision?.costs.find((c) => c.backend === decision.chosen);
  const selfish = decision?.costs.find((c) => c.backend === decision.selfish);

  return (
    <section className="section">
      <div className="section-head">
        <h2>How Abyss scores a replica</h2>
      </div>
      <div className="grid-2">
        <div className="glass panel" style={{ display: "grid", gap: 20, alignContent: "start" }}>
          <Term color={costSegments[0].color} title="Risk to this request">
            The chance that the request misses its first-token or per-token target on that replica, found by simulating the replica with the request
            added.
          </Term>
          <Term color={costSegments[1].color} title="Harm to other requests">
            The same simulation without the request. The difference in how many of the replica's other requests would miss their targets is the cost
            this request puts on them.
          </Term>
          <Term color={costSegments[2].color} title="Cache it would evict">
            A new request pushes older cached prompts out of GPU memory. Each one is valued by how likely it is to be reused and how long it would take
            to rebuild.
          </Term>
          <p className="small muted">
            All three are measured in expected missed targets, so they add up. The request goes to the replica with the lowest total.
            {overrides !== null && ` In this trace that differs from the request's own best option ${fmt.pct(overrides)} of the time.`}
          </p>
        </div>
        <div className="glass panel">
          <div className="panel-head">
            <h3>One decision from the conversation trace</h3>
            {decision && <span className="note tabular">t = {decision.t.toFixed(0)} s</span>}
          </div>
          {error && <div className="error-box">Simulation unavailable: {error}</div>}
          {!decision && !error && <div className="skeleton" style={{ height: 260 }} />}
          {decision && (
            <>
              <StackedBars rows={rows} segments={costSegments} format={(v) => v.toFixed(2)} />
              <p className="secondary" style={{ marginTop: 16 }}>
                A {fmt.num(decision.input)} token prompt with a {fmt.ms(decision.ttft_slo)} first-token target.
                {selfish && chosen && selfish.backend !== chosen.backend
                  ? ` On its own, replica ${selfish.backend} looked best, but sending it there would cause about ${selfish.interference.toFixed(1)} missed targets among requests already running. Abyss sent it to replica ${chosen.backend}, ${chosen.pred_ttft <= decision.ttft_slo ? `where it still meets its target (${fmt.ms(chosen.pred_ttft)}).` : `where it disturbs fewer running requests.`}`
                  : chosen
                    ? ` Abyss sends it to replica ${chosen.backend}, predicted TTFT ${fmt.ms(chosen.pred_ttft)}.`
                    : ""}
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function Term({ color, title, children }: { color: string; title: string; children: React.ReactNode }) {
  return (
    <div className="term" style={{ "--term": color } as React.CSSProperties}>
      <h3>{title}</h3>
      <p className="secondary">{children}</p>
    </div>
  );
}

function Roles({ results }: { results: Loadable<ResultSet[]> }) {
  const rs = results.data?.find((r) => r.name === "conversation-70b");
  const pick = (p: string) => rs?.details.find((d) => d.policy === p);
  const left = pick("round-robin");
  const right = pick("abyss");
  if (!rs || !left || !right) return null;
  const share = (d: NonNullable<typeof left>) => {
    const pf = d.replicas.reduce((a, r) => a + r.prefill_tokens, 0);
    const dc = d.replicas.reduce((a, r) => a + r.decode_tokens, 0);
    return d.replicas.map((r, i) => ({ i, prefill: r.prefill_tokens / pf, decode: r.decode_tokens / dc, busy: r.busy, preempt: r.preemptions }));
  };
  return (
    <section className="section">
      <div className="section-head">
        <h2>Effect on replica workloads</h2>
        <p>
          Abyss has no rule that gives replicas roles. Because it charges for the slowdown a long prompt causes, long prompts end up on one replica while
          the others do most of the token generation. A size-interval rule makes the same split by fixed cutoff; Abyss still uses the other replicas for
          long prompts when they have room.
        </p>
      </div>
      <div className="grid-2">
        {[
          { name: "round-robin", d: left },
          { name: "abyss", d: right },
        ].map(({ name, d }) => (
          <div className="glass panel" key={name}>
            <div className="panel-head">
              <h3>{labelOf(name)}</h3>
              <span className="note tabular">{fmt.pct(d.slo_attainment)} of requests met their targets</span>
            </div>
            <ShareRows rows={share(d)} />
          </div>
        ))}
      </div>
      <p className="small muted">
        Share of prompt (prefill) and output (decode) tokens processed by each replica at the trace's reference rate. The longest prompts pay for this;
        the benchmarks page breaks results down by prompt length.
      </p>
    </section>
  );
}

function ShareRows({ rows }: { rows: { i: number; prefill: number; decode: number }[] }) {
  const max = Math.max(...rows.flatMap((r) => [r.prefill, r.decode]), 0.3);
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="legend">
        <span>
          <i className="swatch" style={{ background: dataColor.prefill }} />
          Prefill tokens
        </span>
        <span>
          <i className="swatch" style={{ background: dataColor.decode }} />
          Decode tokens
        </span>
      </div>
      {rows.map((r) => (
        <div key={r.i} style={{ display: "grid", gridTemplateColumns: "84px 1fr", gap: 12, alignItems: "center" }}>
          <span className="secondary small">Replica {r.i}</span>
          <div style={{ display: "grid", gap: 4 }}>
            {[
              { v: r.prefill, c: dataColor.prefill },
              { v: r.decode, c: dataColor.decode },
            ].map((b, k) => (
              <div key={k} style={{ display: "grid", gridTemplateColumns: "1fr 44px", gap: 8, alignItems: "center" }}>
                <div style={{ height: 9, borderRadius: 4, background: "var(--track)" }}>
                  <div style={{ width: `${(100 * b.v) / max}%`, height: "100%", borderRadius: 4, background: b.c }} />
                </div>
                <span className="small tabular" style={{ textAlign: "right" }}>
                  {fmt.pct(b.v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Summary({ results }: { results: Loadable<ResultSet[]> }) {
  if (!results.data?.length) return null;
  const sets = sortResults(results.data);
  const cols = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "sita"];
  return (
    <section className="section">
      <div className="section-head">
        <h2>All workloads</h2>
        <p>Share of requests meeting their latency targets at each workload's reference rate, default parameters. Every policy sees the same requests and the same replicas.</p>
      </div>
      <div className="glass panel table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Workload</th>
              <th>Load</th>
              {cols.map((c) => (
                <th key={c}>{labelOf(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sets.map((rs) => {
              const scale = rs.detail_scale;
              const at = (p: string) => rs.points.find((pt) => pt.policy === p && pt.scale === scale);
              return (
                <tr key={rs.name}>
                  <td>
                    <div>{rs.title}</div>
                    <div className="small muted">{fleetText(rs.fleet)}</div>
                  </td>
                  <td>{at("abyss") ? fmt.rate(at("abyss")!.rate) : ""}</td>
                  {cols.map((c) => {
                    const pt = at(c);
                    return (
                      <td key={c} style={c === "abyss" ? { color: "var(--ink)", fontWeight: 650 } : undefined}>
                        {pt ? fmt.pct(pt.attainment, 1) : ""}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function Overview({ results }: { results: Loadable<ResultSet[]> }) {
  return (
    <div className="page" style={{ gap: 72 }}>
      <Hero results={results} />
      {results.error && <div className="error-box">Could not load benchmark results: {results.error}</div>}
      <Pricing />
      <Roles results={results} />
      <Summary results={results} />
    </div>
  );
}
