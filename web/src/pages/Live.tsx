import { useEffect, useState } from "react";
import { api, colorOf, Cost, dataColor, fmt, labelOf } from "../api";
import LineChart from "../charts/LineChart";
import { costSegments } from "./Overview";

type Window = { requests: number; rps: number; goodput: number; attainment: number; ttft_p50: number; ttft_p99: number; tpot_p50: number; tpot_p99: number; output_tps: number };

type BackendView = {
  name: string;
  url: string;
  label: string;
  model: string;
  healthy: boolean;
  inflight: number;
  prefilling: number;
  decoding: number;
  served: number;
  twin?: { running: number; waiting: number; kv_usage: number; cached_tokens: number; capacity_tokens: number; host_blocks: number; speed_scale: number; prefill_scale: number };
  scraped_waiting: number;
  scraped_kv_usage: number;
  window: Window;
};

type DecisionLog = { t: number; id: number; input: number; blocks: number; backend: string; action: string; pred_ttft: number; pred_tpot: number; micros: number; costs?: Cost[] };

type Snapshot = {
  time: number;
  policy: string;
  policies: string[];
  totals: { requests: number; completed: number; good: number; rejected: number; errors: number };
  window: Window;
  backends: BackendView[];
  decisions: DecisionLog[];
  series: { t: number; rps: number; attainment: number; ttft_p50: number; ttft_p99: number; tpot_p50: number; inflight: number }[];
};

function useLive(enabled: boolean) {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [lost, setLost] = useState(false);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const tick = () =>
      fetch("/api/live/state")
        .then((r) => r.json())
        .then((s: Snapshot) => {
          if (!alive) return;
          setSnap(s);
          setLost(false);
        })
        .catch(() => alive && setLost(true));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [enabled]);
  return { snap, lost };
}

export default function Live({ attached }: { attached: boolean | null }) {
  const { snap, lost } = useLive(attached === true);
  const [switching, setSwitching] = useState(false);
  if (attached === null) return <div className="page"><div className="skeleton" style={{ height: 420 }} /></div>;
  if (!attached) return <Detached />;
  if (!snap) return <div className="page"><div className="skeleton" style={{ height: 420 }} /></div>;

  const change = (p: string) => {
    setSwitching(true);
    api.setPolicy(p).finally(() => setSwitching(false));
  };
  const w = snap.window;
  const active = snap.series.filter((s) => s.rps > 0);
  return (
    <div className="page">
      <div className="section-head">
        <div style={{ display: "grid", gap: 8 }}>
          <h2>Live fleet</h2>
          <p className="secondary">
            {snap.backends.length} replicas behind the gateway. Figures cover the last 30 seconds.{lost ? " Connection lost, retrying." : ""}
          </p>
        </div>
        <div className="segmented" role="group" aria-label="Routing policy">
          {snap.policies.map((p) => (
            <button key={p} aria-pressed={p === snap.policy} disabled={switching} onClick={() => change(p)}>
              {labelOf(p)}
            </button>
          ))}
        </div>
      </div>
      <div className="kpis glass" style={{ padding: 0 }}>
        <Kpi label="Requests meeting their targets" value={w.requests ? fmt.pct(w.attainment, 1) : "No traffic"} />
        <Kpi label="Requests per second" value={w.rps.toFixed(2)} />
        <Kpi label="Time to first token, p99" value={w.requests ? fmt.ms(w.ttft_p99) : "None yet"} />
        <Kpi label="Time per output token, p99" value={w.requests ? fmt.msFine(w.tpot_p99) : "None yet"} />
      </div>
      <div className="grid-2">
        <div className="glass panel">
          <div className="panel-head">
            <h3>Attainment, 10 second window</h3>
          </div>
          <LineChart
            series={[{ key: "a", label: "Attainment", color: colorOf(snap.policy), points: active.map((s) => ({ x: Math.round(s.t), y: s.attainment })) }]}
            yDomain={[0, 1]}
            yFormat={(v) => fmt.pct(v)}
            xFormat={(v) => `${Math.round(v)}s`}
            markers={false}
            legend={false}
            directLabels={false}
            height={200}
          />
        </div>
        <div className="glass panel">
          <div className="panel-head">
            <h3>Time to first token</h3>
          </div>
          <LineChart
            series={[
              { key: "p50", label: "p50", color: dataColor.p50, points: active.map((s) => ({ x: Math.round(s.t), y: s.ttft_p50 })) },
              { key: "p99", label: "p99", color: dataColor.p99, points: active.map((s) => ({ x: Math.round(s.t), y: s.ttft_p99 })) },
            ]}
            yFormat={(v) => fmt.ms(v)}
            xFormat={(v) => `${Math.round(v)}s`}
            markers={false}
            directLabels
            legend
            height={200}
          />
        </div>
      </div>
      <div className="grid-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(270px, 1fr))" }}>
        {snap.backends.map((b) => (
          <ReplicaCard key={b.name} b={b} />
        ))}
      </div>
      <DecisionFeed decisions={snap.decisions} backends={snap.backends} />
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi">
      <span className="value tabular">{value}</span>
      <span className="label">{label}</span>
    </div>
  );
}

function ReplicaCard({ b }: { b: BackendView }) {
  const kv = b.twin ? b.twin.kv_usage : b.scraped_kv_usage;
  const cached = b.twin ? b.twin.cached_tokens / Math.max(b.twin.capacity_tokens, 1) : 0;
  return (
    <div className="glass panel" style={{ display: "grid", gap: 14, padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
        <h3>{b.name}</h3>
        <span className="small" style={{ color: b.healthy ? "var(--good)" : "var(--critical)" }}>
          {b.healthy ? "Healthy" : "Unreachable"}
        </span>
      </div>
      <span className="small muted">
        {b.model}, {b.label.replace(" x", " TP")}
      </span>
      <div style={{ display: "grid", gap: 6 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }} className="small">
          <span className="secondary">KV memory</span>
          <span className="tabular">
            {fmt.pct(kv)} active{b.twin ? `, ${fmt.pct(cached)} cached` : ""}
          </span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: "var(--track)", display: "flex", overflow: "hidden" }}>
          <div style={{ width: `${100 * kv}%`, background: dataColor.kvActive }} />
          <div style={{ width: `${100 * cached}%`, background: dataColor.kvCached, marginLeft: 2 }} />
        </div>
      </div>
      <div className="grid-2 small tabular" style={{ gap: 6 }}>
        <span className="secondary">Prefilling</span>
        <span style={{ textAlign: "right" }}>{b.prefilling}</span>
        <span className="secondary">Decoding</span>
        <span style={{ textAlign: "right" }}>{b.decoding}</span>
        <span className="secondary">Served</span>
        <span style={{ textAlign: "right" }}>{fmt.num(b.served)}</span>
        <span className="secondary">TTFT p50</span>
        <span style={{ textAlign: "right" }}>{b.window.requests ? fmt.ms(b.window.ttft_p50) : "None"}</span>
        {b.twin && (
          <>
            <span className="secondary">Decode speed</span>
            <span style={{ textAlign: "right" }}>{(1 / b.twin.speed_scale).toFixed(2)}× model</span>
            <span className="secondary">Prefill speed</span>
            <span style={{ textAlign: "right" }}>{(1 / b.twin.prefill_scale).toFixed(2)}× model</span>
          </>
        )}
      </div>
    </div>
  );
}

function DecisionFeed({ decisions, backends }: { decisions: DecisionLog[]; backends: BackendView[] }) {
  const latest = [...decisions].reverse().slice(0, 12);
  const priced = latest.find((d) => d.costs && d.costs.length > 1);
  return (
    <div className="grid-2">
      <div className="glass panel table-scroll">
        <div className="panel-head">
          <h3>Recent decisions</h3>
        </div>
        <table className="data">
          <thead>
            <tr>
              <th>Request</th>
              <th>Prompt</th>
              <th>Replica</th>
              <th>Predicted TTFT</th>
            </tr>
          </thead>
          <tbody>
            {latest.map((d) => (
              <tr key={`${d.id}-${d.t}`}>
                <td>#{d.id}</td>
                <td>{fmt.k(d.input)}</td>
                <td>{d.action === "dispatch" ? d.backend : d.action}</td>
                <td>{d.pred_ttft > 0 ? fmt.ms(d.pred_ttft) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {latest.length > 0 && (
          <p className="small muted" style={{ marginTop: 12 }}>
            Mean decision time {meanMicros(decisions)}
          </p>
        )}
      </div>
      <div className="glass panel">
        <div className="panel-head">
          <h3>Latest price breakdown</h3>
          {priced && <span className="note">Request #{priced.id}</span>}
        </div>
        {priced ? (
          <div style={{ display: "grid", gap: 10 }}>
            <div className="legend">
              {costSegments.map((s) => (
                <span key={s.key}>
                  <i className="swatch" style={{ background: s.color }} />
                  {s.label}
                </span>
              ))}
            </div>
            {priced.costs!.map((c) => {
              const total = Math.max(...priced.costs!.map((x) => x.total), 1e-9);
              const delay = Math.max(c.total - c.own - c.interference - c.cache, 0);
              const parts = [c.own, c.interference, c.cache, delay];
              const name = backends[c.backend]?.name ?? `Replica ${c.backend}`;
              return (
                <div key={c.backend} style={{ display: "grid", gridTemplateColumns: "110px 1fr 56px", gap: 10, alignItems: "center" }}>
                  <span className="small" style={{ fontWeight: name === priced.backend ? 600 : 400, color: name === priced.backend ? "var(--ink)" : "var(--ink-2)" }}>
                    {name}
                  </span>
                  <div style={{ display: "flex", gap: 2, height: 12 }}>
                    {parts.map((v, i) =>
                      v > 0 ? <div key={i} style={{ width: `${(100 * v) / total}%`, background: costSegments[i].color, borderRadius: 3 }} /> : null,
                    )}
                  </div>
                  <span className="small tabular" style={{ textAlign: "right" }}>
                    {c.total.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="secondary">The active policy does not publish prices. Switch to Abyss to see them.</p>
        )}
      </div>
    </div>
  );
}

function meanMicros(ds: DecisionLog[]) {
  const m = ds.reduce((a, d) => a + d.micros, 0) / Math.max(ds.length, 1);
  return m >= 1000 ? `${(m / 1000).toFixed(2)} ms` : `${Math.max(1, Math.round(m))} µs`;
}

function Detached() {
  return (
    <div className="page">
      <div className="glass empty" style={{ gap: 18 }}>
        <h3>No gateway attached</h3>
        <p style={{ maxWidth: "40em" }}>
          This dashboard is running in simulation mode. Start emulated replicas and the gateway to watch Abyss route real HTTP traffic, then open this page
          from the gateway address.
        </p>
        <div className="code">{`abyss emulate --listen :9001 --bias 1.15 &
abyss emulate --listen :9002 &
abyss emulate --listen :9003 --bias 0.9 &
abyss emulate --listen :9004 --hardware a100 &
abyss gateway --config configs/gateway.json
abyss loadgen --target http://localhost:8080 --workload chat --rate 6`}</div>
      </div>
    </div>
  );
}
