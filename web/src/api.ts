export type Dist = { mean: number; p50: number; p90: number; p99: number; max: number };

export type ReplicaStat = {
  name: string;
  requests: number;
  busy: number;
  hit_rate: number;
  preemptions: number;
  evictions: number;
  mean_batch_tokens: number;
  prefill_tokens: number;
  decode_tokens: number;
};

export type LengthBucket = { label: string; requests: number; attainment: number };

export type Summary = {
  policy: string;
  requests: number;
  completed: number;
  rejected: number;
  timed_out: number;
  good: number;
  duration_s: number;
  slo_attainment: number;
  ttft_attainment: number;
  tpot_attainment: number;
  goodput_rps: number;
  output_tok_s: number;
  ttft: Dist;
  tpot: Dist;
  e2e: Dist;
  hit_rate: number;
  preemptions: number;
  cost_per_hour: number;
  usd_per_m_good_tokens: number;
  decision_us: Dist;
  pred_abs_rel_error: Dist;
  replicas: ReplicaStat[];
  length_attainment: LengthBucket[];
};

export type Bucket = {
  t: number;
  arrivals: number;
  attainment: number;
  ttft_p50: number;
  ttft_p99: number;
  tpot_p50: number;
  tpot_p99: number;
  hit_rate: number;
  busy: number[];
  prefill_tokens?: number[];
  decode_tokens?: number[];
};

export type WorkloadSummary = {
  requests: number;
  duration_s: number;
  rate_rps: number;
  mean_input: number;
  p50_input: number;
  p99_input: number;
  mean_output: number;
  reuse_ratio: number;
  sessions: number;
  block_tokens: number;
};

export type ReplicaSpec = { model: string; hardware: string; tp: number; count: number };

export type Preset = {
  name: string;
  title: string;
  description: string;
  cluster: { replicas: ReplicaSpec[]; slo: { ttft_base_s: number; ttft_per_ktok_s: number; tpot_s: number } };
  scales: number[];
};

export type Point = {
  policy: string;
  scale: number;
  rate: number;
  n: number;
  attainment: number;
  attainment_sd: number;
  goodput: number;
  ttft_p50: number;
  ttft_p99: number;
  tpot_p50: number;
  tpot_p99: number;
  hit_rate: number;
  timed_out: number;
  usd_per_m_good_tokens: number;
  decision_us: number;
  pred_error_p50: number;
};

export type Detail = { policy: string; replicas: ReplicaStat[]; length_attainment: LengthBucket[]; slo_attainment: number };

export type ResultSet = {
  name: string;
  title: string;
  preset: Preset;
  workload: WorkloadSummary;
  fleet: string[];
  seeds: number;
  points: Point[];
  detail_scale: number;
  details: Detail[];
};

export type Cost = {
  backend: number;
  own: number;
  interference: number;
  cache: number;
  delay: number;
  total: number;
  pred_ttft: number;
  pred_tpot: number;
  hit_blocks: number;
  evicted_blocks: number;
};

export type DecisionSample = {
  t: number;
  input: number;
  hit_blocks: number;
  chosen: number;
  selfish: number;
  costs: Cost[];
  ttft_slo: number;
};

export type SimRun = { policy: string; summary: Summary; timeline: Bucket[]; error?: string };

export type SimResponse = {
  preset: Preset;
  workload: WorkloadSummary;
  fleet: string[];
  runs: SimRun[];
  decisions: DecisionSample[];
  override_ratio: number;
  wall_s: number;
};

export type Catalog = {
  presets: Preset[];
  policies: string[];
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = body.error?.message ?? body.error ?? msg;
    } catch {

    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  catalog: () => fetch("/api/catalog").then((r) => json<Catalog>(r)),
  results: () => fetch("/api/results").then((r) => json<ResultSet[]>(r)),
  research: () => fetch("/api/research").then((r) => json<Research>(r)),
  simulate: (body: { preset: string; policies: string[]; scale: number; duration?: number; seed?: number }) =>
    fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) =>
      json<SimResponse>(r),
    ),
  setPolicy: (policy: string) =>
    fetch("/api/live/policy", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ policy }) }).then((r) =>
      json<{ policy: string }>(r),
    ),
};

export const policyColor: Record<string, string> = {
  "llm-d": "#2a78d6",
  dynamo: "#eb6834",
  sglang: "#1baf7a",
  abyss: "#b87a00",
  "round-robin": "#d85a8c",
  "llm-d-predicted": "#008300",
  "abyss-selfish": "#6250d6",
  preble: "#e34948",
  sita: "#a3478f",
};

export const dataColor = {
  prefill: "#7b8ba3",
  decode: "#2a78d6",
  p50: "#2a78d6",
  p99: "#eb6834",
  kvActive: "#2a78d6",
  kvCached: "rgba(42, 120, 214, 0.28)",
};

export const policyLabel: Record<string, string> = {
  abyss: "Abyss",
  "abyss-selfish": "Selfish twin",
  "llm-d-predicted": "llm-d predicted latency",
  preble: "Preble-style",
  "llm-d": "llm-d scorer",
  dynamo: "Dynamo KV router",
  sglang: "SGLang cache-aware",
  "round-robin": "Round robin",
  "least-request": "Least request",
  chwbl: "Bounded-load hashing",
  sita: "Size-interval",
  "least-work": "Least work",
  p2c: "Power of two",
  random: "Random",
  "abyss-no-cache": "Abyss without cache term",
  "abyss-no-interference": "Abyss without interference term",
  "abyss-late-binding": "Abyss with late binding",
  "abyss-oracle": "Abyss with known output lengths",
  "abyss-shed": "Abyss with admission control",
};

export const policyOrder = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "sita", "least-work", "round-robin", "least-request", "chwbl"];

export const plotted = (p: string) => p in policyColor;

export const labelSpec = (spec: string) => {
  const [name, args] = spec.split(":");
  return args ? `${labelOf(name)} (${args.split(",").join(", ")})` : labelOf(name);
};

export type TuneChoice = { family: string; best: string; tuning_attainment: number; candidates: number };
export type ResearchSet = { name: string; title: string; choices?: TuneChoice[]; points: Point[]; fleet?: string[] };
export type CalibrationRow = { sample: { kind: string; prompt_tokens: number; batch: number; seconds: number }; predicted_s: number; default_predicted_s: number };
export type Calibration = { chars_per_token: number; median_abs_rel_error: number; max_abs_rel_error: number; default_median_abs_rel_error: number; rows: CalibrationRow[] };
export type Microbench = { cpu: string; rows: { replicas: number; mode: string; micros: number }[] };
export type Research = {
  tuned: ResearchSet[] | null;
  robustness: ResearchSet[] | null;
  sensitivity: ResearchSet[] | null;
  scaling: ResearchSet[] | null;
  calibration: Calibration | null;
  microbench: Microbench | null;
};

export const colorOf = (p: string) => policyColor[p] ?? "#8a94a6";
export const labelOf = (p: string) => policyLabel[p] ?? p;

export const fmt = {
  pct: (v: number, d = 0) => `${(100 * v).toFixed(d)}%`,
  ms: (s: number) => (s >= 10 ? `${s.toFixed(1)} s` : s >= 1 ? `${s.toFixed(2)} s` : `${Math.round(s * 1000)} ms`),
  msFine: (s: number) => (s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1000).toFixed(1)} ms`),
  num: (v: number, d = 0) => v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }),
  k: (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : `${Math.round(v)}`),
  usd: (v: number) => (v >= 100 ? `$${Math.round(v)}` : `$${v.toFixed(2)}`),
  rate: (v: number) => `${v.toFixed(2)} req/s`,
};
