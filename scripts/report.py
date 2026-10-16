import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

PRESETS = ["conversation-70b", "toolagent-70b", "agent-8b", "mixed-hetero"]
TITLES = {
    "conversation-70b": "Mooncake conversation trace, Llama 3.1 70B",
    "toolagent-70b": "Mooncake tool and agent trace, Llama 3.1 70B",
    "agent-8b": "Coding agent sessions, Llama 3.1 8B",
    "mixed-hetero": "Chat, agent and RAG mix, Qwen3 32B",
}
SHORT = {
    "conversation-70b": "Conversation trace",
    "toolagent-70b": "Tool and agent trace",
    "agent-8b": "Coding agent sessions",
    "mixed-hetero": "Mixed traffic, mixed GPUs",
}
INLINE = {
    "conversation-70b": "the conversation trace",
    "toolagent-70b": "the tool and agent trace",
    "agent-8b": "the coding agent sessions",
    "mixed-hetero": "the mixed-GPU fleet",
}
LABELS = {
    "abyss": "Abyss",
    "abyss-selfish": "Selfish twin",
    "llm-d-predicted": "llm-d predicted latency",
    "llm-d": "llm-d scorer",
    "dynamo": "Dynamo KV router",
    "sglang": "SGLang cache-aware",
    "preble": "Preble-style",
    "sita": "Size-interval",
    "least-work": "Least work",
    "round-robin": "Round robin",
    "least-request": "Least request",
    "chwbl": "Bounded-load hashing",
    "abyss-no-interference": "Without interference term",
    "abyss-no-cache": "Without cache term",
    "abyss-oracle": "Known output lengths",
    "abyss-late-binding": "With late binding",
    "abyss-shed": "With admission control",
}
ORDER = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "sita", "least-work", "round-robin", "least-request", "chwbl"]
EXISTING = ["llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "round-robin", "least-request", "chwbl"]
HEURISTIC = ["sita", "least-work"]
BASELINES = EXISTING + HEURISTIC
FAMILIES = ["llm-d", "dynamo", "sglang", "chwbl", "sita"]
PHRASE = {
    "abyss-selfish": "the selfish twin",
    "llm-d-predicted": "llm-d predicted-latency routing",
    "llm-d": "the llm-d scorer",
    "dynamo": "the Dynamo KV router",
    "sglang": "the SGLang router",
    "preble": "Preble-style routing",
    "sita": "the size-interval heuristic",
    "least-work": "least-work routing",
    "round-robin": "round robin",
    "least-request": "least-request routing",
    "chwbl": "bounded-load hashing",
}
ROBUST_GROUPS = [
    ("Engine settings", [
        ("long-prefill-2048", "vLLM long-prefill cap of 2048 tokens"),
        ("chunk-2048", "Step budget of 2048 tokens instead of 8192"),
    ]),
    ("Uniform speed error", [
        ("replicas-40pct-slower", "Every replica 40% slower than modeled"),
        ("replicas-25pct-faster", "Every replica 25% faster than modeled"),
        ("bias-spread-30pct", "Hidden per-replica speed spread of ±30% instead of ±12%"),
    ]),
    ("Model error that one speed factor cannot absorb", [
        ("prefill-gemm-slower", "Matrix-multiply efficiency 27% below the model"),
        ("attention-slower", "Attention efficiency 40% below the model"),
        ("decode-bandwidth-lower", "Memory-bandwidth efficiency 25% below the model"),
        ("prefill-and-attention-slower", "Matrix-multiply and attention efficiency both below the model"),
        ("prefill-faster-decode-slower", "Matrix multiplies 30% faster, memory bandwidth 25% slower"),
    ]),
]
ROBUST_LABELS = {k: v for _, rows in ROBUST_GROUPS for k, v in rows}
MISMATCH = [k for k, _ in ROBUST_GROUPS[2][1]]
SENSITIVITY = [
    ("reference", "Reference: TTFT 1 s + 0.15 s per 1k tokens, TPOT 80 ms"),
    ("tpot-120ms", "TPOT 120 ms"),
    ("tpot-200ms", "TPOT 200 ms"),
    ("ttft-2x", "TTFT 2 s + 0.3 s per 1k tokens"),
    ("ttft-2x-tpot-150ms", "TTFT 2 s + 0.3 s per 1k tokens, TPOT 150 ms"),
    ("tpot-min-16", "TPOT objective only for responses of 16 tokens or more"),
]
SENS_PRESETS = ["conversation-70b", "toolagent-70b"]
TARGETS = [0.8, 0.7]


def load(path):
    with open(os.path.join(RES, path), encoding="utf-8") as f:
        return json.load(f)


def exists(path):
    return os.path.exists(os.path.join(RES, path))


def family(policy):
    return policy.split(":")[0]


def phrase(policy):
    return PHRASE.get(family(policy), label(policy))


def label(spec):
    name, _, args = spec.partition(":")
    base = LABELS.get(name, name)
    if not args:
        return base
    if name == "sita":
        cut, n = args.split(",")
        k = int(float(cut)) // 1024
        n = int(float(n))
        return f"{base} ({k}k cutoff, {n} long {'replica' if n == 1 else 'replicas'})"
    return f"{base} ({args.replace(',', ', ')})"


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def pts_(v):
    return f"{abs(100 * v):.0f}"


def sec(v):
    if v >= 10:
        return f"{v:.1f} s"
    if v >= 1:
        return f"{v:.2f} s"
    return f"{1000 * v:.0f} ms"


def at(points, policy, scale):
    for p in points:
        if p["policy"] == policy and abs(p["scale"] - scale) < 1e-9:
            return p
    return None


def best(points, scale, policies, key="attainment"):
    out = None
    for p in points:
        if abs(p["scale"] - scale) > 1e-9:
            continue
        if family(p["policy"]) not in policies and p["policy"] not in policies:
            continue
        if out is None or p[key] > out[key]:
            out = p
    return out


def scales(points):
    return sorted({p["scale"] for p in points})


def native(points):
    return min(scales(points), key=lambda s: abs(s - 1))


def capacity(points, policy, target):
    xs = sorted([p for p in points if p["policy"] == policy], key=lambda p: p["scale"])
    if not xs:
        return None
    if xs[0]["attainment"] < target:
        return ("below", xs[0]["scale"], xs[0]["rate"])
    for a, b in zip(xs, xs[1:]):
        if b["attainment"] < target:
            f = (a["attainment"] - target) / (a["attainment"] - b["attainment"])
            return ("at", a["scale"] + f * (b["scale"] - a["scale"]), a["rate"] + f * (b["rate"] - a["rate"]))
    return ("above", xs[-1]["scale"], xs[-1]["rate"])


def cap_text(c):
    if c is None:
        return "n/a"
    kind, s, r = c
    if kind == "below":
        return f"below {s:g}×"
    if kind == "above":
        return f"{s:g}× or more"
    return f"{s:.2f}× ({r:.2f} req/s)"


def cap_value(c):
    if c is None:
        return -1
    kind, s, _ = c
    return {"below": s - 1, "above": s + 1}.get(kind, s)


def gap_phrase(diff, where):
    if abs(diff) < 0.01:
        return f"level on {where}"
    return f"{'ahead' if diff > 0 else 'behind'} by {pts_(diff)} {'point' if round(abs(100 * diff)) == 1 else 'points'} on {where}"


def join(parts):
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class Data:
    def __init__(self):
        self.main = {p: load(p + ".json") for p in PRESETS if exists(p + ".json")}
        self.tuned = {p: load(os.path.join("tuned", p + ".json")) for p in PRESETS if exists(os.path.join("tuned", p + ".json"))}
        self.abl = {p: load(os.path.join("ablations", p + ".json")) for p in ["conversation-70b", "agent-8b"] if exists(os.path.join("ablations", p + ".json"))}
        self.robust = {}
        for k in ROBUST_LABELS:
            if exists(os.path.join("robustness", k + ".json")):
                self.robust[k] = load(os.path.join("robustness", k + ".json"))
        self.sens = {}
        for p in SENS_PRESETS:
            for k, _ in SENSITIVITY:
                path = os.path.join("sensitivity", f"{p}-{k}.json")
                if exists(path):
                    self.sens.setdefault(p, {})[k] = load(path)
        self.scaling = {}
        for n in [8, 16, 32, 64]:
            if exists(os.path.join("scaling", f"replicas-{n}.json")):
                self.scaling[n] = load(os.path.join("scaling", f"replicas-{n}.json"))
        self.micro = load("microbench.json") if exists("microbench.json") else None
        cal = sorted(os.listdir(os.path.join(RES, "calibration"))) if os.path.isdir(os.path.join(RES, "calibration")) else []
        self.calib = load(os.path.join("calibration", cal[0])) if cal else None

    def run(self, preset, policy, scale, seed=0):
        for r in self.main[preset]["runs"]:
            if r["policy"] == policy and abs(r["scale"] - scale) < 1e-9 and r["seed"] == seed and r.get("result"):
                return r["result"]
        return None

    def choice(self, preset, fam):
        t = self.tuned.get(preset)
        if not t:
            return None
        c = next((c for c in t["choices"] if c["family"] == fam), None)
        return c["best"] if c else None

    def tuned_best(self, preset, s, families, extra=()):
        pts = self.tuned[preset]["evaluation"]["points"]
        out = None
        cands = []
        for fam in families:
            cands.append(at(pts, fam, s))
            ch = self.choice(preset, fam)
            if ch:
                cands.append(at(pts, ch, s))
        cands += [at(pts, x, s) for x in extra]
        for c in cands:
            if c and (out is None or c["attainment"] > out["attainment"]):
                out = c
        return out

    def tuned_rows(self, preset):
        pts = self.tuned[preset]["evaluation"]["points"]
        out = []
        for s in scales(pts):
            prod = self.tuned_best(preset, s, ["llm-d", "dynamo", "sglang", "chwbl"])
            research = best(pts, s, ["preble", "llm-d-predicted"])
            existing = max([x for x in [prod, research] if x], key=lambda x: x["attainment"])
            heur = self.tuned_best(preset, s, ["sita"], ["least-work"])
            out.append({"scale": s, "abyss": at(pts, "abyss", s), "selfish": at(pts, "abyss-selfish", s), "prod": prod, "research": research, "existing": existing, "heuristic": heur})
        return out

    def headline(self, preset):
        rows = self.tuned_rows(preset)
        return min(rows, key=lambda r: abs(r["scale"] - 1))

    def robust_heuristic(self, pts, s):
        cands = [at(pts, "sita", s), at(pts, "least-work", s)]
        ch = self.choice("conversation-70b", "sita")
        if ch:
            cands.append(at(pts, ch, s))
        cands = [c for c in cands if c]
        return max(cands, key=lambda c: c["attainment"]) if cands else None

    def sens_row(self, preset, key, s):
        d = self.sens[preset][key]["points"]
        pig = at(d, "abyss", s)
        existing = best(d, s, EXISTING)
        heur = best(d, s, HEURISTIC)
        return pig, at(d, "abyss-selfish", s), existing, heur


def loc():
    total = {"go": 0, "test": 0, "ts": 0}
    for base, _, files in os.walk(ROOT):
        if "node_modules" in base or ".git" in base or os.sep + "dist" in base:
            continue
        for f in files:
            p = os.path.join(base, f)
            if f.endswith("_test.go"):
                total["test"] += sum(1 for _ in open(p, encoding="utf-8"))
            elif f.endswith(".go"):
                total["go"] += sum(1 for _ in open(p, encoding="utf-8"))
            elif f.endswith((".ts", ".tsx")) and os.sep + "web" + os.sep in p:
                total["ts"] += sum(1 for _ in open(p, encoding="utf-8"))
    return total


def fleet_text(fleet):
    counts = {}
    for f in fleet:
        counts[f] = counts.get(f, 0) + 1
    parts = []
    for k, n in counts.items():
        name, _, tp = k.rpartition(" x")
        name = name.replace(" SXM", "").replace(" 80GB", "")
        parts.append(f"{n} × {name}" + (f" TP{tp}" if tp and tp != "1" else ""))
    return ", ".join(parts)


def headline_table(D):
    rows = ["| Workload | Fleet | Abyss | Best existing router | Size-interval heuristic | Abyss, by prompt tokens | Heuristic, by prompt tokens |", "|---|---|---|---|---|---|---|"]
    for p in PRESETS:
        if p not in D.tuned or p not in D.main:
            continue
        h = D.headline(p)
        e, x, g = h["existing"], h["heuristic"], h["abyss"]
        top = max(g["attainment"], e["attainment"], x["attainment"])
        ttop = max(g["token_weighted_attainment"], x["token_weighted_attainment"])
        b = lambda v, m: f"**{pct(v)}**" if abs(v - m) < 1e-12 else pct(v)
        rows.append(f"| {TITLES[p]} | {fleet_text(D.main[p]['fleet'])} | {b(g['attainment'], top)} | {b(e['attainment'], top)} ({label(e['policy'])}) | {b(x['attainment'], top)} ({heur_label(x['policy'])}) | {b(g['token_weighted_attainment'], ttop)} | {b(x['token_weighted_attainment'], ttop)} |")
    return "\n".join(rows)


def heur_label(spec):
    if spec == "sita":
        return "default settings"
    if spec == "least-work":
        return "least work"
    return label(spec).replace("Size-interval (", "").rstrip(")")


def headline_sentence(D):
    ex, hx, tw, n = [], [], [], 0
    for p in PRESETS:
        if p not in D.tuned:
            continue
        h = D.headline(p)
        n += 1
        ex.append(h["abyss"]["attainment"] - h["existing"]["attainment"])
        hx.append(gap_phrase(h["abyss"]["attainment"] - h["heuristic"]["attainment"], INLINE[p]))
        if h["abyss"]["token_weighted_attainment"] - h["heuristic"]["token_weighted_attainment"] > 0.005:
            tw.append(INLINE[p])
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}
    lead = (f"Abyss is ahead by {pts_(min(ex))} to {pts_(max(ex))} points" if min(ex) > 0 else f"the difference ranges from {100 * min(ex):+.0f} to {100 * max(ex):+.0f} points")
    return (f"Against the best existing router on each workload, {lead} at the reference rate. "
            f"The strongest alternative is the size-interval heuristic, which sends every prompt above a length cutoff to dedicated replicas: Abyss is {join(hx)}. "
            f"The heuristic gives up on nearly every prompt above its cutoff; weighted by prompt tokens, Abyss is ahead of it on {words.get(len(tw), len(tw))} of {words.get(n, n)} workloads"
            + (f" ({join(tw)})." if 0 < len(tw) < n else "."))


def low_load_sentence(D):
    pig, heur, ex = [], [], []
    for p in PRESETS:
        if p not in D.main:
            continue
        pts = D.main[p]["points"]
        s = scales(pts)[0]
        pig.append(at(pts, "abyss", s)["attainment"])
        heur.append(at(pts, "sita", s)["attainment"])
        ex.append(best(pts, s, EXISTING + ["least-work"])["attainment"])
    s = scales(D.main[PRESETS[0]]["points"])[0]
    return (f"With spare capacity the fixed cutoff is a handicap. At {s:g}× the reference rate Abyss meets both objectives for {pct(min(pig), 0)} to {pct(max(pig), 0)} of requests, "
            f"the size-interval heuristic with default parameters for {pct(min(heur), 0)} to {pct(max(heur), 0)}, and the best other policy for {pct(min(ex), 0)} to {pct(max(ex), 0)}.")


def capacity_rows(D, target):
    out = []
    for p in PRESETS:
        if p not in D.main:
            continue
        pts = D.main[p]["points"]
        pig = capacity(pts, "abyss", target)
        others = [(capacity(pts, pol, target), pol) for pol in BASELINES + ["abyss-selfish"] if at(pts, pol, scales(pts)[0])]
        top = max(others, key=lambda x: cap_value(x[0]))
        out.append((p, pig, top))
    return out


def capacity_sentence(D):
    parts = []
    for p, pig, (c, pol) in capacity_rows(D, TARGETS[0]):
        parts.append(f"{cap_text(pig)} against {cap_text(c)} for {phrase(pol)} on {INLINE[p]}")
    return (f"Measured as the highest load at which at least {pct(TARGETS[0], 0)} of requests meet both objectives, Abyss sustains {'; '.join(parts)}. "
            "Loads are multiples of the reference rate, interpolated between the tested levels.")


def sensitivity_sentence(D):
    if not D.sens:
        return ""
    ex, hx = [], []
    for p in SENS_PRESETS:
        if p not in D.sens:
            continue
        for k, _ in SENSITIVITY:
            if k not in D.sens[p]:
                continue
            pig, _, e, h = D.sens_row(p, k, 1)
            ex.append(pig["attainment"] - e["attainment"])
            hx.append(pig["attainment"] - h["attainment"])
    return (f"The comparison depends on the latency objectives. Across {['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][len(SENSITIVITY)]} variants of the objectives on the two production traces at the reference rate, "
            f"Abyss's difference from the best existing router ranges from {100 * min(ex):+.0f} to {100 * max(ex):+.0f} points, and from the best size-interval setting "
            f"from {100 * min(hx):+.0f} to {100 * max(hx):+.0f} points.")


def mismatch_sentence(D):
    rows = [(k, D.robust[k]["points"]) for k in MISMATCH if k in D.robust]
    if not rows:
        return ""
    pv = [at(pts, "abyss", 1)["attainment"] for _, pts in rows]
    err = [at(pts, "abyss", 1)["pred_error_p50"] for _, pts in rows]
    hx = [at(pts, "abyss", 1)["attainment"] - D.robust_heuristic(pts, 1)["attainment"] for _, pts in rows]
    base = at(D.main["conversation-70b"]["points"], "abyss", 1)
    return (f"Abyss's replica model is the same code as the simulated servers. When the servers' real prefill or decode speed departs from the model in ways a single speed factor cannot absorb, "
            f"Abyss's attainment on the conversation trace at the reference rate ranges from {pct(min(pv))} to {pct(max(pv))} ({pct(base['attainment'])} with an exact model), "
            f"its median TTFT prediction error stays between {pct(min(err), 0)} and {pct(max(err), 0)}, and its difference from the size-interval heuristic ranges from "
            f"{100 * min(hx):+.0f} to {100 * max(hx):+.0f} points.")


def update_readme(D):
    path = os.path.join(ROOT, "README.md")
    s = open(path, encoding="utf-8").read()
    blocks = {
        "results-table": "\n" + headline_table(D) + "\n",
        "headline": headline_sentence(D),
        "low-load": low_load_sentence(D),
        "capacity": capacity_sentence(D),
        "sensitivity": sensitivity_sentence(D),
        "mismatch": mismatch_sentence(D),
        "ablation": ablation_sentence(D),
        "roles": roles_sentence(D),
        "fairness": fairness_sentence(D),
    }
    for key, text in blocks.items():
        pat = r"<!-- " + key + r" -->.*?<!-- /" + key + r" -->"
        if not re.search(pat, s, flags=re.S):
            raise SystemExit(f"README is missing the {key} marker")
        s = re.sub(pat, lambda _: "<!-- " + key + " -->" + text + "<!-- /" + key + " -->", s, flags=re.S)
    open(path, "w", encoding="utf-8", newline="\n").write(s)


def ablation_sentence(D):
    a = D.abl.get("conversation-70b")
    if not a:
        return ""
    pts = a["points"]
    full, noint = at(pts, "abyss", 1), at(pts, "abyss-no-interference", 1)
    return f"The second cost does most of the work: removing it drops Abyss from {pct(full['attainment'])} to {pct(noint['attainment'])} on the conversation trace."


def role_shares(D, policy="abyss"):
    r = D.run("conversation-70b", policy, native(D.main["conversation-70b"]["points"]))
    reps = r["summary"]["replicas"]
    pf = sum(x["prefill_tokens"] for x in reps)
    dc = sum(x["decode_tokens"] for x in reps)
    shares = [(x["prefill_tokens"] / pf, x["decode_tokens"] / dc) for x in reps]
    return max(shares, key=lambda t: t[0] - t[1])


def roles_sentence(D):
    p, d = role_shares(D)
    return (f"Abyss ends up giving replicas different jobs without being told to. In the first conversation window at the reference rate, one replica "
            f"handles {pct(p, 0)} of the prompt processing but only {pct(d, 0)} of the token generation, so the others can stream responses with fewer interruptions. "
            "The size-interval heuristic imposes a similar split by rule.")


def fairness_sentence(D):
    s = native(D.main["conversation-70b"]["points"])
    r = D.run("conversation-70b", "abyss", s)
    buckets = r["summary"]["length_attainment"]
    total = sum(b["requests"] for b in buckets)
    longest = buckets[-1]
    others = []
    for pol in BASELINES + ["abyss-selfish"]:
        x = D.run("conversation-70b", pol, s)
        if x:
            others.append((x["summary"]["length_attainment"][-1]["attainment"], pol))
    top = max(others)
    return (f"The trade-off is that the longest prompts (over 64k tokens, {pct(longest['requests'] / total, 0)} of the conversation trace) meet their targets "
            f"{pct(longest['attainment'], 0)} of the time under Abyss at the reference rate, against {pct(top[0], 0)} for {phrase(top[1])}.")


def md_main_tables(D):
    out = []
    for p in PRESETS:
        if p not in D.main:
            continue
        d = D.main[p]
        pts = d["points"]
        sc = scales(pts)
        out.append(f"### {TITLES[p]}\n")
        out.append("Share of requests meeting both latency targets, mean ± standard deviation over three runs, default parameters for every baseline.\n")
        out.append("| Policy | " + " | ".join(f"{s:g}× ({at(pts, 'abyss', s)['rate']:.2f} req/s)" for s in sc) + " |")
        out.append("|---|" + "---|" * len(sc))
        for pol in ORDER:
            if not at(pts, pol, sc[0]):
                continue
            cells = []
            for s in sc:
                x = at(pts, pol, s)
                v = f"{pct(x['attainment'])} ± {100 * x['attainment_sd']:.1f}" if x else ""
                cells.append(f"**{v}**" if pol == "abyss" else v)
            out.append(f"| {LABELS[pol]} | " + " | ".join(cells) + " |")
        s = native(pts)
        out.append(f"\nAt {s:g}× the reference rate:\n")
        out.append("| Policy | Goodput (req/s) | Token-weighted attainment | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Cache hit rate | Abandoned | $ per M good tokens |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for pol in ORDER:
            x = at(pts, pol, s)
            if not x:
                continue
            cost = f"{x['usd_per_m_good_tokens']:.2f}" if 0 < x["usd_per_m_good_tokens"] < 1e5 else "n/a"
            out.append(f"| {LABELS[pol]} | {x['goodput']:.2f} | {pct(x.get('token_weighted_attainment', 0))} | {sec(x['ttft_p50'])} | {sec(x['ttft_p99'])} | {sec(x['tpot_p50'])} | {sec(x['tpot_p99'])} | {pct(x['hit_rate'])} | {pct(x['timed_out'])} | {cost} |")
        out.append("")
    return "\n".join(out)


def md_capacity(D):
    out = ["| Workload | Policy | " + " | ".join(f"Load at {pct(t, 0)} attainment" for t in TARGETS) + " |", "|---|---|" + "---|" * len(TARGETS)]
    for p in PRESETS:
        if p not in D.main:
            continue
        pts = D.main[p]["points"]
        for pol in ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "preble", "sita", "least-work", "round-robin"]:
            if not at(pts, pol, scales(pts)[0]):
                continue
            cells = [cap_text(capacity(pts, pol, t)) for t in TARGETS]
            name = f"**{LABELS[pol]}**" if pol == "abyss" else LABELS[pol]
            out.append(f"| {SHORT[p]} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_tuned(D):
    out = []
    for p in PRESETS:
        if p not in D.tuned:
            continue
        t = D.tuned[p]
        out.append(f"#### {TITLES[p]}\n")
        out.append("Parameters chosen on the tuning window: " + ", ".join(f"{c['family']} `{c['best']}`" for c in t["choices"]) + ".\n")
        out.append("| Load | Abyss | Selfish twin | Best production router | Best research baseline | Size-interval or least work | Abyss, token-weighted | Heuristic, token-weighted |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in D.tuned_rows(p):
            g = r["abyss"]
            top = max(x["attainment"] for x in [g, r["selfish"], r["prod"], r["research"], r["heuristic"]] if x)

            def cell(x, sd=True, name=True):
                if not x:
                    return "n/a"
                v = pct(x["attainment"]) + (f" ± {100 * x['attainment_sd']:.1f}" if sd else "")
                if abs(x["attainment"] - top) < 1e-12:
                    v = f"**{v}**"
                return v + (f" ({label(x['policy'])})" if name else "")

            out.append(f"| {r['scale']:g}× | {cell(g, name=False)} | {cell(r['selfish'], sd=False, name=False)} | {cell(r['prod'])} | {cell(r['research'])} | {cell(r['heuristic'])} | {pct(g['token_weighted_attainment'])} | {pct(r['heuristic']['token_weighted_attainment'])} |")
        out.append("")
    return "\n".join(out)


def tuned_counts(D):
    p = next(iter(D.tuned.values()))
    return {c["family"]: c["candidates"] for c in p["choices"]}


def md_sensitivity(D):
    out = []
    for p in SENS_PRESETS:
        if p not in D.sens:
            continue
        out.append(f"#### {TITLES[p]}\n")
        out.append("| Objectives | Load | Abyss | Selfish twin | Best existing router | Best size-interval or least-work setting |")
        out.append("|---|---|---|---|---|---|")
        for k, name in SENSITIVITY:
            if k not in D.sens[p]:
                continue
            for s in [0.5, 1]:
                pig, sel, e, h = D.sens_row(p, k, s)
                top = max(x["attainment"] for x in [pig, sel, e, h])
                bold = lambda x: f"**{pct(x['attainment'])}**" if abs(x["attainment"] - top) < 1e-12 else pct(x["attainment"])
                out.append(f"| {name if s == 0.5 else ''} | {s:g}× | {bold(pig)} | {bold(sel)} | {bold(e)} ({label(e['policy'])}) | {bold(h)} ({label(h['policy'])}) |")
        out.append("")
    return "\n".join(out)


def md_ablation(D):
    out = []
    for p, d in D.abl.items():
        pts = d["points"]
        sc = scales(pts)
        out.append(f"#### {TITLES[p]}\n")
        out.append("| Variant | " + " | ".join(f"{s:g}×" for s in sc) + " |")
        out.append("|---|" + "---|" * len(sc))
        for pol in ["abyss", "abyss-no-interference", "abyss-no-cache", "abyss-selfish", "abyss-oracle", "abyss-late-binding", "abyss-shed"]:
            if not at(pts, pol, sc[0]):
                continue
            out.append(f"| {LABELS[pol]} | " + " | ".join(f"{pct(at(pts, pol, s)['attainment'])} ± {100 * at(pts, pol, s)['attainment_sd']:.1f}" for s in sc) + " |")
        out.append("")
    return "\n".join(out)


def md_robust(D):
    out = ["| Condition | Abyss, 0.75× | Existing, 0.75× | Heuristic, 0.75× | Abyss, 1× | Existing, 1× | Heuristic, 1× | Abyss TTFT error, 1× |", "|---|---|---|---|---|---|---|---|"]
    for group, rows in ROBUST_GROUPS:
        present = [(k, n) for k, n in rows if k in D.robust]
        if not present:
            continue
        out.append(f"| *{group}* | | | | | | | |")
        for k, name in present:
            pts = D.robust[k]["points"]
            cells = []
            for s in [0.75, 1]:
                e = best(pts, s, EXISTING)
                h = D.robust_heuristic(pts, s)
                cells += [pct(at(pts, "abyss", s)["attainment"]), f"{pct(e['attainment'])} ({label(e['policy'])})", f"{pct(h['attainment'])} ({label(h['policy'])})"]
            cells.append(pct(at(pts, "abyss", 1)["pred_error_p50"], 0))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def scaling_sentence(D):
    if not D.scaling:
        return ""
    ns = sorted(D.scaling)
    full = [at(D.scaling[n]["points"], "abyss", 1)["attainment"] for n in ns]
    other = [best(D.scaling[n]["points"], 1, ["llm-d", "dynamo", "preble", "sita"]) for n in ns]
    k8 = [at(D.scaling[n]["points"], "abyss-k8", 1) for n in ns]
    k8txt = ", ".join(f"{pct(x['attainment'])} at {n}" for n, x in zip(ns, k8) if x)
    return (f"Pricing every replica keeps attainment between {pct(min(full))} and {pct(max(full))} from {ns[0]} to {ns[-1]} replicas, while the best of "
            f"llm-d, Dynamo, Preble-style and the size-interval heuristic goes from {pct(other[0]['attainment'])} to {pct(other[-1]['attainment'])}. The shortlist of 8 reaches {k8txt} replicas: "
            "it matches full pricing on small fleets but loses quality on large ones, so at scale parallel pricing is the better way to bound decision time.")


def md_scaling(D):
    out = []
    if D.scaling:
        out.append(scaling_sentence(D) + "\n")
        pols = ["abyss", "abyss-k8", "llm-d", "dynamo", "preble", "sita"]
        names = {"abyss-k8": "Abyss, shortlist of 8"}
        out.append("| Replicas | Offered load | " + " | ".join(names.get(p, LABELS.get(p, p)) for p in pols) + " |")
        out.append("|---|---|" + "---|" * len(pols))
        for n, d in sorted(D.scaling.items()):
            pts = d["points"]
            rate = pts[0]["rate"]
            out.append(f"| {n} | {rate:.1f} req/s | " + " | ".join(pct(at(pts, p, 1)["attainment"]) if at(pts, p, 1) else "" for p in pols) + " |")
        out.append("")
    if D.micro:
        out.append(f"Decision time per request, measured with Go's benchmark harness on {D.micro['cpu']}:\n")
        modes = [("all", "Every replica"), ("all-parallel", "Every replica, parallel"), ("k8", "Shortlist of 8")]
        out.append("| Replicas | " + " | ".join(m[1] for m in modes) + " |")
        out.append("|---|" + "---|" * len(modes))
        for n in sorted({r["replicas"] for r in D.micro["rows"]}):
            cells = []
            for m, _ in modes:
                r = next((x for x in D.micro["rows"] if x["replicas"] == n and x["mode"] == m), None)
                cells.append(f"{r['micros'] / 1000:.2f} ms" if r else "")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_lengths(D, preset):
    s = native(D.main[preset]["points"])
    rows = []
    first = D.run(preset, "abyss", s)
    if not first:
        return ""
    buckets = first["summary"]["length_attainment"]
    rows.append("| Policy | " + " | ".join(f"{b['label']} tokens ({b['requests']:,})" for b in buckets if b["requests"]) + " |")
    rows.append("|---|" + "---|" * sum(1 for b in buckets if b["requests"]))
    for pol in ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "preble", "sita", "least-work", "round-robin"]:
        r = D.run(preset, pol, s)
        if not r:
            continue
        rows.append(f"| {LABELS[pol]} | " + " | ".join(pct(b["attainment"], 0) for b in r["summary"]["length_attainment"] if b["requests"]) + " |")
    return "\n".join(rows)


def md_prediction(D):
    rows = ["| Workload | Median relative TTFT prediction error | Mean decision time (simulation) |", "|---|---|---|"]
    for p in PRESETS:
        if p not in D.main:
            continue
        pts = [x for x in D.main[p]["points"] if x["policy"] == "abyss"]
        errs = [x["pred_error_p50"] for x in pts]
        dec = [x["decision_us"] for x in pts]
        rows.append(f"| {TITLES[p]} | {pct(min(errs), 0)} to {pct(max(errs), 0)} | {min(dec) / 1000:.2f} to {max(dec) / 1000:.2f} ms |")
    return "\n".join(rows)


def md_calibration(D):
    c = D.calib
    if not c:
        return "", None
    rows = ["| Measurement | Measured | Fitted model | Default model |", "|---|---|---|---|"]
    for r in c["rows"]:
        smp = r["sample"]
        name = f"Prefill, {smp['prompt_tokens']:,} tokens" if smp["kind"] == "prefill" else f"Decode step, batch {smp['batch']}"
        rows.append(f"| {name} | {1000 * smp['seconds']:.1f} ms | {1000 * r['predicted_s']:.1f} ms | {1000 * r['default_predicted_s']:.1f} ms |")
    return "\n".join(rows), c


def roles_eval(D):
    p, d = role_shares(D)
    sp, sd = role_shares(D, "sita")
    r = D.run("conversation-70b", "round-robin", native(D.main["conversation-70b"]["points"]))
    reps = r["summary"]["replicas"]
    pf = sum(x["prefill_tokens"] for x in reps)
    dc = sum(x["decode_tokens"] for x in reps)
    spread = max(abs(x["prefill_tokens"] / pf - x["decode_tokens"] / dc) for x in reps)
    return (f"In the first window of the conversation trace at the reference rate, round robin gives every replica a share of decode tokens within {pct(spread, 0)} "
            f"of its share of prefill tokens. Under Abyss one replica processes {pct(p, 0)} of all prefill tokens but only {pct(d, 0)} of decode tokens. "
            f"No rule asks for this; it follows from pricing the stall a long prefill imposes on running decodes. The size-interval heuristic produces the split by rule: "
            f"its long-prompt replica processes {pct(sp, 0)} of prefill tokens and {pct(sd, 0)} of decode tokens. The difference is that Abyss still places long prompts "
            "on the other replicas when they have room, while the heuristic never does, which costs the heuristic attainment at low load and on long prompts.")


def write_evaluation(D):
    w = {p: D.main[p]["workload"] for p in D.main}
    cal_table, cal = md_calibration(D)
    abl_conv = D.abl.get("conversation-70b")
    abl_text = ""
    if abl_conv:
        pts = abl_conv["points"]
        full = at(pts, "abyss", 1)["attainment"]
        noint = at(pts, "abyss-no-interference", 1)["attainment"]
        abl_text = f"Removing the interference term on the conversation trace at the reference rate lowers attainment from {pct(full)} to {pct(noint)}."
    counts = tuned_counts(D) if D.tuned else {}
    lines = [
        "# Evaluation",
        "",
        "Every table in this document is generated by `scripts/report.py` from the JSON files in `results/`, which `scripts/reproduce.sh` regenerates. Figures come from `scripts/plot.py`.",
        "",
        "## Method",
        "",
        "All policies run in the same discrete-event fleet simulator (`internal/cluster`). Replicas are instances of the engine model described in [design.md](design.md). Their real step times differ from Abyss's model in two ways the router never sees: a per-replica speed bias drawn uniformly from ±12% and per-step lognormal jitter with σ = 0.06. Policies observe the fleet only through the signals in section 6 of the design document. No policy knows output lengths.",
        "",
        "Abyss's replica twin is the same engine code as the simulated replicas, so in the main experiments its model of each server is exact apart from the hidden bias and jitter. The robustness section adds model errors that change prefill and decode speed by different amounts, which a single speed factor cannot absorb.",
        "",
        "Arrivals are open loop. A client abandons a request with no first token after `max(30 s, 4 × TTFT objective)`; abandoned requests free their memory and count as misses.",
        "",
        "A request meets its objectives if its time to first token is within `base + per_k × prompt_tokens / 1000` and its mean time per output token is within a fixed bound:",
        "",
        "| Preset | TTFT objective | TPOT objective |",
        "|---|---|---|",
        "| conversation-70b, toolagent-70b | 1.0 s + 0.15 s per 1k prompt tokens | 80 ms |",
        "| agent-8b | 0.4 s + 0.03 s per 1k prompt tokens | 30 ms |",
        "| mixed-hetero | 0.8 s + 0.08 s per 1k prompt tokens | 60 ms |",
        "",
        "Loads are multiples of a reference rate. For the Mooncake traces it is the trace's own arrival rate replayed on the fleet in the table below; the fleet size is a choice of this evaluation, not the size of the fleet that served the trace. For the synthetic workloads it is the generator rate set in the preset. At the reference rate the existing routers already miss most objectives on every workload, so the evaluation also reports the load each policy sustains at a fixed attainment.",
        "",
        "Each configuration runs on three independent inputs: three disjoint 20-minute windows of each Mooncake trace, or three generator seeds for synthetic workloads, each with its own noise seed. Attainment is the fraction of all arriving requests that complete within both objectives. Token-weighted attainment weights each request by its prompt length. Goodput counts requests meeting both objectives per second.",
        "",
        "## Workloads",
        "",
        "| Preset | Source | Requests per window | Reference rate | Mean prompt | p99 prompt | Mean output | Reusable prompt tokens | Fleet |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    src = {"conversation-70b": "Mooncake FAST 2025 conversation trace", "toolagent-70b": "Mooncake FAST 2025 tool and agent trace", "agent-8b": "Synthetic coding agent sessions", "mixed-hetero": "Synthetic chat, agent and RAG mix"}
    extra = {"conversation-70b": ", 96 GB host KV per GPU", "toolagent-70b": ", 96 GB host KV per GPU", "agent-8b": "", "mixed-hetero": ""}
    for p in PRESETS:
        if p in w:
            x = w[p]
            lines.append(f"| {p} | {src[p]} | {x['requests']:,} | {x['rate_rps']:.2f} req/s | {x['mean_input']:,.0f} | {x['p99_input']:,} | {x['mean_output']:.0f} | {pct(x['reuse_ratio'], 0)} | {fleet_text(D.main[p]['fleet'])}{extra[p]} |")
    lines += [
        "",
        "## Baselines",
        "",
        "| Policy | Implementation |",
        "|---|---|",
        "| llm-d scorer | Weighted prefix-match fraction (3), normalized queue length (2) and free KV fraction (2), the tuned weights reported by the llm-d project. Exact engine cache contents and gauges refreshed every engine step. |",
        "| llm-d predicted latency | The decision rule of llm-d's predicted-latency scheduler: affinity gate at 80% prefix match, 1% exploration, affinity broken when the sticky replica's predicted TTFT is more than 5 s worse, 80/20 TTFT and TPOT headroom, best fit. Abyss's replica twin stands in for the trained latency model. |",
        "| Dynamo KV router | Minimizes `overlap_weight × prefill_blocks + active_blocks` with weight 1 and temperature 0, with exact engine cache contents. |",
        "| SGLang cache-aware | Per-worker approximate prefix tree; shortest queue when `max - min > 64` and `max > 1.5 × min`; otherwise best match above a 0.3 match ratio, else the smallest tree. |",
        "| Preble-style | Preble's exploit or explore rule: send to the longest cached match when it covers more than half the prompt, otherwise to the replica with the lowest recent load plus prefill cost over a 60 s window. Preble's eviction term and local scheduler are not modeled. |",
        "| Size-interval | Size-interval task assignment (Harchol-Balter et al., 1999). Prompts longer than a cutoff go to a dedicated pool made of the replicas with the slowest prefill, all other prompts to the remaining replicas. Inside each pool the request goes to the replica with the least outstanding work: prompt tokens of requests without a first token, a fixed 64 tokens per decoding request and the request's own uncached tokens, converted to seconds with each replica's prefill speed. Default cutoff 16,384 tokens with a quarter of the fleet in the long pool; exact engine cache contents. |",
        "| Least work | The same work estimate without pools. |",
        "| Bounded-load hashing | Consistent hashing on the first prompt block, 128 virtual nodes per replica, load bound 1.25 × mean. |",
        "| Least request, round robin | Classic balancers on in-flight request counts. |",
        "| Selfish twin | Abyss's replica twins and rollouts, scoring only the request's own cost. |",
        "",
        "All baselines are reimplementations from public documentation, not the original software. In the tables, the existing routers are the llm-d scorer, llm-d predicted latency, Dynamo, SGLang, Preble-style, bounded-load hashing, least request and round robin; the heuristic is the better of size-interval and least work.",
        "",
        "## Main results",
        "",
        "![Attainment against offered load](figures/attainment.svg)",
        "",
        low_load_sentence(D),
        "",
        md_main_tables(D),
        "## Load sustained at a fixed attainment",
        "",
        f"The highest load at which a policy still meets both objectives for at least {pct(TARGETS[0], 0)} and {pct(TARGETS[1], 0)} of requests, interpolated linearly between the tested loads, with default baseline parameters. \"Below\" means the policy misses the target at the lowest tested load; \"or more\" means it never fell below the target. Attainment is not always monotonic in load, so the value is the first crossing.",
        "",
        capacity_sentence(D),
        "",
        md_capacity(D),
        "",
        "## Baselines tuned on held-out data",
        "",
        f"Default parameters can be unfair to a baseline. For each workload, every family's parameters were grid searched on one trace window or seed ({counts.get('llm-d', 0)} settings for llm-d, {counts.get('sglang', 0)} for SGLang, {counts.get('dynamo', 0)} for Dynamo, {counts.get('chwbl', 0)} for bounded-load hashing and {counts.get('sita', 0)} for the size-interval heuristic), choosing the setting with the highest mean attainment across the two load levels. All policies were then evaluated on the other two windows or seeds. The tables report, per family, whichever of default and tuned did better on the held-out data, which favors the baselines.",
        "",
        headline_sentence(D),
        "",
        "![Tuned baselines](figures/tuned.svg)",
        "",
        md_tuned(D),
        "## Sensitivity to the latency objectives",
        "",
        "The production traces rerun with looser objectives, and with the TPOT objective applied only to responses of at least 16 tokens. A single long prefill step can push the mean TPOT of a very short response past its objective, so the last variant measures how much of the result depends on such responses. The size-interval column is the best of five settings in hindsight (default, tuned on the reference objectives, two fixed cutoffs, and least work), which favors the heuristic.",
        "",
        sensitivity_sentence(D),
        "",
        "![Sensitivity](figures/sensitivity.svg)",
        "",
        md_sensitivity(D),
        "## Ablations",
        "",
        "![Ablations](figures/ablation.svg)",
        "",
        md_ablation(D),
        abl_text,
        "",
        "## Robustness",
        "",
        "The conversation trace rerun with different engine settings, with replicas whose real speed differs uniformly from Abyss's model, and with model errors that change prefill and decode speed by different amounts, three windows each. Existing is the strongest of llm-d predicted latency, the llm-d scorer, Dynamo and Preble-style; the heuristic is the best of size-interval (default and tuned on the reference conditions) and least work.",
        "",
        mismatch_sentence(D),
        "",
        "![Robustness](figures/robustness.svg)",
        "",
        md_robust(D),
        "",
        "## Fleet size and decision time",
        "",
        "Mixed traffic on 8 to 64 Llama 3.1 8B replicas with offered load proportional to fleet size, one 300 s run per size. The shortlist variant prices only the 8 replicas with the lowest estimated backlog, plus the replica with the longest cached prefix.",
        "",
        "![Decision time](figures/decision-time.svg)",
        "",
        md_scaling(D),
        "",
        "## Where the gain comes from",
        "",
        "![Prefill and decode share per replica](figures/replica-roles.svg)",
        "",
        roles_eval(D),
        "",
        "## Who pays",
        "",
        "Attainment by prompt length in the first window at the reference rate. Conversation trace:",
        "",
        md_lengths(D, "conversation-70b"),
        "",
        "Coding agent sessions:",
        "",
        md_lengths(D, "agent-8b"),
        "",
        "Abyss maximizes the number of requests that meet their objectives, and the longest prompts are the most expensive to serve, so they gain the least. The size-interval heuristic goes further and gives up on almost every prompt above its cutoff. Operators can raise the weight of a request class with the `X-Abyss-Weight` header.",
        "",
        "## Prediction quality",
        "",
        md_prediction(D),
        "",
        "Prediction error is measured against the real first token of each dispatched request, under hidden speed bias and jitter, unknown output lengths and future arrivals. The robustness table gives the error under model mismatch.",
        "",
        "## Calibration",
        "",
        "![Calibration](figures/calibration.svg)",
        "",
    ]
    if cal:
        lines += [
            f"`abyss calibrate` against an emulated H100 configured to run 30% slower than its model. The default model's median error is {pct(cal['default_median_abs_rel_error'])}; after fitting it is {pct(cal['median_abs_rel_error'])} (maximum {pct(cal['max_abs_rel_error'])}). The emulator uses the same model family, so this verifies the measurement and fitting procedure, not the model's fidelity to real GPUs.",
            "",
            cal_table,
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- Simulation: the main results come from a simulator. The engine model follows vLLM V1 scheduling and the step-time model uses published accelerator specifications with assumed efficiencies, but it has not been validated against real GPUs. `scripts/gpu/` contains the procedure to do so.",
        "- Exact replica model: Abyss's twin and the simulated replicas share code, so structural differences between the twin and a real engine (scheduler details, kernels whose cost does not follow the roofline model) are not represented. The model-error rows in the robustness section cover speed errors only.",
        "- Reimplemented baselines: they follow public documentation and either default or held-out tuned parameters, not the original code.",
        "- Operating point and objectives: at the reference rate the existing routers miss most objectives, and the size of Abyss's lead depends on the objectives, as the sensitivity section shows.",
        "- Request-count objective: optimizing the number of requests that meet their objectives favors short requests, as the length tables show.",
        "- Prompt identity in the gateway: hashing rendered prompt bytes instead of tokens is approximate and affects every cache-aware policy.",
        "- Tail latency under overload: Abyss concentrates the cost of overload on few requests, which reach the client abandonment limit.",
        "",
    ]
    open(os.path.join(ROOT, "docs", "evaluation.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines))


if __name__ == "__main__":
    D = Data()
    write_evaluation(D)
    update_readme(D)
    print("loc", loc())
    print("wrote docs/evaluation.md and updated README.md")
