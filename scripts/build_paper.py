import html
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import (EXISTING, HEURISTIC, INLINE, LABELS, MISMATCH, PRESETS, ROBUST_GROUPS, TARGETS, TITLES, Data, at, best, cap_text, capacity_rows,
                    capacity_sentence, fleet_text, headline_sentence, join, label, loc, low_load_sentence, mismatch_sentence, native, pct, phrase, pts_,
                    role_shares, roles_eval, scales, scaling_sentence, sec, sensitivity_sentence, tuned_counts)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
FIG = "../docs/figures/"

REFS = [
    ("kwon", "W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, I. Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023."),
    ("sglang", "L. Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS 2024."),
    ("sarathi", "A. Agrawal et al. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. OSDI 2024."),
    ("distserve", "Y. Zhong et al. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. OSDI 2024."),
    ("splitwise", "P. Patel et al. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. ISCA 2024."),
    ("mooncake", "R. Qin et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. arXiv:2407.00079, 2024."),
    ("llumnix", "B. Sun et al. Llumnix: Dynamic Scheduling for Large Language Model Serving. OSDI 2024."),
    ("preble", "V. Srivatsa, Z. He, R. Abhyankar, D. Li, Y. Zhang. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. arXiv:2407.00023, 2024."),
    ("dualmap", "Y. Yuan, P. Zuo, B. Wang, Z. Chen, Z. Tan, Z. Yu. DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving. arXiv:2602.06502, 2026."),
    ("cacheroute", "H. Cheng. CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving. arXiv:2608.19677, 2026."),
    ("lodestar", "G. Lim, W. Zhao, B. Godfrey, J. Shan, L. Xu, L. Xie. Lodestar: An Online-Learning LLM Inference Router. arXiv:2606.00946, 2026."),
    ("astrolabe", "W. Da, E. Kalyvianaki. Astrolabe: Balancing Load in LLM Serving with Randomized Prediction-Guided Scheduling. SYSTOR 2026, arXiv:2508.03611."),
    ("poa", "A. Georgiou. The Price of Anarchy in Disaggregated Inference. arXiv:2606.17081, 2026."),
    ("llmdpred", "llm-d project. Predicted-Latency Based Scheduling for LLMs. llm-d.ai blog, 2026."),
    ("llmdprefix", "llm-d project. Prefix cache aware routing. github.com/llm-d/llm-d documentation, 2025."),
    ("dynamo", "NVIDIA. Dynamo KV Router documentation. docs.nvidia.com/dynamo, 2025."),
    ("vidur", "A. Agrawal et al. Vidur: A Large-Scale Simulation Framework for LLM Inference. MLSys 2024."),
    ("vtc", "Y. Sheng et al. Fairness in Serving Large Language Models. OSDI 2024."),
    ("pigou1920", "A. C. Pigou. The Economics of Welfare. Macmillan, 1920."),
    ("beckmann", "M. Beckmann, C. B. McGuire, C. B. Winsten. Studies in the Economics of Transportation. Yale University Press, 1956."),
    ("roughgarden", "T. Roughgarden, É. Tardos. How Bad is Selfish Routing? Journal of the ACM 49(2), 2002."),
    ("sita", "M. Harchol-Balter, M. E. Crovella, C. D. Murta. On Choosing a Task Assignment Policy for a Distributed Server System. Journal of Parallel and Distributed Computing 59(2), 1999."),
    ("p2c", "M. Mitzenmacher. The Power of Two Choices in Randomized Load Balancing. IEEE Transactions on Parallel and Distributed Systems 12(10), 2001."),
    ("karger", "D. Karger et al. Consistent Hashing and Random Trees. STOC 1997."),
    ("chwbl", "V. Mirrokni, M. Thorup, M. Zadimoghaddam. Consistent Hashing with Bounded Loads. SODA 2018."),
    ("roofline", "S. Williams, A. Waterman, D. Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. Communications of the ACM 52(4), 2009."),
]
NUM = {k: i + 1 for i, (k, _) in enumerate(REFS)}


def cite(*keys):
    return "[" + ", ".join(str(NUM[k]) for k in keys) + "]"


FIGS = ["architecture", "attainment", "tuned", "sensitivity", "ablation", "robustness", "replica-roles", "decision-time", "calibration"]
TABS = ["workloads", "main", "capacity", "robust", "scaling"]


def F(name):
    return f"Figure {FIGS.index(name) + 1}"


def T(name):
    return f"Table {TABS.index(name) + 1}"


def table(headers, rows, cls=""):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def fig(name, caption, wide=False):
    cls = "figure wide" if wide else "figure"
    return f'<figure class="{cls}"><img src="{FIG}{name}.svg"><figcaption><b>{F(name)}.</b> {caption}</figcaption></figure>'


def tab(name, caption, content, wide=False):
    cls = "tablebox wide" if wide else "tablebox"
    return f'<div class="{cls}"><div class="tcap"><b>{T(name)}.</b> {caption}</div>{content}</div>'


def calib_sentence(D):
    errs = []
    for k in ["replicas-40pct-slower", "replicas-25pct-faster"]:
        if k in D.robust:
            errs += [x["pred_error_p50"] for x in D.robust[k]["points"] if x["policy"] == "abyss"]
    base = [x["pred_error_p50"] for x in D.main["conversation-70b"]["points"] if x["policy"] == "abyss"]
    if not errs:
        return ""
    return (f" When every replica is 40% slower or 25% faster than modeled, the median error of Abyss's TTFT predictions stays between {pct(min(errs), 0)} and {pct(max(errs), 0)}, "
            f"against {pct(min(base), 0)} to {pct(max(base), 0)} without a systematic error.")


def words(n):
    return {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}.get(n, str(n))


def build(D):
    heads = {p: D.headline(p) for p in PRESETS if p in D.tuned and p in D.main}
    pig_vals = [h["abyss"]["attainment"] for h in heads.values()]
    ex_vals = [h["existing"]["attainment"] for h in heads.values()]
    ex_gaps = [h["abyss"]["attainment"] - h["existing"]["attainment"] for h in heads.values()]
    hx_gaps = {p: h["abyss"]["attainment"] - h["heuristic"]["attainment"] for p, h in heads.items()}
    tw_gaps = {p: h["abyss"]["token_weighted_attainment"] - h["heuristic"]["token_weighted_attainment"] for p, h in heads.items()}
    behind = [p for p, g in hx_gaps.items() if g < -0.005]
    level = [p for p, g in hx_gaps.items() if abs(g) <= 0.005]
    ahead = [p for p, g in hx_gaps.items() if g > 0.005]
    tw_ahead = [p for p, g in tw_gaps.items() if g > 0.005]
    low = scales(D.main[PRESETS[0]]["points"])[0]
    low_pig = [at(D.main[p]["points"], "abyss", low)["attainment"] for p in PRESETS if p in D.main]
    low_heur = [at(D.main[p]["points"], "sita", low)["attainment"] for p in PRESETS if p in D.main]
    mm = [(k, D.robust[k]["points"]) for k in MISMATCH if k in D.robust]
    mm_pig = [at(pts, "abyss", 1)["attainment"] for _, pts in mm]
    base_pig = at(D.main["conversation-70b"]["points"], "abyss", 1)["attainment"]
    abl = D.abl.get("conversation-70b", {}).get("points", [])
    abl_full = at(abl, "abyss", 1)["attainment"] if abl else 0
    abl_noint = at(abl, "abyss-no-interference", 1)["attainment"] if abl else 0
    abl_self = at(abl, "abyss-selfish", 1)["attainment"] if abl else 0
    pf, dc = role_shares(D)
    n_loc = loc()
    micro = D.micro
    m = {}
    if micro:
        for r in micro["rows"]:
            m[(r["replicas"], r["mode"])] = r["micros"] / 1000

    def where(ps):
        return join([INLINE[p] for p in ps])

    if behind:
        heur_abs = (f"Near capacity, a size-interval heuristic that sends every prompt above a tuned length cutoff to dedicated replicas is the strongest alternative and meets more objectives than Abyss on {where(behind)}, "
                    f"by giving up on nearly all long prompts; Abyss is ahead of it on {where(ahead + level) or 'no workload'} and leads on token-weighted attainment on {words(len(tw_ahead))} of {words(len(heads))} workloads. ")
    else:
        heur_abs = (f"A size-interval heuristic that sends every prompt above a tuned length cutoff to dedicated replicas is the strongest alternative; Abyss is ahead of it by {pts_(min(hx_gaps.values()))} to {pts_(max(hx_gaps.values()))} points. ")

    css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.css"), encoding="utf-8").read()
    body = []
    A = body.append

    A('<header class="title">')
    A("<h1>Abyss: Pricing the Externalities of Request Placement in LLM Inference Fleets</h1>")
    A('<div class="authors">Preprint draft, October 2026. Code and results: github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference</div>')
    A("</header>")

    A('<section class="abstract"><h2>Abstract</h2><p>')
    A("Routers in front of LLM inference fleets decide which replica serves each request. Production routers such as the llm-d scheduler, the NVIDIA Dynamo KV router and the SGLang router, and recent predicted-latency schedulers, score each candidate replica from the incoming request's point of view: how much of its prompt is cached there, how long the queue is, or how fast its own first token will arrive. "
      "They ignore what the placement costs everyone else, chiefly the stall a long prefill imposes on decode streams already running on the replica, and the value of the cached prefixes it evicts. "
      "We present Abyss, a router that prices these externalities for every request. Abyss keeps a twin of each replica's scheduler, paged KV memory and prefix cache, synchronized and speed-calibrated from the token streams the router already proxies, and charges each candidate placement the request's own probability of missing its latency objectives, plus the change in the miss probabilities of co-located requests estimated from two counterfactual rollouts, plus the expected cost of rebuilding evicted cache blocks. "
      f"In a simulator that replays public production traces from the Kimi service and synthetic agent workloads, with baseline parameters tuned on held-out trace windows, Abyss meets both latency objectives for {pct(min(pig_vals), 0)} to {pct(max(pig_vals), 0)} of requests at the reference load, against {pct(min(ex_vals), 0)} to {pct(max(ex_vals), 0)} for the best existing router. "
      + heur_abs +
      f"With spare capacity the fixed cutoff is a handicap, and at {low:g}x load Abyss reaches {pct(min(low_pig), 0)} to {pct(max(low_pig), 0)} against {pct(min(low_heur), 0)} to {pct(max(low_heur), 0)} for the heuristic with default parameters. "
      f"The margins depend on the latency objectives, and when the replica model errs in ways a single speed factor cannot absorb, Abyss's attainment on the conversation trace falls from {pct(base_pig, 0)} to as low as {pct(min(mm_pig), 0)}. "
      "All results are from simulation; we release the gateway, the simulator and a calibration tool so they can be checked on GPUs.")
    A("</p></section>")

    A('<div class="cols">')

    A("<h2>1 Introduction</h2>")
    A(f"<p>An LLM serving fleet runs many replicas of a model, each with continuous batching {cite('kwon')}, chunked prefill {cite('sarathi')} and a prefix cache that lets requests sharing a prompt prefix skip recomputation {cite('sglang', 'kwon')}. A router in front of the replicas assigns each request to one of them. The choice matters: a request routed to a replica that already holds its prefix starts faster, and a request routed to a busy replica waits.</p>")
    A(f"<p>Routers in use today make this choice from the incoming request's perspective. The llm-d scheduler combines a prefix-match score with queue length and KV utilization {cite('llmdprefix')}; the Dynamo KV router minimizes a weighted sum of blocks to prefill and active blocks {cite('dynamo')}; predicted-latency schedulers learn the request's own time to first token and per-token time on each replica and pick the best fit {cite('llmdpred', 'lodestar')}. None of them charges a placement for what it does to the requests already on the replica.</p>")
    A(f"<p>Those costs are large in LLM serving. With chunked prefill, a step that carries a long prompt takes as long as the prompt's compute, and every decode stream in the batch waits. A new request's KV blocks evict cached prefixes that other requests were about to reuse. In the language of congestion games, routing each request to its own best replica is selfish routing, and selfish routing can be far from the best outcome for the system {cite('roughgarden')}. The classical remedy is marginal-cost pricing {cite('pigou1920', 'beckmann')}: charge each user the delay it imposes on others. A much older remedy for the same interference is to separate jobs by size {cite('sita')}, and we evaluate it alongside.</p>")
    A("<p>Abyss applies marginal-cost pricing per request. Its contributions are:</p><ul>"
      "<li>A formulation of LLM request routing as marginal-cost placement, with an explicit price for the interference a placement imposes on co-located requests and for the cache it evicts.</li>"
      "<li>A router that estimates those prices from counterfactual rollouts of per-replica twins, kept synchronized and calibrated from the response streams alone, so it needs no changes to the inference servers.</li>"
      f"<li>An open implementation: an OpenAI-compatible gateway, a deterministic fleet simulator with a vLLM V1 style engine model, ten baseline routers, an emulated inference server and a calibration tool, in about {n_loc['go'] // 100 * 100:,} lines of Go plus tests and a dashboard.</li>"
      "<li>An evaluation on public production traces and synthetic agent workloads, against default and held-out-tuned baselines including a size-interval heuristic, with sustained-load, objective-sensitivity, ablation, model-error and scaling measurements.</li></ul>")

    A("<h2>2 Background and motivation</h2>")
    A(f"<p><b>Serving engines.</b> Engines such as vLLM {cite('kwon')} and SGLang {cite('sglang')} run each replica as an iteration-level scheduler. Each step processes a token budget shared by running decodes and chunks of waiting prompts {cite('sarathi')}. KV memory is paged, and full blocks of computed prompt are kept in a prefix cache with least-recently-used eviction, optionally backed by host memory {cite('mooncake')}. Disaggregated systems separate prefill and decode onto different replicas to avoid interference {cite('distserve', 'splitwise')}; Abyss targets the common colocated case.</p>")
    A(f"<p><b>Routers.</b> Besides the production routers above, research systems balance prefix affinity against load {cite('preble', 'dualmap', 'cacheroute')}, predict per-request latency or output length {cite('astrolabe', 'lodestar', 'llmdpred')}, or migrate requests after placement {cite('llumnix')}. A recent study models disaggregated serving as a set of games and switches globally between cache affinity and load balancing near saturation {cite('poa')}. Abyss differs in pricing each placement's externality explicitly, per request and per replica.</p>")
    A("<p><b>An example.</b> Two replicas of Llama 3.1 70B on four H100s each. Replica 0 holds half of an incoming 30,920-token prompt in its prefix cache and is decoding 48 streams with a 24 ms per-token objective; replica 1 is idle. Abyss's twin predicts 2.05 s to first token on replica 0 and 2.78 s on replica 1, and the request's own miss probability is lower on replica 0, so a selfish router picks it. Prefilling the remaining 15,560 tokens there pushes an estimated 20.4 of the 48 streams past their objective, so Abyss places the request on replica 1. The case is a unit test in the repository.</p>")

    A("<h2>3 Problem</h2>")
    A("<p>Request <i>j</i> has a time-to-first-token objective <i>T<sub>j</sub></i> and a time-per-output-token objective <i>P<sub>j</sub></i>. Let <i>M<sub>j</sub></i> = 1 if it misses either, including abandonment. The router minimizes the expected weighted number of misses, <i>E</i>[&Sigma;<sub><i>j</i></sub> <i>w<sub>j</sub></i> <i>M<sub>j</sub></i>], which maximizes goodput. When request <i>q</i> arrives, the own-cost rule places it on the replica minimizing <i>E</i>[<i>M<sub>q</sub></i>]. The marginal-cost rule minimizes</p>")
    A('<div class="eq"><i>E</i>[<i>M<sub>q</sub></i> | <i>s<sub>r</sub></i> &oplus; <i>q</i>] + &Sigma;<sub><i>j</i> on <i>r</i></sub> (<i>E</i>[<i>M<sub>j</sub></i> | <i>s<sub>r</sub></i> &oplus; <i>q</i>] &minus; <i>E</i>[<i>M<sub>j</sub></i> | <i>s<sub>r</sub></i>]) + <i>K<sub>r</sub></i>(<i>q</i>)</div>')
    A(f"<p>where <i>s<sub>r</sub></i> is replica <i>r</i>'s state and <i>K<sub>r</sub></i>(<i>q</i>) the expected future cost of the cache entries <i>q</i> evicts. This is the one-step greedy policy for the fleet objective. In nonatomic routing with convex costs, marginal-cost tolls make the selfish equilibrium optimal {cite('beckmann')}, and for affine latencies selfish routing is within a factor of 4/3 of optimal {cite('roughgarden')}. LLM routing has atomic requests, non-stationary state and deadline objectives, so these guarantees do not transfer; Abyss uses the principle as a design rule and is evaluated empirically.</p>")

    A("<h2>4 Design</h2>")
    A(fig("architecture", "Abyss sits between clients and OpenAI-compatible servers, keeps one twin per replica, prices every placement with two rollouts of that twin, and learns from the token streams it relays.", wide=True))
    A("<p><b>Engine model.</b> The twin is a step-level model of a vLLM V1 engine: a per-step token budget, running requests first, chunked prefill, admission of waiting requests in arrival order when their first chunk fits, paged memory in 16-token pages, a prefix cache keyed by chained block hashes with least-recently-used eviction and deepest-block-first release, an optional host tier with transfer cost, and recompute preemption from the tail of the running list. The same code serves as the simulator's ground-truth replica, the router's twin and the rollout fork. This keeps rollouts consistent with the twin, but it also means the twin is structurally exact in our simulations; Section 6.5 adds model errors.</p>")
    A(f"<p><b>Step time.</b> A roofline model {cite('roofline')} gives each step's time from its composition: linear-layer and attention FLOPs against tensor-parallel compute, weight and KV bytes against HBM bandwidth, all-reduce traffic over the GPU interconnect, and fixed per-step and per-sequence overheads. Efficiencies default to 62% of peak for matrix multiplies, 42% for attention and 80% of memory bandwidth. A calibration tool replaces these defaults with values fitted to measurements of a real server.</p>")
    A("<p><b>Synchronization and calibration.</b> The gateway observes each request's first token, every streamed token and completion. Before each decision every twin advances its clock and reconciles with those observations: requests with a first token are moved to decode, token counts are updated, finished requests are removed, and requests the twin wrongly thinks have started decoding are held one token short of prefill. Two speed scales correct the model. A decode scale tracks the ratio of observed inter-token gaps to the decode part of the twin's simulated steps. A prefill scale tracks the ratio of each request's observed time to first token to the time the twin itself simulated for it, which, unlike the error of the rollout made at routing time, is not biased by the router's choice of the replica that looked best. Each twin also tracks the spread of its own TTFT and TPOT prediction errors, which sets the width of the lognormal error model that turns predictions into miss probabilities.</p>")
    A("<p><b>Pricing.</b> For each candidate replica, Abyss forks the twin, injects the request with an expected output length equal to the mean residual life of recent completions, and steps until its first token and 24 steps beyond. A second fork without the request runs to the same time. The difference in each co-located request's first-token time or token count, plus the request's added per-step decode cost over its expected lifetime, is converted into a change in that request's miss probability given its deadline and progress. Cache blocks the request would evict are taken from the head of the twin's eviction list and valued by a reuse probability learned from per-block reuse gaps times their rebuild time. A small latency term breaks ties when all probabilities are near zero. The request goes to the cheapest replica. Per-request weights let operators protect a class of requests.</p>")
    A("<p><b>Scaling.</b> Forks reuse buffers, hit lookups are memoized against a cache epoch, replicas are priced in parallel, and an optional shortlist prices only the replicas with the lowest estimated backlog plus the one with the longest cached prefix.</p>")

    A("<h2>5 Implementation</h2>")
    A(f"<p>Abyss is written in Go with no third-party dependencies ({n_loc['go']:,} lines of code and {n_loc['test']:,} lines of tests) and a TypeScript dashboard ({n_loc['ts']:,} lines). The gateway proxies OpenAI chat and completion requests, hashes 256-byte blocks of the rendered prompt as a tokenizer-free prefix identity, learns the backend's characters per token from reported usage, parses streamed events for token timing, probes backend health and optionally scrapes vLLM or SGLang gauges. An emulated server runs the engine model in real time behind the same API. <code>abyss calibrate</code> measures prefill latency at six prompt lengths and decode step time at four batch sizes on any OpenAI-compatible server and fits the efficiency parameters by Nelder-Mead.</p>")

    A("<h2>6 Evaluation</h2>")
    A("<p><b>Setup.</b> Every policy runs in the same discrete-event simulator. Each replica's real step times differ from the model by a hidden per-replica bias of up to ±12% and per-step lognormal jitter (σ = 0.06). No policy knows output lengths. Clients abandon a request without a first token after max(30 s, 4 × its TTFT objective). A request meets its objectives if its TTFT is within a base plus an allowance per thousand prompt tokens and its mean TPOT is within a bound (Table 1). Loads are multiples of a reference rate: the trace's own arrival rate replayed on the fleet in Table 1, or the generator rate for synthetic workloads. The fleet sizes are our choice, and at the reference rate the existing routers already miss most objectives, so we also report the load each policy sustains at a fixed attainment. Each point averages three disjoint 20-minute trace windows or three generator seeds.</p>")
    wrows = []
    slo = {"conversation-70b": "1.0 s + 0.15 s/k, 80 ms", "toolagent-70b": "1.0 s + 0.15 s/k, 80 ms", "agent-8b": "0.4 s + 0.03 s/k, 30 ms", "mixed-hetero": "0.8 s + 0.08 s/k, 60 ms"}
    for p in PRESETS:
        if p in D.main:
            w = D.main[p]["workload"]
            wrows.append([TITLES[p], f"{w['rate_rps']:.2f}", f"{w['mean_input']:,.0f}", f"{pct(w['reuse_ratio'], 0)}", fleet_text(D.main[p]["fleet"]), slo[p]])
    A(tab("workloads", "Workloads. Reference rate in requests per second; prompt in tokens; reuse is the share of prompt tokens that repeat an earlier prefix; objectives are TTFT and TPOT.", table(["Workload", "Rate", "Mean prompt", "Reuse", "Fleet", "Objectives"], wrows), wide=True))
    A(f"<p><b>Workloads.</b> Two are public traces from the Kimi service {cite('mooncake')}, which record arrival time, prompt and output length and the chained hashes of 512-token prompt blocks, preserving real prefix sharing. Two are synthetic: coding-agent sessions with shared 8k to 14k token tool prompts that grow by about 1.5k tokens per step, and a mix of chat, agent and retrieval traffic on a fleet of H100, A100 and L40S replicas.</p>")
    A(f"<p><b>Baselines.</b> The llm-d scorer (prefix 3, queue 2, KV 2) and the Dynamo KV router (overlap weight 1) see the engines' exact cache contents. The SGLang cache-aware router uses its approximate tree and balance thresholds. Bounded-load consistent hashing {cite('karger', 'chwbl')}, least request and round robin are classical. The llm-d predicted-latency rule {cite('llmdpred')} (80% affinity gate, 1% exploration, 5 s affinity break, 80/20 headroom, best fit) uses Abyss's twin as its latency predictor. The Preble-style policy {cite('preble')} exploits the longest match when it covers more than half the prompt and otherwise minimizes recent load plus prefill cost. The size-interval heuristic {cite('sita')} sends prompts above a length cutoff to a pool of the replicas with the slowest prefill and all other prompts to the rest, and picks the replica with the least outstanding prefill work inside each pool; least work is the same rule without pools. We call the first eight the existing routers. The selfish twin is Abyss with only the request's own cost. All are reimplementations from public descriptions.</p>")
    counts = tuned_counts(D) if D.tuned else {}
    A(f"<p><b>Tuning.</b> For each workload we grid searched every parameterized baseline on one trace window ({counts.get('llm-d', 0)} settings for llm-d, {counts.get('sglang', 0)} for SGLang, {counts.get('dynamo', 0)} for Dynamo, {counts.get('chwbl', 0)} for bounded-load hashing and {counts.get('sita', 0)} for the size-interval cutoff and pool size) and evaluated all policies on the two windows the search never saw, taking per family the better of default and tuned on the held-out data, which favors the baselines.</p>")

    A("<h3>6.1 Main results</h3>")
    A(fig("attainment", "Share of requests meeting both objectives as offered load grows, default parameters, mean of three runs; bands show one standard deviation.", wide=True))
    hrows = []
    for p, h in heads.items():
        hrows.append([TITLES[p], f"<b>{pct(h['abyss']['attainment'])}</b>", pct(h["selfish"]["attainment"]), f"{pct(h['existing']['attainment'])} ({html.escape(label(h['existing']['policy']))})",
                      f"{pct(h['heuristic']['attainment'])} ({html.escape(label(h['heuristic']['policy']))})", f"{pct(h['abyss']['token_weighted_attainment'], 0)} / {pct(h['heuristic']['token_weighted_attainment'], 0)}"])
    A(tab("main", "Held-out attainment at the reference rate with tuned baselines. The last column weights requests by prompt tokens, for Abyss and the heuristic.", table(["Workload", "Abyss", "Selfish twin", "Best existing router", "Size-interval or least work", "By tokens"], hrows), wide=True))
    A(f"<p>{F('attainment')} shows attainment against load with default parameters and {T('main')} the held-out comparison with tuned baselines at the reference rate. {headline_sentence(D)} {low_load_sentence(D)}</p>")
    A(fig("tuned", "Held-out attainment for Abyss, the best production router with default or tuned parameters, the best research baseline (Preble-style or llm-d predicted latency) and the size-interval heuristic or least work.", wide=True))

    A("<h3>6.2 Sustained load</h3>")
    crows = []
    for t in TARGETS:
        for p, pig, (c, pol) in capacity_rows(D, t):
            crows.append([f"{pct(t, 0)}", TITLES[p], f"<b>{html.escape(cap_text(pig))}</b>", f"{html.escape(cap_text(c))} ({html.escape(label(pol))})"])
    A(tab("capacity", "Highest load, as a multiple of the reference rate, at which at least the target share of requests meets both objectives, default parameters, interpolated between tested loads.", table(["Target", "Workload", "Abyss", "Best other policy"], crows)))
    A(f"<p>Attainment at a single load mixes up two questions: how much traffic a fleet can carry and how it degrades beyond that. {T('capacity')} answers the first. {capacity_sentence(D)}</p>")

    if D.sens:
        A("<h3>6.3 Sensitivity to the objectives</h3>")
        A(f"<p>The objectives decide how much a long prefill step costs. One step carrying an 8k-token chunk of Llama 3.1 70B takes about 0.6 s on four H100s, enough to push a response of a few tokens past an 80 ms mean TPOT on its own. We reran both production traces with looser TPOT and TTFT objectives and with the TPOT objective applied only to responses of at least 16 tokens ({F('sensitivity')}); the size-interval column takes the best of five settings in hindsight. {sensitivity_sentence(D)}</p>")
        A(fig("sensitivity", "Attainment at the reference rate under different latency objectives. The heuristic bar is the best of five size-interval and least-work settings in hindsight.", wide=True))

    A("<h3>6.4 Ablations</h3>")
    if abl:
        ag = D.abl.get("agent-8b", {}).get("points", [])
        cache_bits = []
        for name, pts, sc in [("the coding agent sessions", ag, 0.5), ("the coding agent sessions", ag, 0.75), ("the conversation trace", abl, 1)]:
            a, b = at(pts, "abyss", sc), at(pts, "abyss-no-cache", sc)
            if a and b:
                cache_bits.append(f"{100 * (a['attainment'] - b['attainment']):+.1f} points on {name} at {sc:g}x load")
        orc = [abs(at(pts, "abyss-oracle", sc)["attainment"] - at(pts, "abyss", sc)["attainment"]) for pts in [abl, ag] for sc in scales(pts) if at(pts, "abyss-oracle", sc)]
        lb, sh = at(abl, "abyss-late-binding", 1), at(abl, "abyss-shed", 1)
        A(f"<p>Removing the interference term reduces attainment on the conversation trace at the reference rate from {pct(abl_full)} to {pct(abl_noint)}, and keeping only the request's own cost gives {pct(abl_self)} ({F('ablation')}). The cache term's effect is {'; '.join(cache_bits)}: it helps where prefix reuse is highest and is otherwise within noise. "
          f"Giving the twin the true output length of every request changes attainment by at most {100 * max(orc):.1f} points. Gateway-side queueing gives {pct(lb['attainment'])} at the reference rate, and rejecting requests predicted to exceed twice their TTFT objective gives {pct(sh['attainment'])} while cutting p99 TTFT from {sec(at(abl, 'abyss', 1)['ttft_p99'])} to {sec(sh['ttft_p99'])}.</p>")
        A(fig("ablation", "Ablations on the conversation trace and the coding agent sessions."))

    if D.robust:
        A("<h3>6.5 Robustness and model error</h3>")
        rrows = []
        for group, rows in ROBUST_GROUPS:
            for k, name in rows:
                if k not in D.robust:
                    continue
                pts = D.robust[k]["points"]
                cells = [name]
                for s in [0.75, 1]:
                    cells += [pct(at(pts, "abyss", s)["attainment"]), pct(best(pts, s, EXISTING)["attainment"]), pct(D.robust_heuristic(pts, s)["attainment"])]
                rrows.append(cells)
        A(tab("robust", "Conversation trace under different engine settings, uniform speed errors and model errors that affect prefill and decode differently. Existing is the strongest of llm-d predicted latency, llm-d, Dynamo and Preble-style; heuristic is the best of size-interval (default and tuned) and least work.", table(["Condition", "Abyss 0.75x", "Existing 0.75x", "Heuristic 0.75x", "Abyss 1x", "Existing 1x", "Heuristic 1x"], rrows), wide=True))
        uni = [k for _, rows in ROBUST_GROUPS[:2] for k, _ in rows if k in D.robust]
        pv = [at(D.robust[k]["points"], "abyss", s)["attainment"] for k in uni for s in [0.75, 1]]
        A(f"<p>{T('robust')} and {F('robustness')} rerun the conversation trace with vLLM's long-prefill cap, a smaller step budget, replicas uniformly slower or faster than Abyss's model, a wider spread of hidden per-replica speed, and five model errors that change the efficiency of matrix multiplies, attention or memory bandwidth on the real replicas only. Under the engine and uniform-speed conditions Abyss's attainment ranges from {pct(min(pv))} to {pct(max(pv))}.{calib_sentence(D)} {mismatch_sentence(D)}</p>")
        A(fig("robustness", f"Attainment on the conversation trace under the conditions of {T('robust')}.", wide=True))

    A("<h3>6.6 Where the gain comes from</h3>")
    A(f"<p>{roles_eval(D)} ({F('replica-roles')}). The behavior resembles prefill and decode disaggregation {cite('distserve', 'splitwise')} and size-interval task assignment {cite('sita')}, here arising from prices rather than rules.</p>")
    A(fig("replica-roles", "Share of prefill and decode tokens per replica, conversation trace, first window, reference rate.", wide=True))
    lr = D.run("conversation-70b", "abyss", native(D.main["conversation-70b"]["points"]))
    if lr:
        lb = lr["summary"]["length_attainment"][-1]
        total = sum(b["requests"] for b in lr["summary"]["length_attainment"])
        A(f"<p><b>Who pays.</b> Abyss maximizes the count of requests meeting their objectives, so the most expensive requests gain least: prompts above 64k tokens ({pct(lb['requests'] / total, 0)} of the window) meet their objectives {pct(lb['attainment'], 0)} of the time at the reference rate. The size-interval heuristic gives up on almost every prompt above its cutoff, which is why its token-weighted attainment trails its request count. Fairness-oriented scheduling {cite('vtc')} could be layered on through per-request weights.</p>")

    A("<h3>6.7 Prediction accuracy and overhead</h3>")
    errs = []
    for p in PRESETS:
        if p in D.main:
            errs += [x["pred_error_p50"] for x in D.main[p]["points"] if x["policy"] == "abyss"]
    A(f"<p>Measured against the real first token of each request, the median relative error of Abyss's TTFT prediction ranges from {pct(min(errs), 0)} to {pct(max(errs), 0)} across workloads and loads, under hidden speed bias, jitter, unknown output lengths and unknown future arrivals.</p>")
    if micro:
        A(f"<p>{F('decision-time')} shows the time to price one request, measured with Go's benchmark harness on {html.escape(micro['cpu'])}, a two-core laptop processor. Pricing every replica takes {m.get((8, 'all'), 0):.2f} ms at 8 replicas and {m.get((64, 'all'), 0):.1f} ms at 64; parallel pricing reduces the 64-replica case to {m.get((64, 'all-parallel'), 0):.1f} ms and the shortlist of 8 to {m.get((64, 'k8'), 0):.1f} ms. The size-interval heuristic decides in microseconds.</p>")
        A(fig("decision-time", "Decision time against fleet size."))
    if D.scaling:
        srows = []
        for n, d in sorted(D.scaling.items()):
            pts = d["points"]
            srows.append([str(n), f"{pts[0]['rate']:.1f}"] + [pct(at(pts, p, 1)["attainment"]) if at(pts, p, 1) else "" for p in ["abyss", "abyss-k8", "llm-d", "dynamo", "preble", "sita"]])
        A(tab("scaling", "Mixed traffic on 8 to 64 Llama 3.1 8B replicas, load proportional to fleet size, one 300 s run each.", table(["Replicas", "Req/s", "Abyss", "Abyss k=8", "llm-d", "Dynamo", "Preble", "Size-interval"], srows)))
        A(f"<p>{scaling_sentence(D)}</p>")

    if D.calib:
        c = D.calib
        A("<h3>6.8 Calibration</h3>")
        A(f"<p>To check the calibration path, we ran <code>abyss calibrate</code> against an emulated H100 secretly configured to run 30% slower than its model. The default model's median timing error was {pct(c['default_median_abs_rel_error'], 0)}; after fitting it was {pct(c['median_abs_rel_error'])} (maximum {pct(c['max_abs_rel_error'])}, {F('calibration')}). This verifies the measurement and fitting procedure. Because the emulator uses the same model family, it does not show that the model fits real GPUs, which remains to be measured.</p>")
        A(fig("calibration", "Measured, fitted and default step times relative to the measurement."))

    A("<h2>7 Related work</h2>")
    A(f"<p>Prefix-aware routing in production combines cache affinity with load signals {cite('llmdprefix', 'dynamo')}. Preble balances prefix reuse against load with an exploit-or-explore rule and accounts for eviction cost {cite('preble')}; DualMap maps prefixes to two candidate instances and chooses between them {cite('dualmap')}; CacheRoute plans a periodic routing table that balances affinity against expected load {cite('cacheroute')}. Astrolabe and Lodestar use predicted or learned performance to guide dispatch {cite('astrolabe', 'lodestar')}, and llm-d's predicted-latency scheduler picks the best-fitting replica by predicted headroom {cite('llmdpred')}. These systems optimize for the incoming request or for aggregate balance; Abyss prices the per-request externality on co-located requests. Llumnix migrates requests after placement {cite('llumnix')}, which is complementary. Disaggregation removes prefill and decode interference by construction {cite('distserve', 'splitwise')}, at the cost of transferring KV state; Abyss reduces interference within colocated fleets. Vidur simulates single-replica serving for configuration search {cite('vidur')}; Abyss's simulator models fleets and routing. Load balancing theory offers power-of-two choices {cite('p2c')}, bounded-load hashing {cite('chwbl')} and size-based assignment {cite('sita')}, the last of which is a strong baseline here, and congestion games supply the marginal-cost principle {cite('pigou1920', 'beckmann', 'roughgarden')}; a recent study applies price-of-anarchy analysis to disaggregated inference {cite('poa')}.</p>")

    A("<h2>8 Limitations</h2>")
    A("<p>All headline results come from simulation. The engine and step-time models follow vLLM V1 and published accelerator specifications with assumed efficiencies and have not been validated against GPUs; the repository includes a procedure and scripts to run the same comparison on real vLLM servers. Abyss's twin and the simulated replicas share code, so structural differences between a twin and a real engine are not represented; the model-error experiments cover speed errors only. The size of Abyss's lead depends on the operating point and on the latency objectives, and at the reference rate the existing routers already miss most objectives. Baselines are reimplementations. The objective counts requests, which favors short ones. The gateway identifies prompts by bytes rather than tokens. Under overload Abyss concentrates misses on few requests, which hit the client abandonment limit. Decision cost grows with fleet size; the shortlist bounds it but loses attainment on large fleets, so parallel pricing is the practical option there.</p>")

    A("<h2>9 Conclusion</h2>")
    if behind:
        verdict = (f"A size-interval heuristic that isolates long prompts by rule matches or exceeds Abyss near capacity on {words(len(behind) + len(level))} of {words(len(heads))} workloads, at the cost of the long prompts and of attainment at lower load. "
                   "Much of the benefit therefore comes from keeping long prefills away from latency-sensitive traffic; what pricing adds is doing so adaptively, without a tuned cutoff, and serving long prompts when there is room.")
    else:
        verdict = ("A size-interval heuristic that isolates long prompts by rule recovers part of that gain near capacity, at the cost of the long prompts and of attainment at lower load. "
                   "Much of the benefit therefore comes from keeping long prefills away from latency-sensitive traffic; what pricing adds is doing so adaptively.")
    A(f"<p>Routing a request to the replica that is best for it alone ignores what the placement costs the requests already running there. Pricing that cost explicitly, using counterfactual rollouts of calibrated replica twins, meets latency objectives for more requests than the production and research routers we compared it with in simulation, with default and tuned parameters. {verdict} The next step is to confirm these results on GPU servers.</p>")

    A('<h2>References</h2><ol class="refs">' + "".join(f"<li>{html.escape(t)}</li>" for _, t in REFS) + "</ol>")
    A("</div>")
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Abyss paper draft</title><style>{css}</style></head><body>{"".join(body)}</body></html>'


def main():
    D = Data()
    os.makedirs(OUT, exist_ok=True)
    page = build(D)
    path = os.path.join(OUT, "abyss-paper.html")
    open(path, "w", encoding="utf-8", newline="\n").write(page)
    pdf = os.path.join(OUT, "abyss-paper.pdf")
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    exe = next((e for e in [os.environ.get("BROWSER", ""), chrome, edge, "chromium", "google-chrome"] if e and (os.path.exists(e) or "/" not in e and "\\" not in e)), None)
    profile = tempfile.mkdtemp(prefix="abyss-paper-")
    cmd = [exe, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer", "--print-to-pdf-no-header", f"--print-to-pdf={pdf}", "file:///" + path.replace("\\", "/")]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    shutil.rmtree(profile, ignore_errors=True)
    print("wrote", path, "and", pdf if os.path.exists(pdf) else "(PDF not produced)")


if __name__ == "__main__":
    main()
