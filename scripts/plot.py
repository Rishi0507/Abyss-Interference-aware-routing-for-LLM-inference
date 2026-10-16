import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")

BG = "#ffffff"
PANEL = "#ffffff"
INK = "#14213d"
INK2 = "#4a5874"
INK3 = "#74809a"
GRID = "#e6e9ef"

COLORS = {
    "llm-d": "#2a78d6",
    "dynamo": "#eb6834",
    "sglang": "#1baf7a",
    "abyss": "#b87a00",
    "round-robin": "#d85a8c",
    "llm-d-predicted": "#008300",
    "abyss-selfish": "#6250d6",
    "preble": "#e34948",
    "sita": "#a3478f",
}
HEUR = "#a3478f"
OTHER = "#9aa5b8"
EXISTING = ["llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "round-robin", "least-request", "chwbl"]
HEURISTIC = ["sita", "least-work"]
LABELS = {
    "abyss": "Abyss",
    "abyss-selfish": "Selfish twin",
    "llm-d": "llm-d scorer",
    "dynamo": "Dynamo KV router",
    "sglang": "SGLang cache-aware",
    "round-robin": "Round robin",
    "least-request": "Least request",
    "chwbl": "Bounded-load hashing",
    "llm-d-predicted": "llm-d predicted latency",
    "preble": "Preble-style",
    "sita": "Size-interval",
    "least-work": "Least work",
}
ORDER = ["abyss", "abyss-selfish", "llm-d-predicted", "llm-d", "dynamo", "sglang", "preble", "sita"]
TITLES = {
    "conversation-70b": "Conversation trace, Llama 3.1 70B",
    "toolagent-70b": "Tool and agent trace, Llama 3.1 70B",
    "agent-8b": "Coding agent sessions, Llama 3.1 8B",
    "mixed-hetero": "Mixed traffic, heterogeneous fleet",
}


def setup():
    family = None
    for f in font_manager.findSystemFonts():
        name = os.path.basename(f).lower()
        if "segoeui" in name or "inter" in name:
            family = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams.update(
        {
            "font.family": family or "DejaVu Sans",
            "font.size": 10.5,
            "axes.facecolor": PANEL,
            "figure.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK3,
            "ytick.color": INK3,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def load(name):
    path = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def attainment(names):
    sets = [(n, load(n)) for n in names]
    sets = [(n, d) for n, d in sets if d]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.9), squeeze=False)
    for ax, (name, d) in zip(axes[0], sets):
        for pol in ORDER:
            if pol not in COLORS:
                continue
            pts = sorted([p for p in d["points"] if p["policy"] == pol], key=lambda p: p["rate"])
            if not pts:
                continue
            x = [p["rate"] for p in pts]
            y = [100 * p["attainment"] for p in pts]
            sd = [100 * p["attainment_sd"] for p in pts]
            lw = 3 if pol == "abyss" else 1.8
            ax.fill_between(x, [a - b for a, b in zip(y, sd)], [a + b for a, b in zip(y, sd)], color=COLORS[pol], alpha=0.12, lw=0)
            ax.plot(x, y, color=COLORS[pol], lw=lw, marker="o", ms=5 if pol == "abyss" else 4, mec=PANEL, mew=1.2, label=LABELS[pol], zorder=3 if pol == "abyss" else 2)
        ax.set_title(TITLES.get(name, name), color=INK, fontsize=11.5, loc="left", pad=10)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Offered load (requests per second)")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes[0][0].set_ylabel("Requests meeting their targets")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "attainment")


def roles(name="conversation-70b"):
    d = load(name)
    if not d:
        return
    scales = sorted({r["scale"] for r in d["runs"]}, key=lambda s: abs(s - 1))
    scale = scales[0]
    runs = {r["policy"]: r for r in d["runs"] if r["scale"] == scale and r["seed"] == 0 and r.get("result")}
    pols = [p for p in ["round-robin", "sita", "abyss"] if p in runs]
    fig, axes = plt.subplots(1, len(pols), figsize=(4.4 * len(pols), 3.0), sharey=True, squeeze=False)
    for ax, pol in zip(axes[0], pols):
        reps = runs[pol]["result"]["summary"]["replicas"]
        pf = sum(r["prefill_tokens"] for r in reps)
        dc = sum(r["decode_tokens"] for r in reps)
        idx = range(len(reps))
        ax.barh([i + 0.2 for i in idx], [100 * r["prefill_tokens"] / pf for r in reps], height=0.36, color="#7b8ba3", label="Prefill tokens")
        ax.barh([i - 0.2 for i in idx], [100 * r["decode_tokens"] / dc for r in reps], height=0.36, color="#2a78d6", label="Decode tokens")
        ax.set_yticks(list(idx))
        ax.set_yticklabels([f"Replica {i}" for i in idx], color=INK2)
        ax.invert_yaxis()
        att = runs[pol]["result"]["summary"]["slo_attainment"]
        ax.set_title(f"{LABELS[pol]}: {100 * att:.0f}% met their targets", loc="left", fontsize=11, color=INK, pad=8)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlabel("Share of fleet tokens")
        ax.grid(axis="y", visible=False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, labelcolor=INK2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "replica-roles")


def ablation():
    rows = []
    for name in ["conversation-70b", "agent-8b"]:
        path = os.path.join(RESULTS, "ablations", name + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows.append((name, json.load(f)))
    if not rows:
        return
    names = {
        "abyss": "Abyss",
        "abyss-no-cache": "Without cache term",
        "abyss-no-interference": "Without interference term",
        "abyss-selfish": "Own cost only",
        "abyss-oracle": "Known output lengths",
        "abyss-late-binding": "With late binding",
        "abyss-shed": "With admission control",
    }
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 3.4), squeeze=False)
    for ax, (name, d) in zip(axes[0], rows):
        scales = sorted({p["scale"] for p in d["points"]})
        pols = [p for p in names if any(q["policy"] == p for q in d["points"])]
        width = 0.8 / len(scales)
        shades = ["#9db8dd", "#2a78d6", "#14213d"]
        for k, s in enumerate(scales):
            vals = []
            for p in pols:
                pt = next((q for q in d["points"] if q["policy"] == p and q["scale"] == s), None)
                vals.append(100 * pt["attainment"] if pt else 0)
            ax.barh([i + (k - (len(scales) - 1) / 2) * width for i in range(len(pols))], vals, height=width * 0.9, color=shades[k % len(shades)], label=f"{s:g}x trace rate")
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels([names[p] for p in pols], color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_title(TITLES.get(name, name), loc="left", fontsize=11, color=INK, pad=8)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(scales), fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    save(fig, "ablation")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    svg = os.path.join(OUT, f"{name}.svg")
    with open(svg, "rb") as f:
        data = f.read()
    with open(svg, "wb") as f:
        f.write(data.replace(b"\r\n", b"\n"))
    print("wrote", os.path.join("docs", "figures", name + ".svg"))


def best_of(points, scale, families):
    best = None
    for p in points:
        if p["scale"] != scale or p["policy"].split(":")[0] not in families:
            continue
        if best is None or p["attainment"] > best["attainment"]:
            best = p
    return best


def tuned():
    names = [n for n in ["conversation-70b", "toolagent-70b", "agent-8b", "mixed-hetero"] if os.path.exists(os.path.join(RESULTS, "tuned", n + ".json"))]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.3), squeeze=False)
    groups = [
        ("Abyss", ["abyss"], COLORS["abyss"]),
        ("Best production router, default or tuned", ["llm-d", "dynamo", "sglang", "chwbl"], "#2a78d6"),
        ("Best research baseline", ["preble", "llm-d-predicted"], OTHER),
        ("Size-interval or least work, default or tuned", HEURISTIC, HEUR),
    ]
    for ax, name in zip(axes[0], names):
        with open(os.path.join(RESULTS, "tuned", name + ".json"), encoding="utf-8") as f:
            pts = json.load(f)["evaluation"]["points"]
        scales = sorted({p["scale"] for p in pts})
        width = 0.8 / len(groups)
        for gi, (label, fams, color) in enumerate(groups):
            xs, ys = [], []
            for si, sc in enumerate(scales):
                b = best_of(pts, sc, fams)
                if b:
                    xs.append(si + (gi - (len(groups) - 1) / 2) * width)
                    ys.append(100 * b["attainment"])
            ax.bar(xs, ys, width=width * 0.92, color=color, label=label)
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([f"{x:g}x load" for x in scales], color=INK2)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_title(TITLES.get(name, name), loc="left", fontsize=11, color=INK, pad=8)
        ax.grid(axis="x", visible=False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, labelcolor=INK2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "tuned")


ROBUST = [
    ("long-prefill-2048", "Long-prefill cap 2048"),
    ("chunk-2048", "Step budget 2048"),
    ("replicas-40pct-slower", "Replicas 40% slower"),
    ("replicas-25pct-faster", "Replicas 25% faster"),
    ("bias-spread-30pct", "Speed spread 30%"),
    ("prefill-gemm-slower", "Matmul efficiency -27%"),
    ("attention-slower", "Attention efficiency -40%"),
    ("decode-bandwidth-lower", "Bandwidth efficiency -25%"),
    ("prefill-and-attention-slower", "Matmul and attention lower"),
    ("prefill-faster-decode-slower", "Matmul +30%, bandwidth -25%"),
]
SENSITIVITY = [
    ("reference", "Reference objectives"),
    ("tpot-120ms", "TPOT 120 ms"),
    ("tpot-200ms", "TPOT 200 ms"),
    ("ttft-2x", "TTFT objective doubled"),
    ("ttft-2x-tpot-150ms", "TTFT doubled, TPOT 150 ms"),
    ("tpot-min-16", "TPOT only for 16+ tokens"),
]


def three_bars(ax, rows, title):
    idx = range(len(rows))
    h = 0.26
    for k, (key, label, color) in enumerate([("pig", "Abyss", COLORS["abyss"]), ("ex", "Best existing router", OTHER), ("heur", "Size-interval or least work", HEUR)]):
        ax.barh([i + (k - 1) * h for i in idx], [r[key] for r in rows], height=h * 0.9, color=color, label=label)
    ax.set_yticks(list(idx))
    ax.set_yticklabels([r["label"] for r in rows], color=INK2)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.grid(axis="y", visible=False)


def summarize(pts, sc):
    heur = [p for p in pts if p["scale"] == sc and p["policy"].split(":")[0] in HEURISTIC]
    return {
        "pig": 100 * (best_of(pts, sc, ["abyss"]) or {"attainment": 0})["attainment"],
        "ex": 100 * (best_of(pts, sc, EXISTING) or {"attainment": 0})["attainment"],
        "heur": 100 * max([p["attainment"] for p in heur] or [0]),
    }


def robustness():
    rows = []
    for key, label in ROBUST:
        path = os.path.join(RESULTS, "robustness", key + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows.append((label, json.load(f)["points"]))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 0.62 * len(rows) + 1.4), sharey=True)
    for ax, sc in zip(axes, [0.75, 1]):
        data = [dict(summarize(pts, sc), label=label) for label, pts in rows]
        three_bars(ax, data, f"Conversation trace at {sc:g}x load")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "robustness")


def sensitivity():
    panels = []
    for preset in ["conversation-70b", "toolagent-70b"]:
        data = []
        for key, label in SENSITIVITY:
            path = os.path.join(RESULTS, "sensitivity", f"{preset}-{key}.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                pts = json.load(f)["points"]
            data.append(dict(summarize(pts, 1), label=label))
        if data:
            panels.append((preset, data))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 0.62 * len(SENSITIVITY) + 1.4), sharey=True, squeeze=False)
    for ax, (preset, data) in zip(axes[0], panels):
        three_bars(ax, data, f"{TITLES[preset]}, 1x load")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, labelcolor=INK2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "sensitivity")


def scaling():
    path = os.path.join(RESULTS, "microbench.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        mb = json.load(f)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for mode, label, color in [("all", "Every replica", COLORS["abyss"]), ("all-parallel", "Every replica, parallel", "#2a78d6"), ("k8", "Shortlist of 8", "#1baf7a")]:
        pts = sorted([r for r in mb["rows"] if r["mode"] == mode], key=lambda r: r["replicas"])
        ax.plot([r["replicas"] for r in pts], [r["micros"] / 1000 for r in pts], marker="o", color=color, lw=2, mec=PANEL, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64])
    ax.set_xticklabels(["8", "16", "32", "64"])
    ax.set_xlabel("Replicas")
    ax.set_ylabel("Decision time (ms)")
    ax.set_title("Time to price one request", loc="left", fontsize=11, color=INK, pad=8)
    ax.legend(fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    save(fig, "decision-time")


def calibration():
    d = os.path.join(RESULTS, "calibration")
    if not os.path.isdir(d):
        return
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files:
        return
    with open(os.path.join(d, files[0]), encoding="utf-8") as f:
        c = json.load(f)
    rows = c["rows"]
    labels = []
    for r in rows:
        smp = r["sample"]
        labels.append(f"prefill {smp['prompt_tokens']}" if smp["kind"] == "prefill" else f"decode b={smp['batch']}")
    meas = [r["sample"]["seconds"] for r in rows]
    fit = [r["predicted_s"] for r in rows]
    dflt = [r["default_predicted_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    idx = range(len(rows))
    ax.bar([i - 0.27 for i in idx], [1.0 for _ in meas], width=0.26, color="#14213d", label="Measured")
    ax.bar(list(idx), [a / m for a, m in zip(fit, meas)], width=0.26, color=COLORS["abyss"], label="Fitted model")
    ax.bar([i + 0.27 for i in idx], [a / m for a, m in zip(dflt, meas)], width=0.26, color="#9aa5b8", label="Default model")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=30, ha="right", color=INK2, fontsize=9)
    ax.set_ylabel("Time relative to measured")
    ax.set_ylim(0, 1.25)
    ax.set_title("Calibration against an emulated H100 running 30% slow", loc="left", fontsize=11, color=INK, pad=8)
    ax.legend(fontsize=9, labelcolor=INK2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "calibration")


if __name__ == "__main__":
    setup()
    presets = sys.argv[1:] or ["conversation-70b", "toolagent-70b", "agent-8b", "mixed-hetero"]
    attainment(presets)
    roles()
    ablation()
    tuned()
    robustness()
    sensitivity()
    scaling()
    calibration()
