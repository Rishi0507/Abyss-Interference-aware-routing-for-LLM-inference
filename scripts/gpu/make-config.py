import glob
import json
import os

model_key = os.environ.get("MODEL_KEY", "llama-3.1-8b")
hardware = os.environ.get("HARDWARE", "h100")
token_scale = float(os.environ.get("TOKEN_SCALE", "0.25"))
ttft_base = float(os.environ.get("TTFT_BASE", "1.0"))
ttft_per_k = float(os.environ.get("TTFT_PER_K", "0.15"))
tpot = float(os.environ.get("TPOT", "0.08"))

backends = []
cpts = []
for path in sorted(glob.glob("results/gpu/calibration-*.json")):
    port = path.rsplit("-", 1)[1].split(".")[0]
    with open(path) as f:
        cal = json.load(f)
    cpts.append(cal.get("chars_per_token", 4))
    backends.append({
        "name": f"gpu-{port}",
        "url": f"http://localhost:{port}",
        "model": model_key,
        "hardware": hardware,
        "tp": 1,
        "scrape_metrics": True,
        "efficiency": cal["efficiency"],
    })

config = {
    "listen": ":8080",
    "policy": "abyss",
    "chars_per_token": sum(cpts) / len(cpts) if cpts else 4,
    "slo": {"ttft_base_s": ttft_base, "ttft_per_ktok_s": ttft_per_k / token_scale, "tpot_s": tpot},
    "backends": backends,
}
os.makedirs("results/gpu", exist_ok=True)
with open("results/gpu/gateway.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"wrote results/gpu/gateway.json with {len(backends)} backends")
