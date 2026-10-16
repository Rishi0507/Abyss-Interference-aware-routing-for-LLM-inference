import glob
import json
import os

rows = []
for path in sorted(glob.glob("results/gpu/runs/*.json")):
    with open(path) as f:
        s = json.load(f)["summary"]
    rows.append((os.path.basename(path)[:-5], s))
rows.sort(key=lambda r: -r[1]["attainment"])
print(f"{'policy':<16}{'met':>8}{'ttft p50':>11}{'ttft p99':>11}{'tpot p50':>11}{'tpot p99':>11}{'failed':>8}")
for name, s in rows:
    print(f"{name:<16}{100 * s['attainment']:7.1f}%{1000 * s['ttft']['p50']:9.0f}ms{1000 * s['ttft']['p99']:9.0f}ms"
          f"{1000 * s['tpot']['p50']:9.1f}ms{1000 * s['tpot']['p99']:9.1f}ms{s['failed'] + s['rejected']:8d}")
