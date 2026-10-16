import json
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
cpu = ""
rows = []
for line in open(src, encoding="utf-8"):
    if line.startswith("cpu:"):
        cpu = line.split(":", 1)[1].strip()
    m = re.match(r"BenchmarkAbyssDecision/replicas=(\d+)/(\S+?)-\d+\s+\d+\s+([\d.]+) ns/op", line)
    if m:
        rows.append({"replicas": int(m.group(1)), "mode": m.group(2), "micros": float(m.group(3)) / 1000})
json.dump({"cpu": cpu, "rows": rows}, open(dst, "w", encoding="utf-8", newline="\n"), indent=2)
print(f"wrote {dst} with {len(rows)} rows")
