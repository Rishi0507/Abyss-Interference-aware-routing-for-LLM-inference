import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preset, family = sys.argv[1], sys.argv[2]
path = os.path.join(ROOT, "results", "tuned", preset + ".json")
choice = family
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        for c in json.load(f)["choices"]:
            if c["family"] == family and c["best"]:
                choice = c["best"]
print(choice)
