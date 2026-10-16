#!/usr/bin/env sh
set -e
BIN=${BIN:-./abyss}
PAR=${PAR:-2}
mkdir -p results/scaling
for n in 8 16 32 64; do
  rate=$(python -c "print(2.4 * $n)")
  $BIN bench --preset agent-8b --workload mixed --rate "$rate" --duration 300 --replicas "$n" --seeds 1 --scales 1 \
    --timeline=false --parallel "$PAR" --policies "abyss,abyss-k8,llm-d,dynamo,preble,sita" --out "results/scaling/replicas-$n.json"
done
