#!/usr/bin/env sh
set -e
BIN=${BIN:-./abyss}
PAR=${PAR:-3}
mkdir -p results/sensitivity
for preset in ${PRESETS:-conversation-70b toolagent-70b}; do
  SITA=$(python scripts/tuned-choice.py "$preset" sita)
  POLICIES="abyss,abyss-selfish,llm-d-predicted,llm-d,dynamo,preble,round-robin,least-work,sita,sita:16384,1,sita:32768,1,$SITA"
  run() {
    name=$1
    shift
    $BIN bench --preset "$preset" --seeds 3 --scales 0.5,1 --timeline=false --parallel "$PAR" \
      --policies "$POLICIES" --out "results/sensitivity/$preset-$name.json" "$@"
  }
  run reference
  run tpot-120ms --slo-tpot 0.12
  run tpot-200ms --slo-tpot 0.2
  run ttft-2x --slo-ttft 2 --slo-ttft-per-k 0.3
  run ttft-2x-tpot-150ms --slo-ttft 2 --slo-ttft-per-k 0.3 --slo-tpot 0.15
  run tpot-min-16 --slo-tpot-min-tokens 16
done
