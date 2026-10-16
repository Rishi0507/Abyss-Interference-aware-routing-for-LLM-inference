#!/usr/bin/env sh
set -e
BIN=${BIN:-./abyss}
PAR=${PAR:-3}
POLICIES="round-robin,least-request,chwbl,sglang,llm-d,dynamo,preble,least-work,sita,llm-d-predicted,abyss-selfish,abyss"
for preset in ${PRESETS:-conversation-70b toolagent-70b agent-8b mixed-hetero}; do
  $BIN bench --preset "$preset" --seeds 3 --parallel "$PAR" --timeline=false --policies "$POLICIES" --out "results/$preset.json"
done
