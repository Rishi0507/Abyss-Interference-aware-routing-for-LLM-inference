#!/usr/bin/env sh
set -e
PRESET=${PRESET:-conversation-70b}
SCALE=${SCALE:-0.5}
TOKEN_SCALE=${TOKEN_SCALE:-0.25}
DURATION=${DURATION:-600}
POLICIES=${POLICIES:-"round-robin llm-d dynamo sglang preble abyss-selfish abyss"}
CPT=$(python -c "import json; print(json.load(open('results/gpu/gateway.json'))['chars_per_token'])")
mkdir -p results/gpu/runs
for policy in $POLICIES; do
  sh scripts/gpu/stop-vllm.sh
  sh scripts/gpu/start-vllm.sh
  ./abyss gateway --config results/gpu/gateway.json --policy "$policy" > "results/gpu/logs/gateway-$policy.log" 2>&1 &
  gw=$!
  until curl -sf http://localhost:8080/healthz > /dev/null; do sleep 1; done
  ./abyss loadgen --preset "$PRESET" --scale "$SCALE" --duration "$DURATION" --token-scale "$TOKEN_SCALE" \
    --output-scale 1 --max-output 1024 --chars-per-token "$CPT" --target http://localhost:8080 \
    --out "results/gpu/runs/$policy.json"
  kill "$gw"
  wait "$gw" 2>/dev/null || true
done
sh scripts/gpu/stop-vllm.sh
