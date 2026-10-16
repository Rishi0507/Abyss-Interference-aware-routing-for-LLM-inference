#!/usr/bin/env sh
for f in results/gpu/logs/vllm-*.pid; do
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null && rm -f "$f"
done
sleep 5
