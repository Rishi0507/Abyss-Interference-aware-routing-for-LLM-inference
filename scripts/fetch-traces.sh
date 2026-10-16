#!/usr/bin/env sh
set -e
BASE=https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release
mkdir -p traces
for f in traces/conversation_trace.jsonl traces/toolagent_trace.jsonl traces/synthetic_trace.jsonl arxiv-trace/mooncake_trace.jsonl; do
  name=$(basename "$f")
  if [ ! -s "traces/$name" ]; then
    echo "downloading $name"
    curl -fsSL -o "traces/$name" "$BASE/$f"
  fi
done
ls -l traces
