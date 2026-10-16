#!/usr/bin/env sh
set -e
BIN=${BIN:-./abyss}
PAR=${PAR:-3}
mkdir -p results/ablations
$BIN bench --preset conversation-70b --seeds 3 --scales 0.75,1 --timeline=false --parallel "$PAR" \
  --policies abyss,abyss-selfish,abyss-no-interference,abyss-no-cache,abyss-late-binding,abyss-oracle,abyss-shed \
  --out results/ablations/conversation-70b.json
$BIN bench --preset agent-8b --seeds 3 --scales 0.5,0.75 --timeline=false --parallel "$PAR" \
  --policies abyss,abyss-selfish,abyss-no-interference,abyss-no-cache,abyss-oracle \
  --out results/ablations/agent-8b.json
