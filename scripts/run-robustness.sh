#!/usr/bin/env sh
set -e
BIN=${BIN:-./abyss}
PAR=${PAR:-3}
SITA=$(python scripts/tuned-choice.py conversation-70b sita)
POLICIES="abyss,abyss-selfish,llm-d-predicted,llm-d,dynamo,preble,least-work,sita,$SITA"
mkdir -p results/robustness
run() {
  name=$1
  shift
  $BIN bench --preset conversation-70b --seeds 3 --scales 0.75,1 --timeline=false --parallel "$PAR" \
    --policies "$POLICIES" --out "results/robustness/$name.json" "$@"
}
run long-prefill-2048 --long-prefill 2048
run chunk-2048 --max-batched-tokens 2048
run replicas-40pct-slower --bias-mean 1.4
run replicas-25pct-faster --bias-mean 0.75
run bias-spread-30pct --bias-spread 0.3
run prefill-gemm-slower --hidden-gemm 0.73
run attention-slower --hidden-attention 0.6
run decode-bandwidth-lower --hidden-bandwidth 0.75
run prefill-and-attention-slower --hidden-gemm 0.73 --hidden-attention 0.6
run prefill-faster-decode-slower --hidden-gemm 1.3 --hidden-bandwidth 0.75
