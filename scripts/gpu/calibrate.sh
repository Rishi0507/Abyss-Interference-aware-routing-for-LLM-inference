#!/usr/bin/env sh
set -e
GPUS=${GPUS:-4}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
MODEL_KEY=${MODEL_KEY:-llama-3.1-8b}
HARDWARE=${HARDWARE:-h100}
mkdir -p results/gpu
i=0
while [ "$i" -lt "$GPUS" ]; do
  port=$((8001 + i))
  ./abyss calibrate --target "http://localhost:$port" --model "$MODEL_KEY" --hardware "$HARDWARE" \
    --model-name "$MODEL" --out "results/gpu/calibration-$port.json"
  i=$((i + 1))
done
