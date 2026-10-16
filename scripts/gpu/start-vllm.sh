#!/usr/bin/env sh
set -e
GPUS=${GPUS:-4}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
MAX_LEN=${MAX_LEN:-65536}
mkdir -p results/gpu/logs
i=0
while [ "$i" -lt "$GPUS" ]; do
  port=$((8001 + i))
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" \
    --port "$port" \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --max-model-len "$MAX_LEN" \
    --enable-prefix-caching \
    > "results/gpu/logs/vllm-$port.log" 2>&1 &
  echo $! > "results/gpu/logs/vllm-$port.pid"
  i=$((i + 1))
done
i=0
while [ "$i" -lt "$GPUS" ]; do
  port=$((8001 + i))
  until curl -sf "http://localhost:$port/health" > /dev/null; do sleep 5; done
  echo "server on port $port is ready"
  i=$((i + 1))
done
