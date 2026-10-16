# Running the comparison on real GPUs

The results in this repository come from the simulator. These scripts run the same comparison against real vLLM servers so that the simulated results can be checked. They have not yet been run on GPUs, so paths and flags may need adjustment for a given environment.

## What you need

- One machine with N NVIDIA GPUs (the scripts default to 4) and a recent vLLM installed (`pip install vllm`).
- Access to the model weights, for example `meta-llama/Llama-3.1-8B-Instruct` with a Hugging Face token in `HF_TOKEN`.
- Go 1.22 or newer to build `abyss`, and Python 3 for the summary.
- The Mooncake traces: `sh scripts/fetch-traces.sh`.

## Steps

| Step | Command | Effect |
|---|---|---|
| 1 | `make build` | Builds the `abyss` binary |
| 2 | `sh scripts/gpu/start-vllm.sh` | Starts one vLLM server per GPU on ports 8001 to 800N |
| 3 | `sh scripts/gpu/calibrate.sh` | Measures each server and writes `results/gpu/calibration-*.json` |
| 4 | `python scripts/gpu/make-config.py` | Writes `results/gpu/gateway.json` from the calibration |
| 5 | `sh scripts/gpu/run-policies.sh` | Replays the same trace window through the gateway once per policy |
| 6 | `python scripts/gpu/compare.py` | Prints the comparison table |
| 7 | `sh scripts/gpu/stop-vllm.sh` | Stops the servers |

`run-policies.sh` restarts the vLLM servers between policies so every policy starts with an empty prefix cache.

## Settings that matter

| Variable | Default | Meaning |
|---|---|---|
| `GPUS` | 4 | Number of servers, one per GPU |
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model served by vLLM |
| `MODEL_KEY`, `HARDWARE` | `llama-3.1-8b`, `h100` | Starting point for Abyss's performance model |
| `PRESET` | `conversation-70b` | Trace and objectives used by the load generator |
| `SCALE` | 0.5 | Arrival rate multiplier |
| `TOKEN_SCALE` | 0.25 | Prompt length multiplier, keeps prompts within the model's context |
| `DURATION` | 600 | Seconds of trace to replay |
| `POLICIES` | `round-robin llm-d dynamo sglang preble abyss-selfish abyss` | Policies to compare |

The latency objectives come from the preset and are applied by the load generator to the client-side measurements, so every policy is judged on the same requests and the same clock.
