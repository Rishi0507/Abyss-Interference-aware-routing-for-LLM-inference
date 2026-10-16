# Operations

## Building

| Target | Effect |
|---|---|
| `make build` | Builds `./abyss` with the dashboard embedded from `internal/ui/dist`, which is committed |
| `make web` | Rebuilds the dashboard (requires Node 20 or newer) |
| `make test` | Runs the Go tests |

Without `make`, `go build -o abyss ./cmd/abyss` is sufficient. The binary can also be installed with `go install github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/cmd/abyss@latest`.

## Gateway

```sh
abyss gateway --config configs/gateway.json [--listen :8080] [--policy abyss]
```

### Configuration

| Field | Default | Meaning |
|---|---|---|
| `listen` | `:8080` | Address for the OpenAI-compatible API and the dashboard |
| `policy` | `abyss` | Routing policy at start; can be changed at runtime |
| `slo.ttft_base_s`, `slo.ttft_per_ktok_s` | 0.5, 0.05 | TTFT objective: base plus allowance per thousand prompt tokens |
| `slo.tpot_s` | 0.05 | Mean time per output token objective |
| `block_bytes` | 256 | Prompt bytes per prefix block |
| `chars_per_token` | 4 | Starting estimate of prompt bytes per token; corrected online from backend usage reports |
| `max_defer_s` | 5 | Upper bound on how long a deferring policy may hold a request |
| `health_path` | `/health` | Probed on every backend every two seconds |
| `engine` | vLLM defaults | Scheduler settings the twins assume: `max_num_seqs`, `max_batched_tokens`, `page_tokens`, `prefix_caching`, `host_gb_per_gpu` |
| `backends[].name`, `url` | | Identity and base URL of each server |
| `backends[].model`, `hardware`, `tp` | `llama-3.1-8b`, `h100`, 1 | Starting performance model for the twin; see `abyss catalog` |
| `backends[].scrape_metrics` | false | Read vLLM or SGLang Prometheus gauges every 250 ms |
| `backends[].efficiency` | defaults | Fitted performance parameters from `abyss calibrate` (the `efficiency` object of its output) |

The twins correct a wrong `hardware` or `tp` through online calibration, but a close starting point shortens the warm-up.

### Request headers

| Header | Effect |
|---|---|
| `X-Abyss-Weight` | Weight of this request's miss probability in every price, from 0 to 100, default 1 |

Responses carry `X-Abyss-Backend` with the name of the replica that served them.

### HTTP API

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions` | Routed and streamed to a backend |
| `GET /v1/models` | Forwarded to the first healthy backend |
| `GET /healthz` | Gateway liveness |
| `GET /metrics` | Prometheus metrics |
| `GET /api/live/state` | Fleet snapshot as JSON |
| `GET /api/live/stream` | Fleet snapshots as server-sent events, every 500 ms |
| `POST /api/live/policy` | Switch policy, body `{"policy": "llm-d"}` |
| `GET /api/results`, `POST /api/simulate` | Dashboard data and the simulation lab |

### Metrics

| Metric | Type |
|---|---|
| `abyss_requests_total`, `abyss_requests_completed_total`, `abyss_requests_good_total` | counter |
| `abyss_requests_rejected_total`, `abyss_requests_failed_total` | counter |
| `abyss_slo_attainment_ratio`, `abyss_ttft_p99_seconds`, `abyss_tpot_p99_seconds` | gauge, last 30 s |
| `abyss_backend_inflight{backend}`, `abyss_backend_healthy{backend}` | gauge |
| `abyss_twin_kv_usage_ratio{backend}`, `abyss_twin_speed_scale{backend}`, `abyss_twin_prefill_scale{backend}` | gauge |
| `abyss_chars_per_token` | gauge |

`abyss_twin_speed_scale` is the ratio of observed to modeled decode step time and `abyss_twin_prefill_scale` the same ratio for prefill, learned from time to first token. A value far from 1 on one backend points at a throttled GPU, a misconfigured server, or a wrong entry in the configuration; a large difference between the two scales points at a performance model that needs recalibration.

## Emulator

`abyss emulate` serves the OpenAI API with the engine model running in real time, emits vLLM-compatible metrics, and honors `max_tokens` exactly.

| Flag | Default | Meaning |
|---|---|---|
| `--listen` | `:9001` | Address |
| `--model`, `--hardware`, `--tp` | `llama-3.1-8b`, `h100`, 1 | Deployment being emulated |
| `--bias` | 1 | Multiplier on modeled step time, to emulate slower or faster hardware |
| `--jitter` | 0.05 | Lognormal step-time noise |
| `--speed` | 1 | Simulated seconds per wall-clock second |
| `--max-batched-tokens` | 8192 | Scheduler token budget |
| `--kv-tokens`, `--host-gb` | derived, 0 | KV capacity override and host offload size per GPU |

## Load generator

`abyss loadgen` replays any preset, trace or synthetic workload against an HTTP endpoint and reports client-side TTFT, TPOT and attainment.

```sh
abyss loadgen --target http://localhost:8080 --preset conversation-70b --scale 0.5 --token-scale 0.125 --duration 300
```

Prompt text is synthesized from the trace's block hashes so that requests sharing a prefix in the trace share it byte for byte on the wire. `--token-scale` shrinks prompts and `--output-scale` shrinks outputs, so a laptop can replay a production trace against emulated replicas.

## Simulator

| Command | Effect |
|---|---|
| `abyss sim --preset agent-8b --scale 0.75 -v` | One load level with per-replica details |
| `abyss bench --preset agent-8b --seeds 3 --parallel 2` | Full sweep written to `results/agent-8b.json` |
| `abyss trace --preset mixed-hetero` | Workload statistics and replica capacity |

Overrides such as `--model`, `--hardware`, `--tp`, `--replicas`, `--slo-ttft`, `--slo-ttft-per-k`, `--slo-tpot`, `--workload` and `--trace` apply to any preset. `--slo-tpot-min-tokens N` applies the TPOT objective only to responses of at least N tokens.

`--hidden-gemm`, `--hidden-attention` and `--hidden-bandwidth` multiply the real replicas' matrix-multiply, attention and memory-bandwidth efficiencies without telling any policy, which makes the router's performance model wrong by different amounts for prefill and decode. `--bias-mean` and `--bias-spread` change all step times uniformly.

## Calibration

```sh
abyss calibrate --target http://10.0.0.11:8000 --model llama-3.1-70b --hardware h100 --tp 4 --model-name meta-llama/Llama-3.1-70B-Instruct --out node-a.json
```

The command measures time to first token at six prompt lengths and the gap between tokens at four decode batch sizes, prints measured against fitted and default predictions, and writes the fitted parameters. Use the server's own `--max-batched-tokens` value. Run it on an otherwise idle server; concurrent traffic inflates the measurements. Copy the `efficiency` object into the backend's entry in the gateway configuration, or pass the file to the simulator with `--calibration`.

## Tuning baselines

```sh
abyss tune --preset conversation-70b --scales 0.75,1 --tune-seed 0 --eval-seeds 1,2
```

Grid searches the parameters of the llm-d, Dynamo, SGLang, bounded-load hashing and size-interval baselines on one trace window or seed, then evaluates every policy on held-out windows and writes `results/tuned/<preset>.json`. Parameterized policies can also be named directly, for example `--policies llm-d:8,2,0,dynamo:4,sita:32768,1`.

## Reproducing everything

`sh scripts/reproduce.sh` runs the main sweeps, ablations, tuning, robustness, objective sensitivity and scaling experiments and the decision-time benchmark, then regenerates the figures. `python scripts/report.py` rebuilds `docs/evaluation.md` and the README results from `results/`, and `python scripts/build_paper.py` rebuilds the paper draft in `paper/`. On the two-core laptop used to produce the published results, the full set takes about 75 minutes.

## Real GPUs

`scripts/gpu/` contains a procedure for running the comparison on vLLM servers; see its README.
