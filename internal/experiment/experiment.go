package experiment

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/cluster"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/policy"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

type WorkloadRef struct {
	Trace    string  `json:"trace,omitempty"`
	Kind     string  `json:"kind,omitempty"`
	Rate     float64 `json:"rate,omitempty"`
	Duration float64 `json:"duration,omitempty"`
	Start    float64 `json:"start,omitempty"`
	Seed     uint64  `json:"seed,omitempty"`
	Burst    float64 `json:"burstiness,omitempty"`
}

type Preset struct {
	Name        string         `json:"name"`
	Title       string         `json:"title"`
	Description string         `json:"description"`
	Workload    WorkloadRef    `json:"workload"`
	Cluster     cluster.Config `json:"cluster"`
	Scales      []float64      `json:"scales"`
}

var TraceDir = "traces"

func baseEngine() engine.Config {
	c := engine.DefaultConfig()
	c.MaxBatchedTokens = 8192
	return c
}

func offloadEngine(gb float64) engine.Config {
	c := baseEngine()
	c.HostGBPerGPU = gb
	return c
}

func Presets() []Preset {
	noise := cluster.Noise{BiasSpread: 0.12, Jitter: 0.06}
	return []Preset{
		{
			Name:        "conversation-70b",
			Title:       "Mooncake conversation trace, Llama 3.1 70B",
			Description: "Production multi-turn chat trace from Kimi (FAST 2025 release) replayed on four H100 TP4 replicas.",
			Workload:    WorkloadRef{Trace: "conversation_trace.jsonl", Duration: 1200},
			Cluster: cluster.Config{
				Replicas: []cluster.ReplicaSpec{{Model: "llama-3.1-70b", Hardware: "h100", TP: 4, Count: 4}},
				Engine:   offloadEngine(96),
				SLO:      workload.SLO{TTFTBase: 1.0, TTFTPerK: 0.15, TPOT: 0.08},
				Noise:    noise, Seed: 1, Precise: true,
			},
			Scales: []float64{0.25, 0.5, 0.75, 1, 1.25},
		},
		{
			Name:        "toolagent-70b",
			Title:       "Mooncake tool and agent trace, Llama 3.1 70B",
			Description: "Production tool-use and agent trace from Kimi (FAST 2025 release) replayed on four H100 TP4 replicas.",
			Workload:    WorkloadRef{Trace: "toolagent_trace.jsonl", Duration: 1200},
			Cluster: cluster.Config{
				Replicas: []cluster.ReplicaSpec{{Model: "llama-3.1-70b", Hardware: "h100", TP: 4, Count: 4}},
				Engine:   offloadEngine(96),
				SLO:      workload.SLO{TTFTBase: 1.0, TTFTPerK: 0.15, TPOT: 0.08},
				Noise:    noise, Seed: 1, Precise: true,
			},
			Scales: []float64{0.25, 0.5, 0.75, 1, 1.25},
		},
		{
			Name:        "agent-8b",
			Title:       "Coding agent sessions, Llama 3.1 8B",
			Description: "Synthetic coding agent sessions with long shared tool prompts and growing context on eight H100 replicas.",
			Workload:    WorkloadRef{Kind: "agent", Rate: 12, Duration: 900, Seed: 3},
			Cluster: cluster.Config{
				Replicas: []cluster.ReplicaSpec{{Model: "llama-3.1-8b", Hardware: "h100", TP: 1, Count: 8}},
				Engine:   baseEngine(),
				SLO:      workload.SLO{TTFTBase: 0.4, TTFTPerK: 0.03, TPOT: 0.03},
				Noise:    noise, Seed: 2, Precise: true,
			},
			Scales: []float64{0.25, 0.5, 0.75, 1},
		},
		{
			Name:        "mixed-hetero",
			Title:       "Mixed traffic on a heterogeneous fleet, Qwen3 32B",
			Description: "Chat, agent and RAG traffic on a fleet of H100, A100 and L40S replicas with different speeds and cache sizes.",
			Workload:    WorkloadRef{Kind: "mixed", Rate: 8, Duration: 900, Seed: 5},
			Cluster: cluster.Config{
				Replicas: []cluster.ReplicaSpec{
					{Model: "qwen3-32b", Hardware: "h100", TP: 2, Count: 2},
					{Model: "qwen3-32b", Hardware: "a100", TP: 2, Count: 2},
					{Model: "qwen3-32b", Hardware: "l40s", TP: 4, Count: 2},
				},
				Engine: baseEngine(),
				SLO:    workload.SLO{TTFTBase: 0.8, TTFTPerK: 0.08, TPOT: 0.06},
				Noise:  noise, Seed: 3, Precise: true,
			},
			Scales: []float64{0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75},
		},
	}
}

func Find(name string) (Preset, error) {
	for _, p := range Presets() {
		if p.Name == name {
			return p, nil
		}
	}
	var names []string
	for _, p := range Presets() {
		names = append(names, p.Name)
	}
	return Preset{}, fmt.Errorf("unknown preset %q (known: %s)", name, strings.Join(names, ", "))
}

var (
	cacheMu sync.Mutex
	cache   = map[string]*workload.Workload{}
)

func Load(ref WorkloadRef) (*workload.Workload, error) {
	key := fmt.Sprintf("%+v", ref)
	cacheMu.Lock()
	defer cacheMu.Unlock()
	if w, ok := cache[key]; ok {
		return w, nil
	}
	var w *workload.Workload
	var err error
	if ref.Trace != "" {
		path := ref.Trace
		if !filepath.IsAbs(path) {
			if _, e := os.Stat(path); e != nil {
				path = filepath.Join(TraceDir, ref.Trace)
			}
		}
		w, err = workload.LoadMooncake(path)
		if err != nil {
			return nil, fmt.Errorf("%w (run scripts/fetch-traces.sh to download the public traces)", err)
		}
		if ref.Start > 0 || ref.Duration > 0 {
			w = w.Window(ref.Start, ref.Duration)
		}
	} else {
		w, err = workload.Generate(workload.Spec{Kind: ref.Kind, Rate: ref.Rate, Duration: ref.Duration, Seed: ref.Seed, Burst: ref.Burst})
		if err != nil {
			return nil, err
		}
	}
	cache[key] = w
	return w, nil
}

type Run struct {
	Policy string          `json:"policy"`
	Scale  float64         `json:"scale"`
	Seed   int             `json:"seed"`
	Result *cluster.Result `json:"result"`
	Err    string          `json:"error,omitempty"`
}

type Point struct {
	Policy     string  `json:"policy"`
	Scale      float64 `json:"scale"`
	Rate       float64 `json:"rate"`
	N          int     `json:"n"`
	Attainment float64 `json:"attainment"`
	AttainSD   float64 `json:"attainment_sd"`
	TokenAtt   float64 `json:"token_weighted_attainment"`
	Goodput    float64 `json:"goodput"`
	TTFTP50    float64 `json:"ttft_p50"`
	TTFTP99    float64 `json:"ttft_p99"`
	TPOTP50    float64 `json:"tpot_p50"`
	TPOTP99    float64 `json:"tpot_p99"`
	HitRate    float64 `json:"hit_rate"`
	TimedOut   float64 `json:"timed_out"`
	CostPerM   float64 `json:"usd_per_m_good_tokens"`
	DecisionUS float64 `json:"decision_us"`
	PredError  float64 `json:"pred_error_p50"`
}

type Sweep struct {
	Preset   Preset           `json:"preset"`
	Workload workload.Summary `json:"workload"`
	Fleet    []string         `json:"fleet"`
	Policies []string         `json:"policies"`
	Seeds    int              `json:"seeds"`
	Points   []Point          `json:"points"`
	Runs     []Run            `json:"runs"`
}

type Spec struct {
	Policies []string
	Scales   []float64
	Seeds    int
	SeedList []int
	Timeline bool
	Parallel int
}

func variant(p Preset, k int) WorkloadRef {
	ref := p.Workload
	if k == 0 {
		return ref
	}
	if ref.Trace != "" {
		ref.Start += float64(k) * ref.Duration
	} else {
		ref.Seed += uint64(k) * 1009
	}
	return ref
}

func RunSweep(p Preset, spec Spec, progress func(done, total int, r Run)) (*Sweep, error) {
	if len(spec.Scales) == 0 {
		spec.Scales = p.Scales
	}
	if spec.Seeds <= 0 {
		spec.Seeds = 1
	}
	seeds := spec.SeedList
	if len(seeds) == 0 {
		for k := 0; k < spec.Seeds; k++ {
			seeds = append(seeds, k)
		}
	}
	loads := map[int]*workload.Workload{}
	for _, k := range seeds {
		w, err := Load(variant(p, k))
		if err != nil {
			return nil, err
		}
		if len(w.Items) == 0 {
			return nil, fmt.Errorf("workload window %d is empty", k)
		}
		loads[k] = w
	}
	sw := &Sweep{Preset: p, Workload: loads[seeds[0]].Summarize(), Policies: spec.Policies, Seeds: len(seeds)}
	type job struct {
		pol   string
		scale float64
		seed  int
	}
	var jobs []job
	for _, k := range seeds {
		for _, sc := range spec.Scales {
			for _, pol := range spec.Policies {
				jobs = append(jobs, job{pol, sc, k})
			}
		}
	}
	runs := make([]Run, len(jobs))
	var mu sync.Mutex
	done := 0
	var wg sync.WaitGroup
	par := spec.Parallel
	if par <= 0 {
		par = runtime.NumCPU()
	}
	sem := make(chan struct{}, max(par, 1))
	for i, j := range jobs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, j job) {
			defer wg.Done()
			defer func() { <-sem }()
			run := Run{Policy: j.pol, Scale: j.scale, Seed: j.seed}
			cfg := p.Cluster
			cfg.Seed += uint64(j.seed) * 7919
			pol, err := policy.New(j.pol, cfg.Seed)
			if err != nil {
				run.Err = err.Error()
			} else {
				res, err := cluster.Run(cfg, loads[j.seed].Scale(j.scale), pol, cluster.Options{Timeline: spec.Timeline})
				if err != nil {
					run.Err = err.Error()
				}
				run.Result = res
			}
			runs[i] = run
			mu.Lock()
			done++
			if progress != nil {
				progress(done, len(jobs), run)
			}
			mu.Unlock()
		}(i, j)
	}
	wg.Wait()
	sort.SliceStable(runs, func(a, b int) bool {
		if runs[a].Scale != runs[b].Scale {
			return runs[a].Scale < runs[b].Scale
		}
		if runs[a].Policy != runs[b].Policy {
			return indexOf(spec.Policies, runs[a].Policy) < indexOf(spec.Policies, runs[b].Policy)
		}
		return runs[a].Seed < runs[b].Seed
	})
	sw.Runs = runs
	for _, r := range runs {
		if r.Result != nil {
			sw.Fleet = r.Result.Fleet
			break
		}
	}
	sw.Points = Aggregate(runs, sw.Workload.Rate)
	return sw, nil
}

func Aggregate(runs []Run, baseRate float64) []Point {
	type key struct {
		pol   string
		scale float64
	}
	groups := map[key][]*cluster.Result{}
	var order []key
	for _, r := range runs {
		if r.Result == nil {
			continue
		}
		k := key{r.Policy, r.Scale}
		if _, ok := groups[k]; !ok {
			order = append(order, k)
		}
		groups[k] = append(groups[k], r.Result)
	}
	var out []Point
	for _, k := range order {
		g := groups[k]
		pt := Point{Policy: k.pol, Scale: k.scale, Rate: baseRate * k.scale, N: len(g)}
		var att []float64
		for _, res := range g {
			s := res.Summary
			att = append(att, s.Attainment)
			pt.Goodput += s.Goodput
			pt.TokenAtt += s.TokenAttain
			pt.TTFTP50 += s.TTFT.P50
			pt.TTFTP99 += s.TTFT.P99
			pt.TPOTP50 += s.TPOT.P50
			pt.TPOTP99 += s.TPOT.P99
			pt.HitRate += s.HitRate
			pt.TimedOut += float64(s.TimedOut) / math.Max(float64(s.Requests), 1)
			pt.CostPerM += s.CostPerMGood
			pt.DecisionUS += s.DecisionMicros.Mean
			pt.PredError += s.PredError.P50
		}
		n := float64(len(g))
		pt.Goodput /= n
		pt.TokenAtt /= n
		pt.TTFTP50 /= n
		pt.TTFTP99 /= n
		pt.TPOTP50 /= n
		pt.TPOTP99 /= n
		pt.HitRate /= n
		pt.TimedOut /= n
		pt.CostPerM /= n
		pt.DecisionUS /= n
		pt.PredError /= n
		var mean float64
		for _, a := range att {
			mean += a
		}
		mean /= n
		var v float64
		for _, a := range att {
			v += (a - mean) * (a - mean)
		}
		pt.Attainment = mean
		if n > 1 {
			pt.AttainSD = math.Sqrt(v / (n - 1))
		}
		out = append(out, pt)
	}
	return out
}

func indexOf(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return len(xs)
}
