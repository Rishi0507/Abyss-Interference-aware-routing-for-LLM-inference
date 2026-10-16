package cluster

import (
	"testing"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/policy"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

func smallConfig() Config {
	ec := engine.DefaultConfig()
	return Config{
		Replicas: []ReplicaSpec{{Model: "llama-3.1-8b", Hardware: "h100", TP: 1, Count: 3}},
		Engine:   ec,
		SLO:      workload.SLO{TTFTBase: 0.5, TTFTPerK: 0.05, TPOT: 0.05},
		Noise:    Noise{BiasSpread: 0.1, Jitter: 0.05},
		Seed:     3,
		Precise:  true,
	}
}

func TestEveryPolicyCompletesTheWorkload(t *testing.T) {
	w, err := workload.Generate(workload.Spec{Kind: "mixed", Rate: 3, Duration: 60, Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range policy.Names() {
		p, _ := policy.New(name, 1)
		res, err := Run(smallConfig(), w, p, Options{Timeline: true})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		s := res.Summary
		if s.Requests != len(w.Items) {
			t.Fatalf("%s: %d records for %d items", name, s.Requests, len(w.Items))
		}
		if s.Completed+s.Rejected+s.TimedOut != s.Requests {
			t.Fatalf("%s: %d completed, %d rejected, %d timed out of %d", name, s.Completed, s.Rejected, s.TimedOut, s.Requests)
		}
		if s.Attainment <= 0 || s.Attainment > 1 {
			t.Fatalf("%s: attainment %.3f", name, s.Attainment)
		}
	}
}

func TestSimulationIsDeterministic(t *testing.T) {
	w, _ := workload.Generate(workload.Spec{Kind: "agent", Rate: 4, Duration: 60, Seed: 5})
	run := func() float64 {
		p, _ := policy.New("abyss", 1)
		res, err := Run(smallConfig(), w, p, Options{})
		if err != nil {
			t.Fatal(err)
		}
		return res.Summary.TTFT.Mean + res.Summary.TPOT.Mean + res.Summary.HitRate
	}
	if a, b := run(), run(); a != b {
		t.Fatalf("runs differ: %v vs %v", a, b)
	}
}

func TestParallelPricingMatchesSerial(t *testing.T) {
	w, _ := workload.Generate(workload.Spec{Kind: "mixed", Rate: 4, Duration: 60, Seed: 8})
	run := func(parallel bool) float64 {
		p := policy.NewAbyss("abyss", policy.DefaultAbyss())
		p.SetParallel(parallel)
		res, err := Run(smallConfig(), w, p, Options{})
		if err != nil {
			t.Fatal(err)
		}
		return res.Summary.TTFT.Mean + res.Summary.TPOT.Mean + res.Summary.Attainment
	}
	if a, b := run(false), run(true); a != b {
		t.Fatalf("parallel pricing changed results: %v vs %v", a, b)
	}
}
