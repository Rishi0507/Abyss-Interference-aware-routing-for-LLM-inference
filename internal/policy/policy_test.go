package policy

import (
	"math"
	"testing"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

func fleet(t *testing.T, n int) *Fleet {
	t.Helper()
	rep, err := perf.NewReplica(perf.Deployment{ModelKey: "llama-3.1-8b", HardwareKey: "h100", TP: 1})
	if err != nil {
		t.Fatal(err)
	}
	cfg := engine.DefaultConfig()
	cfg.BlockTokens = 64
	f := &Fleet{BlockTokens: 64}
	for i := 0; i < n; i++ {
		f.Backends = append(f.Backends, &Backend{Index: i, Rep: rep, EngineCfg: cfg, Healthy: true})
	}
	return f
}

func request(id int64, now float64, input int, hashes []uint64) *Request {
	return &Request{ID: id, Arrival: now, Input: input, Hashes: hashes, TTFTSLO: 1.0, TPOTSLO: 0.05, Backend: -1, FirstAt: -1, DoneAt: -1}
}

func dispatchTo(p Policy, f *Fleet, r *Request, b int) {
	r.Backend = b
	r.Dispatched = f.Now
	f.Backends[b].Inflight = append(f.Backends[b].Inflight, r)
	p.Dispatched(f, r)
}

func TestRegistryKnowsAllPolicies(t *testing.T) {
	for _, n := range []string{"round-robin", "least-request", "p2c", "chwbl", "sglang", "llm-d", "dynamo", "abyss", "abyss-selfish"} {
		if _, err := New(n, 1); err != nil {
			t.Errorf("%s: %v", n, err)
		}
	}
	if _, err := New("nope", 1); err == nil {
		t.Error("expected error for unknown policy")
	}
}

func TestEveryPolicySkipsUnhealthyBackends(t *testing.T) {
	for _, n := range Names() {
		p, _ := New(n, 3)
		f := fleet(t, 3)
		f.Backends[0].Healthy = false
		f.Backends[2].Healthy = false
		for i := 0; i < 5; i++ {
			d := p.Route(f, request(int64(i), 0, 512, nil))
			if d.Action == Dispatch && d.Backend != 1 {
				t.Fatalf("%s routed to unhealthy backend %d", n, d.Backend)
			}
		}
	}
}

func TestAbyssAvoidsReplicaWithLongPrefillInFlight(t *testing.T) {
	f := fleet(t, 2)
	p := NewAbyss("abyss", DefaultAbyss())
	big := request(1, 0, 60000, nil)
	big.TTFTSLO = 30
	dispatchTo(p, f, big, 0)
	f.Now = 0.05
	d := p.Route(f, request(2, 0.05, 2000, nil))
	if d.Backend != 1 {
		t.Fatalf("expected the idle replica, got %d", d.Backend)
	}
}

func TestAbyssPrefersCachedPrefixWhenIdle(t *testing.T) {
	f := fleet(t, 2)
	p := NewAbyss("abyss", DefaultAbyss())
	hashes := make([]uint64, 120)
	for i := range hashes {
		hashes[i] = uint64(i + 1)
	}
	warm := request(1, 0, 120*64, hashes)
	dispatchTo(p, f, warm, 1)
	warm.FirstAt, warm.Generated, warm.LastAt = 1.0, 1, 1.0
	p.FirstToken(f, warm)
	warm.Done, warm.DoneAt = true, 1.2
	f.Backends[1].Remove(warm)
	f.Now = 1.2
	p.Finished(f, warm)
	f.Now = 2
	d := p.Route(f, request(2, 2, 120*64+100, hashes))
	if d.Backend != 1 {
		t.Fatalf("expected the replica holding the prefix, got %d", d.Backend)
	}
}

func TestViolationProbabilityIsMonotone(t *testing.T) {
	prev := 0.0
	for _, x := range []float64{0.2, 0.5, 0.9, 1.0, 1.1, 2, 5} {
		v := violation(x, 1, 0.2, 0.1)
		if v < prev {
			t.Fatalf("violation not monotone at %v", x)
		}
		prev = v
	}
	if math.Abs(violation(1, 1, 0.2, 0.1)-0.5) > 1e-9 {
		t.Fatal("prediction equal to the objective should give probability one half")
	}
}

func TestLengthModelMeanResidualLife(t *testing.T) {
	l := newLengthModel(100)
	for i := 1; i <= 1000; i++ {
		l.observe(i)
	}
	if r := l.remaining(0, 0); math.Abs(float64(r)-500) > 25 {
		t.Fatalf("expected remaining near 500, got %d", r)
	}
	if r := l.remaining(900, 0); r < 25 || r > 60 {
		t.Fatalf("expected remaining near 50 after 900 tokens, got %d", r)
	}
	if r := l.remaining(0, 64); r != 64 {
		t.Fatalf("max_tokens should cap the estimate, got %d", r)
	}
}

func TestReuseModelDecaysForAbandonedBlocks(t *testing.T) {
	m := newReuseModel(3600)
	for k := 0; k < 6; k++ {
		m.touch([]uint64{7}, float64(k)*10)
	}
	fresh := m.prob(7, 55, 120)
	stale := m.prob(7, 400, 120)
	if !(fresh > 0.8 && stale < 0.05) {
		t.Fatalf("fresh %.3f stale %.3f", fresh, stale)
	}
}

func twoReplicaScenario(t *testing.T, p Policy, decodes int, decodeSLO float64) Decision {
	t.Helper()
	rep, err := perf.NewReplica(perf.Deployment{ModelKey: "llama-3.1-70b", HardwareKey: "h100", TP: 4})
	if err != nil {
		t.Fatal(err)
	}
	cfg := engine.DefaultConfig()
	cfg.BlockTokens = 512
	f := &Fleet{BlockTokens: 512}
	for i := 0; i < 2; i++ {
		f.Backends = append(f.Backends, &Backend{Index: i, Rep: rep, EngineCfg: cfg, Healthy: true})
	}
	prefix := make([]uint64, 40)
	for i := range prefix {
		prefix[i] = uint64(1000 + i)
	}
	warm := request(1, 0, 40*512, prefix)
	warm.TTFTSLO, warm.TPOTSLO = 10, 0.08
	dispatchTo(p, f, warm, 0)
	f.Now = 2
	warm.FirstAt, warm.LastAt, warm.Generated = 2, 2, 1
	p.FirstToken(f, warm)
	warm.Done, warm.DoneAt = true, 2.1
	f.Backends[0].Remove(warm)
	p.Finished(f, warm)
	for k := 0; k < decodes; k++ {
		f.Now = 2.2 + 0.001*float64(k)
		d := request(int64(10+k), f.Now, 1024, nil)
		d.TTFTSLO, d.TPOTSLO = 2, decodeSLO
		dispatchTo(p, f, d, 0)
	}
	f.Now = 4
	for _, d := range append([]*Request(nil), f.Backends[0].Inflight...) {
		d.FirstAt, d.LastAt, d.Generated = 3.5, 4, 25
		p.FirstToken(f, d)
	}
	hashes := append([]uint64(nil), prefix[:30]...)
	for i := 0; i < 30; i++ {
		hashes = append(hashes, uint64(5000+i))
	}
	q := request(99, 4, 60*512+200, hashes)
	q.TTFTSLO, q.TPOTSLO = 1+0.15*float64(q.Input)/1000, 0.08
	d := p.Route(f, q)
	for _, c := range d.Costs {
		t.Logf("%s, decode target %.0f ms, replica %d: cached blocks %d, ttft %.2fs, own %.3f, interference %.3f, total %.3f", p.Name(), 1000*decodeSLO, c.Backend, c.HitBlocks, c.PredTTFT, c.Own, c.Interference, c.Total)
	}
	return d
}

func explainAbyss(selfish bool) *Abyss {
	c := DefaultAbyss()
	c.Selfish = selfish
	c.Explain = true
	name := "abyss"
	if selfish {
		name = "abyss-selfish"
	}
	return NewAbyss(name, c)
}

func TestAbyssPricesTheStallOnRunningStreams(t *testing.T) {
	if d := twoReplicaScenario(t, explainAbyss(true), 48, 0.024); d.Backend != 0 {
		t.Fatalf("selfish routing should pick the cache-hot replica 0, picked %d", d.Backend)
	}
	d := twoReplicaScenario(t, explainAbyss(false), 48, 0.024)
	if d.Backend != 1 {
		t.Fatalf("Abyss should move the prompt off the replica with 48 streams near their target, picked %d", d.Backend)
	}
	if d.Costs[0].Interference <= d.Costs[1].Interference {
		t.Fatalf("interference on the busy replica (%.3f) should exceed the idle one (%.3f)", d.Costs[0].Interference, d.Costs[1].Interference)
	}
}

func TestSITASeparatesLongPrompts(t *testing.T) {
	p, err := New("sita", 1)
	if err != nil {
		t.Fatal(err)
	}
	f := fleet(t, 4)
	if d := p.Route(f, request(1, 0, 40000, nil)); d.Backend != 3 {
		t.Fatalf("long prompt went to %d, want the long pool (3)", d.Backend)
	}
	busy := request(2, 0, 9000, nil)
	dispatchTo(p, f, busy, 0)
	for i := int64(3); i < 6; i++ {
		d := p.Route(f, request(i, 0, 2000, nil))
		if d.Backend == 3 || d.Backend == 0 {
			t.Fatalf("short prompt went to %d", d.Backend)
		}
	}
	q, _ := New("sita:4096,2", 1)
	if d := q.Route(fleet(t, 4), request(7, 0, 6000, nil)); d.Backend < 2 {
		t.Fatalf("with two long replicas a 6k prompt went to %d", d.Backend)
	}
}
