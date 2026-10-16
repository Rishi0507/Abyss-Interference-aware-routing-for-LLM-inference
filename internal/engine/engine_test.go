package engine

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

func replica(t *testing.T, model, hw string, tp int) *perf.Replica {
	t.Helper()
	r, err := perf.NewReplica(perf.Deployment{ModelKey: model, HardwareKey: hw, TP: tp})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

type sink struct{ first, done int }

func (s *sink) OnFirstToken(*Request, float64) { s.first++ }
func (s *sink) OnFinish(*Request, float64)     { s.done++ }

func chain(prefix, n int, salt uint64) []uint64 {
	out := make([]uint64, n)
	for i := range out {
		if i < prefix {
			out[i] = uint64(i + 1)
		} else {
			out[i] = salt<<32 | uint64(i+1)
		}
	}
	return out
}

func drain(e *Engine) {
	for e.Busy() {
		e.Step()
	}
}

func TestRequestsCompleteAndMemoryReturns(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BlockTokens = 64
	cfg.KVTokenOverride = 64 * 400
	cfg.HostGBPerGPU = 1
	e := New(cfg, replica(t, "llama-3.1-8b", "h100", 1))
	s := &sink{}
	e.Listener = s
	rng := rand.New(rand.NewPCG(1, 2))
	var reqs []*Request
	for i := 0; i < 200; i++ {
		in := 200 + rng.IntN(6000)
		r := NewRequest(int64(i), 0, in, 1+rng.IntN(300), chain(rng.IntN(20), in/64, uint64(i)))
		reqs = append(reqs, r)
		e.Add(r, float64(i)*0.01)
		if i%7 == 0 {
			e.AdvanceTo(float64(i) * 0.01)
		}
	}
	drain(e)
	if s.first != 200 || s.done != 200 {
		t.Fatalf("first=%d done=%d", s.first, s.done)
	}
	for _, r := range reqs {
		if r.Generated != r.OutputLen || r.FirstAt < 0 || r.FinishAt < r.FirstAt {
			t.Fatalf("request %d: generated %d of %d", r.ID, r.Generated, r.OutputLen)
		}
	}
	if e.UsedTokens() != e.CachedTokens() {
		t.Fatalf("leak: used %d cached %d", e.UsedTokens(), e.CachedTokens())
	}
	if e.Stats.Preemptions == 0 {
		t.Log("no preemptions exercised")
	}
}

func TestPrefixHitSkipsPrefill(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BlockTokens = 64
	e := New(cfg, replica(t, "llama-3.1-8b", "h100", 1))
	h := chain(64, 64, 9)
	a := NewRequest(1, 0, 64*64, 4, h)
	e.Add(a, 0)
	drain(e)
	b := NewRequest(2, e.Now(), 64*64, 4, h)
	e.Add(b, e.Now())
	drain(e)
	if b.Hit != 63*64 {
		t.Fatalf("hit %d, want %d", b.Hit, 63*64)
	}
	if b.FirstAt-b.Arrival >= a.FirstAt-a.Arrival {
		t.Fatalf("cached request was not faster: %.4f vs %.4f", b.FirstAt-b.Arrival, a.FirstAt-a.Arrival)
	}
}

func TestHostTierServesEvictedBlocks(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BlockTokens = 64
	cfg.KVTokenOverride = 64 * 80
	cfg.HostGBPerGPU = 4
	e := New(cfg, replica(t, "llama-3.1-8b", "h100", 1))
	hot := chain(60, 60, 1)
	e.Add(NewRequest(1, 0, 60*64, 2, hot), 0)
	drain(e)
	for i := 0; i < 4; i++ {
		e.Add(NewRequest(int64(10+i), e.Now(), 60*64, 2, chain(0, 60, uint64(100+i))), e.Now())
		drain(e)
	}
	r := NewRequest(99, e.Now(), 60*64, 2, hot)
	e.Add(r, e.Now())
	drain(e)
	if e.Stats.HostHitTokens == 0 || r.Hit == 0 {
		t.Fatalf("expected host tier hit, got hit=%d host=%d", r.Hit, e.Stats.HostHitTokens)
	}
}

func TestForkMatchesEngineWithoutCache(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PrefixCaching = false
	e := New(cfg, replica(t, "llama-3.1-70b", "h100", 4))
	for i := 0; i < 20; i++ {
		e.Add(NewRequest(int64(i), 0, 2000+100*i, 50, nil), 0)
	}
	e.AdvanceTo(0.5)
	f := NewFork()
	g := f.From(e, func(r *Request) int { return r.OutputLen - r.Generated })
	for g.Busy() {
		g.Step()
	}
	drain(e)
	if math.Abs(g.Now()-e.Now()) > 1e-9 {
		t.Fatalf("fork finished at %.6f, engine at %.6f", g.Now(), e.Now())
	}
}

func TestPreemptionUnderPressure(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BlockTokens = 64
	cfg.KVTokenOverride = 16 * 400
	e := New(cfg, replica(t, "llama-3.1-8b", "h100", 1))
	s := &sink{}
	e.Listener = s
	for i := 0; i < 12; i++ {
		e.Add(NewRequest(int64(i), 0, 300, 400, chain(0, 4, uint64(i))), 0)
	}
	drain(e)
	if s.done != 12 {
		t.Fatalf("done %d", s.done)
	}
	if e.Stats.Preemptions == 0 {
		t.Fatal("expected preemptions under memory pressure")
	}
}
