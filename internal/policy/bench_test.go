package policy

import (
	"fmt"
	"testing"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

func loadedFleet(b *testing.B, p *Abyss, n, perReplica int) (*Fleet, *Request) {
	b.Helper()
	rep, err := perf.NewReplica(perf.Deployment{ModelKey: "llama-3.1-8b", HardwareKey: "h100", TP: 1})
	if err != nil {
		b.Fatal(err)
	}
	cfg := engine.DefaultConfig()
	cfg.BlockTokens = 64
	f := &Fleet{BlockTokens: 64}
	for i := 0; i < n; i++ {
		f.Backends = append(f.Backends, &Backend{Index: i, Rep: rep, EngineCfg: cfg, Healthy: true})
	}
	id := int64(0)
	for k := 0; k < perReplica; k++ {
		for i := 0; i < n; i++ {
			id++
			hashes := make([]uint64, 64)
			for j := range hashes {
				hashes[j] = uint64(id)<<20 | uint64(j)
			}
			r := request(id, float64(k)*0.01, 4096+64*(int(id)%32), hashes)
			r.TTFTSLO, r.TPOTSLO = 2, 0.04
			f.Now = float64(k) * 0.01
			dispatchTo(p, f, r, i)
			if k < perReplica*3/4 {
				r.FirstAt, r.LastAt, r.Generated = f.Now, f.Now, 1+k
			}
		}
	}
	f.Now = float64(perReplica)*0.01 + 0.5
	q := request(id+1, f.Now, 6000, make([]uint64, 90))
	q.TTFTSLO, q.TPOTSLO = 2, 0.04
	p.Route(f, q)
	return f, q
}

func BenchmarkAbyssDecision(b *testing.B) {
	for _, n := range []int{8, 16, 32, 64} {
		for _, mode := range []struct {
			name     string
			k        int
			parallel bool
		}{{"all", 0, false}, {"all-parallel", 0, true}, {"k8", 8, false}} {
			b.Run(fmt.Sprintf("replicas=%d/%s", n, mode.name), func(b *testing.B) {
				c := DefaultAbyss()
				c.Candidates = mode.k
				c.Parallel = mode.parallel
				p := NewAbyss("abyss", c)
				f, q := loadedFleet(b, p, n, 24)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					p.Route(f, q)
				}
			})
		}
	}
}
