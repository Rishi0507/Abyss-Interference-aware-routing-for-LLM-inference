package policy

import (
	"math"
	"sort"
)

func init() {
	register("sita", func(seed uint64) Policy { return &sita{cutoff: 16384, long: -1} })
	register("least-work", func(seed uint64) Policy { return &sita{label: "least-work"} })
	registerParams("sita", func(seed uint64, a []float64) (Policy, error) {
		if err := want(a, 2); err != nil {
			return nil, err
		}
		return &sita{cutoff: int(a[0]), long: int(a[1])}, nil
	})
}

type sita struct {
	hooks
	named
	label  string
	cutoff int
	long   int
	pool   []bool
	ppt    []float64
}

func (p *sita) Name() string {
	if p.label != "" {
		return p.label
	}
	return p.nameOr("sita")
}

func (p *sita) ensure(f *Fleet) {
	if len(p.ppt) == len(f.Backends) {
		return
	}
	n := len(f.Backends)
	p.ppt = make([]float64, n)
	p.pool = make([]bool, n)
	for i, b := range f.Backends {
		p.ppt[i] = 1
		if b.Rep != nil {
			chunk := max(b.EngineCfg.MaxBatchedTokens, 1)
			p.ppt[i] = b.Rep.PrefillTime(8192, 4096, chunk) / 8192
		}
	}
	k := p.long
	if k < 0 {
		k = max(1, int(math.Round(float64(n)/4)))
	}
	if p.cutoff <= 0 || k <= 0 || k >= n {
		return
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		if p.ppt[idx[a]] != p.ppt[idx[b]] {
			return p.ppt[idx[a]] > p.ppt[idx[b]]
		}
		return idx[a] > idx[b]
	})
	for _, i := range idx[:k] {
		p.pool[i] = true
	}
}

func (p *sita) Route(f *Fleet, r *Request) Decision {
	p.ensure(f)
	long := p.cutoff > 0 && r.Input > p.cutoff
	bt := max(f.BlockTokens, 1)
	best, bestW := -1, math.Inf(1)
	for pass := 0; pass < 2 && best < 0; pass++ {
		for i, b := range f.Backends {
			if !b.Healthy || pass == 0 && p.cutoff > 0 && p.pool[i] != long {
				continue
			}
			tokens := 0.0
			for _, x := range b.Inflight {
				if x.FirstAt < 0 {
					tokens += float64(x.Input)
				} else {
					tokens += 64
				}
			}
			hit := 0
			if b.Match != nil && len(r.Hashes) > 0 {
				hit = b.Match(r.Hashes)
			}
			tokens += float64(max(r.Input-hit*bt, 1))
			if w := tokens * p.ppt[i]; w < bestW {
				best, bestW = i, w
			}
		}
	}
	if best < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	return dispatch(best)
}
