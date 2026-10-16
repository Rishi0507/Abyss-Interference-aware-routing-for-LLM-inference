package policy

import (
	"math"
	"math/rand/v2"
	"sort"
)

func init() {
	register("round-robin", func(seed uint64) Policy { return &roundRobin{} })
	register("random", func(seed uint64) Policy { return &random{rng: newRNG(seed)} })
	register("least-request", func(seed uint64) Policy { return &leastRequest{rng: newRNG(seed)} })
	register("p2c", func(seed uint64) Policy { return &powerOfTwo{rng: newRNG(seed)} })
	register("chwbl", func(seed uint64) Policy { return newCHWBL(1.25) })
	register("sglang", func(seed uint64) Policy { return newCacheAware(seed) })
	register("llm-d", func(seed uint64) Policy { return newWeighted(seed, 3, 2, 2) })
	register("dynamo", func(seed uint64) Policy { return &dynamo{weight: 1} })
	registerParams("llm-d", func(seed uint64, a []float64) (Policy, error) {
		if err := want(a, 3); err != nil {
			return nil, err
		}
		return newWeighted(seed, a[0], a[1], a[2]), nil
	})
	registerParams("dynamo", func(seed uint64, a []float64) (Policy, error) {
		if err := want(a, 1); err != nil {
			return nil, err
		}
		return &dynamo{weight: a[0]}, nil
	})
	registerParams("sglang", func(seed uint64, a []float64) (Policy, error) {
		if err := want(a, 3); err != nil {
			return nil, err
		}
		p := newCacheAware(seed)
		p.cacheThresh, p.absThresh, p.relThresh = a[0], int(a[1]), a[2]
		return p, nil
	})
	registerParams("chwbl", func(seed uint64, a []float64) (Policy, error) {
		if err := want(a, 1); err != nil {
			return nil, err
		}
		return newCHWBL(a[0]), nil
	})
}

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, seed*0x9e3779b97f4a7c15+1)) }

func dispatch(i int) Decision { return Decision{Action: Dispatch, Backend: i} }

type roundRobin struct {
	hooks
	next int
}

func (p *roundRobin) Name() string { return "round-robin" }
func (p *roundRobin) Route(f *Fleet, r *Request) Decision {
	n := len(f.Backends)
	for k := 0; k < n; k++ {
		i := (p.next + k) % n
		if f.Backends[i].Healthy {
			p.next = i + 1
			return dispatch(i)
		}
	}
	return Decision{Action: Reject, Reason: "no healthy backend"}
}

type random struct {
	hooks
	rng *rand.Rand
}

func (p *random) Name() string { return "random" }
func (p *random) Route(f *Fleet, r *Request) Decision {
	h := f.Healthy()
	if len(h) == 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	return dispatch(h[p.rng.IntN(len(h))].Index)
}

type leastRequest struct {
	hooks
	rng *rand.Rand
}

func (p *leastRequest) Name() string { return "least-request" }
func (p *leastRequest) Route(f *Fleet, r *Request) Decision {
	best, bestN, ties := -1, math.MaxInt, 0
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		n := b.Outstanding()
		if n < bestN {
			best, bestN, ties = b.Index, n, 1
		} else if n == bestN {
			ties++
			if p.rng.IntN(ties) == 0 {
				best = b.Index
			}
		}
	}
	if best < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	return dispatch(best)
}

type powerOfTwo struct {
	hooks
	rng *rand.Rand
}

func (p *powerOfTwo) Name() string { return "p2c" }
func (p *powerOfTwo) Route(f *Fleet, r *Request) Decision {
	h := f.Healthy()
	if len(h) == 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	a := h[p.rng.IntN(len(h))]
	b := h[p.rng.IntN(len(h))]
	if outstandingTokens(b) < outstandingTokens(a) {
		a = b
	}
	return dispatch(a.Index)
}

func outstandingTokens(b *Backend) float64 {
	var t float64
	for _, r := range b.Inflight {
		if r.FirstAt < 0 {
			t += float64(r.Input)
		} else {
			t += 1
		}
	}
	return t
}

type chwbl struct {
	hooks
	named
	load  float64
	ring  []uint64
	owner []int
	built int
}

func newCHWBL(load float64) *chwbl { return &chwbl{load: load} }

func (p *chwbl) Name() string { return p.nameOr("chwbl") }

func (p *chwbl) build(n int) {
	type pt struct {
		h uint64
		o int
	}
	var pts []pt
	for i := 0; i < n; i++ {
		for v := 0; v < 128; v++ {
			pts = append(pts, pt{splitmix(uint64(i)<<32 | uint64(v)), i})
		}
	}
	sort.Slice(pts, func(a, b int) bool { return pts[a].h < pts[b].h })
	p.ring = p.ring[:0]
	p.owner = p.owner[:0]
	for _, x := range pts {
		p.ring = append(p.ring, x.h)
		p.owner = append(p.owner, x.o)
	}
	p.built = n
}

func (p *chwbl) Route(f *Fleet, r *Request) Decision {
	n := len(f.Backends)
	if p.built != n {
		p.build(n)
	}
	key := uint64(r.Session)
	if len(r.Hashes) > 0 {
		key = r.Hashes[0]
	}
	total := 1
	healthy := 0
	for _, b := range f.Backends {
		if b.Healthy {
			total += b.Outstanding()
			healthy++
		}
	}
	if healthy == 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	limit := math.Ceil(p.load * float64(total) / float64(healthy))
	h := splitmix(key)
	i := sort.Search(len(p.ring), func(i int) bool { return p.ring[i] >= h })
	for k := 0; k < len(p.ring); k++ {
		b := f.Backends[p.owner[(i+k)%len(p.ring)]]
		if b.Healthy && float64(b.Outstanding()+1) <= limit {
			return dispatch(b.Index)
		}
	}
	return dispatch(f.Backends[p.owner[i%len(p.ring)]].Index)
}

func splitmix(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

type approxTree struct {
	entries map[uint64]float64
	limit   int
}

func (t *approxTree) match(hashes []uint64) int {
	n := 0
	for n < len(hashes) {
		if _, ok := t.entries[hashes[n]]; !ok {
			break
		}
		n++
	}
	return n
}

func (t *approxTree) insert(hashes []uint64, now float64) {
	for _, h := range hashes {
		t.entries[h] = now
	}
	if t.limit > 0 && len(t.entries) > t.limit+t.limit/8 {
		type kv struct {
			h uint64
			t float64
		}
		all := make([]kv, 0, len(t.entries))
		for h, at := range t.entries {
			all = append(all, kv{h, at})
		}
		sort.Slice(all, func(a, b int) bool {
			if all[a].t != all[b].t {
				return all[a].t < all[b].t
			}
			return all[a].h < all[b].h
		})
		for _, x := range all[:len(all)-t.limit] {
			delete(t.entries, x.h)
		}
	}
}

type cacheAware struct {
	hooks
	named
	rng         *rand.Rand
	trees       []*approxTree
	cacheThresh float64
	absThresh   int
	relThresh   float64
}

func newCacheAware(seed uint64) *cacheAware {
	return &cacheAware{rng: newRNG(seed), cacheThresh: 0.3, absThresh: 64, relThresh: 1.5}
}

func (p *cacheAware) Name() string { return p.nameOr("sglang") }

func (p *cacheAware) ensure(f *Fleet) {
	for len(p.trees) < len(f.Backends) {
		b := f.Backends[len(p.trees)]
		limit := 0
		if b.Rep != nil && f.BlockTokens > 0 {
			limit = 2 * b.Rep.KVTokens / f.BlockTokens
		}
		p.trees = append(p.trees, &approxTree{entries: make(map[uint64]float64), limit: limit})
	}
}

func (p *cacheAware) Route(f *Fleet, r *Request) Decision {
	p.ensure(f)
	minL, maxL := math.MaxInt, -1
	minB := -1
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		l := b.Outstanding()
		if l < minL || (l == minL && p.rng.IntN(2) == 0) {
			minL, minB = l, b.Index
		}
		if l > maxL {
			maxL = l
		}
	}
	if minB < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	if maxL-minL > p.absThresh && float64(maxL) > p.relThresh*float64(minL) {
		return dispatch(minB)
	}
	best, bestRate := -1, -1.0
	smallest, smallestSize := -1, math.MaxInt
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		t := p.trees[b.Index]
		rate := 0.0
		if len(r.Hashes) > 0 {
			rate = float64(t.match(r.Hashes)) / float64(len(r.Hashes))
		}
		if rate > bestRate {
			best, bestRate = b.Index, rate
		}
		if len(t.entries) < smallestSize {
			smallest, smallestSize = b.Index, len(t.entries)
		}
	}
	if bestRate > p.cacheThresh {
		return dispatch(best)
	}
	return dispatch(smallest)
}

func (p *cacheAware) Dispatched(f *Fleet, r *Request) {
	p.ensure(f)
	p.trees[r.Backend].insert(r.Hashes, f.Now)
}

type weighted struct {
	hooks
	named
	rng                  *rand.Rand
	wPrefix, wQueue, wKV float64
	approx               *cacheAware
}

func newWeighted(seed uint64, wp, wq, wk float64) *weighted {
	return &weighted{rng: newRNG(seed), wPrefix: wp, wQueue: wq, wKV: wk, approx: newCacheAware(seed)}
}

func (p *weighted) Name() string { return p.nameOr("llm-d") }

func (p *weighted) Route(f *Fleet, r *Request) Decision {
	p.approx.ensure(f)
	minQ, maxQ := math.MaxInt, -1
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		q := b.Gauges.Waiting
		minQ = min(minQ, q)
		maxQ = max(maxQ, q)
	}
	best, bestScore, ties := -1, math.Inf(-1), 0
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		prefix := 0.0
		if len(r.Hashes) > 0 {
			var m int
			if b.Match != nil {
				m = b.Match(r.Hashes)
			} else {
				m = p.approx.trees[b.Index].match(r.Hashes)
			}
			prefix = float64(m) / float64(len(r.Hashes))
		}
		queue := 1.0
		if maxQ > minQ {
			queue = float64(maxQ-b.Gauges.Waiting) / float64(maxQ-minQ)
		}
		kv := 1 - b.Gauges.KVUsage
		s := p.wPrefix*prefix + p.wQueue*queue + p.wKV*kv
		if s > bestScore+1e-12 {
			best, bestScore, ties = b.Index, s, 1
		} else if math.Abs(s-bestScore) <= 1e-12 {
			ties++
			if p.rng.IntN(ties) == 0 {
				best = b.Index
			}
		}
	}
	if best < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	return dispatch(best)
}

func (p *weighted) Dispatched(f *Fleet, r *Request) { p.approx.Dispatched(f, r) }

type dynamo struct {
	hooks
	named
	weight float64
}

func (p *dynamo) Name() string { return p.nameOr("dynamo") }

func (p *dynamo) Route(f *Fleet, r *Request) Decision {
	bt := float64(max(f.BlockTokens, 1))
	reqBlocks := math.Ceil(float64(r.Input) / bt)
	best, bestCost := -1, math.Inf(1)
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		overlap := 0
		if b.Match != nil && len(r.Hashes) > 0 {
			overlap = b.Match(r.Hashes)
		}
		prefill := reqBlocks - float64(overlap)
		active := reqBlocks
		for _, x := range b.Inflight {
			active += math.Ceil(float64(x.Input+x.Generated) / bt)
		}
		c := p.weight*prefill + active
		if c < bestCost {
			best, bestCost = b.Index, c
		}
	}
	if best < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	return dispatch(best)
}
