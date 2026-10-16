package engine

import (
	"math"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

type State uint8

const (
	Queued State = iota
	Active
	Done
)

type Request struct {
	ID        int64
	Session   int64
	Arrival   float64
	InputLen  int
	OutputLen int
	MaxTokens int
	Hashes    []uint64

	State     State
	Target    int
	Computed  int
	KV        int
	Generated int
	Hit       int
	HitHint   int
	HostHint  int
	StartAt   float64
	FirstAt   float64
	FinishAt  float64
	Preempted int
	Dead      bool
	Ref       any

	shared    []*block
	nShared   int
	pages     int
	ownPages  int
	forkIdx   int32
	hintEpoch uint64
	bt        int
}

func (r *Request) blockTokens() int {
	if r.bt <= 0 {
		return 1 << 30
	}
	return r.bt
}

func NewRequest(id int64, arrival float64, input, output int, hashes []uint64) *Request {
	return &Request{ID: id, Arrival: arrival, InputLen: input, OutputLen: output, Hashes: hashes, Target: input, StartAt: -1, FirstAt: -1, FinishAt: -1}
}

func (r *Request) Prefilling() bool { return r.Computed < r.Target }
func (r *Request) Remaining() int   { return r.Target - r.Computed }

type Config struct {
	MaxNumSeqs       int     `json:"max_num_seqs"`
	MaxBatchedTokens int     `json:"max_batched_tokens"`
	PageTokens       int     `json:"page_tokens"`
	BlockTokens      int     `json:"block_tokens"`
	PrefixCaching    bool    `json:"prefix_caching"`
	MaxModelLen      int     `json:"max_model_len"`
	LongPrefill      int     `json:"long_prefill_token_threshold,omitempty"`
	HostGBPerGPU     float64 `json:"host_gb_per_gpu"`
	KVTokenOverride  int     `json:"kv_tokens,omitempty"`
	Jitter           float64 `json:"-"`
}

func DefaultConfig() Config {
	return Config{MaxNumSeqs: 256, MaxBatchedTokens: 8192, PageTokens: 16, BlockTokens: 512, PrefixCaching: true, MaxModelLen: 131072}
}

type Listener interface {
	OnFirstToken(r *Request, t float64)
	OnFinish(r *Request, t float64)
}

type Stats struct {
	Steps          int64
	BusyTime       float64
	PrefillBusy    float64
	PrefillTokens  int64
	DecodeTokens   int64
	HitTokens      int64
	HostHitTokens  int64
	PromptTokens   int64
	Preemptions    int64
	Admitted       int64
	BatchTokensSum int64
}

type slot struct {
	r *Request
	q int
}

type Engine struct {
	Cfg        Config
	Rep        *perf.Replica
	Listener   Listener
	TimeScale  func(base float64) float64
	SplitScale func(full, decode float64) (float64, float64)
	Stats      Stats

	kv        kvStore
	cache     *cacheKV
	waiting   []*Request
	running   []*Request
	batch     []slot
	busy      bool
	stepStart float64
	stepEnd   float64
	stepCost  float64
	now       float64
	blockPgs  int

	reqBuf []Request
}

func New(cfg Config, rep *perf.Replica) *Engine {
	if cfg.PageTokens <= 0 {
		cfg.PageTokens = 16
	}
	if cfg.BlockTokens <= 0 || cfg.BlockTokens%cfg.PageTokens != 0 {
		cfg.BlockTokens = cfg.PageTokens * ((cfg.BlockTokens + cfg.PageTokens - 1) / cfg.PageTokens)
		if cfg.BlockTokens == 0 {
			cfg.BlockTokens = cfg.PageTokens
		}
	}
	kvTokens := rep.KVTokens
	if cfg.KVTokenOverride > 0 {
		kvTokens = cfg.KVTokenOverride
	}
	e := &Engine{Cfg: cfg, Rep: rep, blockPgs: cfg.BlockTokens / cfg.PageTokens}
	e.cache = newCacheKV(kvTokens/cfg.PageTokens, e.blockPgs, cfg.PrefixCaching)
	if cfg.HostGBPerGPU > 0 && cfg.PrefixCaching {
		blocks := int(cfg.HostGBPerGPU * 1e9 / rep.KVBytesPerTokenGPU() / float64(cfg.BlockTokens))
		e.cache.host = newHostTier(blocks)
	}
	e.kv = e.cache
	return e
}

func (e *Engine) Now() float64           { return e.now }
func (e *Engine) Busy() bool             { return e.busy }
func (e *Engine) StepEnd() float64       { return e.stepEnd }
func (e *Engine) Waiting() []*Request    { return e.waiting }
func (e *Engine) Running() []*Request    { return e.running }
func (e *Engine) CapacityTokens() int    { return e.kv.totalPages() * e.Cfg.PageTokens }
func (e *Engine) UsedTokens() int        { return e.kv.usedPages() * e.Cfg.PageTokens }
func (e *Engine) CachedTokens() int      { return e.kv.cachedPages() * e.Cfg.PageTokens }
func (e *Engine) Evictions() int64       { return e.cache.evictions }
func (e *Engine) CacheBlocks() int       { return len(e.cache.blocks) }
func (e *Engine) Contains(h uint64) bool { return e.cache.contains(h) }

func (e *Engine) KVUsage() float64 {
	t := e.kv.totalPages()
	if t == 0 {
		return 0
	}
	return float64(e.kv.usedPages()-e.kv.cachedPages()) / float64(t)
}

func (e *Engine) SetCacheHooks(onStore, onEvict func(hash uint64)) {
	e.cache.onStore = onStore
	e.cache.onEvict = onEvict
}

func (e *Engine) EvictionOrder(fn func(hash uint64) bool) { e.cache.evictionOrder(fn) }

func (e *Engine) MatchBlocks(hashes []uint64) int {
	a, b := e.MatchTiers(hashes)
	return a + b
}

func (e *Engine) MatchTiers(hashes []uint64) (int, int) {
	if !e.Cfg.PrefixCaching {
		return 0, 0
	}
	n := 0
	for n < len(hashes) {
		if _, ok := e.cache.blocks[hashes[n]]; !ok {
			break
		}
		n++
	}
	m := 0
	if h := e.cache.host; h != nil {
		for n+m < len(hashes) && h.has(hashes[n+m]) {
			m++
		}
	}
	return n, m
}

func (e *Engine) HostEnabled() bool { return e.cache.host != nil }

func (e *Engine) HostContains(h uint64) bool { return e.cache.host != nil && e.cache.host.has(h) }

func (e *Engine) HostEvictionOrder(fn func(hash uint64) bool) {
	if e.cache.host != nil {
		e.cache.host.order(fn)
	}
}

func (e *Engine) HostFreeBlocks() int {
	if e.cache.host == nil {
		return 0
	}
	return e.cache.host.cap - e.cache.host.n
}

func (e *Engine) HostBlocks() int {
	if e.cache.host == nil {
		return 0
	}
	return e.cache.host.n
}

func (e *Engine) Fits(r *Request) bool {
	need := r.Target + r.OutputLen
	if r.OutputLen <= 0 || r.OutputLen > 1<<30 {
		need = r.Target + 1
	}
	return need <= e.Cfg.MaxModelLen && need <= e.CapacityTokens()
}

func (e *Engine) Add(r *Request, now float64) {
	e.now = math.Max(e.now, now)
	r.State = Queued
	e.waiting = append(e.waiting, r)
	if !e.busy {
		e.startStep()
	}
}

func (e *Engine) NextEvent() float64 {
	if e.busy {
		return e.stepEnd
	}
	return math.Inf(1)
}

func (e *Engine) AdvanceTo(t float64) {
	for e.busy && e.stepEnd <= t {
		e.Step()
	}
	if t > e.now {
		e.now = t
	}
	if !e.busy {
		e.startStep()
	}
}

func (e *Engine) Step() {
	if !e.busy {
		return
	}
	e.now = e.stepEnd
	e.busy = false
	e.completeStep()
	e.startStep()
}

func (e *Engine) pagesFor(r *Request, extra int) int {
	priv := r.KV + extra - r.nShared*e.Cfg.BlockTokens
	if priv < 0 {
		priv = 0
	}
	return (priv+e.Cfg.PageTokens-1)/e.Cfg.PageTokens - r.pages
}

func (e *Engine) startStep() {
	if len(e.running) == 0 && len(e.waiting) == 0 {
		return
	}
	e.batch = e.batch[:0]
	var b, bd perf.Batch
	budget := e.Cfg.MaxBatchedTokens
	preempted := false
	for i := 0; i < len(e.running) && budget > 0; i++ {
		r := e.running[i]
		q := 1
		decode := !r.Prefilling()
		if !decode {
			q = e.chunk(r.Remaining())
			if q > budget {
				q = budget
			}
		}
		need := e.pagesFor(r, q)
		ok := true
		for need > 0 && !e.kv.alloc(need) {
			victim := e.running[len(e.running)-1]
			e.preempt(victim)
			preempted = true
			if victim == r {
				ok = false
				break
			}
		}
		if !ok {
			break
		}
		if need > 0 {
			r.pages += need
		}
		e.batch = append(e.batch, slot{r, q})
		b.Add(q, r.KV, decode || q == r.Remaining())
		if decode {
			bd.Add(1, r.KV, true)
		}
		budget -= q
	}
	if !preempted {
		for len(e.waiting) > 0 && budget > 0 && len(e.running) < e.Cfg.MaxNumSeqs {
			r := e.waiting[0]
			fresh := r.Computed == 0
			hitBlocks, hostBlocks := 0, 0
			if fresh && e.Cfg.PrefixCaching {
				maxBlocks := (r.Target - 1) / e.Cfg.BlockTokens
				hitBlocks = e.kv.match(r, maxBlocks)
				hostBlocks = e.kv.matchHost(r, hitBlocks, maxBlocks)
			}
			if fresh {
				e.kv.acquire(r, hitBlocks)
			}
			if hostBlocks > 0 && !e.kv.alloc(hostBlocks*e.blockPgs) {
				e.kv.release(r)
				break
			}
			hit := (hitBlocks + hostBlocks) * e.Cfg.BlockTokens
			if fresh {
				e.kv.acquireHost(r, hitBlocks, hostBlocks)
				r.Computed = hit
				r.KV = hit
			}
			q := e.chunk(r.Remaining())
			if q > budget {
				q = budget
			}
			need := e.pagesFor(r, q)
			if !e.kv.alloc(need) {
				if fresh {
					e.kv.release(r)
					r.Computed, r.KV = 0, 0
				}
				break
			}
			b.LoadTokens += hostBlocks * e.Cfg.BlockTokens
			e.Stats.HostHitTokens += int64(hostBlocks * e.Cfg.BlockTokens)
			r.pages += need
			if r.Preempted == 0 || r.Hit == 0 {
				r.Hit = hit
			}
			e.Stats.HitTokens += int64(hit)
			e.Stats.PromptTokens += int64(r.Target)
			e.Stats.Admitted++
			e.waiting = e.waiting[1:]
			if r.StartAt < 0 {
				r.StartAt = e.now
			}
			r.State = Active
			e.running = append(e.running, r)
			e.batch = append(e.batch, slot{r, q})
			b.Add(q, r.KV, q == r.Remaining())
			budget -= q
		}
	}
	if len(e.batch) == 0 {
		return
	}
	dt := e.Rep.StepTime(b)
	pre := 0.0
	switch {
	case e.SplitScale != nil:
		dec := dt
		if bd.Tokens < b.Tokens {
			dec = math.Min(e.Rep.StepTime(bd), dt)
		}
		d, p := e.SplitScale(dt, dec)
		dt, pre = d+p, p
	case e.TimeScale != nil:
		dt = e.TimeScale(dt)
	}
	e.stepCost = dt
	e.stepStart = e.now
	e.stepEnd = e.now + dt
	e.busy = true
	e.Stats.Steps++
	e.Stats.BusyTime += dt
	e.Stats.PrefillBusy += pre
	e.Stats.BatchTokensSum += int64(b.Tokens)
}

func (e *Engine) preempt(r *Request) {
	for i := len(e.running) - 1; i >= 0; i-- {
		if e.running[i] == r {
			e.running = append(e.running[:i], e.running[i+1:]...)
			break
		}
	}
	for i := len(e.batch) - 1; i >= 0; i-- {
		if e.batch[i].r == r {
			e.batch = append(e.batch[:i], e.batch[i+1:]...)
		}
	}
	e.kv.release(r)
	r.Computed, r.KV = 0, 0
	r.Target = r.InputLen + r.Generated
	r.Preempted++
	r.State = Queued
	e.Stats.Preemptions++
	e.waiting = append(e.waiting, nil)
	copy(e.waiting[1:], e.waiting)
	e.waiting[0] = r
}

func (e *Engine) completeStep() {
	t := e.now
	finished := false
	for _, s := range e.batch {
		r := s.r
		if r.Dead || r.State != Active {
			continue
		}
		r.KV += s.q
		if r.Prefilling() {
			r.Computed += s.q
			e.Stats.PrefillTokens += int64(s.q)
			if r.Prefilling() {
				e.publishBlocks(r)
				continue
			}
		} else {
			e.Stats.DecodeTokens++
		}
		r.Generated++
		e.publishBlocks(r)
		if r.FirstAt < 0 {
			r.FirstAt = t
			if e.Listener != nil {
				e.Listener.OnFirstToken(r, t)
			}
		}
		if r.Generated >= r.OutputLen {
			r.State = Done
			r.FinishAt = t
			e.kv.release(r)
			finished = true
			if e.Listener != nil {
				e.Listener.OnFinish(r, t)
			}
		}
	}
	if finished {
		k := 0
		for _, r := range e.running {
			if r.State == Active && !r.Dead {
				e.running[k] = r
				k++
			}
		}
		for i := k; i < len(e.running); i++ {
			e.running[i] = nil
		}
		e.running = e.running[:k]
	}
}

func (e *Engine) publishBlocks(r *Request) {
	bt := e.Cfg.BlockTokens
	for r.nShared < len(r.Hashes) && r.KV >= (r.nShared+1)*bt {
		e.kv.publish(r, r.nShared)
	}
}

func (e *Engine) Remove(r *Request) {
	if r.State == Done {
		return
	}
	r.Dead = true
	r.State = Done
	e.kv.release(r)
	for i, x := range e.running {
		if x == r {
			e.running = append(e.running[:i], e.running[i+1:]...)
			break
		}
	}
	for i, x := range e.waiting {
		if x == r {
			e.waiting = append(e.waiting[:i], e.waiting[i+1:]...)
			break
		}
	}
}

func (e *Engine) ForceDecoding(r *Request, generated int) {
	if r.State == Done {
		return
	}
	if r.State == Queued {
		for i, x := range e.waiting {
			if x == r {
				e.waiting = append(e.waiting[:i], e.waiting[i+1:]...)
				break
			}
		}
		if r.Computed == 0 && e.Cfg.PrefixCaching {
			hitBlocks := e.kv.match(r, (r.Target-1)/e.Cfg.BlockTokens)
			e.kv.acquire(r, hitBlocks)
			r.Hit = hitBlocks * e.Cfg.BlockTokens
		}
		r.State = Active
		e.running = append(e.running, r)
	}
	if generated < 1 {
		generated = 1
	}
	r.Computed = r.Target
	r.Generated = generated
	r.KV = r.Target + generated - 1
	need := e.pagesFor(r, 0)
	if need > 0 && e.kv.alloc(need) {
		r.pages += need
	}
	e.publishBlocks(r)
	if r.FirstAt < 0 {
		r.FirstAt = e.now
	}
}

func (e *Engine) HoldPrefill(r *Request) {
	if r.State != Active || r.Prefilling() || r.Preempted > 0 {
		return
	}
	r.Computed = r.Target - 1
	r.KV = r.Computed
	r.Generated = 0
	r.FirstAt = -1
}

func (e *Engine) SetGenerated(r *Request, g int) {
	if r.State != Active || r.Prefilling() || g <= r.Generated {
		return
	}
	d := g - r.Generated
	r.Generated = g
	r.KV += d
	need := e.pagesFor(r, 0)
	if need > 0 && e.kv.alloc(need) {
		r.pages += need
	}
	e.publishBlocks(r)
}

type Fork struct {
	Engine
	base *Engine
}

func NewFork() *Fork { return &Fork{} }

func (f *Fork) From(src *Engine, remaining func(r *Request) int) *Engine {
	e := &f.Engine
	e.Cfg = src.Cfg
	e.Rep = src.Rep
	e.TimeScale = src.TimeScale
	e.SplitScale = src.SplitScale
	e.Listener = nil
	e.Stats = Stats{}
	e.blockPgs = src.blockPgs
	free := src.kv.freePages() + src.kv.cachedPages()
	pool := &poolKV{total: src.kv.totalPages(), used: src.kv.totalPages() - free, blockPages: src.blockPgs}
	e.kv = pool
	e.cache = src.cache
	n := len(src.running) + len(src.waiting) + 1
	if cap(e.reqBuf) < n {
		e.reqBuf = make([]Request, 0, 2*n)
	}
	e.reqBuf = e.reqBuf[:0]
	e.running = e.running[:0]
	e.waiting = e.waiting[:0]
	copyReq := func(r *Request) *Request {
		r.forkIdx = int32(len(e.reqBuf))
		e.reqBuf = append(e.reqBuf, *r)
		c := &e.reqBuf[len(e.reqBuf)-1]
		c.shared = nil
		c.Ref = r
		c.OutputLen = r.Generated + remaining(r)
		if c.OutputLen <= c.Generated {
			c.OutputLen = c.Generated + 1
		}
		if c.Computed == 0 && src.Cfg.PrefixCaching {
			r.bt = src.Cfg.BlockTokens
			c.HitHint, c.HostHint = src.cache.hints(r)
		}
		return c
	}
	for _, r := range src.running {
		e.running = append(e.running, copyReq(r))
	}
	for _, r := range src.waiting {
		e.waiting = append(e.waiting, copyReq(r))
	}
	e.batch = e.batch[:0]
	for _, s := range src.batch {
		if s.r.State == Active && !s.r.Dead {
			e.batch = append(e.batch, slot{&e.reqBuf[s.r.forkIdx], s.q})
		}
	}
	e.busy = src.busy
	e.stepStart, e.stepEnd, e.stepCost = src.stepStart, src.stepEnd, src.stepCost
	e.now = src.now
	f.base = src
	return e
}

func (f *Fork) Inject(r Request) *Request {
	e := &f.Engine
	if len(e.reqBuf) == cap(e.reqBuf) {
		panic("fork buffer exhausted")
	}
	r.shared = nil
	r.StartAt, r.FirstAt, r.FinishAt = -1, -1, -1
	r.bt = f.base.Cfg.BlockTokens
	r.hintEpoch = 0
	r.HitHint, r.HostHint = f.base.cache.hints(&r)
	e.reqBuf = append(e.reqBuf, r)
	c := &e.reqBuf[len(e.reqBuf)-1]
	c.State = Queued
	e.waiting = append(e.waiting, c)
	if !e.busy {
		e.startStep()
	}
	return c
}

func (f *Fork) Requests() []Request { return f.reqBuf }

func (e *Engine) StepCost() float64 { return e.stepCost }

func (e *Engine) chunk(n int) int {
	if t := e.Cfg.LongPrefill; t > 0 && n > t {
		return t
	}
	return n
}
