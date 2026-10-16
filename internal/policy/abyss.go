package policy

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
)

type AbyssConfig struct {
	Interference bool    `json:"interference"`
	Cache        bool    `json:"cache"`
	Selfish      bool    `json:"selfish"`
	LateBinding  bool    `json:"late_binding"`
	Shed         bool    `json:"shed"`
	Oracle       bool    `json:"oracle"`
	Epsilon      float64 `json:"epsilon"`
	Eta          float64 `json:"eta"`
	SigmaFloor   float64 `json:"sigma_floor"`
	Horizon      float64 `json:"horizon_slo"`
	TPOTWindow   int     `json:"tpot_window_steps"`
	DeferQueue   float64 `json:"defer_queue_slo"`
	MaxDefer     float64 `json:"max_defer_slo"`
	Explain      bool    `json:"explain"`
	Parallel     bool    `json:"parallel"`
	Candidates   int     `json:"candidates"`
}

func DefaultAbyss() AbyssConfig {
	return AbyssConfig{
		Interference: true,
		Cache:        true,
		LateBinding:  false,
		Epsilon:      0.05,
		Eta:          1.0,
		SigmaFloor:   0.12,
		Horizon:      3,
		TPOTWindow:   24,
		DeferQueue:   0.35,
		MaxDefer:     0.5,
	}
}

func init() {
	variant := func(name string, mod func(*AbyssConfig)) {
		register(name, func(seed uint64) Policy {
			c := DefaultAbyss()
			mod(&c)
			return NewAbyss(name, c)
		})
	}
	variant("abyss", func(c *AbyssConfig) {})
	variant("abyss-shed", func(c *AbyssConfig) { c.Shed = true })
	variant("abyss-selfish", func(c *AbyssConfig) { c.Selfish = true })
	variant("abyss-no-cache", func(c *AbyssConfig) { c.Cache = false })
	variant("abyss-no-interference", func(c *AbyssConfig) { c.Interference = false })
	variant("abyss-late-binding", func(c *AbyssConfig) { c.LateBinding = true })
	variant("abyss-oracle", func(c *AbyssConfig) { c.Oracle = true })
	variant("abyss-k8", func(c *AbyssConfig) { c.Candidates = 8 })
	variant("abyss-k4", func(c *AbyssConfig) { c.Candidates = 4 })
}

type Abyss struct {
	cfg     AbyssConfig
	name    string
	twins   []*twin
	lengths *lengthModel
	pop     *reuseModel
	refSLO  float64
}

func NewAbyss(name string, cfg AbyssConfig) *Abyss {
	return &Abyss{cfg: cfg, name: name, lengths: newLengthModel(256), pop: newReuseModel(3600)}
}

func (p *Abyss) Name() string        { return p.name }
func (p *Abyss) Config() AbyssConfig { return p.cfg }
func (p *Abyss) SetExplain(on bool)  { p.cfg.Explain = on }
func (p *Abyss) SetParallel(on bool) { p.cfg.Parallel = on }

func (p *Abyss) Twin(i int) *engine.Engine {
	if i < len(p.twins) {
		return p.twins[i].eng
	}
	return nil
}

func (p *Abyss) TwinScale(i int) float64 {
	if i < len(p.twins) {
		return p.twins[i].scale
	}
	return 1
}

func (p *Abyss) TwinPrefillScale(i int) float64 {
	if i < len(p.twins) {
		return p.twins[i].pscale
	}
	return 1
}

func (p *Abyss) ensure(f *Fleet) {
	for len(p.twins) < len(f.Backends) {
		b := f.Backends[len(p.twins)]
		p.twins = append(p.twins, newTwin(b, f.Now))
	}
}

func owner(r *engine.Request) *Request {
	switch x := r.Ref.(type) {
	case *Request:
		return x
	case *engine.Request:
		if pr, ok := x.Ref.(*Request); ok {
			return pr
		}
	}
	return nil
}

func (p *Abyss) remaining(r *engine.Request) int {
	pr := owner(r)
	if pr != nil && p.cfg.Oracle && pr.Oracle > 0 {
		return max(pr.Oracle-r.Generated, 1)
	}
	limit := 0
	if pr != nil {
		limit = pr.MaxTokens
	}
	return p.lengths.remaining(r.Generated, limit)
}

func (p *Abyss) expectedOutput(r *Request) int {
	if p.cfg.Oracle && r.Oracle > 0 {
		return r.Oracle
	}
	return p.lengths.remaining(0, r.MaxTokens)
}

func (p *Abyss) Predict(f *Fleet, r *Request) []Cost {
	p.ensure(f)
	now := f.Now
	if p.refSLO == 0 {
		p.refSLO = r.TTFTSLO
	}
	p.refSLO = 0.98*p.refSLO + 0.02*r.TTFTSLO
	var healthy []*Backend
	for _, b := range f.Backends {
		if b.Healthy {
			p.twins[b.Index].sync(now)
			healthy = append(healthy, b)
		}
	}
	cands := p.shortlist(healthy, r)
	costs := make([]Cost, len(cands))
	if p.cfg.Parallel && len(cands) > 1 {
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.GOMAXPROCS(0))
		for i, b := range cands {
			wg.Add(1)
			sem <- struct{}{}
			go func(i int, b *Backend) {
				defer wg.Done()
				costs[i] = p.price(f, b, p.twins[b.Index], r)
				<-sem
			}(i, b)
		}
		wg.Wait()
	} else {
		for i, b := range cands {
			costs[i] = p.price(f, b, p.twins[b.Index], r)
		}
	}
	return costs
}

func (p *Abyss) shortlist(bs []*Backend, r *Request) []*Backend {
	k := p.cfg.Candidates
	if k <= 0 || len(bs) <= k {
		return bs
	}
	type scored struct {
		b   *Backend
		est float64
		hit int
	}
	ss := make([]scored, len(bs))
	bestHit := 0
	for i, b := range bs {
		tw := p.twins[b.Index]
		backlog := 0
		for _, x := range tw.eng.Running() {
			backlog += x.Remaining()
		}
		for _, x := range tw.eng.Waiting() {
			backlog += x.Remaining()
		}
		hit := tw.eng.MatchBlocks(r.Hashes)
		own := r.Input - hit*tw.eng.Cfg.BlockTokens
		ss[i] = scored{b, float64(backlog+own)*tw.prefillPerToken*tw.pscale + 0.001*float64(len(tw.eng.Running())), hit}
		if hit > ss[bestHit].hit {
			bestHit = i
		}
	}
	keep := ss[bestHit].b
	sort.SliceStable(ss, func(i, j int) bool { return ss[i].est < ss[j].est })
	out := make([]*Backend, 0, k+1)
	for _, s := range ss[:k] {
		out = append(out, s.b)
	}
	for _, b := range out {
		if b == keep {
			return out
		}
	}
	return append(out, keep)
}

func (p *Abyss) Route(f *Fleet, r *Request) Decision {
	now := f.Now
	costs := p.Predict(f, r)
	best := -1
	for i, c := range costs {
		if best < 0 || c.Total < costs[best].Total {
			best = i
		}
	}
	if best < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	c := costs[best]
	d := Decision{Action: Dispatch, Backend: c.Backend}
	waited := now - r.Arrival
	if p.cfg.LateBinding && waited < p.cfg.MaxDefer*r.TTFTSLO {
		if c.Delay > p.cfg.DeferQueue*r.TTFTSLO && waited+c.PredTTFT > r.TTFTSLO {
			d.Action = Defer
			d.Reason = "backlog"
		}
	}
	if p.cfg.Shed && d.Action == Dispatch && waited+c.PredTTFT > 2*r.TTFTSLO {
		d.Action = Reject
		d.Reason = "infeasible"
	}
	if d.Action == Dispatch {
		r.PredTTFT = waited + c.PredTTFT
		r.PredTPOT = c.PredTPOT
	}
	if p.cfg.Explain {
		sort.Slice(costs, func(i, j int) bool { return costs[i].Backend < costs[j].Backend })
		d.Costs = costs
	}
	return d
}

func (p *Abyss) price(f *Fleet, b *Backend, tw *twin, r *Request) Cost {
	now := f.Now
	src := tw.eng
	outQ := p.expectedOutput(r)
	q := engine.Request{ID: r.ID, Arrival: now, InputLen: r.Input, OutputLen: outQ, Target: r.Input, Hashes: r.Hashes, StartAt: -1, FirstAt: -1, FinishAt: -1, Ref: r}
	cost := Cost{Backend: b.Index, HitBlocks: src.MatchBlocks(r.Hashes)}
	horizon := now + p.cfg.Horizon*r.TTFTSLO + 1

	with := tw.with.From(src, p.remaining)
	qc := tw.with.Inject(q)
	for qc.FirstAt < 0 && with.Busy() && with.StepEnd() <= horizon {
		with.Step()
	}
	tFirst := with.Now()
	ttft := qc.FirstAt - now
	if qc.FirstAt < 0 {
		ttft = tFirst - now + float64(qc.Remaining())*tw.prefillPerToken*tw.pscale
	}
	start := qc.StartAt
	if start < 0 {
		start = tFirst
	}
	if src.Busy() {
		cost.Delay = math.Max(0, start-src.StepEnd())
	} else {
		cost.Delay = math.Max(0, start-now)
	}
	cost.PredTTFT = ttft

	var extDelay float64
	if !p.cfg.Selfish && p.cfg.Interference {
		cost.Interference, extDelay = p.interference(src, tFirst, now, tw, r, outQ)
	}

	steps := 0
	t0 := with.Now()
	for steps < p.cfg.TPOTWindow && with.Busy() && qc.State == engine.Active {
		with.Step()
		steps++
	}
	running := len(src.Running()) + 1
	longRun := src.Rep.DecodeStepTime(running, tw.meanCtx(src, r.Input+outQ/2)) * tw.scale / (1 - math.Min(tw.prefillLoad, 0.85))
	tpot := longRun
	if steps > 0 {
		window := (with.Now() - t0) / float64(steps)
		w := math.Min(float64(steps)/float64(max(outQ-1, 1)), 1)
		tpot = w*window + (1-w)*math.Max(longRun, 0.5*window)
	}
	cost.PredTPOT = tpot

	sT, sP := r.TTFTSLO, r.TPOTSLO
	pT := violation(ttft, sT, tw.sigTTFT, p.cfg.SigmaFloor)
	pP := violation(tpot, sP, tw.sigTPOT, p.cfg.SigmaFloor)
	cost.Own = r.weight() * (1 - (1-pT)*(1-pP))
	delay := ttft/sT + 0.25*tpot/sP + extDelay

	if !p.cfg.Selfish && p.cfg.Cache {
		val, n := p.cacheExternality(src, tw, r, cost.HitBlocks, now, outQ)
		cost.Evicted = n
		cost.Cache = p.cfg.Eta * val / math.Max(p.refSLO, 1e-3)
	}
	cost.Total = cost.Own + cost.Interference + cost.Cache + p.cfg.Epsilon*delay
	return cost
}

func (p *Abyss) interference(src *engine.Engine, tFirst, now float64, tw *twin, r *Request, outQ int) (float64, float64) {
	base := tw.base.From(src, p.remaining)
	for base.Busy() && base.StepEnd() <= tFirst {
		base.Step()
	}
	bReqs := tw.base.Requests()
	wReqs := tw.with.Requests()
	n := min(len(bReqs), len(wReqs)-1)
	span := math.Max(tFirst-now, 1e-6)
	itl := span
	if base.Stats.Steps > 0 {
		itl = span / float64(base.Stats.Steps)
	}
	running := len(src.Running())
	ctx := tw.meanCtx(src, 0)
	extraStep := 0.0
	if running > 0 {
		with := src.Rep.DecodeStepTime(running+1, (ctx*float64(running)+float64(r.Input))/float64(running+1))
		without := src.Rep.DecodeStepTime(running, ctx)
		extraStep = math.Max(0, with-without) * tw.scale
	}
	var dp, dd float64
	for k := 0; k < n; k++ {
		a := &bReqs[k]
		w := &wReqs[k]
		pr := owner(a)
		if pr == nil {
			continue
		}
		if pr.FirstAt < 0 {
			var bt, wt float64
			switch {
			case a.FirstAt >= 0 && w.FirstAt >= 0:
				bt, wt = a.FirstAt, w.FirstAt
			case a.FirstAt >= 0:
				bt, wt = a.FirstAt, tFirst+itl
			default:
				continue
			}
			bt -= pr.Arrival
			wt -= pr.Arrival
			if wt <= bt {
				continue
			}
			dp += pr.weight() * (violation(wt, pr.TTFTSLO, tw.sigTTFT, p.cfg.SigmaFloor) - violation(bt, pr.TTFTSLO, tw.sigTTFT, p.cfg.SigmaFloor))
			dd += (wt - bt) / pr.TTFTSLO
			continue
		}
		lost := float64(a.Generated - w.Generated)
		remain := math.Max(float64(a.OutputLen-a.Generated), 1)
		delay := math.Max(0, lost)*itl + extraStep*math.Min(remain, float64(outQ))
		if delay <= 0 {
			continue
		}
		total := math.Max(float64(pr.Generated)-1+remain, 1)
		elapsed := math.Max(now-pr.FirstAt, 0)
		baseTPOT := (elapsed + remain*itl) / total
		withTPOT := baseTPOT + delay/total
		dp += pr.weight() * (violation(withTPOT, pr.TPOTSLO, tw.sigTPOT, p.cfg.SigmaFloor) - violation(baseTPOT, pr.TPOTSLO, tw.sigTPOT, p.cfg.SigmaFloor))
		dd += 0.25 * delay / total / pr.TPOTSLO
	}
	return math.Max(dp, 0), dd
}

func (p *Abyss) cacheExternality(src *engine.Engine, tw *twin, r *Request, hit int, now float64, outQ int) (float64, int) {
	bt := src.Cfg.BlockTokens
	pt := src.Cfg.PageTokens
	newTokens := (len(r.Hashes)-hit)*bt + (r.Input - len(r.Hashes)*bt) + outQ
	needPages := (newTokens + pt - 1) / pt
	free := (src.CapacityTokens() - src.UsedTokens()) / pt
	if needPages <= free {
		return 0, 0
	}
	blockPages := bt / pt
	k := (needPages - free + blockPages - 1) / blockPages
	residency := tw.residency()
	var val float64
	n := 0
	src.EvictionOrder(func(h uint64) bool {
		if n >= k {
			return false
		}
		val += p.pop.prob(h, now, residency) * tw.blockCost * tw.pscale
		n++
		return true
	})
	return val, n
}

func (p *Abyss) Dispatched(f *Fleet, r *Request) {
	p.ensure(f)
	p.pop.touch(r.Hashes, f.Now)
	tw := p.twins[r.Backend]
	tw.sync(f.Now)
	er := engine.NewRequest(r.ID, f.Now, r.Input, math.MaxInt32, r.Hashes)
	er.Ref = r
	tw.reqs[r.ID] = er
	tw.eng.Add(er, f.Now)
}

func (p *Abyss) FirstToken(f *Fleet, r *Request) {
	if r.Backend >= len(p.twins) || r.PredTTFT <= 0 {
		return
	}
	tw := p.twins[r.Backend]
	e := math.Log(math.Max(r.FirstAt-r.Arrival, 1e-4) / r.PredTTFT)
	tw.sigTTFT = math.Sqrt(0.95*tw.sigTTFT*tw.sigTTFT + 0.05*e*e)
}

func (p *Abyss) Finished(f *Fleet, r *Request) {
	p.lengths.observe(r.Generated)
	if r.Backend >= len(p.twins) {
		return
	}
	tw := p.twins[r.Backend]
	if r.PredTPOT > 0 && r.Generated > 1 {
		actual := (r.DoneAt - r.FirstAt) / float64(r.Generated-1)
		e := math.Log(math.Max(actual, 1e-5) / r.PredTPOT)
		tw.sigTPOT = math.Sqrt(0.95*tw.sigTPOT*tw.sigTPOT + 0.05*e*e)
	}
	if er, ok := tw.reqs[r.ID]; ok {
		tw.sync(f.Now)
		tw.eng.Remove(er)
		delete(tw.reqs, r.ID)
		delete(tw.seen, r.ID)
		delete(tw.simFirst, r.ID)
	}
}

func violation(pred, slo, sig, floor float64) float64 {
	if slo <= 0 || pred <= 0 {
		return 0
	}
	s := math.Sqrt(sig*sig + floor*floor)
	z := (math.Log(pred) - math.Log(slo)) / s
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}
