package cluster

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/metrics"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/policy"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

type ReplicaSpec struct {
	Model      string           `json:"model"`
	Hardware   string           `json:"hardware"`
	TP         int              `json:"tp"`
	Count      int              `json:"count"`
	Efficiency *perf.Efficiency `json:"efficiency,omitempty"`
}

type Noise struct {
	BiasMean   float64 `json:"bias_mean,omitempty"`
	BiasSpread float64 `json:"bias_spread"`
	Jitter     float64 `json:"jitter"`
	GEMM       float64 `json:"gemm_factor,omitempty"`
	Attention  float64 `json:"attention_factor,omitempty"`
	Bandwidth  float64 `json:"bandwidth_factor,omitempty"`
}

type Config struct {
	Replicas []ReplicaSpec `json:"replicas"`
	Engine   engine.Config `json:"engine"`
	SLO      workload.SLO  `json:"slo"`
	Noise    Noise         `json:"noise"`
	Seed     uint64        `json:"seed"`
	Retry    float64       `json:"retry_interval_s"`
	Precise  bool          `json:"precise_prefix_index"`
	Drain    float64       `json:"drain_s"`
	Timeout  Timeout       `json:"client_timeout"`
}

type Timeout struct {
	Factor float64 `json:"ttft_slo_factor"`
	Floor  float64 `json:"floor_s"`
}

type Options struct {
	KeepRecords bool
	Timeline    bool
	Buckets     int
	Explain     func(t float64, r *policy.Request, d policy.Decision)
}

type Bucket struct {
	T          float64   `json:"t"`
	Arrivals   int       `json:"arrivals"`
	Attainment float64   `json:"attainment"`
	TTFTp50    float64   `json:"ttft_p50"`
	TTFTp99    float64   `json:"ttft_p99"`
	TPOTp50    float64   `json:"tpot_p50"`
	TPOTp99    float64   `json:"tpot_p99"`
	HitRate    float64   `json:"hit_rate"`
	Busy       []float64 `json:"busy"`
	Prefill    []float64 `json:"prefill_tokens"`
	Decode     []float64 `json:"decode_tokens"`
}

type Result struct {
	Summary  metrics.Summary  `json:"summary"`
	Timeline []Bucket         `json:"timeline,omitempty"`
	Records  []metrics.Record `json:"-"`
	Fleet    []string         `json:"fleet"`
	Wall     float64          `json:"wall_s"`
}

type replica struct {
	eng     *engine.Engine
	backend *policy.Backend
	busy    []float64
	prefill []float64
	decode  []float64
}

type sim struct {
	cfg      Config
	w        *workload.Workload
	pol      policy.Policy
	opt      Options
	fleet    *policy.Fleet
	reps     []*replica
	reqs     []policy.Request
	recs     []metrics.Record
	deferred []*policy.Request
	timeouts deadlineHeap
	retryAt  float64
	decide   []float64
	bucket   float64
	t0       float64
}

func Build(cfg Config, blockTokens int) ([]*perf.Replica, []engine.Config, error) {
	var reps []*perf.Replica
	var cfgs []engine.Config
	for _, s := range cfg.Replicas {
		n := max(s.Count, 1)
		for i := 0; i < n; i++ {
			d := perf.Deployment{ModelKey: s.Model, HardwareKey: s.Hardware, TP: s.TP}
			if s.Efficiency != nil {
				d.Eff = *s.Efficiency
			}
			r, err := perf.NewReplica(d)
			if err != nil {
				return nil, nil, err
			}
			ec := cfg.Engine
			if ec.MaxNumSeqs == 0 {
				ec = engine.DefaultConfig()
			}
			ec.BlockTokens = blockTokens
			reps = append(reps, r)
			cfgs = append(cfgs, ec)
		}
	}
	if len(reps) == 0 {
		return nil, nil, fmt.Errorf("fleet has no replicas")
	}
	return reps, cfgs, nil
}

func Run(cfg Config, w *workload.Workload, pol policy.Policy, opt Options) (*Result, error) {
	start := time.Now()
	if cfg.Retry <= 0 {
		cfg.Retry = 0.02
	}
	if cfg.Drain <= 0 {
		cfg.Drain = 900
	}
	if cfg.Timeout.Factor <= 0 {
		cfg.Timeout = Timeout{Factor: 4, Floor: 30}
	}
	perfs, ecfgs, err := Build(cfg, w.BlockTokens)
	if err != nil {
		return nil, err
	}
	s := &sim{cfg: cfg, w: w, pol: pol, opt: opt, fleet: &policy.Fleet{BlockTokens: w.BlockTokens}}
	if len(w.Items) == 0 {
		return nil, fmt.Errorf("empty workload")
	}
	s.t0 = w.Items[0].Arrival
	span := w.Items[len(w.Items)-1].Arrival - s.t0
	if opt.Buckets <= 0 {
		opt.Buckets = 60
		s.opt.Buckets = 60
	}
	s.bucket = math.Max(span/float64(opt.Buckets), 1)
	rng := rand.New(rand.NewPCG(cfg.Seed+11, cfg.Seed*31+7))
	var costPerHour float64
	var names []string
	for i, rp := range perfs {
		truth := rp
		if n := cfg.Noise; n.GEMM > 0 || n.Attention > 0 || n.Bandwidth > 0 {
			truth = rp.Scaled(n.GEMM, n.Attention, n.Bandwidth)
		}
		e := engine.New(ecfgs[i], truth)
		bias := 1.0
		if cfg.Noise.BiasMean > 0 {
			bias = cfg.Noise.BiasMean
		}
		if cfg.Noise.BiasSpread > 0 {
			bias += cfg.Noise.BiasSpread * (2*rng.Float64() - 1)
		}
		jitter := cfg.Noise.Jitter
		nr := rand.New(rand.NewPCG(cfg.Seed+uint64(i)*977, 3))
		e.TimeScale = func(d float64) float64 {
			if jitter > 0 {
				return d * bias * math.Exp(jitter*nr.NormFloat64()-jitter*jitter/2)
			}
			return d * bias
		}
		b := &policy.Backend{Index: i, Name: fmt.Sprintf("r%d", i), Rep: rp, EngineCfg: ecfgs[i], Healthy: true}
		if cfg.Precise {
			eng := e
			b.Match = func(h []uint64) int { return eng.MatchBlocks(h) }
		}
		rep := &replica{eng: e, backend: b, busy: make([]float64, opt.Buckets+2), prefill: make([]float64, opt.Buckets+2), decode: make([]float64, opt.Buckets+2)}
		e.Listener = &listener{s: s, rep: rep}
		s.reps = append(s.reps, rep)
		s.fleet.Backends = append(s.fleet.Backends, b)
		costPerHour += rp.PricePerHour()
		names = append(names, rp.Label)
	}
	s.reqs = make([]policy.Request, len(w.Items))
	s.recs = make([]metrics.Record, len(w.Items))
	s.retryAt = math.Inf(1)
	s.loop()
	dur := math.Max(span, 1)
	res := &Result{Fleet: names}
	res.Summary = metrics.Summarize(pol.Name(), s.recs, dur, costPerHour, s.decide, len(w.Classes))
	for _, r := range s.reps {
		st := r.eng.Stats
		rs := metrics.ReplicaStat{Name: r.eng.Rep.Label, Busy: st.BusyTime / dur, Preemptions: st.Preemptions, Evictions: r.eng.Evictions(), Prefill: st.PrefillTokens, Decode: st.DecodeTokens}
		if st.PromptTokens > 0 {
			rs.HitRate = float64(st.HitTokens) / float64(st.PromptTokens)
		}
		if st.Steps > 0 {
			rs.MeanBatch = float64(st.BatchTokensSum) / float64(st.Steps)
		}
		res.Summary.Preemptions += st.Preemptions
		res.Summary.PrefillTokens += st.PrefillTokens
		res.Summary.Replicas = append(res.Summary.Replicas, rs)
	}
	for i := range s.recs {
		if !s.recs[i].Rejected && s.recs[i].Replica >= 0 && s.recs[i].Replica < len(res.Summary.Replicas) {
			res.Summary.Replicas[s.recs[i].Replica].Requests++
		}
	}
	if opt.Timeline {
		res.Timeline = s.timeline()
	}
	if opt.KeepRecords {
		res.Records = s.recs
	}
	res.Wall = time.Since(start).Seconds()
	return res, nil
}

func (s *sim) loop() {
	items := s.w.Items
	next := 0
	end := items[len(items)-1].Arrival + s.cfg.Drain
	for {
		tA := math.Inf(1)
		if next < len(items) {
			tA = items[next].Arrival
		}
		tE, ei := math.Inf(1), -1
		for i, r := range s.reps {
			if t := r.eng.NextEvent(); t < tE {
				tE, ei = t, i
			}
		}
		tR := math.Inf(1)
		if len(s.deferred) > 0 {
			tR = s.retryAt
		}
		tT := math.Inf(1)
		if len(s.timeouts) > 0 {
			tT = s.timeouts[0].at
		}
		t := math.Min(math.Min(tA, tT), math.Min(tE, tR))
		if math.IsInf(t, 1) || t > end {
			break
		}
		s.fleet.Now = t
		switch {
		case tT == t:
			s.expire(t)
		case tE == t:
			r := s.reps[ei]
			before := r.eng.StepEnd()
			stepStart := before - lastCost(r.eng)
			pf, dc := r.eng.Stats.PrefillTokens, r.eng.Stats.DecodeTokens
			r.eng.Step()
			s.accountBusy(r, stepStart, before)
			if s.opt.Timeline {
				if k := int((stepStart - s.t0) / s.bucket); k >= 0 && k < len(r.prefill) {
					r.prefill[k] += float64(r.eng.Stats.PrefillTokens - pf)
					r.decode[k] += float64(r.eng.Stats.DecodeTokens - dc)
				}
			}
			s.observe(r, t)
		case tA == t:
			s.arrive(next, t)
			next++
		default:
			s.retry(t)
		}
	}
	for i := range s.recs {
		rec := &s.recs[i]
		if rec.FinishAt < 0 && !rec.Rejected && rec.Dispatch < 0 && !rec.TimedOut {
			rec.Rejected = true
		}
	}
}

type deadline struct {
	at float64
	id int64
}

type deadlineHeap []deadline

func (h deadlineHeap) Len() int           { return len(h) }
func (h deadlineHeap) Less(i, j int) bool { return h[i].at < h[j].at }
func (h deadlineHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *deadlineHeap) Push(x any)        { *h = append(*h, x.(deadline)) }
func (h *deadlineHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func (s *sim) expire(t float64) {
	d := heap.Pop(&s.timeouts).(deadline)
	pr := &s.reqs[d.id]
	rec := &s.recs[d.id]
	if pr.FirstAt >= 0 || pr.Done || rec.Rejected {
		return
	}
	rec.TimedOut = true
	pr.Done = true
	pr.DoneAt = t
	if pr.Dispatched < 0 {
		for i, x := range s.deferred {
			if x == pr {
				s.deferred = append(s.deferred[:i], s.deferred[i+1:]...)
				break
			}
		}
		return
	}
	r := s.reps[pr.Backend]
	if er, ok := pr.Ext.(*engine.Request); ok {
		r.eng.Remove(er)
	}
	r.backend.Remove(pr)
	s.pol.Finished(s.fleet, pr)
	s.kick()
}

func lastCost(e *engine.Engine) float64 { return e.StepCost() }

func (s *sim) accountBusy(r *replica, a, b float64) {
	if !s.opt.Timeline {
		return
	}
	for a < b {
		i := int((a - s.t0) / s.bucket)
		if i < 0 {
			i = 0
		}
		if i >= len(r.busy) {
			return
		}
		edge := s.t0 + float64(i+1)*s.bucket
		seg := math.Min(b, edge) - a
		r.busy[i] += seg
		a += seg
		if seg <= 0 {
			return
		}
	}
}

func (s *sim) observe(r *replica, t float64) {
	for _, er := range r.eng.Running() {
		pr := er.Ref.(*policy.Request)
		if er.Generated != pr.Generated {
			pr.Generated = er.Generated
			pr.LastAt = t
		}
	}
	b := r.backend
	b.Gauges = policy.Gauges{Waiting: len(r.eng.Waiting()), Running: len(r.eng.Running()), KVUsage: r.eng.KVUsage(), At: t}
}

func (s *sim) arrive(i int, t float64) {
	it := &s.w.Items[i]
	pr := &s.reqs[i]
	promptBlocks := it.Input / s.w.BlockTokens
	hashes := it.Hashes
	if promptBlocks < len(hashes) {
		hashes = hashes[:promptBlocks]
	}
	*pr = policy.Request{ID: it.ID, Session: it.Session, Class: it.Class, Arrival: t, Input: it.Input, Hashes: hashes,
		TTFTSLO: s.cfg.SLO.TTFT(it.Input), TPOTSLO: s.cfg.SLO.TPOT, Backend: -1, Dispatched: -1, FirstAt: -1, DoneAt: -1, Oracle: it.Output}
	rec := &s.recs[i]
	*rec = metrics.Record{ID: it.ID, Class: it.Class, Input: it.Input, Output: it.Output, Arrival: t, Dispatch: -1, FirstAt: -1, FinishAt: -1, Replica: -1, TTFTSLO: pr.TTFTSLO, TPOTSLO: pr.TPOTSLO, TPOTMin: s.cfg.SLO.TPOTMinTokens}
	fits := false
	for _, r := range s.reps {
		probe := engine.Request{Target: it.Input, OutputLen: it.Output}
		if r.eng.Fits(&probe) {
			fits = true
			break
		}
	}
	if !fits {
		rec.Rejected = true
		return
	}
	heap.Push(&s.timeouts, deadline{at: t + math.Max(s.cfg.Timeout.Floor, s.cfg.Timeout.Factor*pr.TTFTSLO), id: it.ID})
	s.route(pr)
}

func (s *sim) route(pr *policy.Request) bool {
	t0 := time.Now()
	d := s.pol.Route(s.fleet, pr)
	s.decide = append(s.decide, float64(time.Since(t0).Nanoseconds())/1e3)
	if s.opt.Explain != nil {
		s.opt.Explain(s.fleet.Now, pr, d)
	}
	switch d.Action {
	case policy.Dispatch:
		s.dispatch(pr, d.Backend)
		return true
	case policy.Reject:
		s.recs[pr.ID].Rejected = true
		return true
	default:
		if pr.Deferrals == 0 {
			s.deferred = append(s.deferred, pr)
			sort.SliceStable(s.deferred, func(i, j int) bool {
				return s.deferred[i].Arrival+s.deferred[i].TTFTSLO < s.deferred[j].Arrival+s.deferred[j].TTFTSLO
			})
		}
		pr.Deferrals++
		if s.retryAt > s.fleet.Now+s.cfg.Retry {
			s.retryAt = s.fleet.Now + s.cfg.Retry
		}
		return false
	}
}

func (s *sim) retry(t float64) {
	s.retryAt = math.Inf(1)
	keep := s.deferred[:0]
	tries := 0
	pending := append([]*policy.Request(nil), s.deferred...)
	for _, pr := range pending {
		if tries >= 6 {
			keep = append(keep, pr)
			continue
		}
		tries++
		t0 := time.Now()
		d := s.pol.Route(s.fleet, pr)
		s.decide = append(s.decide, float64(time.Since(t0).Nanoseconds())/1e3)
		switch d.Action {
		case policy.Dispatch:
			s.dispatch(pr, d.Backend)
		case policy.Reject:
			s.recs[pr.ID].Rejected = true
		default:
			pr.Deferrals++
			keep = append(keep, pr)
		}
	}
	s.deferred = keep
	if len(s.deferred) > 0 {
		s.retryAt = t + s.cfg.Retry
	}
}

func (s *sim) kick() {
	if len(s.deferred) > 0 && s.retryAt > s.fleet.Now+0.002 {
		s.retryAt = s.fleet.Now + 0.002
	}
}

func (s *sim) dispatch(pr *policy.Request, b int) {
	if b < 0 || b >= len(s.reps) {
		s.recs[pr.ID].Rejected = true
		return
	}
	now := s.fleet.Now
	r := s.reps[b]
	it := &s.w.Items[pr.ID]
	pr.Backend = b
	pr.Dispatched = now
	r.backend.Inflight = append(r.backend.Inflight, pr)
	rec := &s.recs[pr.ID]
	rec.Dispatch = now
	rec.Replica = b
	rec.PredTTFT = pr.PredTTFT
	er := engine.NewRequest(it.ID, pr.Arrival, it.Input, it.Output, it.Hashes)
	er.Ref = pr
	pr.Ext = er
	s.pol.Dispatched(s.fleet, pr)
	r.eng.Add(er, now)
}

type listener struct {
	s   *sim
	rep *replica
}

func (l *listener) OnFirstToken(er *engine.Request, t float64) {
	pr := er.Ref.(*policy.Request)
	pr.FirstAt = t
	pr.LastAt = t
	pr.Generated = er.Generated
	rec := &l.s.recs[pr.ID]
	rec.FirstAt = t
	rec.Hit = er.Hit
	l.s.pol.FirstToken(l.s.fleet, pr)
	l.s.kick()
}

func (l *listener) OnFinish(er *engine.Request, t float64) {
	pr := er.Ref.(*policy.Request)
	pr.Done = true
	pr.DoneAt = t
	pr.Generated = er.Generated
	pr.LastAt = t
	rec := &l.s.recs[pr.ID]
	rec.FinishAt = t
	rec.Preempted = er.Preempted
	l.rep.backend.Remove(pr)
	l.s.pol.Finished(l.s.fleet, pr)
	l.s.kick()
}

func (s *sim) timeline() []Bucket {
	n := s.opt.Buckets
	out := make([]Bucket, n)
	type acc struct {
		ttft, tpot []float64
		good, all  int
		hit, in    float64
	}
	accs := make([]acc, n)
	for i := range s.recs {
		r := &s.recs[i]
		k := int((r.Arrival - s.t0) / s.bucket)
		if k >= n {
			k = n - 1
		}
		a := &accs[k]
		a.all++
		if r.Good() {
			a.good++
		}
		if r.Completed() {
			a.ttft = append(a.ttft, r.TTFT())
			if r.Output > 1 {
				a.tpot = append(a.tpot, r.TPOT())
			}
			a.hit += float64(r.Hit)
			a.in += float64(r.Input)
		}
	}
	for k := range out {
		a := &accs[k]
		b := &out[k]
		b.T = float64(k) * s.bucket
		b.Arrivals = a.all
		if a.all > 0 {
			b.Attainment = float64(a.good) / float64(a.all)
		}
		sort.Float64s(a.ttft)
		sort.Float64s(a.tpot)
		b.TTFTp50 = metrics.Quantile(a.ttft, 0.5)
		b.TTFTp99 = metrics.Quantile(a.ttft, 0.99)
		b.TPOTp50 = metrics.Quantile(a.tpot, 0.5)
		b.TPOTp99 = metrics.Quantile(a.tpot, 0.99)
		if a.in > 0 {
			b.HitRate = a.hit / a.in
		}
		b.Busy = make([]float64, len(s.reps))
		b.Prefill = make([]float64, len(s.reps))
		b.Decode = make([]float64, len(s.reps))
		for i, r := range s.reps {
			if k < len(r.busy) {
				b.Prefill[i] = r.prefill[k]
				b.Decode[i] = r.decode[k]
				b.Busy[i] = r.busy[k] / s.bucket
			}
		}
	}
	return out
}
