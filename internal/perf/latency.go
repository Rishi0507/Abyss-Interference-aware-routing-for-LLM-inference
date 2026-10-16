package perf

import (
	"fmt"
	"math"
)

type Efficiency struct {
	GEMM         float64 `json:"gemm"`
	GEMMHalf     float64 `json:"gemm_half_tokens"`
	Attention    float64 `json:"attention"`
	Bandwidth    float64 `json:"bandwidth"`
	StepOverhead float64 `json:"step_overhead_s"`
	SeqOverhead  float64 `json:"seq_overhead_s"`
	Reserve      float64 `json:"reserve_bytes"`
}

func DefaultEfficiency() Efficiency {
	return Efficiency{
		GEMM:         0.62,
		GEMMHalf:     96,
		Attention:    0.42,
		Bandwidth:    0.80,
		StepOverhead: 0.0018,
		SeqOverhead:  12e-6,
		Reserve:      3 * giga,
	}
}

type Deployment struct {
	ModelKey    string     `json:"model"`
	HardwareKey string     `json:"hardware"`
	TP          int        `json:"tp"`
	MemUtil     float64    `json:"mem_util"`
	Eff         Efficiency `json:"efficiency"`
}

type Replica struct {
	Model    Model
	HW       Hardware
	TP       int
	Eff      Efficiency
	MemUtil  float64
	Label    string
	KVTokens int

	weightBytesGPU  float64
	denseBytesGPU   float64
	expertBytesGPU  float64
	kvBytesTokenGPU float64
	flopsPerToken   float64
	headFlops       float64
	attnCoef        float64
}

func NewReplica(d Deployment) (*Replica, error) {
	m, err := LookupModel(d.ModelKey)
	if err != nil {
		return nil, err
	}
	hw, err := LookupHardware(d.HardwareKey)
	if err != nil {
		return nil, err
	}
	if d.TP <= 0 {
		d.TP = 1
	}
	if d.MemUtil <= 0 {
		d.MemUtil = 0.9
	}
	if d.Eff.GEMM == 0 {
		d.Eff = DefaultEfficiency()
	}
	r := &Replica{Model: m, HW: hw, TP: d.TP, Eff: d.Eff, MemUtil: d.MemUtil}
	r.Label = fmt.Sprintf("%s x%d", hw.Name, d.TP)
	tp := float64(d.TP)
	total := m.TotalParams() * m.WeightBytes
	r.weightBytesGPU = total / tp
	if m.Experts > 0 {
		per := 3 * float64(m.Hidden) * float64(m.ExpertFFN) * m.WeightBytes
		r.expertBytesGPU = per * float64(m.Experts*m.Layers) / tp
		r.denseBytesGPU = r.weightBytesGPU - r.expertBytesGPU
	} else {
		r.denseBytesGPU = r.weightBytesGPU
	}
	kvHeadsGPU := math.Ceil(float64(m.KVHeads) / tp)
	r.kvBytesTokenGPU = 2 * float64(m.Layers) * kvHeadsGPU * float64(m.HeadDim) * m.KVBytes
	r.flopsPerToken = 2 * m.ActiveLayerParams()
	r.headFlops = 2 * float64(m.Hidden) * float64(m.Vocab)
	r.attnCoef = 4 * float64(m.Layers*m.Heads*m.HeadDim)
	free := hw.MemBytes*d.MemUtil - r.weightBytesGPU - d.Eff.Reserve
	if free <= 0 {
		return nil, fmt.Errorf("%s does not fit on %d x %s", m.Name, d.TP, hw.Name)
	}
	r.KVTokens = int(free / r.kvBytesTokenGPU)
	return r, nil
}

type Batch struct {
	LoadTokens int
	Tokens     int
	Seqs       int
	Samples    int
	AttnPairs  float64
	KVTokens   float64
}

func (b *Batch) Add(q, ctx int, sample bool) {
	b.Tokens += q
	b.Seqs++
	if sample {
		b.Samples++
	}
	b.AttnPairs += float64(q) * (float64(ctx) + float64(q+1)/2)
	b.KVTokens += float64(ctx + q)
}

func (r *Replica) StepTime(b Batch) float64 {
	if b.Tokens == 0 {
		return 0
	}
	load := 0.0
	if b.LoadTokens > 0 && r.HW.HostBW > 0 {
		load = float64(b.LoadTokens) * r.kvBytesTokenGPU / r.HW.HostBW
	}
	tp := float64(r.TP)
	n := float64(b.Tokens)
	gemmEff := r.Eff.GEMM * n / (n + r.Eff.GEMMHalf)
	linear := r.flopsPerToken*n + r.headFlops*float64(b.Samples)
	attn := r.attnCoef * b.AttnPairs
	tc := linear/(tp*r.HW.FLOPS*gemmEff) + attn/(tp*r.HW.FLOPS*r.Eff.Attention)
	weights := r.denseBytesGPU
	if r.expertBytesGPU > 0 {
		k := float64(r.Model.ActiveExperts) / float64(r.Model.Experts)
		weights += r.expertBytesGPU * (1 - math.Pow(1-k, n))
	}
	kv := (b.KVTokens + n) * r.kvBytesTokenGPU
	tm := (weights + kv) / (r.HW.MemBW * r.Eff.Bandwidth)
	t := math.Max(tc, tm)
	if r.TP > 1 {
		msg := n * float64(r.Model.Hidden) * 2
		t += 2 * float64(r.Model.Layers) * (2*(tp-1)/tp*msg/r.HW.LinkBW + r.HW.LinkLatency)
	}
	return t + load + r.Eff.StepOverhead + r.Eff.SeqOverhead*float64(b.Seqs)
}

func (r *Replica) Scaled(gemm, attention, bandwidth float64) *Replica {
	c := *r
	if gemm > 0 {
		c.Eff.GEMM *= gemm
	}
	if attention > 0 {
		c.Eff.Attention *= attention
	}
	if bandwidth > 0 {
		c.Eff.Bandwidth *= bandwidth
	}
	return &c
}

func (r *Replica) LoadTime(tokens int) float64 {
	if r.HW.HostBW <= 0 {
		return 0
	}
	return float64(tokens) * r.kvBytesTokenGPU / r.HW.HostBW
}

func (r *Replica) PrefillTime(tokens, ctx, chunk int) float64 {
	if chunk <= 0 {
		chunk = 8192
	}
	t := 0.0
	for tokens > 0 {
		q := tokens
		if q > chunk {
			q = chunk
		}
		var b Batch
		b.Add(q, ctx, tokens == q)
		t += r.StepTime(b)
		ctx += q
		tokens -= q
	}
	return t
}

func (r *Replica) DecodeStepTime(seqs int, avgCtx float64) float64 {
	var b Batch
	b.Tokens = seqs
	b.Seqs = seqs
	b.Samples = seqs
	b.AttnPairs = float64(seqs) * (avgCtx + 1)
	b.KVTokens = float64(seqs) * (avgCtx + 1)
	return r.StepTime(b)
}

func (r *Replica) KVBytesPerTokenGPU() float64 { return r.kvBytesTokenGPU }
func (r *Replica) WeightBytesGPU() float64     { return r.weightBytesGPU }
func (r *Replica) PricePerHour() float64       { return r.HW.PricePerHour * float64(r.TP) }
