package perf

import (
	"fmt"
	"sort"
	"strings"
)

type Hardware struct {
	Name         string  `json:"name"`
	Vendor       string  `json:"vendor"`
	FLOPS        float64 `json:"flops"`
	MemBW        float64 `json:"mem_bw"`
	MemBytes     float64 `json:"mem_bytes"`
	LinkBW       float64 `json:"link_bw"`
	LinkLatency  float64 `json:"link_latency"`
	HostBW       float64 `json:"host_bw"`
	PricePerHour float64 `json:"price_per_hour"`
}

type Model struct {
	Name          string  `json:"name"`
	Layers        int     `json:"layers"`
	Hidden        int     `json:"hidden"`
	Heads         int     `json:"heads"`
	KVHeads       int     `json:"kv_heads"`
	HeadDim       int     `json:"head_dim"`
	FFN           int     `json:"ffn"`
	Vocab         int     `json:"vocab"`
	Experts       int     `json:"experts"`
	ActiveExperts int     `json:"active_experts"`
	ExpertFFN     int     `json:"expert_ffn"`
	WeightBytes   float64 `json:"weight_bytes"`
	KVBytes       float64 `json:"kv_bytes"`
	TiedEmbedding bool    `json:"tied_embedding"`
}

const (
	tera = 1e12
	giga = 1e9
)

var hardware = map[string]Hardware{
	"h100":   {Name: "H100 SXM", Vendor: "NVIDIA", FLOPS: 989 * tera, MemBW: 3.35 * tera, MemBytes: 80 * giga, LinkBW: 450 * giga, LinkLatency: 6e-6, HostBW: 25 * giga, PricePerHour: 2.49},
	"h200":   {Name: "H200 SXM", Vendor: "NVIDIA", FLOPS: 989 * tera, MemBW: 4.8 * tera, MemBytes: 141 * giga, LinkBW: 450 * giga, LinkLatency: 6e-6, HostBW: 25 * giga, PricePerHour: 3.39},
	"b200":   {Name: "B200", Vendor: "NVIDIA", FLOPS: 2250 * tera, MemBW: 8.0 * tera, MemBytes: 180 * giga, LinkBW: 900 * giga, LinkLatency: 5e-6, HostBW: 25 * giga, PricePerHour: 5.49},
	"a100":   {Name: "A100 SXM 80GB", Vendor: "NVIDIA", FLOPS: 312 * tera, MemBW: 2.039 * tera, MemBytes: 80 * giga, LinkBW: 300 * giga, LinkLatency: 8e-6, HostBW: 12 * giga, PricePerHour: 1.29},
	"l40s":   {Name: "L40S", Vendor: "NVIDIA", FLOPS: 362 * tera, MemBW: 0.864 * tera, MemBytes: 48 * giga, LinkBW: 32 * giga, LinkLatency: 15e-6, HostBW: 12 * giga, PricePerHour: 0.86},
	"mi300x": {Name: "MI300X", Vendor: "AMD", FLOPS: 1307 * tera, MemBW: 5.3 * tera, MemBytes: 192 * giga, LinkBW: 448 * giga, LinkLatency: 8e-6, HostBW: 25 * giga, PricePerHour: 2.29},
}

var models = map[string]Model{
	"llama-3.1-8b":      {Name: "Llama 3.1 8B", Layers: 32, Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, FFN: 14336, Vocab: 128256, WeightBytes: 2, KVBytes: 2},
	"llama-3.1-70b":     {Name: "Llama 3.1 70B", Layers: 80, Hidden: 8192, Heads: 64, KVHeads: 8, HeadDim: 128, FFN: 28672, Vocab: 128256, WeightBytes: 2, KVBytes: 2},
	"llama-3.1-70b-fp8": {Name: "Llama 3.1 70B FP8", Layers: 80, Hidden: 8192, Heads: 64, KVHeads: 8, HeadDim: 128, FFN: 28672, Vocab: 128256, WeightBytes: 1, KVBytes: 1},
	"qwen3-32b":         {Name: "Qwen3 32B", Layers: 64, Hidden: 5120, Heads: 64, KVHeads: 8, HeadDim: 128, FFN: 25600, Vocab: 151936, WeightBytes: 2, KVBytes: 2},
	"qwen3-30b-a3b":     {Name: "Qwen3 30B A3B", Layers: 48, Hidden: 2048, Heads: 32, KVHeads: 4, HeadDim: 128, Vocab: 151936, Experts: 128, ActiveExperts: 8, ExpertFFN: 768, WeightBytes: 2, KVBytes: 2},
	"mixtral-8x7b":      {Name: "Mixtral 8x7B", Layers: 32, Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, Vocab: 32000, Experts: 8, ActiveExperts: 2, ExpertFFN: 14336, WeightBytes: 2, KVBytes: 2},
}

func LookupHardware(key string) (Hardware, error) {
	h, ok := hardware[strings.ToLower(key)]
	if !ok {
		return Hardware{}, fmt.Errorf("unknown hardware %q (known: %s)", key, strings.Join(HardwareKeys(), ", "))
	}
	return h, nil
}

func LookupModel(key string) (Model, error) {
	m, ok := models[strings.ToLower(key)]
	if !ok {
		return Model{}, fmt.Errorf("unknown model %q (known: %s)", key, strings.Join(ModelKeys(), ", "))
	}
	return m, nil
}

func HardwareKeys() []string { return sortedKeys(hardware) }
func ModelKeys() []string    { return sortedKeys(models) }

func sortedKeys[T any](m map[string]T) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func (m Model) attnParams() float64 {
	q := float64(m.Heads * m.HeadDim)
	kv := float64(m.KVHeads * m.HeadDim)
	d := float64(m.Hidden)
	return d*q + 2*d*kv + q*d
}

func (m Model) mlpParams() (total, active float64) {
	d := float64(m.Hidden)
	if m.Experts > 0 {
		per := 3 * d * float64(m.ExpertFFN)
		router := d * float64(m.Experts)
		return per*float64(m.Experts) + router, per*float64(m.ActiveExperts) + router
	}
	p := 3 * d * float64(m.FFN)
	return p, p
}

func (m Model) embedParams() float64 {
	e := float64(m.Vocab * m.Hidden)
	if m.TiedEmbedding {
		return e
	}
	return 2 * e
}

func (m Model) TotalParams() float64 {
	mlp, _ := m.mlpParams()
	return float64(m.Layers)*(m.attnParams()+mlp) + m.embedParams()
}

func (m Model) ActiveLayerParams() float64 {
	_, act := m.mlpParams()
	return float64(m.Layers) * (m.attnParams() + act)
}

func (m Model) KVBytesPerToken() float64 {
	return 2 * float64(m.Layers*m.KVHeads*m.HeadDim) * m.KVBytes
}
