package perf

import (
	"math"
	"testing"
)

func TestParameterCounts(t *testing.T) {
	cases := map[string]float64{"llama-3.1-8b": 8.03e9, "llama-3.1-70b": 70.6e9, "qwen3-32b": 32.8e9, "mixtral-8x7b": 46.7e9}
	for key, want := range cases {
		m, err := LookupModel(key)
		if err != nil {
			t.Fatal(err)
		}
		if got := m.TotalParams(); math.Abs(got-want)/want > 0.02 {
			t.Errorf("%s: %.2fB params, want about %.2fB", key, got/1e9, want/1e9)
		}
	}
}

func TestKVCapacityMatchesPublishedFigures(t *testing.T) {
	r, err := NewReplica(Deployment{ModelKey: "llama-3.1-70b", HardwareKey: "h100", TP: 4})
	if err != nil {
		t.Fatal(err)
	}
	if r.KVTokens < 350_000 || r.KVTokens > 450_000 {
		t.Fatalf("70B TP4 KV capacity %d tokens, expected roughly 400k", r.KVTokens)
	}
}

func TestDecodeIsMemoryBoundAndPrefillComputeBound(t *testing.T) {
	r, _ := NewReplica(Deployment{ModelKey: "llama-3.1-8b", HardwareKey: "h100", TP: 1})
	one := r.DecodeStepTime(1, 1024)
	weights := r.WeightBytesGPU() / (r.HW.MemBW * r.Eff.Bandwidth)
	if one < weights || one > weights*1.6 {
		t.Fatalf("single decode step %.2fms, weight streaming floor %.2fms", one*1e3, weights*1e3)
	}
	p := r.PrefillTime(8192, 0, 8192)
	flops := 2 * r.Model.ActiveLayerParams() * 8192 / (r.HW.FLOPS * r.Eff.GEMM)
	if p < flops {
		t.Fatalf("prefill %.1fms is faster than the compute floor %.1fms", p*1e3, flops*1e3)
	}
	if r.DecodeStepTime(64, 1024) > 3*one {
		t.Fatal("batched decode should amortize weight reads")
	}
}

func TestFasterHardwareIsFaster(t *testing.T) {
	a, _ := NewReplica(Deployment{ModelKey: "qwen3-32b", HardwareKey: "a100", TP: 2})
	h, _ := NewReplica(Deployment{ModelKey: "qwen3-32b", HardwareKey: "h100", TP: 2})
	if h.PrefillTime(4096, 0, 8192) >= a.PrefillTime(4096, 0, 8192) {
		t.Fatal("H100 prefill should beat A100")
	}
	if h.DecodeStepTime(32, 4096) >= a.DecodeStepTime(32, 4096) {
		t.Fatal("H100 decode should beat A100")
	}
}

func TestModelThatDoesNotFitIsRejected(t *testing.T) {
	if _, err := NewReplica(Deployment{ModelKey: "llama-3.1-70b", HardwareKey: "l40s", TP: 1}); err == nil {
		t.Fatal("70B on a single L40S should not fit")
	}
}
