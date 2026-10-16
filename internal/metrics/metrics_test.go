package metrics

import (
	"math"
	"testing"
)

func rec(input, output int, arrival, first, finish float64) Record {
	return Record{Input: input, Output: output, Arrival: arrival, Dispatch: arrival, FirstAt: first, FinishAt: finish, Replica: 0, TTFTSLO: 1, TPOTSLO: 0.05}
}

func TestGoodRequiresBothObjectives(t *testing.T) {
	cases := []struct {
		name string
		r    Record
		good bool
	}{
		{"both met", rec(100, 11, 0, 0.5, 0.9), true},
		{"ttft missed", rec(100, 11, 0, 1.5, 1.9), false},
		{"tpot missed", rec(100, 11, 0, 0.5, 1.5), false},
		{"single token", rec(100, 1, 0, 0.5, 0.5), true},
	}
	for _, c := range cases {
		if got := c.r.Good(); got != c.good {
			t.Errorf("%s: Good() = %v, want %v", c.name, got, c.good)
		}
	}
	r := rec(100, 11, 0, 0.5, 0.9)
	r.TimedOut = true
	if r.Good() {
		t.Error("timed out request counted as good")
	}
	r = rec(100, 11, 0, 0.5, 0.9)
	r.Rejected = true
	if r.Good() {
		t.Error("rejected request counted as good")
	}
}

func TestTPOTMinTokensExemptsShortResponses(t *testing.T) {
	r := rec(100, 4, 0, 0.5, 2.0)
	if r.Good() {
		t.Fatal("slow short response should miss without a minimum")
	}
	r.TPOTMin = 16
	if !r.Good() {
		t.Fatal("short response should be judged on TTFT only")
	}
	r = rec(100, 20, 0, 0.5, 5.0)
	r.TPOTMin = 16
	if r.Good() {
		t.Fatal("long response still needs its TPOT objective")
	}
}

func TestSummarizeAttainmentAndTokenWeighting(t *testing.T) {
	recs := []Record{
		rec(1000, 11, 0, 0.5, 0.9),
		rec(3000, 11, 1, 1.4, 1.8),
		rec(4000, 11, 2, 4.0, 4.4),
		{Input: 2000, Output: 11, Arrival: 3, Dispatch: -1, FirstAt: -1, FinishAt: -1, Replica: -1, TimedOut: true, TTFTSLO: 1, TPOTSLO: 0.05},
	}
	s := Summarize("p", recs, 10, 3.6, nil, 1)
	if s.Good != 2 || s.Completed != 3 || s.TimedOut != 1 {
		t.Fatalf("good %d completed %d timed out %d", s.Good, s.Completed, s.TimedOut)
	}
	if math.Abs(s.Attainment-0.5) > 1e-12 {
		t.Errorf("attainment %v, want 0.5", s.Attainment)
	}
	if math.Abs(s.TokenAttain-0.4) > 1e-12 {
		t.Errorf("token-weighted attainment %v, want 0.4", s.TokenAttain)
	}
	if math.Abs(s.Goodput-0.2) > 1e-12 {
		t.Errorf("goodput %v, want 0.2", s.Goodput)
	}
	if math.Abs(s.TTFTAttain-0.5) > 1e-12 || math.Abs(s.TPOTAttain-0.75) > 1e-12 {
		t.Errorf("ttft %v tpot %v", s.TTFTAttain, s.TPOTAttain)
	}
	want := 3.6 * 10 / 3600 / 22 * 1e6
	if math.Abs(s.CostPerMGood-want) > 1e-9 {
		t.Errorf("cost per M good tokens %v, want %v", s.CostPerMGood, want)
	}
}

func TestQuantileInterpolates(t *testing.T) {
	xs := []float64{1, 2, 3, 4}
	if q := Quantile(xs, 0.5); q != 2.5 {
		t.Errorf("median %v", q)
	}
	if q := Quantile(xs, 1); q != 4 {
		t.Errorf("max %v", q)
	}
	if q := Quantile(nil, 0.5); q != 0 {
		t.Errorf("empty %v", q)
	}
}
