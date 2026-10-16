package metrics

import (
	"math"
	"sort"
)

type Record struct {
	ID        int64   `json:"id"`
	Class     uint8   `json:"class"`
	Input     int     `json:"input"`
	Output    int     `json:"output"`
	Arrival   float64 `json:"arrival"`
	Dispatch  float64 `json:"dispatch"`
	FirstAt   float64 `json:"first_at"`
	FinishAt  float64 `json:"finish_at"`
	Replica   int     `json:"replica"`
	Hit       int     `json:"hit"`
	Preempted int     `json:"preempted"`
	Rejected  bool    `json:"rejected"`
	TimedOut  bool    `json:"timed_out"`
	TTFTSLO   float64 `json:"ttft_slo"`
	TPOTSLO   float64 `json:"tpot_slo"`
	TPOTMin   int     `json:"tpot_min_tokens,omitempty"`
	PredTTFT  float64 `json:"pred_ttft"`
}

func (r *Record) TTFT() float64 { return r.FirstAt - r.Arrival }

func (r *Record) TPOT() float64 {
	if r.Output <= 1 {
		return 0
	}
	return (r.FinishAt - r.FirstAt) / float64(r.Output-1)
}

func (r *Record) Completed() bool {
	return !r.Rejected && !r.TimedOut && r.FinishAt >= 0 && r.FirstAt >= 0
}

func (r *Record) TPOTMet() bool { return r.Output < r.TPOTMin || r.TPOT() <= r.TPOTSLO }

func (r *Record) Good() bool {
	return r.Completed() && r.TTFT() <= r.TTFTSLO && r.TPOTMet()
}

type Dist struct {
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P90  float64 `json:"p90"`
	P99  float64 `json:"p99"`
	Max  float64 `json:"max"`
}

func NewDist(xs []float64) Dist {
	if len(xs) == 0 {
		return Dist{}
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	var sum float64
	for _, x := range s {
		sum += x
	}
	return Dist{Mean: sum / float64(len(s)), P50: Quantile(s, 0.5), P90: Quantile(s, 0.9), P99: Quantile(s, 0.99), Max: s[len(s)-1]}
}

func Quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	f := pos - float64(lo)
	return sorted[lo]*(1-f) + sorted[hi]*f
}

type ReplicaStat struct {
	Name        string  `json:"name"`
	Requests    int     `json:"requests"`
	Busy        float64 `json:"busy"`
	HitRate     float64 `json:"hit_rate"`
	Preemptions int64   `json:"preemptions"`
	Evictions   int64   `json:"evictions"`
	MeanBatch   float64 `json:"mean_batch_tokens"`
	Prefill     int64   `json:"prefill_tokens"`
	Decode      int64   `json:"decode_tokens"`
}

type Summary struct {
	Policy           string         `json:"policy"`
	Requests         int            `json:"requests"`
	Completed        int            `json:"completed"`
	Rejected         int            `json:"rejected"`
	TimedOut         int            `json:"timed_out"`
	Good             int            `json:"good"`
	Duration         float64        `json:"duration_s"`
	Attainment       float64        `json:"slo_attainment"`
	TTFTAttain       float64        `json:"ttft_attainment"`
	TPOTAttain       float64        `json:"tpot_attainment"`
	TokenAttain      float64        `json:"token_weighted_attainment"`
	Goodput          float64        `json:"goodput_rps"`
	Throughput       float64        `json:"output_tok_s"`
	TTFT             Dist           `json:"ttft"`
	TPOT             Dist           `json:"tpot"`
	E2E              Dist           `json:"e2e"`
	HitRate          float64        `json:"hit_rate"`
	Preemptions      int64          `json:"preemptions"`
	PrefillTokens    int64          `json:"prefill_tokens"`
	CostPerHour      float64        `json:"cost_per_hour"`
	CostPerMGood     float64        `json:"usd_per_m_good_tokens"`
	DecisionMicros   Dist           `json:"decision_us"`
	PredError        Dist           `json:"pred_abs_rel_error"`
	Replicas         []ReplicaStat  `json:"replicas"`
	ClassAttainment  []float64      `json:"class_attainment,omitempty"`
	LengthAttainment []LengthBucket `json:"length_attainment"`
}

type LengthBucket struct {
	Label      string  `json:"label"`
	Requests   int     `json:"requests"`
	Attainment float64 `json:"attainment"`
}

var lengthEdges = []int{4096, 16384, 65536}
var lengthLabels = []string{"<4k", "4k-16k", "16k-64k", ">64k"}

func lengthBucket(n int) int {
	for i, e := range lengthEdges {
		if n < e {
			return i
		}
	}
	return len(lengthEdges)
}

func Summarize(policy string, recs []Record, duration, costPerHour float64, decisions []float64, classes int) Summary {
	s := Summary{Policy: policy, Requests: len(recs), Duration: duration, CostPerHour: costPerHour}
	var ttft, tpot, e2e, perr []float64
	var hit, input float64
	var goodTokens float64
	var ttftOK, tpotOK int
	classGood := make([]int, classes)
	classTotal := make([]int, classes)
	for i := range recs {
		r := &recs[i]
		if int(r.Class) < classes {
			classTotal[r.Class]++
		}
		if r.Rejected {
			s.Rejected++
			continue
		}
		if r.TimedOut {
			s.TimedOut++
			continue
		}
		if !r.Completed() {
			continue
		}
		s.Completed++
		ttft = append(ttft, r.TTFT())
		if r.Output > 1 {
			tpot = append(tpot, r.TPOT())
		}
		e2e = append(e2e, r.FinishAt-r.Arrival)
		hit += float64(r.Hit)
		input += float64(r.Input)
		if r.TTFT() <= r.TTFTSLO {
			ttftOK++
		}
		if r.TPOTMet() {
			tpotOK++
		}
		if r.Good() {
			s.Good++
			goodTokens += float64(r.Output)
			if int(r.Class) < classes {
				classGood[r.Class]++
			}
		}
		if r.PredTTFT > 0 && r.TTFT() > 0 {
			perr = append(perr, math.Abs(r.PredTTFT-r.TTFT())/r.TTFT())
		}
	}
	s.TTFT, s.TPOT, s.E2E = NewDist(ttft), NewDist(tpot), NewDist(e2e)
	s.PredError = NewDist(perr)
	s.DecisionMicros = NewDist(decisions)
	if s.Requests > 0 {
		s.Attainment = float64(s.Good) / float64(s.Requests)
		s.TTFTAttain = float64(ttftOK) / float64(s.Requests)
		s.TPOTAttain = float64(tpotOK) / float64(s.Requests)
	}
	if duration > 0 {
		s.Goodput = float64(s.Good) / duration
	}
	var out float64
	for i := range recs {
		if recs[i].Completed() {
			out += float64(recs[i].Output)
		}
	}
	if duration > 0 {
		s.Throughput = out / duration
	}
	if input > 0 {
		s.HitRate = hit / input
	}
	if goodTokens > 0 {
		s.CostPerMGood = costPerHour * duration / 3600 / goodTokens * 1e6
	}
	var goodIn, allIn float64
	for i := range recs {
		allIn += float64(recs[i].Input)
		if recs[i].Good() {
			goodIn += float64(recs[i].Input)
		}
	}
	if allIn > 0 {
		s.TokenAttain = goodIn / allIn
	}
	lb := make([]LengthBucket, len(lengthLabels))
	for i := range lb {
		lb[i].Label = lengthLabels[i]
	}
	lg := make([]int, len(lengthLabels))
	for i := range recs {
		k := lengthBucket(recs[i].Input)
		lb[k].Requests++
		if recs[i].Good() {
			lg[k]++
		}
	}
	for i := range lb {
		if lb[i].Requests > 0 {
			lb[i].Attainment = float64(lg[i]) / float64(lb[i].Requests)
		}
	}
	s.LengthAttainment = lb
	if classes > 1 {
		s.ClassAttainment = make([]float64, classes)
		for c := range classes {
			if classTotal[c] > 0 {
				s.ClassAttainment[c] = float64(classGood[c]) / float64(classTotal[c])
			}
		}
	}
	return s
}
