package workload

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

type Item struct {
	ID      int64    `json:"id"`
	Session int64    `json:"session"`
	Arrival float64  `json:"arrival"`
	Input   int      `json:"input"`
	Output  int      `json:"output"`
	Hashes  []uint64 `json:"-"`
	Class   uint8    `json:"class"`
}

type SLO struct {
	TTFTBase      float64 `json:"ttft_base_s"`
	TTFTPerK      float64 `json:"ttft_per_ktok_s"`
	TPOT          float64 `json:"tpot_s"`
	TPOTMinTokens int     `json:"tpot_min_tokens,omitempty"`
}

func (s SLO) TTFT(input int) float64 { return s.TTFTBase + s.TTFTPerK*float64(input)/1000 }

func (s SLO) TPOTMet(tpot float64, tokens int) bool {
	return tokens < s.TPOTMinTokens || tpot <= s.TPOT
}

type Workload struct {
	Name        string   `json:"name"`
	BlockTokens int      `json:"block_tokens"`
	Items       []Item   `json:"-"`
	Classes     []string `json:"classes"`
}

type Summary struct {
	Requests    int     `json:"requests"`
	Duration    float64 `json:"duration_s"`
	Rate        float64 `json:"rate_rps"`
	MeanInput   float64 `json:"mean_input"`
	P50Input    int     `json:"p50_input"`
	P99Input    int     `json:"p99_input"`
	MeanOutput  float64 `json:"mean_output"`
	ReuseRatio  float64 `json:"reuse_ratio"`
	Sessions    int     `json:"sessions"`
	BlockTokens int     `json:"block_tokens"`
}

func (w *Workload) Summarize() Summary {
	s := Summary{Requests: len(w.Items), BlockTokens: w.BlockTokens}
	if len(w.Items) == 0 {
		return s
	}
	in := make([]int, len(w.Items))
	seen := make(map[uint64]struct{}, len(w.Items)*8)
	sessions := make(map[int64]struct{})
	var hit, tot float64
	var sumIn, sumOut float64
	for i, it := range w.Items {
		in[i] = it.Input
		sumIn += float64(it.Input)
		sumOut += float64(it.Output)
		sessions[it.Session] = struct{}{}
		promptBlocks := it.Input / w.BlockTokens
		k := 0
		for k < promptBlocks && k < len(it.Hashes) {
			if _, ok := seen[it.Hashes[k]]; !ok {
				break
			}
			k++
		}
		hit += float64(k * w.BlockTokens)
		tot += float64(it.Input)
		for _, h := range it.Hashes {
			seen[h] = struct{}{}
		}
	}
	sort.Ints(in)
	s.Duration = w.Items[len(w.Items)-1].Arrival - w.Items[0].Arrival
	if s.Duration > 0 {
		s.Rate = float64(len(w.Items)) / s.Duration
	}
	s.MeanInput = sumIn / float64(len(w.Items))
	s.MeanOutput = sumOut / float64(len(w.Items))
	s.P50Input = in[len(in)/2]
	s.P99Input = in[int(0.99*float64(len(in)-1))]
	s.ReuseRatio = hit / tot
	s.Sessions = len(sessions)
	return s
}

func (w *Workload) Scale(rate float64) *Workload {
	out := *w
	out.Items = make([]Item, len(w.Items))
	copy(out.Items, w.Items)
	if rate > 0 && rate != 1 {
		t0 := w.Items[0].Arrival
		for i := range out.Items {
			out.Items[i].Arrival = t0 + (out.Items[i].Arrival-t0)/rate
		}
	}
	return &out
}

func (w *Workload) Window(start, dur float64) *Workload {
	out := *w
	out.Items = nil
	if len(w.Items) == 0 {
		return &out
	}
	t0 := w.Items[0].Arrival + start
	for _, it := range w.Items {
		if it.Arrival < t0 {
			continue
		}
		if dur > 0 && it.Arrival >= t0+dur {
			break
		}
		it.Arrival -= t0
		it.ID = int64(len(out.Items))
		out.Items = append(out.Items, it)
	}
	return &out
}

type mooncakeRow struct {
	Timestamp float64  `json:"timestamp"`
	Input     int      `json:"input_length"`
	Output    int      `json:"output_length"`
	HashIDs   []uint64 `json:"hash_ids"`
}

func LoadMooncake(path string) (*Workload, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadMooncake(f, nameFromPath(path))
}

func ReadMooncake(r io.Reader, name string) (*Workload, error) {
	w := &Workload{Name: name, BlockTokens: 512, Classes: []string{"trace"}}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	var id int64
	parents := make(map[uint64]int64)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row mooncakeRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("line %d: %w", id+1, err)
		}
		full := row.Input / 512
		if full > len(row.HashIDs) {
			full = len(row.HashIDs)
		}
		hashes := make([]uint64, full)
		for i := 0; i < full; i++ {
			hashes[i] = mix(row.HashIDs[i] + 0x9e3779b97f4a7c15)
		}
		session := id
		if len(row.HashIDs) > 1 {
			if s, ok := parents[row.HashIDs[0]<<20^row.HashIDs[1]]; ok {
				session = s
			} else {
				parents[row.HashIDs[0]<<20^row.HashIDs[1]] = id
			}
		}
		out := row.Output
		if out < 1 {
			out = 1
		}
		w.Items = append(w.Items, Item{ID: id, Session: session, Arrival: row.Timestamp / 1000, Input: row.Input, Output: out, Hashes: hashes})
		id++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(w.Items, func(i, j int) bool { return w.Items[i].Arrival < w.Items[j].Arrival })
	for i := range w.Items {
		w.Items[i].ID = int64(i)
	}
	return w, nil
}

func nameFromPath(p string) string {
	p = strings.ReplaceAll(p, "\\", "/")
	if i := strings.LastIndex(p, "/"); i >= 0 {
		p = p[i+1:]
	}
	return strings.TrimSuffix(p, ".jsonl")
}

func mix(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

func chain(parent, content uint64) uint64 {
	return mix(parent*0x100000001b3 ^ mix(content))
}

func clamp(x, lo, hi float64) float64 { return math.Max(lo, math.Min(hi, x)) }
