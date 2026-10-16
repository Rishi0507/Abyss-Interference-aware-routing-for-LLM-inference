package workload

import (
	"strings"
	"testing"
)

func TestMooncakeParsingKeepsFullBlocksOnly(t *testing.T) {
	trace := `{"timestamp": 0, "input_length": 1100, "output_length": 5, "hash_ids": [1, 2, 3]}
{"timestamp": 250, "input_length": 1024, "output_length": 0, "hash_ids": [1, 2]}
`
	w, err := ReadMooncake(strings.NewReader(trace), "t")
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Items) != 2 || w.BlockTokens != 512 {
		t.Fatalf("%d items, block %d", len(w.Items), w.BlockTokens)
	}
	if len(w.Items[0].Hashes) != 2 {
		t.Fatalf("partial block kept: %d hashes", len(w.Items[0].Hashes))
	}
	if w.Items[0].Hashes[0] != w.Items[1].Hashes[0] || w.Items[0].Hashes[1] != w.Items[1].Hashes[1] {
		t.Fatal("shared prefix lost")
	}
	if w.Items[1].Arrival != 0.25 || w.Items[1].Output != 1 {
		t.Fatalf("arrival %.3f output %d", w.Items[1].Arrival, w.Items[1].Output)
	}
}

func TestSyntheticSessionsShareGrowingPrefixes(t *testing.T) {
	w, err := Generate(Spec{Kind: "agent", Rate: 2, Duration: 300, Seed: 9})
	if err != nil {
		t.Fatal(err)
	}
	bySession := map[int64][]Item{}
	for _, it := range w.Items {
		bySession[it.Session] = append(bySession[it.Session], it)
	}
	checked := 0
	for _, items := range bySession {
		for i := 1; i < len(items); i++ {
			prev, cur := items[i-1], items[i]
			n := prev.Input / w.BlockTokens
			for k := 0; k < n; k++ {
				if prev.Hashes[k] != cur.Hashes[k] {
					t.Fatalf("session turn %d diverges at block %d", i, k)
				}
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no multi-step sessions generated")
	}
	if s := w.Summarize(); s.ReuseRatio < 0.5 {
		t.Fatalf("agent reuse ratio %.2f is implausibly low", s.ReuseRatio)
	}
}

func TestScaleCompressesArrivals(t *testing.T) {
	w, _ := Generate(Spec{Kind: "chat", Rate: 4, Duration: 120, Seed: 1})
	s := w.Scale(2)
	last := len(w.Items) - 1
	if got, want := s.Items[last].Arrival-s.Items[0].Arrival, (w.Items[last].Arrival-w.Items[0].Arrival)/2; got-want > 1e-9 || want-got > 1e-9 {
		t.Fatalf("scaled span %.3f, want %.3f", got, want)
	}
}
