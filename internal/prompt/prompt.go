package prompt

import (
	"bytes"
	"encoding/json"
	"strings"
)

type Hasher struct {
	BlockBytes    int
	CharsPerToken float64
}

func Default() Hasher { return Hasher{BlockBytes: 256, CharsPerToken: 4} }

func (h Hasher) BlockTokens() int { return int(float64(h.BlockBytes) / h.CharsPerToken) }

func (h Hasher) Tokens(text []byte) int {
	n := int(float64(len(text))/h.CharsPerToken + 0.5)
	if n < 1 {
		n = 1
	}
	return n
}

func (h Hasher) Hashes(text []byte) []uint64 {
	n := len(text) / h.BlockBytes
	out := make([]uint64, n)
	var parent uint64 = 0xcbf29ce484222325
	for i := 0; i < n; i++ {
		x := parent
		for _, c := range text[i*h.BlockBytes : (i+1)*h.BlockBytes] {
			x ^= uint64(c)
			x *= 0x100000001b3
		}
		x ^= x >> 33
		x *= 0xff51afd7ed558ccd
		x ^= x >> 33
		parent = x
		out[i] = x
	}
	return out
}

type Request struct {
	Model     string
	Stream    bool
	MaxTokens int
	Text      []byte
	Session   string
}

func Parse(body []byte) (Request, map[string]any, error) {
	var raw map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return Request{}, nil, err
	}
	var r Request
	r.Model, _ = raw["model"].(string)
	r.Stream, _ = raw["stream"].(bool)
	for _, k := range []string{"max_completion_tokens", "max_tokens"} {
		if v, ok := raw[k].(json.Number); ok {
			if n, err := v.Int64(); err == nil && n > 0 {
				r.MaxTokens = int(n)
				break
			}
		}
	}
	if u, ok := raw["user"].(string); ok {
		r.Session = u
	}
	var buf strings.Builder
	if msgs, ok := raw["messages"].([]any); ok {
		for _, m := range msgs {
			mm, _ := m.(map[string]any)
			role, _ := mm["role"].(string)
			buf.WriteString("<|")
			buf.WriteString(role)
			buf.WriteString("|>")
			writeContent(&buf, mm["content"])
			buf.WriteString("\n")
		}
	} else {
		switch p := raw["prompt"].(type) {
		case string:
			buf.WriteString(p)
		case []any:
			for _, x := range p {
				if s, ok := x.(string); ok {
					buf.WriteString(s)
				}
			}
		}
	}
	r.Text = []byte(buf.String())
	return r, raw, nil
}

func writeContent(b *strings.Builder, c any) {
	switch v := c.(type) {
	case string:
		b.WriteString(v)
	case []any:
		for _, part := range v {
			if pm, ok := part.(map[string]any); ok {
				if t, ok := pm["text"].(string); ok {
					b.WriteString(t)
				}
			}
		}
	}
}
