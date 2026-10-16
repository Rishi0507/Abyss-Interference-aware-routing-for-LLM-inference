package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestResultsSkipsFilesThatAreNotSweeps(t *testing.T) {
	dir := t.TempDir()
	sweep := `{"preset":{"name":"p","title":"P"},"fleet":["H100 SXM x1"],"points":[{"policy":"abyss","scale":1,"attainment":0.5}],"runs":[]}`
	if err := os.WriteFile(filepath.Join(dir, "p.json"), []byte(sweep), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "microbench.json"), []byte(`{"cpu":"x","rows":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	s := New(dir, nil, nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/results", nil))
	var out []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0]["name"] != "p" {
		t.Fatalf("got %v, want only the sweep", out)
	}
}
