package main

import (
	"reflect"
	"testing"
)

func TestSplitPoliciesKeepsParameterLists(t *testing.T) {
	got := splitPolicies("abyss, llm-d:8,2,0,sita:16384,1, dynamo:4,round-robin,sita:16384,1")
	want := []string{"abyss", "llm-d:8,2,0", "sita:16384,1", "dynamo:4", "round-robin"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}
