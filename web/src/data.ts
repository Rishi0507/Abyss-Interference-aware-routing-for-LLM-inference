import { useEffect, useState } from "react";
import { api, Catalog, ResultSet } from "./api";

export type Loadable<T> = { data?: T; error?: string; loading: boolean };

export function useResults(): Loadable<ResultSet[]> {
  const [state, setState] = useState<Loadable<ResultSet[]>>({ loading: true });
  useEffect(() => {
    api
      .results()
      .then((data) => setState({ data: data ?? [], loading: false }))
      .catch((e: Error) => setState({ error: e.message, loading: false }));
  }, []);
  return state;
}

export function useCatalog(): Loadable<Catalog> {
  const [state, setState] = useState<Loadable<Catalog>>({ loading: true });
  useEffect(() => {
    api
      .catalog()
      .then((data) => setState({ data, loading: false }))
      .catch((e: Error) => setState({ error: e.message, loading: false }));
  }, []);
  return state;
}

export const resultOrder = ["conversation-70b", "toolagent-70b", "agent-8b", "mixed-hetero"];

export function sortResults(rs: ResultSet[]) {
  return [...rs].sort((a, b) => {
    const ia = resultOrder.indexOf(a.name);
    const ib = resultOrder.indexOf(b.name);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}

export function fleetText(fleet: string[] | null | undefined) {
  const counts = new Map<string, number>();
  (fleet ?? []).forEach((f) => counts.set(f, (counts.get(f) ?? 0) + 1));
  return [...counts.entries()].map(([k, n]) => `${n} × ${k.replace(" x", " TP")}`).join(", ");
}
