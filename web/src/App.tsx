import { useEffect, useState } from "react";
import Overview from "./pages/Overview";
import Benchmarks from "./pages/Benchmarks";
import Lab from "./pages/Lab";
import Live from "./pages/Live";
import ResearchPage from "./pages/Research";
import { useResults } from "./data";

const pages = [
  { path: "", label: "Overview" },
  { path: "benchmarks", label: "Benchmarks" },
  { path: "lab", label: "Lab" },
  { path: "research", label: "Research" },
  { path: "live", label: "Live" },
];

function useRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("?")[0];
  const [route, setRoute] = useState(read());
  useEffect(() => {
    const on = () => {
      setRoute(read());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="11" fill="none" stroke="var(--accent)" strokeWidth="3" />
      <path d="M16 5v22" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const route = useRoute();
  const results = useResults();
  const [live, setLive] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("/api/live/state")
      .then((r) => r.json())
      .then((s) => setLive(s.attached !== false))
      .catch(() => setLive(false));
  }, []);

  const page = pages.find((p) => p.path === route) ? route : "";

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="wordmark" href="#/">
            <Mark />
            abyss
          </a>
          <nav className="nav" aria-label="Primary">
            {pages.map((p) => (
              <a key={p.path} href={`#/${p.path}`} aria-current={page === p.path ? "page" : undefined}>
                {p.label}
              </a>
            ))}
          </nav>
          <span className="meta">{live === null ? "" : live ? "Connected to a gateway" : "Simulation only"}</span>
        </div>
      </header>
      <main className="shell">
        {page === "" && <Overview results={results} />}
        {page === "benchmarks" && <Benchmarks results={results} />}
        {page === "lab" && <Lab />}
        {page === "research" && <ResearchPage />}
        {page === "live" && <Live attached={live} />}
        <footer className="foot">
          <span>Abyss: request routing for LLM inference fleets</span>
          <span>Apache 2.0</span>
        </footer>
      </main>
    </>
  );
}
