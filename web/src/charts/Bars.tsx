import { useState } from "react";
import { useWidth } from "./scale";

export type BarRow = { key: string; label: string; color: string; value: number; text: string; emphasis?: boolean; detail?: string };

export function HBars({ rows, max, height = 26 }: { rows: BarRow[]; max?: number; height?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const labelW = Math.min(190, width * 0.38);
  const valueW = 64;
  const barW = Math.max(width - labelW - valueW - 12, 20);
  const top = max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={width} height={rows.length * (height + 8)} role="img" aria-label="bar chart" style={{ display: "block" }}>
        {rows.map((r, i) => {
          const y = i * (height + 8);
          const w = Math.max((r.value / top) * barW, r.value > 0 ? 3 : 0);
          return (
            <g key={r.key} transform={`translate(0,${y})`} onMouseEnter={() => setHover(r.key)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={-2} width={width} height={height + 4} fill={hover === r.key ? "var(--hover)" : "transparent"} rx={8} />
              <text x={0} y={height / 2} dy="0.32em" fontSize={13} fill={r.emphasis ? "var(--ink)" : "var(--ink-2)"} fontWeight={r.emphasis ? 600 : 450}>
                {r.label}
              </text>
              <rect x={labelW} y={height * 0.18} width={barW} height={height * 0.64} fill="var(--track)" rx={4} />
              <rect x={labelW} y={height * 0.18} width={w} height={height * 0.64} fill={r.color} rx={4} />
              <text x={labelW + barW + 10} y={height / 2} dy="0.32em" fontSize={13} fill="var(--ink)" className="tabular" fontWeight={r.emphasis ? 600 : 450}>
                {r.text}
              </text>
            </g>
          );
        })}
      </svg>
      {hover && rows.find((r) => r.key === hover)?.detail && (
        <div className="tooltip" style={{ right: 0, top: rows.findIndex((r) => r.key === hover) * (height + 8) + height + 4 }}>
          {rows.find((r) => r.key === hover)!.detail}
        </div>
      )}
    </div>
  );
}

export type Segment = { key: string; label: string; color: string };

export type StackRow = {
  key: string;
  label: string;
  values: Record<string, number>;
  marker?: string;
  emphasis?: boolean;
  note?: string;
};

export function StackedBars({ rows, segments, max, format }: { rows: StackRow[]; segments: Segment[]; max?: number; format: (v: number) => string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const h = 30;
  const labelW = Math.min(150, width * 0.3);
  const valueW = 150;
  const barW = Math.max(width - labelW - valueW - 8, 40);
  const totals = rows.map((r) => segments.reduce((a, s) => a + Math.max(r.values[s.key] ?? 0, 0), 0));
  const top = max ?? Math.max(...totals, 1e-9);
  const hoverRow = rows.find((r) => r.key === hover);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <div className="legend" style={{ marginBottom: 12 }}>
        {segments.map((s) => (
          <span key={s.key}>
            <i className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <svg width={width} height={rows.length * (h + 10)} role="img" aria-label="stacked bars" style={{ display: "block" }}>
        {rows.map((r, i) => {
          let x = labelW;
          const y = i * (h + 10);
          return (
            <g key={r.key} transform={`translate(0,${y})`} onMouseEnter={() => setHover(r.key)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={-3} width={width} height={h + 6} rx={9} fill={r.emphasis ? "var(--accent-soft)" : hover === r.key ? "var(--hover)" : "transparent"} />
              <text x={10} y={h / 2} dy="0.32em" fontSize={13} fill={r.emphasis ? "var(--ink)" : "var(--ink-2)"} fontWeight={r.emphasis ? 600 : 450}>
                {r.label}
              </text>
              {segments.map((s) => {
                const v = Math.max(r.values[s.key] ?? 0, 0);
                const w = (v / top) * barW;
                const rect = w > 0.5 ? <rect key={s.key} x={x} y={h * 0.2} width={Math.max(w - 2, 1)} height={h * 0.6} fill={s.color} rx={3} /> : null;
                x += w;
                return rect;
              })}
              <text x={labelW + barW + 8} y={h / 2} dy="0.32em" fontSize={12.5} fill="var(--ink)" className="tabular">
                {format(totals[i])}
              </text>
              {r.marker && (
                <text x={labelW + barW + 52} y={h / 2} dy="0.32em" fontSize={12} fill="var(--ink-2)">
                  {r.marker}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hoverRow && (
        <div className="tooltip" style={{ left: labelW, top: rows.indexOf(hoverRow) * (h + 10) + h + 34 }}>
          <div className="title">{hoverRow.label}</div>
          {segments.map((s) => (
            <div className="row" key={s.key}>
              <span>
                <i className="swatch" style={{ background: s.color }} />
                {s.label}
              </span>
              <span>{format(hoverRow.values[s.key] ?? 0)}</span>
            </div>
          ))}
          {hoverRow.note && <div className="muted" style={{ marginTop: 6 }}>{hoverRow.note}</div>}
        </div>
      )}
    </div>
  );
}

export function HeatStrip({
  rows,
  buckets,
  label,
  format,
}: {
  rows: { key: string; label: string; values: number[]; raw: number[] }[];
  buckets: number;
  label: (i: number) => string;
  format: (v: number) => string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const labelW = 120;
  const cw = Math.max((width - labelW) / Math.max(buckets, 1), 2);
  const h = 22;
  const color = (v: number) => {
    const t = Math.max(0, Math.min(v, 1));
    const a = [236, 241, 248];
    const b = [28, 92, 171];
    const c = a.map((x, i) => Math.round(x + (b[i] - x) * Math.pow(t, 0.85)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={width} height={rows.length * (h + 4)} role="img" aria-label="replica utilization over time" style={{ display: "block" }}>
        {rows.map((r, ri) => (
          <g key={r.key} transform={`translate(0,${ri * (h + 4)})`}>
            <text x={0} y={h / 2} dy="0.32em" fontSize={12} fill="var(--ink-2)">
              {r.label}
            </text>
            {r.values.slice(0, buckets).map((v, ci) => (
              <rect
                key={ci}
                x={labelW + ci * cw}
                y={0}
                width={Math.max(cw - 1.5, 1)}
                height={h}
                rx={2}
                fill={color(v)}
                onMouseEnter={() => setHover({ r: ri, c: ci })}
                onMouseLeave={() => setHover(null)}
              />
            ))}
          </g>
        ))}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: Math.min(labelW + hover.c * cw, width - 180), top: (hover.r + 1) * (h + 4) + 4 }}>
          <div className="title">{rows[hover.r].label}</div>
          <div className="row">
            <span>{label(hover.c)}</span>
            <span>{format(rows[hover.r].raw[hover.c] ?? 0)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
