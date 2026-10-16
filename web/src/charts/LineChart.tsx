import { useMemo, useState } from "react";
import { extent, linear, niceTicks, useWidth } from "./scale";

export type SeriesPoint = { x: number; y: number; lo?: number; hi?: number };
export type Series = { key: string; label: string; color: string; points: SeriesPoint[]; emphasis?: boolean };

type Props = {
  series: Series[];
  height?: number;
  yDomain?: [number, number];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  xLabel?: string;
  yLabel?: string;
  markers?: boolean;
  directLabels?: boolean;
  legend?: boolean;
  logY?: boolean;
};

const M = { top: 14, right: 18, bottom: 40, left: 54 };

export default function LineChart({
  series,
  height = 300,
  yDomain,
  xFormat = (v) => String(v),
  yFormat = (v) => String(v),
  xLabel,
  yLabel,
  markers = true,
  directLabels = true,
  legend = true,
  logY = false,
}: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const labelSpace = directLabels && series.length <= 4 ? 120 : 0;
  const right = M.right + labelSpace;
  const iw = Math.max(width - M.left - right, 40);
  const ih = height - M.top - M.bottom;

  const xs = useMemo(() => {
    const set = new Set<number>();
    series.forEach((s) => s.points.forEach((p) => set.add(p.x)));
    return [...set].sort((a, b) => a - b);
  }, [series]);

  const [x0, x1] = extent(xs);
  const allY = series.flatMap((s) => s.points.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]));
  const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-6)) : v);
  let [y0, y1] = yDomain ?? extent(logY ? allY.filter((v) => v > 0) : allY);
  if (!yDomain && logY) {
    y0 = y0 / 1.6;
    y1 = y1 * 1.6;
  } else if (!yDomain) {
    y0 = Math.min(0, y0);
    y1 = y1 * 1.06;
  }
  const yTicks = logY ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  if (!yDomain && !logY && yTicks.length) y1 = Math.max(y1, yTicks[yTicks.length - 1]);
  const sx = linear(x0, x1, 0, iw);
  const syRaw = linear(tf(y0), tf(y1), ih, 0);
  const sy = (v: number) => syRaw(tf(v));
  const xTicks = xs.length <= 8 ? xs : niceTicks(x0, x1, 6);

  const paths = series.map((s) => {
    const pts = [...s.points].sort((a, b) => a.x - b.x);
    const line = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    let band = "";
    if (pts.some((p) => p.lo !== undefined)) {
      const top = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.hi ?? p.y).toFixed(1)}`).join("");
      const bottom = [...pts]
        .reverse()
        .map((p) => `L${sx(p.x).toFixed(1)},${sy(p.lo ?? p.y).toFixed(1)}`)
        .join("");
      band = `${top}${bottom}Z`;
    }
    return { s, pts, line, band };
  });

  const labels = useMemo(() => {
    if (!directLabels || series.length > 4) return [];
    const items = paths
      .filter((p) => p.pts.length)
      .map((p) => ({ key: p.s.key, label: p.s.label, color: p.s.color, y: sy(p.pts[p.pts.length - 1].y) }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i++) {
      if (items[i].y - items[i - 1].y < 16) items[i].y = items[i - 1].y + 16;
    }
    return items;
  }, [paths, directLabels, series.length]);

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = (e.target as SVGRectElement).getBoundingClientRect();
    const px = e.clientX - rect.left;
    let best = 0;
    let dist = Infinity;
    xs.forEach((x, i) => {
      const d = Math.abs(sx(x) - px);
      if (d < dist) {
        dist = d;
        best = i;
      }
    });
    setHover(best);
  };

  const hx = hover !== null ? xs[hover] : null;
  const rows =
    hx === null
      ? []
      : series
          .map((s) => ({ s, p: s.points.find((p) => p.x === hx) }))
          .filter((r) => r.p)
          .sort((a, b) => (b.p!.y ?? 0) - (a.p!.y ?? 0));

  return (
    <div ref={ref} style={{ position: "relative" }}>
      {legend && series.length > 1 && (
        <div className="legend" style={{ marginBottom: 10 }}>
          {series.map((s) => (
            <span key={s.key}>
              <i className="swatch" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <svg width={width} height={height} role="img" aria-label={yLabel ? `${yLabel} by ${xLabel}` : "line chart"} style={{ display: "block", overflow: "visible" }}>
        <g transform={`translate(${M.left},${M.top})`}>
          {yTicks.map((t) => (
            <g key={t} transform={`translate(0,${sy(t)})`}>
              <line x1={0} x2={iw} stroke="var(--grid)" />
              <text x={-10} dy="0.32em" textAnchor="end" fontSize={11.5} fill="var(--ink-3)" className="tabular">
                {yFormat(t)}
              </text>
            </g>
          ))}
          <line x1={0} x2={iw} y1={ih} y2={ih} stroke="var(--axis)" />
          {xTicks.map((t) => (
            <text key={t} x={sx(t)} y={ih + 20} textAnchor="middle" fontSize={11.5} fill="var(--ink-3)" className="tabular">
              {xFormat(t)}
            </text>
          ))}
          {xLabel && (
            <text x={iw / 2} y={ih + 36} textAnchor="middle" fontSize={12} fill="var(--ink-2)">
              {xLabel}
            </text>
          )}
          {yLabel && (
            <text transform={`translate(${-44},${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--ink-2)">
              {yLabel}
            </text>
          )}
          {paths.map(({ s, band }) => band && <path key={`b-${s.key}`} d={band} fill={s.color} opacity={0.13} />)}
          {paths.map(({ s, line }) => (
            <path key={`l-${s.key}`} d={line} fill="none" stroke={s.color} strokeWidth={s.emphasis ? 3 : 2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {markers &&
            paths.map(({ s, pts }) =>
              pts.map((p) => (
                <circle key={`${s.key}-${p.x}`} cx={sx(p.x)} cy={sy(p.y)} r={s.emphasis ? 4.5 : 3.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
              )),
            )}
          {hx !== null && <line x1={sx(hx)} x2={sx(hx)} y1={0} y2={ih} stroke="var(--axis)" strokeDasharray="3 3" />}
          {labels.map((l) => (
            <text key={l.key} x={iw + 10} y={l.y} dy="0.32em" fontSize={12} fill="var(--ink-2)">
              <tspan fill={l.color}>&#9632; </tspan>
              {l.label}
            </text>
          ))}
          <rect width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </g>
      </svg>
      {hx !== null && rows.length > 0 && (
        <div className="tooltip" style={{ left: Math.min(M.left + sx(hx) + 14, width - 200), top: 30 }}>
          <div className="title">{xFormat(hx)}</div>
          {rows.map(({ s, p }) => (
            <div className="row" key={s.key}>
              <span>
                <i className="swatch" style={{ background: s.color }} />
                {s.label}
              </span>
              <span>{yFormat(p!.y)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const a = Math.floor(Math.log10(Math.max(lo, 1e-4)));
  const b = Math.ceil(Math.log10(Math.max(hi, 1e-4)));
  const mults = b - a <= 2 ? [1, 2, 5] : [1, 3];
  for (let e = a; e <= b; e++) {
    for (const m of mults) {
      const v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  if (out.length < 2) return [lo, hi];
  return out;
}
