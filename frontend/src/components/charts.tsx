"use client";

import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { featureLabel, featureValue } from "@/lib/features";
import { fmt } from "@/lib/format";
import type { DailyRow, Decision, ExpectedCost, Explanation, Monitoring } from "@/lib/types";

/** Width of a container in CSS pixels, so SVG text is drawn at its real size, not scaled. */
function useWidth<T extends HTMLElement>(fallback = 640): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(300, Math.round(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

const AXIS = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 11 }, tickLine: false };
const GRID = { stroke: "var(--grid)", vertical: false };
const TOOLTIP = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    fontSize: 12.5,
    color: "var(--ink)",
  },
  labelStyle: { color: "var(--ink-2)", marginBottom: 4 },
  itemStyle: { color: "var(--ink)", padding: 0 },
  cursor: { fill: "var(--accent-wash)", stroke: "var(--axis)" },
};
const LEGEND = {
  wrapperStyle: { fontSize: 12.5 },
  iconSize: 10,
  formatter: (value: string) => <span style={{ color: "var(--ink-2)" }}>{value}</span>,
};
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

/** Fraud dollars stopped vs missed per replayed day. */
export function DailyFraudChart({ rows }: { rows: DailyRow[] }) {
  const data = rows.map((r) => ({
    date: r.date.slice(5),
    stopped: Math.round(r.fraud_blocked_amount),
    missed: Math.round(r.fraud_missed_amount),
  }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 4, bottom: 0 }} barCategoryGap="20%">
        <CartesianGrid {...GRID} />
        <XAxis dataKey="date" {...AXIS} minTickGap={16} />
        <YAxis {...AXIS} width={56} tickFormatter={(v) => fmt.compactUsd(v)} />
        <Tooltip {...TOOLTIP} formatter={(v: number, name: string) => [fmt.usd(v), name]} />
        <Legend {...LEGEND} />
        <Bar dataKey="stopped" name="Fraud $ stopped" stackId="a" fill="var(--series-1)" maxBarSize={24} />
        <Bar dataKey="missed" name="Fraud $ missed" stackId="a" fill="var(--series-2)" maxBarSize={24} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Share of transactions scored at or above each threshold: the alert volume per threshold. */
export function ExceedanceChart({ m }: { m: Monitoring }) {
  if (!m.reference) return null;
  const floor = 1e-5;
  const data = m.thresholds.map((t, i) => ({
    t: "≥" + (t >= 0.01 ? (t * 100).toFixed(0) : (t * 100).toPrecision(1)) + "%",
    reference: Math.max(m.reference![i], floor),
    live: m.live ? Math.max(m.live[i], floor) : undefined,
  }));
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="t" {...AXIS} />
        <YAxis
          {...AXIS}
          width={52}
          scale="log"
          domain={["auto", "auto"]}
          allowDataOverflow
          tickFormatter={(v: number) => (v * 100).toPrecision(1) + "%"}
        />
        <Tooltip {...TOOLTIP} formatter={(v: number, name: string) => [(v * 100).toPrecision(3) + "%", name]} />
        <Legend {...LEGEND} />
        <Line dataKey="reference" name="Validation (Feb-Apr 2020)" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, strokeWidth: 2, stroke: "var(--surface)", fill: "var(--series-1)" }} isAnimationActive={false} />
        {m.live && (
          <Line dataKey="live" name="Live replay" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 4, strokeWidth: 2, stroke: "var(--surface)", fill: "var(--series-2)" }} isAnimationActive={false} />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

const POLICIES = ["static", "monthly expanding", "monthly sliding 6m", "drift-triggered", "static GBDT + online FTRL"];

export function BacktestChart({ bt }: { bt: Record<string, { month: string; pr_auc: number }[]> }) {
  const pols = POLICIES.filter((p) => bt[p]);
  const months = bt[pols[0]]?.map((r) => r.month) ?? [];
  const data = months.map((month, i) => Object.fromEntries([["month", month], ...pols.map((p) => [p, bt[p][i].pr_auc])]));
  return (
    <ResponsiveContainer width="100%" height={270}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="month" {...AXIS} minTickGap={24} />
        <YAxis {...AXIS} width={44} domain={[0.94, 1]} tickFormatter={(v: number) => v.toFixed(2)} />
        <Tooltip {...TOOLTIP} formatter={(v: number, name: string) => [v.toFixed(4), name]} />
        <Legend {...LEGEND} />
        {pols.map((p, i) => (
          <Line key={p} dataKey={p} stroke={SERIES[i]} strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ImportanceChart({ rows, top = 15 }: { rows: { feature: string; gain_share: number }[]; top?: number }) {
  const data = rows.slice(0, top).map((r) => ({ name: featureLabel(r.feature), share: r.gain_share }));
  return (
    <ResponsiveContainer width="100%" height={top * 26 + 20}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 40, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" {...AXIS} tickFormatter={(v: number) => fmt.pct(v, 0)} />
        <YAxis type="category" dataKey="name" {...AXIS} width={210} interval={0} />
        <Tooltip {...TOOLTIP} formatter={(v: number) => [fmt.pct(v, 1), "share of split gain"]} />
        <Bar dataKey="share" fill="var(--series-1)" barSize={14} radius={[0, 4, 4, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * TreeSHAP waterfall: start at the model's base log-odds, add each feature's
 * contribution (largest first), end at this transaction's margin. Red pushes
 * towards fraud, blue towards legitimate.
 */
export function Waterfall({ explanation, top = 10 }: { explanation: Explanation; top?: number }) {
  const shown = explanation.contributions.slice(0, top);
  const rest = explanation.contributions.slice(top).reduce((a, c) => a + c.contribution, 0);
  const steps = [...shown.map((c) => ({ label: featureLabel(c.feature), detail: featureValue(c.feature, c.value), delta: c.contribution }))];
  if (Math.abs(rest) > 1e-9) steps.push({ label: `${explanation.contributions.length - top} other features`, detail: "", delta: rest });

  let run = explanation.base_value;
  const bars = steps.map((s) => {
    const from = run;
    run += s.delta;
    return { ...s, from, to: run };
  });
  const values = [explanation.base_value, ...bars.map((b) => b.to)];
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo) * 0.05 || 1;
  const [ref, W] = useWidth<HTMLDivElement>();
  const L = Math.min(240, Math.round(W * 0.4)), R = 64, rowH = 32, H = (bars.length + 2) * rowH + 24;
  const x = (v: number) => L + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - L - R);
  const prob = (z: number) => 1 / (1 + Math.exp(-z));

  return (
    <div ref={ref}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" role="img" aria-label="Contribution of each feature to the fraud score">
      {[lo, (lo + hi) / 2, hi].map((v, i) => (
        <g key={i}>
          <line x1={x(v)} x2={x(v)} y1={4} y2={H - 22} stroke="var(--grid)" />
          <text x={x(v)} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--muted)">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      <Row y={rowH * 0 + 8} label="Base rate (average transaction)" detail={`p ≈ ${fmt.prob(prob(explanation.base_value))}`}>
        <line x1={x(explanation.base_value)} x2={x(explanation.base_value)} y1={2} y2={rowH - 6} stroke="var(--ink-2)" strokeWidth={2} />
      </Row>
      {bars.map((b, i) => {
        const left = x(Math.min(b.from, b.to));
        const width = Math.max(2, Math.abs(x(b.to) - x(b.from)));
        const up = b.delta > 0;
        return (
          <Row key={i} y={rowH * (i + 1) + 8} label={b.label} detail={b.detail}>
            <rect x={left} y={6} width={width} height={rowH - 16} rx={3} fill={up ? "var(--diverge-pos)" : "var(--diverge-neg)"} />
            <text x={Math.max(x(b.from), x(b.to)) + 4} y={rowH / 2 + 2} fontSize={11} fill="var(--ink-2)">
              {(up ? "+" : "") + b.delta.toFixed(2)}
            </text>
          </Row>
        );
      })}
      <Row y={rowH * (bars.length + 1) + 8} label="This transaction (raw model)" detail={`p ≈ ${fmt.prob(prob(explanation.margin))}`}>
        <line x1={x(explanation.margin)} x2={x(explanation.margin)} y1={2} y2={rowH - 6} stroke="var(--ink)" strokeWidth={2} />
      </Row>
      <text x={W - R + 4} y={H - 6} fontSize={11} fill="var(--muted)">
        log-odds
      </text>
    </svg>
    </div>
  );
}

function Row({ y, label, detail, children }: { y: number; label: string; detail: string; children: React.ReactNode }) {
  return (
    <g transform={`translate(0, ${y})`}>
      <text x={0} y={12} fontSize={12} fill="var(--ink)">
        {label.length > 34 ? label.slice(0, 33) + "…" : label}
      </text>
      <text x={0} y={25} fontSize={11} fill="var(--muted)">
        {detail}
      </text>
      {children}
    </g>
  );
}

/** Expected cost of each action given p and amount; the policy picks the smallest. */
export function CostBars({ expected, chosen }: { expected: ExpectedCost; chosen: Decision }) {
  const max = Math.max(...Object.values(expected), 1e-9);
  return (
    <div className="grid gap-2">
      {(["approve", "review", "decline"] as Decision[]).map((d) => (
        <div key={d} className="grid grid-cols-[80px_minmax(0,1fr)_76px] items-center gap-2 text-[12.5px]">
          <span className={d === chosen ? "font-semibold text-ink" : "text-ink-2"}>{d}</span>
          <span className="h-2.5 overflow-hidden rounded-r-[4px]">
            <span
              className="block h-full rounded-r-[4px]"
              style={{
                width: `${Math.max(1, (expected[d] / max) * 100)}%`,
                background: d === chosen ? "var(--accent)" : "var(--axis)",
              }}
            />
          </span>
          <span className="tabular text-right text-ink-2">{fmt.usd2(expected[d])}</span>
        </div>
      ))}
    </div>
  );
}

/** Point estimate with its bootstrap interval, one row per model or strategy. */
export function IntervalChart({
  rows,
  format = (v: number) => v.toFixed(3),
}: {
  rows: { label: string; point: number; low?: number; high?: number }[];
  format?: (v: number) => string;
}) {
  const vals = rows.flatMap((r) => [r.point, r.low ?? r.point, r.high ?? r.point]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.06 || 0.01;
  const [ref, W] = useWidth<HTMLDivElement>();
  const L = Math.min(300, Math.round(W * 0.42)), R = 72, rowH = 30, H = rows.length * rowH + 26;
  const x = (v: number) => L + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - L - R);
  return (
    <div ref={ref}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" role="img" aria-label="Estimates with 95% intervals">
      {[lo, (lo + hi) / 2, hi].map((v, i) => (
        <g key={i}>
          <line x1={x(v)} x2={x(v)} y1={0} y2={H - 22} stroke="var(--grid)" />
          <text x={x(v)} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--muted)">
            {format(v)}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = i * rowH + rowH / 2;
        return (
          <g key={r.label}>
            <title>{`${r.label}: ${format(r.point)}${r.low != null ? ` (${format(r.low)} to ${format(r.high!)})` : ""}`}</title>
            <text x={0} y={y + 4} fontSize={12} fill="var(--ink)">
              {r.label.length > Math.floor(L / 7) ? r.label.slice(0, Math.floor(L / 7) - 1) + "…" : r.label}
            </text>
            {r.low != null && <line x1={x(r.low)} x2={x(r.high!)} y1={y} y2={y} stroke="var(--series-1)" strokeWidth={2} strokeLinecap="round" />}
            <circle cx={x(r.point)} cy={y} r={5} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
            <text x={W - R + 6} y={y + 4} fontSize={11} fill="var(--ink-2)">
              {format(r.point)}
            </text>
          </g>
        );
      })}
    </svg>
    </div>
  );
}
