"use client";

import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { Shuffle } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CostBars, Waterfall } from "@/components/charts";
import { ReasonBars } from "@/components/feed";
import { Card, DecisionBadge, Empty, buttonClass } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import type { Decision } from "@/lib/types";

const PRESETS = [
  { label: "Coffee, 9am, near home", category: "food_dining", amount: 6.5, hour: 9, distance: 2 },
  { label: "Groceries, 6pm", category: "grocery_pos", amount: 84, hour: 18, distance: 5 },
  { label: "Online electronics, 2am", category: "shopping_net", amount: 1240, hour: 2, distance: 5 },
  { label: "Online, 11pm, 800 km away", category: "misc_net", amount: 890, hour: 23, distance: 800 },
];
const SWEEP = [5, 15, 40, 80, 150, 300, 500, 800, 1200, 2000, 3500];
const DECISION_COLOR: Record<Decision, string> = {
  approve: "var(--good)",
  review: "var(--warning)",
  decline: "var(--critical)",
};

// amount slider is logarithmic: $1 .. $10,000
const toAmount = (s: number) => Math.round(Math.pow(10, (s / 1000) * 4) * 100) / 100;
const toSlider = (a: number) => Math.round((Math.log10(Math.max(a, 1)) / 4) * 1000);

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function WhatIf() {
  const params = useSearchParams();
  const router = useRouter();
  const options = useQuery({ queryKey: ["options"], queryFn: api.options, staleTime: Infinity });
  const [card, setCard] = useState<number | null>(params.get("card") ? Number(params.get("card")) : null);
  const [merchant, setMerchant] = useState<number | null>(params.get("merchant") ? Number(params.get("merchant")) : null);
  const [category, setCategory] = useState(params.get("category") ?? "shopping_net");
  const [amount, setAmount] = useState(params.get("amount") ? Number(params.get("amount")) : 120);
  const [hour, setHour] = useState<number>(params.get("hour") ? Number(params.get("hour")) : 14);
  const [distance, setDistance] = useState(5);

  // fill defaults once the option lists arrive
  useEffect(() => {
    if (!options.data) return;
    if (card == null) setCard(options.data.cards[0]?.card_id ?? null);
    if (merchant == null) setMerchant(options.data.merchants[0]?.merchant_id ?? null);
  }, [options.data, card, merchant]);

  const inputs = useDebounced({ card, merchant, category, amount, hour, distance }, 180);
  const ready = inputs.card != null && inputs.merchant != null;
  const body = (amt: number) => ({
    card_id: inputs.card!,
    merchant_id: inputs.merchant!,
    category: inputs.category,
    amount: amt,
    hour: inputs.hour,
    distance_km: inputs.distance,
  });

  const result = useQuery({
    queryKey: ["whatif", inputs],
    queryFn: () => api.whatif(body(inputs.amount)),
    enabled: ready,
    placeholderData: keepPreviousData,
  });
  const sweep = useQuery({
    queryKey: ["sweep", { ...inputs, amount: 0 }],
    queryFn: async () => {
      const rs = await Promise.all(SWEEP.map((a) => api.whatif(body(a))));
      return SWEEP.map((a, i) => ({ amount: a, p: Math.max(rs[i].p_fraud, 1e-5), decision: rs[i].decision }));
    },
    enabled: ready,
    placeholderData: keepPreviousData,
  });
  const commit = useMutation({
    mutationFn: () =>
      api.scoreManual({ card_id: card!, merchant_id: merchant!, category, amount, away: distance > 100, distance_km: distance }),
    onSuccess: (item) => router.push(`/transaction?id=${item.txn_id}`),
  });

  const o = options.data;
  if (options.error) return <Card><Empty>Cannot load cards and merchants from the API.</Empty></Card>;
  const r = result.data;

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">What-if explorer</h1>
        <p className="mt-1 max-w-[95ch] text-[13px] text-ink-2">
          Change one thing about a transaction and watch the score respond. Each answer reads the card&apos;s live state
          at the replay clock (its velocity, habits and chargeback history) without recording anything, so you can probe
          the model freely. Commit it to send it through the real scoring path.
        </p>
      </div>
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)]">
        <Card title="Transaction">
          <div className="mb-4 flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                className={buttonClass("plain", true)}
                onClick={() => {
                  setCategory(p.category);
                  setAmount(p.amount);
                  setHour(p.hour);
                  setDistance(p.distance);
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="grid gap-4 text-[12.5px]">
            <label className="grid gap-1 text-ink-2">
              Card
              <span className="flex gap-2">
                <select className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-2 py-1.5 text-ink" value={card ?? ""} onChange={(e) => setCard(Number(e.target.value))}>
                  {o?.cards.map((c) => (
                    <option key={c.card_id} value={c.card_id}>
                      {c.label}
                    </option>
                  ))}
                </select>
                <button
                  className={buttonClass("plain", true)}
                  title="Random card"
                  onClick={() => o && setCard(o.cards[Math.floor(Math.random() * o.cards.length)].card_id)}
                >
                  <Shuffle size={13} />
                </button>
              </span>
            </label>
            <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <label className="grid gap-1 text-ink-2">
                Category
                <select className="w-full min-w-0 rounded-lg border border-line bg-surface px-2 py-1.5 text-ink" value={category} onChange={(e) => setCategory(e.target.value)}>
                  {o?.categories.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label className="grid gap-1 text-ink-2">
                Merchant
                <select className="w-full min-w-0 rounded-lg border border-line bg-surface px-2 py-1.5 text-ink" value={merchant ?? ""} onChange={(e) => setMerchant(Number(e.target.value))}>
                  {o?.merchants.map((m) => (
                    <option key={m.merchant_id} value={m.merchant_id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <Slider label="Amount" value={fmt.usd2(amount)}>
              <input type="range" min={0} max={1000} value={toSlider(amount)} onChange={(e) => setAmount(toAmount(Number(e.target.value)))} className="w-full" aria-label="Amount" />
            </Slider>
            <Slider label="Hour of day" value={fmt.hour(hour)}>
              <input type="range" min={0} max={23} value={hour} onChange={(e) => setHour(Number(e.target.value))} className="w-full" aria-label="Hour of day" />
            </Slider>
            <Slider label="Distance from the cardholder's home" value={`${distance} km`}>
              <input type="range" min={0} max={2000} step={5} value={distance} onChange={(e) => setDistance(Number(e.target.value))} className="w-full" aria-label="Distance from home" />
            </Slider>
            <div className="flex flex-wrap items-center gap-2 border-t border-grid pt-3">
              <button className={buttonClass("primary")} disabled={!ready || commit.isPending} onClick={() => commit.mutate()}>
                Commit as a real transaction
              </button>
              <span className="text-[12px] text-muted">Scored at the replay clock; it then counts in the card&apos;s history.</span>
            </div>
            {commit.error && <p className="text-[12px] text-critical-ink">{String(commit.error.message)}</p>}
          </div>
        </Card>

        <div className="grid gap-4">
          <Card>
            {r ? (
              <div className={result.isFetching ? "opacity-70 transition-opacity" : "transition-opacity"}>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                  <span className="text-[44px] leading-none font-semibold">{fmt.prob(r.p_fraud)}</span>
                  <div>
                    <DecisionBadge decision={r.decision} size="lg" />
                    <div className="text-[12px] text-muted">calibrated probability of fraud</div>
                  </div>
                </div>
                <div className="mt-4 grid gap-5 md:grid-cols-2">
                  <div>
                    <h3 className="mb-2 text-[12.5px] font-semibold text-ink-2">Expected cost of each action</h3>
                    <CostBars expected={r.expected_cost} chosen={r.decision} />
                  </div>
                  <div>
                    <h3 className="mb-2 text-[12.5px] font-semibold text-ink-2">Biggest effects (log-odds)</h3>
                    <ReasonBars reasons={r.explanation.contributions.slice(0, 5)} />
                  </div>
                </div>
              </div>
            ) : (
              <Empty>{ready ? "Scoring…" : "Loading cards…"}</Empty>
            )}
          </Card>
          <Card
            title="How the decision changes with the amount"
            subtitle="Same card, merchant, hour and distance; only the amount varies. Dots show the policy's decision."
          >
            {sweep.data ? <SweepChart data={sweep.data} /> : <Empty>…</Empty>}
          </Card>
          {r && (
            <Card title="Full breakdown" subtitle="From the model's base rate to this transaction, largest effects first">
              <Waterfall explanation={r.explanation} top={8} />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function Slider({ label, value, children }: { label: string; value: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1">
      <div className="flex justify-between text-ink-2">
        <span>{label}</span>
        <span className="tabular font-semibold text-ink">{value}</span>
      </div>
      {children}
    </div>
  );
}

function SweepChart({ data }: { data: { amount: number; p: number; decision: Decision }[] }) {
  return (
    <>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="amount" scale="log" type="number" domain={[4, 4000]} ticks={[5, 15, 40, 150, 500, 1200, 3500]} tickFormatter={(v: number) => "$" + v} stroke="var(--axis)" tick={{ fill: "var(--muted)", fontSize: 11 }} tickLine={false} />
          <YAxis scale="log" domain={[1e-5, 1]} allowDataOverflow ticks={[1e-4, 1e-3, 1e-2, 1e-1, 1]} tickFormatter={(v: number) => (v >= 0.01 ? (v * 100).toFixed(0) : (v * 100).toPrecision(1)) + "%"} stroke="var(--axis)" tick={{ fill: "var(--muted)", fontSize: 11 }} tickLine={false} width={48} />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12.5 }}
            labelFormatter={(v: number) => fmt.usd2(v)}
            formatter={(v: number, _n, item) => [`${fmt.prob(v)} → ${(item.payload as { decision: string }).decision}`, "p(fraud)"]}
          />
          <Line
            dataKey="p"
            stroke="var(--axis)"
            strokeWidth={2}
            isAnimationActive={false}
            dot={(props: { cx?: number; cy?: number; payload?: { decision: Decision }; index?: number }) => (
              <circle key={props.index} cx={props.cx} cy={props.cy} r={5} fill={DECISION_COLOR[props.payload!.decision]} stroke="var(--surface)" strokeWidth={2} />
            )}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex flex-wrap gap-4 text-[12px] text-ink-2">
        {(["approve", "review", "decline"] as Decision[]).map((d) => (
          <DecisionBadge key={d} decision={d} />
        ))}
      </div>
    </>
  );
}

export default function WhatIfPage() {
  return (
    <Suspense fallback={<Empty>Loading…</Empty>}>
      <WhatIf />
    </Suspense>
  );
}
