"use client";

import { ArrowRight, Play } from "lucide-react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { DailyFraudChart } from "@/components/charts";
import { useLive } from "@/components/live-feed";
import { Card, DecisionBadge, Empty, OutcomeTag, Stat, buttonClass } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

const PIPELINE = [
  { title: "Authorization", body: "card, merchant, amount, time, location" },
  { title: "Streaming features", body: "26 point-in-time aggregates from deques, O(1) per event" },
  { title: "Row + derived features", body: "39 inputs, same NumPy code as training" },
  { title: "LightGBM", body: "gradient-boosted trees, tuned on rolling-origin CV" },
  { title: "Isotonic calibration", body: "scores become trustworthy probabilities" },
  { title: "Cost policy", body: "cheapest of approve / review / decline given p and $" },
];

export default function Overview() {
  const { state, items, control } = useLive();
  const running = state?.running ?? false;
  const daily = useQuery({ queryKey: ["daily"], queryFn: api.daily, refetchInterval: running ? 3000 : 15000 });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models, staleTime: 60_000 });
  const s = state?.stats;
  const alerts = items.filter((i) => i.decision !== "approve").slice(0, 8);

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Overview</h1>
        <p className="mt-1 max-w-[90ch] text-[13px] text-ink-2">
          The deployment period (Jun 21 - Dec 31 2020) is replayed through the production scorer one authorization at a
          time. Ground truth is used only to count outcomes on this page; the model never sees it.
        </p>
      </div>

      <section className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6" aria-label="Running totals">
        <Stat label="Transactions scored" value={s ? fmt.int(s.transactions) : "—"} sub={state && s?.transactions ? `p50 ${fmt.ms(state.latency_p50)} · p99 ${fmt.ms(state.latency_p99)}` : "per-transaction latency"} />
        <Stat
          label="Fraud stopped"
          value={s ? fmt.compactUsd(s.fraud_blocked_amount) : "—"}
          sub={s?.fraud_amount ? `${fmt.pct(s.fraud_blocked_amount / s.fraud_amount)} of ${fmt.compactUsd(s.fraud_amount)} · ${fmt.int(s.frauds)} frauds` : "no fraud seen yet"}
        />
        <Stat label="Fraud missed" value={s ? fmt.compactUsd(s.fraud_missed_amount) : "—"} sub="approved, later charged back" />
        <Stat label="False declines" value={s ? fmt.int(s.false_declines) : "—"} sub="legitimate customers declined" />
        <Stat label="Sent to review" value={s ? fmt.int(s.reviews) : "—"} sub={s?.transactions ? `${fmt.pct(s.reviews / s.transactions, 2)} of transactions` : "analyst cases"} />
        <Stat hero label="Saved vs approving everything" value={s ? fmt.compactUsd(s.saving_vs_approve_all) : "—"} sub="after review and false-decline costs" />
      </section>

      {s && s.transactions === 0 && (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-[13px] text-ink-2">
              Nothing replayed yet. Start the replay and watch decisions arrive, or try the what-if explorer.
            </p>
            <div className="flex gap-2">
              <button className={buttonClass("primary")} onClick={() => control("start", 100)}>
                <Play size={14} /> Start at 100 txn/s
              </button>
              <Link href="/whatif" className={buttonClass("plain")}>
                What-if explorer <ArrowRight size={13} />
              </Link>
            </div>
          </div>
        </Card>
      )}

      <Card
        title="How one transaction is scored"
        subtitle={
          models.data
            ? `Model ${models.data.production.version} · ${models.data.production.trees} trees · ${models.data.production.n_features} features · feature spec ${models.data.production.feature_spec}`
            : "Every step runs per request"
        }
      >
        <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6">
          {PIPELINE.map((step, i) => (
            <li key={step.title} className="relative rounded-lg border border-line bg-surface-2 px-3 py-2.5">
              <div className="text-[11.5px] text-muted">step {i + 1}</div>
              <div className="text-[13px] font-semibold">{step.title}</div>
              <div className="mt-0.5 text-[12px] text-ink-2">{step.body}</div>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-[12.5px] text-muted">
          Feature definitions compile both to PostgreSQL window queries (training) and to this streaming engine (serving);
          the replay check found every served probability identical to the offline one across all 555,719 test
          transactions.
        </p>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card title="Fraud dollars per day" subtitle="Stopped (declined or sent to review) vs missed (approved)">
          {daily.data?.length ? <DailyFraudChart rows={daily.data} /> : <Empty>Appears once the replay has run.</Empty>}
        </Card>
        <Card
          title="Latest alerts"
          action={
            <Link href="/live" className="text-[12.5px] text-ink-2 hover:underline">
              Live stream →
            </Link>
          }
        >
          {alerts.length ? (
            <ul className="grid gap-1">
              {alerts.map((a) => (
                <li key={a.txn_id}>
                  <Link
                    href={`/transaction?id=${a.txn_id}`}
                    className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-accent-wash"
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-[13px]">
                        {fmt.usd2(a.amount)} · {a.category}
                      </span>
                      <span className="text-[11.5px] text-muted">
                        {fmt.time(a.ts)} · {fmt.card(a.card_id)} · p {fmt.prob(a.p_fraud)}
                      </span>
                    </span>
                    <span className="flex items-center gap-2">
                      <DecisionBadge decision={a.decision} />
                      <OutcomeTag outcome={a.outcome} />
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>No alerts yet.</Empty>
          )}
        </Card>
      </div>
    </div>
  );
}
