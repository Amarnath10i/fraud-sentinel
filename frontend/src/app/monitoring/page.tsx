"use client";

import { useQuery } from "@tanstack/react-query";
import { BacktestChart, ExceedanceChart } from "@/components/charts";
import { useLive } from "@/components/live-feed";
import { Card, Empty, Stat, StatusPill } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

export default function MonitoringPage() {
  const { state } = useLive();
  const m = useQuery({ queryKey: ["monitoring"], queryFn: api.monitoring, refetchInterval: 5000 });
  const d = m.data;
  const ratio = d?.alert_ratio;
  const off = ratio ? Math.max(ratio, 1 / ratio) : null;
  const tone = off == null ? null : off < 1.5 ? "good" : off < 2 ? "warning" : "critical";

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Monitoring</h1>
        <p className="mt-1 max-w-[95ch] text-[13px] text-ink-2">
          Fraud labels arrive weeks late as chargebacks, so a live model cannot be judged on accuracy in real time.
          What can be watched immediately is whether its scores still look like they did on the calibration period.
        </p>
      </div>
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Alert volume vs validation"
          value={ratio ? ratio.toFixed(2) + "×" : "—"}
          sub={tone ? <StatusPill tone={tone} label={tone === "good" ? "stable" : tone === "warning" ? "shifting" : "significant drift"} /> : `${fmt.int(d?.live_n ?? 0)} live scores (needs 1,000)`}
        />
        <Stat label="Score PSI (fixed bins)" value={d?.psi != null ? d.psi.toFixed(3) : "—"} sub="barely moves on rare-event scores" />
        <Stat label="Latency p50" value={state?.stats.transactions ? fmt.ms(state.latency_p50) : "—"} sub="full online path" />
        <Stat label="Latency p99" value={state?.stats.transactions ? fmt.ms(state.latency_p99) : "—"} sub="features + model + policy + reasons" />
      </section>
      <div className="grid items-start gap-4 xl:grid-cols-2">
        <Card
          title="Alert volume at each threshold"
          subtitle="Share of transactions scored at or above each probability. Why not PSI alone: 97% of scores are ~0, so PSI weights bins by mass and halving the alert volume moves it by only ~0.01."
        >
          {d ? <ExceedanceChart m={d} /> : <Empty>Loading…</Empty>}
        </Card>
        <Card title="Retraining policies, 18-month backtest" subtitle="PR-AUC by month for each policy (from reports/backtest)">
          {d?.backtest ? <BacktestChart bt={d.backtest} /> : <Empty>Run <code>uv run sentinel backtest</code> to generate it.</Empty>}
        </Card>
      </div>
    </div>
  );
}
