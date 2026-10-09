"use client";

import { Suspense } from "react";
import { SlidersHorizontal } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CostBars, Waterfall } from "@/components/charts";
import { CardState } from "@/components/feed";
import { Card, DecisionBadge, Empty, OutcomeTag, buttonClass } from "@/components/ui";
import { api } from "@/lib/api";
import { featureLabel, featureValue } from "@/lib/features";
import { fmt } from "@/lib/format";

function TransactionPageInner() {
  const id = useSearchParams().get("id") ?? "";
  const qc = useQueryClient();
  const txn = useQuery({ queryKey: ["txn", id], queryFn: () => api.transaction(id) });
  const resolve = useMutation({
    mutationFn: (fraud: boolean) => api.resolve(id, fraud),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["txn", id] });
      qc.invalidateQueries({ queryKey: ["queue"] });
    },
  });

  if (txn.isLoading) return <Empty>Loading…</Empty>;
  if (!txn.data)
    return (
      <Card>
        <Empty>
          This transaction is not in the recent replay window (the server keeps the latest 50,000). Pick one from the{" "}
          <Link className="underline" href="/live">
            live stream
          </Link>
          .
        </Empty>
      </Card>
    );

  const t = txn.data;
  const hour = new Date(t.ts * 1000).getUTCHours();
  const whatif = `/whatif?card=${t.card_id}&merchant=${t.merchant_id}&category=${t.category}&amount=${t.amount}&hour=${hour}`;

  return (
    <div className="grid gap-4">
      <div>
        <div className="text-[12.5px] text-muted">
          Transaction {t.txn_id} · {fmt.dateTime(t.ts)} UTC
        </div>
        <h1 className="mt-1 text-[20px] font-semibold tracking-tight">
          {fmt.usd2(t.amount)} at {t.merchant} <span className="font-normal text-ink-2">· {t.category}</span>
        </h1>
      </div>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="grid gap-4">
          <Card
            title="Why the model scored it this way"
            subtitle="TreeSHAP contributions in log-odds, largest first. They add up exactly from the base rate to this transaction's score; calibration then maps that score to the probability shown."
          >
            {t.explanation ? <Waterfall explanation={t.explanation} /> : <Empty>No explanation stored for this transaction.</Empty>}
          </Card>

          {t.explanation && (
            <Card title="Every model input" subtitle={`${t.explanation.contributions.length} features, sorted by size of effect`}>
              <div className="max-h-[420px] overflow-auto">
                <table className="w-full text-[12.5px]">
                  <thead className="sticky top-0 bg-surface text-left text-ink-2">
                    <tr>
                      <th className="border-b border-grid px-2 py-1.5 font-semibold">Feature</th>
                      <th className="border-b border-grid px-2 py-1.5 text-right font-semibold">Value</th>
                      <th className="border-b border-grid px-2 py-1.5 text-right font-semibold">Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {t.explanation.contributions.map((c) => (
                      <tr key={c.feature} className="border-b border-grid">
                        <td className="px-2 py-1">{featureLabel(c.feature)}</td>
                        <td className="tabular px-2 py-1 text-right">{featureValue(c.feature, c.value)}</td>
                        <td className="tabular px-2 py-1 text-right" style={{ color: c.contribution > 0.005 ? "var(--critical-ink)" : undefined }}>
                          {c.contribution > 0 ? "+" : ""}
                          {c.contribution.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>

        <div className="grid gap-4 xl:sticky xl:top-20">
          <Card>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="text-[38px] leading-none font-semibold">{fmt.prob(t.p_fraud)}</span>
              <span className="text-[12.5px] text-muted">calibrated probability of fraud</span>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <DecisionBadge decision={t.decision} size="lg" />
              <OutcomeTag outcome={t.outcome} />
              <span className="text-[12px] text-muted">scored in {fmt.ms(t.latency_ms)}</span>
            </div>
            <h3 className="mt-5 mb-2 text-[12.5px] font-semibold text-ink-2">Expected cost of each action</h3>
            <CostBars expected={t.expected_cost} chosen={t.decision} />
            <p className="mt-2 text-[12px] text-muted">
              Approving risks p × amount; a review costs analyst time and stops 95% of fraud; a decline risks losing a
              legitimate customer. The policy takes the cheapest, so the same probability can lead to different decisions
              for different amounts.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {t.in_queue && (
                <>
                  <button className={buttonClass("danger", true)} disabled={resolve.isPending} onClick={() => resolve.mutate(true)}>
                    Confirm fraud
                  </button>
                  <button className={buttonClass("plain", true)} disabled={resolve.isPending} onClick={() => resolve.mutate(false)}>
                    Mark legitimate
                  </button>
                </>
              )}
              <Link href={whatif} className={buttonClass("plain", true)}>
                <SlidersHorizontal size={13} /> Open in what-if
              </Link>
            </div>
            {resolve.isSuccess && (
              <p className="mt-2 text-[12px] text-ink-2">
                {resolve.variables ? "Chargeback sent into the online state; the card's next transactions see it." : "Removed from the review queue."}
              </p>
            )}
          </Card>
          <Card>
            <CardState cardId={t.card_id} />
          </Card>
        </div>
      </div>
    </div>
  );
}

export default function TransactionPage() {
  return (
    <Suspense fallback={<Empty>Loading…</Empty>}>
      <TransactionPageInner />
    </Suspense>
  );
}
