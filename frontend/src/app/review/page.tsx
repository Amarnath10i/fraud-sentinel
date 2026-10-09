"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, Empty, buttonClass } from "@/components/ui";
import { api } from "@/lib/api";
import { featureLabel } from "@/lib/features";
import { fmt } from "@/lib/format";

export default function ReviewPage() {
  const qc = useQueryClient();
  const queue = useQuery({ queryKey: ["queue"], queryFn: api.queue, refetchInterval: 2000 });
  const resolve = useMutation({
    mutationFn: ({ id, fraud }: { id: string; fraud: boolean }) => api.resolve(id, fraud),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["queue"] }),
  });
  const items = queue.data ?? [];
  const total = items.reduce((a, i) => a + i.saving, 0);

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Review queue</h1>
        <p className="mt-1 max-w-[90ch] text-[13px] text-ink-2">
          Cases the policy sent to an analyst, highest expected saving first (the same top-k idea as the capacity
          policy). Confirming fraud posts a chargeback into the online state, so the card&apos;s next transactions are
          scored with that knowledge, exactly as in production.
        </p>
      </div>
      <Card title={`${items.length} open cases`} subtitle={items.length ? `${fmt.usd2(total)} expected saving if all are reviewed` : undefined}>
        {items.length ? (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[13px]">
              <thead className="text-left text-[12px] text-ink-2">
                <tr>
                  {["Time", "Card", "Merchant", "Amount", "P(fraud)", "Expected saving", "Top reason", ""].map((h, i) => (
                    <th key={i} className={`border-b border-grid px-2 py-1.5 font-semibold ${[3, 4, 5].includes(i) ? "text-right" : ""}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {items.map((it) => (
                  <tr key={it.txn_id} className="border-b border-grid">
                    <td className="tabular px-2 py-1.5 whitespace-nowrap">{fmt.dateTime(it.ts).slice(5, 16)}</td>
                    <td className="px-2 py-1.5">
                      <Link href={`/cards/${it.card_id}`} className="hover:underline">
                        {fmt.card(it.card_id)}
                      </Link>
                    </td>
                    <td className="max-w-[220px] px-2 py-1.5">
                      <Link href={`/transactions/${it.txn_id}`} className="block truncate hover:underline">
                        {it.merchant}
                      </Link>
                      <div className="text-[11.5px] text-muted">{it.category}</div>
                    </td>
                    <td className="tabular px-2 py-1.5 text-right">{fmt.usd2(it.amount)}</td>
                    <td className="tabular px-2 py-1.5 text-right">{fmt.prob(it.p_fraud)}</td>
                    <td className="tabular px-2 py-1.5 text-right">{fmt.usd2(it.saving)}</td>
                    <td className="px-2 py-1.5">{it.reasons[0] ? featureLabel(it.reasons[0].feature) : "—"}</td>
                    <td className="px-2 py-1.5">
                      <div className="flex gap-1.5">
                        <button className={buttonClass("danger", true)} disabled={resolve.isPending} onClick={() => resolve.mutate({ id: it.txn_id, fraud: true })}>
                          Fraud
                        </button>
                        <button className={buttonClass("plain", true)} disabled={resolve.isPending} onClick={() => resolve.mutate({ id: it.txn_id, fraud: false })}>
                          Legit
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>Nothing waiting for review. Start the replay from the top bar.</Empty>
        )}
      </Card>
    </div>
  );
}
