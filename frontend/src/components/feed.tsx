"use client";

import clsx from "clsx";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { featureLabel, featureValue } from "@/lib/features";
import { fmt } from "@/lib/format";
import type { FeedItem, Reason } from "@/lib/types";
import { DecisionBadge, Empty, OutcomeTag, ProbabilityMeter, buttonClass } from "./ui";

export function FeedTable({
  items,
  selected,
  onSelect,
  previousTop = Infinity,
  maxHeight = 640,
}: {
  items: FeedItem[];
  selected?: string | null;
  onSelect?: (item: FeedItem) => void;
  previousTop?: number;
  maxHeight?: number;
}) {
  if (!items.length) return <Empty>No transactions yet. Start the replay from the top bar.</Empty>;
  return (
    <div className="overflow-auto rounded-lg" style={{ maxHeight }}>
      <table className="w-full border-collapse text-[13px]">
        <thead className="sticky top-0 z-[1] bg-surface">
          <tr className="text-left text-[12px] text-ink-2">
            {["Time", "Card", "Merchant", "Amount", "P(fraud)", "Decision", "Outcome"].map((h) => (
              <th key={h} className={clsx("border-b border-grid px-2 py-1.5 font-semibold", (h === "Amount" || h === "P(fraud)") && "text-right")}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr
              key={it.txn_id}
              tabIndex={0}
              onClick={() => onSelect?.(it)}
              onKeyDown={(e) => e.key === "Enter" && onSelect?.(it)}
              className={clsx(
                "cursor-pointer border-b border-grid hover:bg-accent-wash",
                selected === it.txn_id && "bg-accent-wash",
                it.seq > previousTop && it.decision !== "approve" && "flash",
              )}
            >
              <td className="tabular px-2 py-1.5 whitespace-nowrap">{fmt.time(it.ts)}</td>
              <td className="px-2 py-1.5 whitespace-nowrap">
                <Link href={`/cards/${it.card_id}`} onClick={(e) => e.stopPropagation()} className="hover:underline">
                  {fmt.card(it.card_id)}
                </Link>
              </td>
              <td className="max-w-[200px] px-2 py-1.5">
                <div className="truncate" title={it.merchant}>
                  {it.merchant}
                </div>
                <div className="text-[11.5px] text-muted">{it.category}</div>
              </td>
              <td className="tabular px-2 py-1.5 text-right whitespace-nowrap">{fmt.usd2(it.amount)}</td>
              <td className="tabular px-2 py-1.5 text-right whitespace-nowrap">
                {fmt.prob(it.p_fraud)}
                <ProbabilityMeter p={it.p_fraud} decision={it.decision} />
              </td>
              <td className="px-2 py-1.5 whitespace-nowrap">
                <DecisionBadge decision={it.decision} />
              </td>
              <td className="px-2 py-1.5">
                <OutcomeTag outcome={it.outcome} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ReasonBars({ reasons }: { reasons: Reason[] }) {
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution)), 1e-9);
  return (
    <div className="grid gap-2.5">
      {reasons.map((r) => (
        <div key={r.feature}>
          <div className="flex justify-between gap-2 text-[12.5px]">
            <span>{featureLabel(r.feature)}</span>
            <span className="tabular whitespace-nowrap text-ink-2">
              {featureValue(r.feature, r.value)} · {r.contribution > 0 ? "+" : ""}
              {r.contribution.toFixed(2)}
            </span>
          </div>
          <div className="mt-1 h-2">
            <div
              className="h-2 rounded-r-[4px]"
              style={{
                width: `${Math.max(2, (Math.abs(r.contribution) / max) * 100)}%`,
                background: r.contribution > 0 ? "var(--diverge-pos)" : "var(--diverge-neg)",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

const CARD_FEATURES = [
  "card__count_1h",
  "card__count_24h",
  "card__sum_amount_24h",
  "card__max_amount_7d",
  "card__distinct_merchant_24h",
  "card__mean_amount_all",
  "card__since_last",
  "card__cb_count_all",
];

export function CardState({ cardId }: { cardId: number }) {
  const card = useQuery({ queryKey: ["card", cardId], queryFn: () => api.card(cardId), refetchInterval: 4000 });
  if (!card.data) return <p className="text-[12.5px] text-muted">{card.error ? "Card not available." : "Loading card…"}</p>;
  const c = card.data;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-[13px] font-semibold">
          <Link href={`/cards/${c.card_id}`} className="hover:underline">
            Card {c.masked}
          </Link>{" "}
          · {c.city}
        </h3>
      </div>
      <p className="mb-2 text-[12px] text-muted">
        {c.age ? `${Math.round(c.age)} years old · ` : ""}
        {c.job}
      </p>
      <h4 className="mt-3 mb-1.5 text-[12.5px] font-semibold text-ink-2">Live card state (streaming feature engine)</h4>
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 text-[12.5px]">
        {CARD_FEATURES.map((f) => (
          <div key={f} className="contents">
            <dt className="text-ink-2">{featureLabel(f)}</dt>
            <dd className="tabular text-right">{featureValue(f, c.features[f])}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function Inspector({ item }: { item: FeedItem | null }) {
  if (!item) return <Empty>Select a transaction to see why it was scored that way.</Empty>;
  return (
    <div>
      <div className="text-[12px] text-muted">
        {fmt.dateTime(item.ts)} UTC · {item.category}
      </div>
      <div className="mt-0.5 mb-3 font-semibold">
        {fmt.usd2(item.amount)} at {item.merchant}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-[34px] leading-none font-semibold">{fmt.prob(item.p_fraud)}</span>
        <span className="text-[12.5px] text-muted">probability of fraud</span>
        <DecisionBadge decision={item.decision} />
        <OutcomeTag outcome={item.outcome} />
      </div>
      <h4 className="mt-4 mb-2 text-[12.5px] font-semibold text-ink-2">Why: largest contributions (log-odds, TreeSHAP)</h4>
      {item.reasons.length ? (
        <ReasonBars reasons={item.reasons} />
      ) : (
        <p className="text-[12.5px] text-muted">
          Approved. Reason codes are computed only for review and decline; open the investigation for the full breakdown.
        </p>
      )}
      <Link href={`/transactions/${item.txn_id}`} className={clsx(buttonClass("plain", true), "mt-4")}>
        Investigate <ArrowRight size={13} />
      </Link>
      <div className="mt-5 border-t border-grid pt-4">
        <CardState cardId={item.card_id} />
      </div>
    </div>
  );
}
