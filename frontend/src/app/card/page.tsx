"use client";

import { Suspense } from "react";
import { SlidersHorizontal } from "lucide-react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { CardState, FeedTable } from "@/components/feed";
import { Card, Empty, buttonClass } from "@/components/ui";
import { api } from "@/lib/api";

function CardPageInner() {
  const id = useSearchParams().get("id") ?? "";
  const router = useRouter();
  const card = useQuery({ queryKey: ["card", Number(id)], queryFn: () => api.card(id), refetchInterval: 4000 });

  if (card.error) return <Card><Empty>Unknown card.</Empty></Card>;
  if (!card.data) return <Empty>Loading…</Empty>;
  const c = card.data;

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-[12.5px] text-muted">Card profile</div>
          <h1 className="mt-1 text-[20px] font-semibold tracking-tight">
            {c.masked} <span className="font-normal text-ink-2">· {c.city}</span>
          </h1>
        </div>
        <Link href={`/whatif?card=${c.card_id}`} className={buttonClass("plain")}>
          <SlidersHorizontal size={14} /> What-if for this card
        </Link>
      </div>
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Card>
          <CardState cardId={c.card_id} />
        </Card>
        <Card title="Recent transactions on this card" subtitle="From this replay session, newest first">
          {c.recent.length ? (
            <FeedTable items={c.recent} onSelect={(it) => router.push(`/transaction?id=${it.txn_id}`)} maxHeight={520} />
          ) : (
            <Empty>No transactions on this card in the replay yet.</Empty>
          )}
        </Card>
      </div>
    </div>
  );
}

export default function CardPage() {
  return (
    <Suspense fallback={<Empty>Loading…</Empty>}>
      <CardPageInner />
    </Suspense>
  );
}
