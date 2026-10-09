"use client";

import { useEffect, useMemo, useState } from "react";
import { FeedTable, Inspector } from "@/components/feed";
import { useLive } from "@/components/live-feed";
import { Card, Segmented } from "@/components/ui";
import type { FeedItem } from "@/lib/types";

type Filter = "all" | "flagged" | "fraud";

export default function LivePage() {
  const { items, previousTop } = useLive();
  const [filter, setFilter] = useState<Filter>("all");
  const [follow, setFollow] = useState(true);
  const [selected, setSelected] = useState<FeedItem | null>(null);

  const shown = useMemo(
    () =>
      items
        .filter((i) => (filter === "flagged" ? i.decision !== "approve" : filter === "fraud" ? i.truth === true : true))
        .slice(0, 250),
    [items, filter],
  );

  // follow mode: the inspector jumps to each new alert as it arrives
  const latestAlert = items.find((i) => i.decision !== "approve");
  useEffect(() => {
    if (follow && latestAlert && latestAlert.txn_id !== selected?.txn_id) setSelected(latestAlert);
  }, [follow, latestAlert, selected?.txn_id]);

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Live stream</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          Newest first. Outcomes compare the decision with ground truth, which the scorer never sees.
        </p>
      </div>
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <Segmented
              label="Filter transactions"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: "All" },
                { value: "flagged", label: "Review or decline" },
                { value: "fraud", label: "Actual fraud" },
              ]}
            />
            <label className="inline-flex items-center gap-2 text-[12.5px] text-ink-2">
              <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
              Follow latest alert
            </label>
          </div>
          <FeedTable
            items={shown}
            selected={selected?.txn_id}
            previousTop={previousTop}
            onSelect={(it) => {
              setFollow(false);
              setSelected(it);
            }}
          />
        </Card>
        <Card className="xl:sticky xl:top-20">
          <Inspector item={selected} />
        </Card>
      </div>
    </div>
  );
}
