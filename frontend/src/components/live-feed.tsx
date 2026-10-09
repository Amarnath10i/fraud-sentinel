"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { DemoState, FeedItem } from "@/lib/types";

const MAX_ITEMS = 2000;

interface LiveValue {
  state: DemoState | null;
  /** newest first */
  items: FeedItem[];
  /** seq of the newest item before the latest poll, to highlight what just arrived */
  previousTop: number;
  error: string | null;
  control: (action: "start" | "pause" | "reset" | "rate", rate?: number) => Promise<void>;
}

const LiveContext = createContext<LiveValue | null>(null);

/**
 * One poller for the whole app. The backend returns only items newer than
 * `after`, so each poll is small; on the first poll it also returns recent
 * alerts so the alert views are not empty after a page reload.
 */
export function LiveFeedProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<DemoState | null>(null);
  const [items, setItems] = useState<FeedItem[]>([]);
  const [previousTop, setPreviousTop] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const itemsRef = useRef<FeedItem[]>([]);
  const after = useRef(0);
  const running = useRef(false);
  const lastCount = useRef(0);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const v = await api.state(after.current);
        if (!alive) return;
        // someone reset the replay (another tab, the built-in dashboard): start over
        if (v.stats.transactions < lastCount.current) {
          after.current = 0;
          itemsRef.current = [];
          setItems([]);
          lastCount.current = 0;
          timer = setTimeout(tick, 0);
          return;
        }
        const first = after.current === 0;
        lastCount.current = v.stats.transactions;
        if (v.items.length) after.current = v.items[v.items.length - 1].seq;
        const prev = itemsRef.current;
        let next = [...v.items].reverse().concat(prev);
        if (first && v.alerts?.length) {
          const seen = new Set(next.map((i) => i.txn_id));
          next = next.concat(v.alerts.filter((a) => !seen.has(a.txn_id))).sort((a, b) => b.seq - a.seq);
        }
        if (v.items.length || first) {
          itemsRef.current = next.slice(0, MAX_ITEMS);
          setPreviousTop(prev[0]?.seq ?? 0);
          setItems(itemsRef.current);
        }
        setState(v);
        setError(null);
        running.current = v.running;
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
      if (alive) timer = setTimeout(tick, running.current ? 600 : 2000);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  const control = useCallback(async (action: "start" | "pause" | "reset" | "rate", rate?: number) => {
    const r = await api.control(action, rate);
    running.current = r.running;
    if (action === "reset") {
      after.current = 0;
      lastCount.current = 0;
      itemsRef.current = [];
      setItems([]);
    }
    setState((s) => (s ? { ...s, running: r.running, rate: r.rate } : s));
  }, []);

  return (
    <LiveContext.Provider value={{ state, items, previousTop, error, control }}>{children}</LiveContext.Provider>
  );
}

export function useLive(): LiveValue {
  const v = useContext(LiveContext);
  if (!v) throw new Error("useLive must be used inside LiveFeedProvider");
  return v;
}
