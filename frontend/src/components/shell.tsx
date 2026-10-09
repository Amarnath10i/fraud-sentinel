"use client";

import clsx from "clsx";
import {
  Activity,
  BarChart3,
  FlaskConical,
  Gauge,
  Inbox,
  LayoutDashboard,
  Moon,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Sun,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, API_URL } from "@/lib/api";
import { fmt } from "@/lib/format";
import { useLive } from "./live-feed";
import { buttonClass } from "./ui";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/live", label: "Live stream", icon: Activity },
  { href: "/review", label: "Review queue", icon: Inbox },
  { href: "/whatif", label: "What-if explorer", icon: SlidersHorizontal },
  { href: "/models", label: "Model", icon: BarChart3 },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/monitoring", label: "Monitoring", icon: Gauge },
];

function ThemeToggle() {
  const [dark, setDark] = useState(false);
  useEffect(() => setDark(document.documentElement.classList.contains("dark")), []);
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* storage unavailable: the toggle still works for this page */
    }
    setDark(next);
  };
  return (
    <button onClick={toggle} className={buttonClass("plain")} aria-label="Toggle light or dark theme">
      {dark ? <Sun size={15} /> : <Moon size={15} />}
    </button>
  );
}

function ReplayControls() {
  const { state, control } = useLive();
  const running = state?.running ?? false;
  const rate = String(Math.round(state?.rate ?? 25));
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        className={buttonClass("primary")}
        disabled={!state}
        onClick={() => control(running ? "pause" : "start", Number(rate))}
      >
        {running ? <Pause size={14} /> : <Play size={14} />}
        {running ? "Pause" : state && state.progress > 0 ? "Resume" : "Start replay"}
      </button>
      <select
        aria-label="Replay speed"
        value={["5", "25", "100", "500"].includes(rate) ? rate : "25"}
        onChange={(e) => control("rate", Number(e.target.value))}
        className="rounded-lg border border-line bg-surface px-2 py-1.5 text-[13px]"
      >
        {["5", "25", "100", "500"].map((r) => (
          <option key={r} value={r}>
            {r} txn/s
          </option>
        ))}
      </select>
      <button className={buttonClass("plain")} onClick={() => control("reset")} title="Rewind to deploy time">
        <RotateCcw size={14} />
        Reset
      </button>
    </div>
  );
}

function BackendDown({ error }: { error: string }) {
  return (
    <div className="mb-4 rounded-xl border border-critical bg-surface p-4 text-[13px]">
      <div className="font-semibold text-critical-ink">Cannot reach the scoring API at {API_URL}</div>
      <p className="mt-1 text-ink-2">
        Start it from the repository root with <code className="rounded bg-surface-2 px-1">uv run sentinel serve</code>{" "}
        (PostgreSQL running, production model registered), then this page reconnects by itself.
      </p>
      <p className="mt-1 text-muted">{error}</p>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { state, error } = useLive();
  const queue = useQuery({ queryKey: ["queue"], queryFn: api.queue, refetchInterval: 3000 });
  const active = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="border-b border-grid bg-surface lg:sticky lg:top-0 lg:h-screen lg:border-r lg:border-b-0">
        <div className="flex items-center justify-between px-4 py-4">
          <Link href="/" className="font-semibold tracking-tight">
            fraud-sentinel
            <span className="block text-[11.5px] font-normal text-muted">real-time card fraud scoring</span>
          </Link>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-2 pb-2 lg:flex-col lg:overflow-visible">
          {NAV.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              aria-current={active(href) ? "page" : undefined}
              className={clsx(
                "flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px]",
                active(href) ? "bg-accent-wash font-semibold text-ink" : "text-ink-2 hover:bg-surface-2",
              )}
            >
              <Icon size={16} aria-hidden />
              {label}
              {href === "/review" && !!queue.data?.length && (
                <span className="ml-auto rounded-full bg-surface-2 px-1.5 text-[11.5px] text-ink-2">
                  {queue.data.length}
                </span>
              )}
            </Link>
          ))}
        </nav>
        {state && (
          <div className="hidden px-4 pt-4 text-[12px] text-muted lg:block">
            <div>Model</div>
            <div className="font-medium text-ink-2">{state.model}</div>
            <div className="mt-2">Latency p50 / p99</div>
            <div className="tabular font-medium text-ink-2">
              {state.stats.transactions ? `${fmt.ms(state.latency_p50)} / ${fmt.ms(state.latency_p99)}` : "—"}
            </div>
          </div>
        )}
      </aside>

      <div className="min-w-0">
        <header className="sticky top-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-grid bg-page/90 px-4 py-3 backdrop-blur lg:px-6">
          <div className="tabular text-[13px] text-ink-2" title="Replay clock (UTC)">
            {state?.clock ? fmt.dateTime(state.clock) + " UTC" : "—"}
          </div>
          <div className="hidden h-1 w-32 overflow-hidden rounded-full bg-grid sm:block" title="Share of the deployment period replayed">
            <div className="h-full bg-accent" style={{ width: `${((state?.progress ?? 0) * 100).toFixed(2)}%` }} />
          </div>
          <div className="flex-1" />
          <ReplayControls />
          <ThemeToggle />
        </header>
        <main className="mx-auto max-w-[1400px] px-4 py-5 lg:px-6">
          {error && <BackendDown error={error} />}
          {children}
        </main>
      </div>
    </div>
  );
}
