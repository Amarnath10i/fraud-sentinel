import clsx from "clsx";
import { AlertTriangle, Check, X } from "lucide-react";
import type { Decision } from "@/lib/types";

export function Card({
  title,
  subtitle,
  action,
  children,
  className,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={clsx("min-w-0 rounded-xl border border-line bg-surface p-4", className)}>
      {(title || action) && (
        <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            {title && <h2 className="text-[14px] font-semibold">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[12.5px] text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  sub,
  hero,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  hero?: boolean;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-[12.5px] text-ink-2">{label}</div>
      <div className={clsx("mt-0.5 font-semibold break-words", hero ? "text-[30px] leading-tight" : "text-[24px]")}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-[12px] text-muted">{sub}</div>}
    </div>
  );
}

const DECISION_STYLE: Record<Decision, { icon: React.ReactNode; dot: string }> = {
  approve: { icon: <Check size={11} strokeWidth={3} />, dot: "bg-good text-white" },
  review: { icon: <AlertTriangle size={10} strokeWidth={3} />, dot: "bg-warning text-[#3a2a00]" },
  decline: { icon: <X size={11} strokeWidth={3} />, dot: "bg-critical text-white" },
};

/** Status always travels with an icon and a label, never colour alone. */
export function DecisionBadge({ decision, size = "sm" }: { decision: Decision; size?: "sm" | "lg" }) {
  const s = DECISION_STYLE[decision];
  return (
    <span className={clsx("inline-flex items-center gap-1.5", size === "lg" ? "text-[15px] font-semibold" : "text-[12.5px]")}>
      <span className={clsx("inline-grid size-4 place-items-center rounded-full", s.dot)} aria-hidden>
        {s.icon}
      </span>
      {decision}
    </span>
  );
}

/** Generic good/warning/critical state with icon and label (e.g. model stage, drift). */
export function StatusPill({ tone, label }: { tone: "good" | "warning" | "critical"; label: string }) {
  const map = { good: "approve", warning: "review", critical: "decline" } as const;
  const s = DECISION_STYLE[map[tone]];
  return (
    <span className="inline-flex items-center gap-1.5 text-[12.5px]">
      <span className={clsx("inline-grid size-4 place-items-center rounded-full", s.dot)} aria-hidden>
        {s.icon}
      </span>
      {label}
    </span>
  );
}

export function OutcomeTag({ outcome }: { outcome: string }) {
  if (outcome === "ok") return <span className="text-[12.5px] text-muted">legit</span>;
  if (outcome === "manual")
    return <span className="rounded-full border border-accent px-2 text-[11.5px] text-ink-2">manual</span>;
  const fraud = outcome.includes("fraud");
  return (
    <span
      className={clsx(
        "whitespace-nowrap rounded-full border px-2 text-[11.5px]",
        fraud ? "border-critical text-critical-ink" : "border-line text-ink-2",
      )}
    >
      {outcome}
    </span>
  );
}

export function ProbabilityMeter({ p, decision }: { p: number; decision: Decision }) {
  const color = decision === "decline" ? "bg-critical" : decision === "review" ? "bg-warning" : "bg-good";
  return (
    <span className="ml-1.5 inline-block h-1.5 w-14 overflow-hidden rounded-full bg-grid align-middle" aria-hidden>
      <span className={clsx("block h-full rounded-full", color)} style={{ width: `${Math.max(3, Math.min(100, p * 100))}%` }} />
    </span>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="px-2 py-10 text-center text-[13px] text-muted">{children}</div>;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex overflow-hidden rounded-lg border border-line">
      {options.map((o) => (
        <button
          key={o.value}
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={clsx(
            "px-3 py-1.5 text-[13px]",
            value === o.value ? "bg-surface-2 font-semibold text-ink" : "bg-surface text-ink-2 hover:bg-surface-2",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export const buttonClass = (variant: "primary" | "plain" | "danger" = "plain", small = false) =>
  clsx(
    "inline-flex items-center gap-1.5 rounded-lg border font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent disabled:opacity-50",
    small ? "px-2.5 py-1 text-[12.5px]" : "px-3 py-1.5 text-[13px]",
    variant === "primary" && "border-transparent bg-accent text-white hover:brightness-110",
    variant === "plain" && "border-line bg-surface text-ink hover:bg-surface-2",
    variant === "danger" && "border-critical bg-surface text-critical-ink hover:bg-surface-2",
  );
