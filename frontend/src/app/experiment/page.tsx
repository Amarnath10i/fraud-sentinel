"use client";

import { Suspense } from "react";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useQuery } from "@tanstack/react-query";
import { BacktestChart, IntervalChart } from "@/components/charts";
import { Card, Empty } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import { REPORT_BLURBS } from "@/lib/reports";
import { parseInterval, parseTables, type Table } from "@/lib/tables";

function column(table: Table, header: string) {
  const i = table.headers.findIndex((h) => h.toLowerCase().startsWith(header.toLowerCase()));
  return i;
}

function intervalRows(table: Table, labelCol: number, valueCol: number) {
  return table.rows
    .map((r) => ({ label: r[labelCol], ...parseInterval(r[valueCol] ?? "") }))
    .filter((r): r is { label: string; point: number; low?: number; high?: number } => typeof r.point === "number");
}

/** A chart for the reports where one makes the result obvious at a glance. */
function KeyChart({ name, tables }: { name: string; tables: Table[] }) {
  if (["model_comparison", "imbalance", "tuning"].includes(name) && tables[0]) {
    const t = tables[0];
    const rows = intervalRows(t, 0, column(t, "PR-AUC"));
    return (
      <Card title="PR-AUC on the test period" subtitle="Point estimate and 95% card-level bootstrap interval">
        <IntervalChart rows={rows} />
      </Card>
    );
  }
  if (name === "decisions") {
    const t = tables.find((x) => x.headers[0] === "policy");
    if (!t) return null;
    const rows = intervalRows(t, 0, column(t, "total cost")).filter((r) => r.point < 200_000);
    return (
      <Card title="Total cost of each policy on the test period" subtitle="95% interval; approving everything ($1.13M) is left off the scale">
        <IntervalChart rows={rows} format={(v) => fmt.compactUsd(v)} />
      </Card>
    );
  }
  if (name === "backtest") {
    const t = tables.find((x) => x.headers[0] === "month");
    if (!t) return null;
    const bt = Object.fromEntries(
      t.headers.slice(1).map((policy, j) => [policy, t.rows.map((r) => ({ month: r[0], pr_auc: Number(r[j + 1]) }))]),
    );
    return (
      <Card title="PR-AUC by month" subtitle="Each month scored by the model that policy would have had in production">
        <BacktestChart bt={bt} />
      </Card>
    );
  }
  return null;
}

function ReportPageInner() {
  const name = useSearchParams().get("name") ?? "";
  const report = useQuery({ queryKey: ["report", name], queryFn: () => api.report(name) });
  if (report.error) return <Card><Empty>No report named “{name}”.</Empty></Card>;
  if (!report.data) return <Empty>Loading…</Empty>;
  const tables = parseTables(report.data.markdown);

  return (
    <div className="grid gap-4">
      <div>
        <Link href="/experiments" className="inline-flex items-center gap-1 text-[12.5px] text-ink-2 hover:underline">
          <ArrowLeft size={13} /> All experiments
        </Link>
        {REPORT_BLURBS[name] && <p className="mt-2 max-w-[90ch] text-[13px] text-ink-2">{REPORT_BLURBS[name]}</p>}
      </div>
      <KeyChart name={name} tables={tables} />
      <Card>
        <article className="report">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.data.markdown}</ReactMarkdown>
        </article>
      </Card>
    </div>
  );
}

export default function ReportPage() {
  return (
    <Suspense fallback={<Empty>Loading…</Empty>}>
      <ReportPageInner />
    </Suspense>
  );
}
