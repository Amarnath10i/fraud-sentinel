"use client";

import { useQuery } from "@tanstack/react-query";
import { ImportanceChart } from "@/components/charts";
import { Card, Empty, Stat, StatusPill } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

export default function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  if (!models.data) return <Empty>{models.error ? "Cannot load the model registry." : "Loading…"}</Empty>;
  const { production: p, registry } = models.data;
  const params = Object.entries(p.meta.params ?? {});

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Model</h1>
        <p className="mt-1 max-w-[90ch] text-[13px] text-ink-2">
          LightGBM with isotonic calibration, trained on Jan 2019 - Jan 2020 with labels as known on deploy day
          (chargebacks reported by then, 60-day maturity for legitimate labels), calibrated on Feb - Apr 2020.
        </p>
      </div>
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Production version" value={<span className="text-[17px]">{p.version}</span>} sub={`feature spec ${p.feature_spec}`} />
        <Stat label="Trees" value={fmt.int(p.trees)} sub={`${p.n_features} input features`} />
        <Stat label="Validation PR-AUC" value={p.meta.metrics.valid_pr_auc?.toFixed(4) ?? "—"} sub="Feb - Apr 2020, labels as of deploy" />
        <Stat label="Label cutoff" value={<span className="text-[17px]">{p.meta.label_cutoff.slice(0, 10)}</span>} sub="nothing after this date was seen" />
      </section>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <Card title="What the trees split on" subtitle="Share of total split gain, top 15 of the features">
          <ImportanceChart rows={p.importance} />
        </Card>
        <div className="grid gap-4">
          <Card title="Hyperparameters" subtitle="Optuna TPE search on rolling-origin time-series CV">
            {params.length ? (
              <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 text-[12.5px]">
                {params.map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-ink-2">{k}</dt>
                    <dd className="tabular text-right">{Number.isInteger(v) ? v : v.toPrecision(3)}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-[12.5px] text-muted">Library defaults.</p>
            )}
          </Card>
          <Card title="Decision costs" subtitle="What each mistake costs; the policy minimizes the expected total">
            <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 text-[12.5px]">
              <dt className="text-ink-2">Approve a fraud</dt>
              <dd className="text-right">its full amount</dd>
              <dt className="text-ink-2">Review a case</dt>
              <dd className="tabular text-right">{fmt.usd2(p.costs.review_cost)} + {fmt.usd2(p.costs.review_friction)} friction</dd>
              <dt className="text-ink-2">Review catches</dt>
              <dd className="tabular text-right">{fmt.pct(p.costs.review_catch_rate, 0)} of frauds</dd>
              <dt className="text-ink-2">Decline a legitimate customer</dt>
              <dd className="tabular text-right">
                {fmt.usd2(p.costs.decline_fixed)} + {fmt.pct(p.costs.decline_margin, 0)} of amount
              </dd>
            </dl>
          </Card>
        </div>
      </div>

      <Card title="Registry" subtitle="Every trained model; a partial unique index in PostgreSQL allows only one in production">
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead className="text-left text-[12px] text-ink-2">
              <tr>
                {["Version", "Created (UTC)", "Stage", "Training window", "Validation PR-AUC", "Trees / leaves"].map((h) => (
                  <th key={h} className="border-b border-grid px-2 py-1.5 font-semibold">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {registry.map((r) => (
                <tr key={r.version} className="border-b border-grid">
                  <td className="px-2 py-1.5 font-medium">{r.version}</td>
                  <td className="tabular px-2 py-1.5">{r.created_at.slice(0, 19).replace("T", " ")}</td>
                  <td className="px-2 py-1.5">
                    {r.stage === "production" ? (
                      <StatusPill tone="good" label="production" />
                    ) : (
                      <span className="text-muted">{r.stage}</span>
                    )}
                  </td>
                  <td className="tabular px-2 py-1.5">
                    {r.train_start.slice(0, 10)} → {r.train_end.slice(0, 10)}
                  </td>
                  <td className="tabular px-2 py-1.5">{r.metrics.valid_pr_auc?.toFixed(4) ?? "—"}</td>
                  <td className="tabular px-2 py-1.5">{r.params.num_leaves ? `${r.params.num_leaves} leaves` : "defaults"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
