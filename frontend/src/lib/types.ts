// Shapes returned by the FastAPI backend (src/sentinel/serve).

export type Decision = "approve" | "review" | "decline";

export interface Reason {
  feature: string;
  value: number | null;
  contribution: number;
}

export interface FeedItem {
  seq: number;
  txn_id: string;
  ts: number;
  card_id: number;
  merchant: string;
  merchant_id: number;
  category: string;
  amount: number;
  p_fraud: number;
  decision: Decision;
  reasons: Reason[];
  truth: boolean | null; // ground truth, display only; null for manual transactions
  manual: boolean;
  latency_ms: number;
  outcome: string;
}

export interface Stats {
  transactions: number;
  frauds: number;
  fraud_amount: number;
  fraud_blocked_amount: number;
  fraud_missed_amount: number;
  reviews: number;
  declines: number;
  false_declines: number;
  chargebacks: number;
  cost: number;
  manual: number;
  saving_vs_approve_all: number;
}

export interface DemoState {
  running: boolean;
  rate: number;
  clock: number | null;
  progress: number;
  stats: Stats;
  latency_p50: number;
  latency_p99: number;
  items: FeedItem[];
  alerts: FeedItem[];
  model: string;
}

export interface CardView {
  card_id: number;
  masked: string;
  city: string;
  job: string;
  age: number | null;
  features: Record<string, number | null>;
  recent: FeedItem[];
}

export interface Explanation {
  base_value: number;
  margin: number;
  contributions: Reason[];
}

export type ExpectedCost = Record<Decision, number>;

export interface TransactionDetail extends FeedItem {
  in_queue: boolean;
  expected_cost: ExpectedCost;
  explanation: Explanation | null;
}

export interface WhatIfResult {
  p_fraud: number;
  decision: Decision;
  expected_cost: ExpectedCost;
  explanation: Explanation;
  clock: number | null;
}

export interface QueueItem extends FeedItem {
  saving: number;
}

export interface Options {
  cards: { card_id: number; label: string }[];
  merchants: { merchant_id: number; name: string }[];
  categories: string[];
}

export interface DailyRow {
  date: string;
  transactions: number;
  frauds: number;
  fraud_amount: number;
  fraud_blocked_amount: number;
  fraud_missed_amount: number;
  reviews: number;
  declines: number;
  false_declines: number;
  cost: number;
  saving_vs_approve_all: number;
}

export interface Monitoring {
  live_n: number;
  psi: number | null;
  alert_ratio?: number;
  thresholds: number[];
  reference?: number[];
  live?: number[];
  backtest?: Record<string, { month: string; pr_auc: number }[]>;
  registry: { version: string; created_at: string; stage: string; metrics: Record<string, number> }[];
}

export interface RegistryRow {
  version: string;
  created_at: string;
  stage: "production" | "candidate" | "archived";
  algorithm: string;
  feature_set: string;
  train_start: string;
  train_end: string;
  label_cutoff: string;
  params: Record<string, number>;
  metrics: Record<string, number>;
}

export interface ModelsView {
  production: {
    version: string;
    feature_spec: string;
    n_features: number;
    trees: number;
    costs: Record<string, number>;
    meta: {
      train_start: string;
      train_end: string;
      label_cutoff: string;
      params: Record<string, number>;
      best_iteration: number;
      metrics: Record<string, number>;
    };
    importance: { feature: string; gain_share: number }[];
  };
  registry: RegistryRow[];
}

export interface ReportMeta {
  name: string;
  title: string;
}

export interface Report extends ReportMeta {
  markdown: string;
}
