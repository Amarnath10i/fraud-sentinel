import type {
  CardView,
  DailyRow,
  DemoState,
  ModelsView,
  Monitoring,
  Options,
  QueueItem,
  Report,
  ReportMeta,
  TransactionDetail,
  WhatIfResult,
  FeedItem,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_URL + path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (res.status === 204 ? null : await res.json()) as T;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<{ status: string; model: string; state_clock: number }>("/health"),
  state: (after: number) => request<DemoState>(`/v1/demo/state?after=${after}`),
  control: (action: "start" | "pause" | "reset" | "rate", rate?: number) =>
    post<{ running: boolean; rate: number }>("/v1/demo/control", { action, rate }),
  daily: () => request<DailyRow[]>("/v1/demo/daily"),
  options: () => request<Options>("/v1/demo/options"),
  card: (id: number | string) => request<CardView>(`/v1/demo/cards/${id}`),
  transaction: (id: string) => request<TransactionDetail>(`/v1/demo/transactions/${encodeURIComponent(id)}`),
  queue: () => request<QueueItem[]>("/v1/demo/queue"),
  resolve: (id: string, fraud: boolean) =>
    post<{ ok: boolean }>(`/v1/demo/queue/${encodeURIComponent(id)}`, { fraud }),
  whatif: (body: {
    card_id: number;
    merchant_id: number;
    category: string;
    amount: number;
    hour: number | null;
    distance_km: number;
  }) => post<WhatIfResult>("/v1/demo/whatif", body),
  scoreManual: (body: {
    card_id: number;
    merchant_id: number;
    category: string;
    amount: number;
    away: boolean;
    distance_km?: number;
  }) =>
    post<FeedItem>("/v1/demo/score", body),
  monitoring: () => request<Monitoring>("/v1/demo/monitoring"),
  models: () => request<ModelsView>("/v1/models"),
  reports: () => request<ReportMeta[]>("/v1/reports"),
  report: (name: string) => request<Report>(`/v1/reports/${encodeURIComponent(name)}`),
};
