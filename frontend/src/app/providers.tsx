"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { LiveFeedProvider } from "@/components/live-feed";
import { AppShell } from "@/components/shell";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 1_000 },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <LiveFeedProvider>
        <AppShell>{children}</AppShell>
      </LiveFeedProvider>
    </QueryClientProvider>
  );
}
