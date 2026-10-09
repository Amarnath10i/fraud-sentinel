# fraud-sentinel frontend

Next.js 15 (App Router) + TypeScript + Tailwind CSS 4, talking to the FastAPI
scoring service. TanStack Query handles caching and polling, Recharts the
standard charts; the SHAP waterfall and interval plots are hand-written SVG.

```bash
# terminal 1, repository root: the API (needs PostgreSQL and a registered model)
uv run sentinel serve

# terminal 2
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` if the API is not on `http://127.0.0.1:8000`; the API
allows `http://localhost:3000` through CORS by default (`SENTINEL_CORS_ORIGINS`).

| Page | What it shows |
|---|---|
| `/` | Running totals in dollars, how one transaction is scored, fraud stopped vs missed per day, latest alerts |
| `/live` | The replay as it happens, with an inspector that follows each new alert |
| `/transactions/[id]` | One decision: probability, expected cost of each action, full TreeSHAP waterfall, every model input |
| `/cards/[id]` | A card's live streaming state and recent transactions |
| `/review` | The analyst queue; confirming fraud sends a chargeback into the online state |
| `/whatif` | Change amount, hour, distance, category or merchant and watch the score respond, without recording anything |
| `/models` | Production model, feature importance, hyperparameters, decision costs, registry history |
| `/experiments` | Every generated report, rendered, with its key result charted |
| `/monitoring` | Alert-rate drift vs the calibration period, latency, the 18-month retraining backtest |

Checks: `npm run lint`, `npm run typecheck`, `npm test`, `npm run build` (all run in CI).
