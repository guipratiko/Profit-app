const configuredApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
const CLIENT_API_PROXY_BASE = "/api/backend";

export const API_BASE_URL = configuredApiBaseUrl || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8021" : "");

export type Asset = {
  ticker: string;
  name: string;
  website?: string | null;
  logo_url?: string | null;
};
export type PriceRow = { date: string; close: number; open?: number; high?: number; low?: number; volume?: number };
export type PaperSignal = {
  signal_id: string;
  run_id?: string;
  ticker: string;
  signal_date: string;
  horizon?: string;
  decision: string;
  operational_action?: string;
  block_reason?: string | null;
  confidence?: number;
  probability_up?: number;
  probability_win?: number;
  probability_loss?: number;
  probability_timeout?: number;
  expected_return?: number;
  net_expected_return?: number;
  reference_price?: number;
  suggested_entry?: number;
  stop_loss?: number;
  partial_target?: number;
  target_price?: number;
  max_shares?: number;
  risk_amount?: number;
  reward_risk_ratio?: number;
  created_at?: string;
  thesis_json?: string;
};
export type FusionPrediction = {
  ticker: string;
  signal_date: string;
  fused_score?: number;
  fused_direction?: string;
  explanation_json?: string;
};
export type TechnicalPrediction = {
  run_id?: string;
  ticker: string;
  date?: string;
  target_name?: string;
  predicted_direction?: string;
  probability_down?: number;
  probability_sideways?: number;
  probability_up?: number;
  expected_return?: number;
  calibration_method?: string;
  inference_version?: string;
  conformal_interval_low?: number;
  conformal_interval_high?: number;
  conformal_alpha?: number;
  conformal_quantile?: number;
};
export type PredictionPayload = {
  ticker: string;
  model_run_id?: string | null;
  status?: string;
  technical_prediction?: TechnicalPrediction | null;
  fusion_prediction?: FusionPrediction | null;
  paper_signal?: PaperSignal | null;
};
export type Position = {
  position_id: string;
  signal_id?: string;
  run_id?: string;
  ticker: string;
  opened_at?: string;
  horizon?: string;
  stop_loss?: number;
  partial_target?: number;
  target_price?: number;
  status: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_return: number;
  realized_return?: number | null;
  last_evaluated_at?: string | null;
  metadata_json?: string;
  created_at?: string;
};
export type RiskAlert = {
  alert_id: string;
  position_id?: string;
  ticker: string;
  evaluated_at?: string;
  action: string;
  severity: string;
  reason: string;
  current_price?: number;
  unrealized_return: number;
  metadata_json?: string;
  created_at?: string;
};
export type RealPosition = {
  position_id: string;
  ticker: string;
  quantity: number;
  entry_price: number;
  entry_at: string;
  cost_basis: number;
  current_price: number;
  market_price_date?: string | null;
  market_value: number;
  unrealized_pnl: number;
  unrealized_return: number;
  last_updated_at?: string | null;
  notes?: string | null;
  created_at?: string;
};
export type RealPositionCreateInput = {
  ticker: string;
  quantity: number;
  entry_price: number;
  entry_at: string;
  notes?: string | null;
};
export type RealPositionUpdateInput = RealPositionCreateInput;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE_URL) {
    throw new Error("Defina NEXT_PUBLIC_API_BASE_URL com a URL publica do backend FastAPI antes de publicar o frontend.");
  }
  const response = await fetch(`${CLIENT_API_PROXY_BASE}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function withQuery(path: string, params: Record<string, string | number | boolean | undefined>) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

export const api = {
  assets: () => request<{ assets: Asset[] }>("/assets"),
  prices: (ticker: string, limit = 1300) => request<{ ticker: string; rows: PriceRow[] }>(`/prices/${ticker}?limit=${limit}`),
  predictions: () => request<{ predictions: PredictionPayload[] }>("/predictions"),
  prediction: (ticker: string) => request<PredictionPayload>(`/predictions/${ticker}`),
  explanation: (ticker: string) => request<{ ticker: string; explanation: unknown }>(`/predictions/${ticker}/explanation`),
  paperSignals: () => request<{ signals: PaperSignal[] }>("/paper/signals?limit=200"),
  positions: () => request<{ positions: Position[] }>("/portfolio/positions?limit=200"),
  alerts: () => request<{ alerts: RiskAlert[] }>("/portfolio/alerts?limit=200"),
  auditConselheiro: () => request<Record<string, unknown>>("/portfolio/audit-conselheiro", { method: "POST" }),
  realPositions: () => request<{ positions: RealPosition[] }>('/portfolio/real/positions?limit=200'),
  registerRealPosition: (payload: RealPositionCreateInput) => request<Record<string, unknown>>('/portfolio/real/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }),
  updateRealPosition: (positionId: string, payload: RealPositionUpdateInput) => request<Record<string, unknown>>(`/portfolio/real/${positionId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }),
  deleteRealPosition: (positionId: string) => request<Record<string, unknown>>(`/portfolio/real/${positionId}`, { method: 'DELETE' }),
  markToMarket: (options?: { refreshPrices?: boolean; refreshPeriod?: string }) => request<Record<string, unknown>>(
    withQuery('/portfolio/mark-to-market', {
      refresh_prices: options?.refreshPrices,
      refresh_period: options?.refreshPeriod
    }),
    { method: 'POST' }
  ),
  alphaMetrics: () => request<Record<string, any>>("/alpha/metrics"),
  paperGate: () => request<Record<string, any>>("/validation/paper-gate"),
  refreshStatus: () => request<Record<string, unknown>>("/refresh/status"),
  refreshRun: (options?: { maxStalenessDays?: number; refitWindowDays?: number; skipPriceUpdate?: boolean; asyncMode?: boolean; forceRefresh?: boolean }) => request<Record<string, unknown>>(
    withQuery("/refresh/run", {
      max_staleness_days: options?.maxStalenessDays,
      refit_window_days: options?.refitWindowDays,
      skip_price_update: options?.skipPriceUpdate,
      async_mode: options?.asyncMode,
      force_refresh: options?.forceRefresh
    }),
    { method: "POST" }
  )
};

export function assetLogoUrl(asset?: Pick<Asset, "ticker" | "logo_url"> | null) {
  if (!asset?.ticker) return "";
  return `${CLIENT_API_PROXY_BASE}/assets/${encodeURIComponent(asset.ticker)}/logo`;
}

export function parseJson<T>(text?: string | null): T | null {
  if (!text) return null;
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}
