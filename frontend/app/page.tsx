"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  ChevronDown,
  CircleDollarSign,
  Edit3,
  ExternalLink,
  Filter,
  Plus,
  RefreshCw,
  Search,
  Save,
  ShieldCheck,
  Trash2,
  TrendingUp,
  WalletCards,
  X
} from "lucide-react";
import { api, parseJson, type Asset, type PaperSignal, type Position, type PredictionPayload, type PriceRow, type RealPosition, type RiskAlert } from "@/lib/api";
import { AssetLogo } from "@/components/asset-logo";
import { TradingViewSymbolInfo } from "@/components/tradingview-symbol-info";
import { tradingViewChartUrl, tradingViewSymbolPageUrl } from "@/lib/tradingview";
import { cn, fmtMoney, fmtNumber, fmtPercent } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PriceChart } from "@/components/price-chart";
import { GateChart } from "@/components/gate-chart";

type Horizon = "7d" | "3m" | "1y";
type DashboardTab = "trending" | "predictions" | "investments";
type IntentFilter = "ALL" | "BUY" | "SELL" | "NO_OPERATE";
type Thesis = Record<string, any>;
type Tone = "neutral" | "good" | "warn" | "bad" | "info";
type PortfolioIntent = {
  asset: Asset;
  ticker: string;
  name: string;
  signal?: PaperSignal | null;
  position?: Position;
  alert?: RiskAlert;
  thesis: Thesis | null;
  alertMeta: Record<string, any> | null;
  intentLabel: string;
  intentTone: Tone;
  statusLabel: string;
  statusTone: Tone;
  whenLabel: string;
  timeLabel: string;
  reasonLabel: string;
  reviewLabel: string;
  whyLines: string[];
  entryPrice?: number;
  currentPrice?: number;
  stopLoss?: number;
  partialTarget?: number;
  targetPrice?: number;
  trailingStop?: number;
  daysRemaining?: number | null;
};
type RealPositionFormState = {
  positionId: string | null;
  ticker: string;
  quantity: string;
  entryPrice: string;
  entryAt: string;
  notes: string;
};
type RealPortfolioRow = {
  asset: Asset | null;
  position: RealPosition;
  assetName: string;
  intentLabel: string;
  intentTone: Tone;
  whenLabel: string;
  timeLabel: string;
  reasonLabel: string;
  updatedLabel: string;
};

const REVIEW_TIME_LABEL = "Apos fechamento (~18:00 BRT)";
const INTENT_FILTERS: Array<{ value: IntentFilter; label: string; tone: Tone }> = [
  { value: "ALL", label: "Todos", tone: "neutral" },
  { value: "BUY", label: "Comprar", tone: "good" },
  { value: "SELL", label: "Vender", tone: "bad" },
  { value: "NO_OPERATE", label: "Nao operar", tone: "warn" }
];
const HORIZON_TO_DAYS: Record<string, number> = {
  "7d": 7,
  "21d": 21,
  "3m": 63,
  "63d": 63,
  "1y": 252,
  "252d": 252
};

const REASON_LABELS: Record<string, string> = {
  target_price_reached: "Alvo atingido",
  partial_target_reached: "Parcial no alvo",
  hard_stop_breached: "Stop rompido",
  trailing_stop_protected_profit: "Trailing stop protegeu lucro",
  horizon_elapsed: "Horizonte encerrado",
  residual_expected_value_non_positive: "Valor esperado residual ficou negativo",
  profit_buffer_available_trail_stop: "Lucro pede stop mais justo",
  risk_within_policy: "Risco dentro da politica",
  reward_risk_below_minimum: "Risco-retorno abaixo do minimo",
  event_within_48h: "Evento recente pede cautela",
  regime_divergence_high: "Regime e contexto divergem",
  kelly_edge_below_minimum: "Edge insuficiente para alocacao",
  position_size_zero: "Tamanho calculado ficou zerado",
  volatility_percentile_above_80: "Volatilidade acima do limite",
  probability_win_below_enter_threshold: "Probabilidade de ganho abaixo do limiar",
  net_expected_return_not_positive: "Retorno esperado liquido nao compensa",
  win_minus_loss_edge_below_minimum: "Vantagem win-loss insuficiente",
  probability_up_below_strategy_threshold: "Probabilidade de alta abaixo do limiar",
  confidence_below_minimum: "Confianca abaixo do minimo",
  data_stale: "Preco de referencia estava defasado",
  volatility_above_limit: "Volatilidade acima do limite",
  rounded_to_zero_shares: "Quantidade calculada ficou zerada",
  exposure_caps_exhausted: "Limite de exposicao esgotado",
  strategy_gate_failed: "Gate estrategico bloqueou a entrada",
  beats_buy_hold_average: "Ainda nao supera o buy and hold medio"
};

const TECHNICAL_VALIDATION_REASONS = new Set([
  "technical_fallback_validation_edge_failed",
  "technical_fallback_test_min_trades_failed",
  "technical_fallback_test_return_positive_failed",
  "technical_fallback_test_avg_return_positive_failed",
  "technical_fallback_profitable_tickers_acceptable_failed",
  "trade_outcome_min_test_trades_failed",
  "trade_outcome_avg_return_positive_failed",
  "trade_outcome_win_rate_acceptable_failed",
  "strategy_gate_failed",
]);

type DashboardState = {
  assets: Asset[];
  predictions: Record<string, PredictionPayload>;
  prices: PriceRow[];
  signals: PaperSignal[];
  positions: Position[];
  realPositions: RealPosition[];
  alerts: RiskAlert[];
  alpha: Record<string, any> | null;
  gate: Record<string, any> | null;
  refresh: Record<string, unknown> | null;
};

type RefreshJob = {
  job_id?: string | null;
  status?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  result?: Record<string, unknown> | null;
};

const horizonLimits: Record<Horizon, number> = { "7d": 7, "3m": 90, "1y": 252 };

function toLocalDateTimeInputValue(date = new Date()) {
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function createEmptyRealForm(ticker = ""): RealPositionFormState {
  return {
    positionId: null,
    ticker,
    quantity: "1",
    entryPrice: "",
    entryAt: toLocalDateTimeInputValue(),
    notes: ""
  };
}

function createRealFormFromPosition(position: RealPosition): RealPositionFormState {
  const parsedEntryAt = new Date(position.entry_at);
  return {
    positionId: position.position_id,
    ticker: position.ticker,
    quantity: String(position.quantity),
    entryPrice: String(position.entry_price),
    entryAt: toLocalDateTimeInputValue(Number.isNaN(parsedEntryAt.valueOf()) ? new Date() : parsedEntryAt),
    notes: position.notes || ""
  };
}

function badgeTone(value?: string | null): Tone {
  const normalized = (value || "").toLowerCase();
  if (normalized.includes("enter") || normalized.includes("up") || normalized.includes("passed")) return "good";
  if (normalized.includes("watch") || normalized.includes("mixed") || normalized.includes("collecting") || normalized.includes("monitoring")) return "warn";
  if (normalized.includes("no") || normalized.includes("down") || normalized.includes("failed") || normalized.includes("bloque")) return "bad";
  if (normalized.includes("technical") || normalized.includes("sideways")) return "info";
  return "neutral";
}

function actionLabel(action?: string | null) {
  if (!action) return "N/A";
  if (action === "ENTER_LONG") return "Comprar";
  if (action === "WATCHLIST") return "Observar";
  if (action === "NO_TRADE") return "Nao operar";
  return action.replaceAll("_", " ");
}

function auditorAction(position: Position, alert?: RiskAlert) {
  const action = (alert?.action || "").toLowerCase();
  if (action.includes("close") || action.includes("exit")) return "Vender";
  if (action.includes("partial")) return "Vender parcial";
  if (position.status === "open") return "Manter";
  return "Comprar bloqueado";
}

function thesisFor(signal?: PaperSignal | null): Thesis | null {
  return parseJson<Thesis>(signal?.thesis_json || null);
}

function signalForTicker(signals: PaperSignal[], ticker: string) {
  return signals.find((signal) => signal.ticker === ticker);
}

function latestAlert(alerts: RiskAlert[], ticker: string) {
  return alerts.find((alert) => alert.ticker === ticker);
}

function latestPosition(positions: Position[], ticker: string) {
  return positions.find((position) => position.ticker === ticker);
}

function formatShortDate(value?: string | null) {
  if (!value) return "-";
  const normalized = value.length === 10 ? `${value}T00:00:00` : value;
  const date = new Date(normalized);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "2-digit" }).format(date);
}

function formatDateTime(value?: string | null) {
  if (!value) return REVIEW_TIME_LABEL;
  const normalized = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(value) ? `${value}Z` : value;
  const date = new Date(normalized);
  if (Number.isNaN(date.valueOf())) return value;
  return `${new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC"
  }).format(date)} UTC`;
}

function reasonTokens(reason?: string | null) {
  if (!reason) return [] as string[];
  return reason
    .split(",")
    .map((token) => token.trim().replace(/^aggressive_alpha_override:/, ""))
    .filter(Boolean);
}

function uniqueReasonTokens(reason?: string | null) {
  const tokens = reasonTokens(reason);
  return Array.from(new Set(tokens.map((token) => token.toLowerCase())));
}

function formatReason(reason?: string | null, fallback = "Sem bloqueios ativos") {
  const tokens = uniqueReasonTokens(reason);
  if (!tokens.length) return fallback;
  return tokens.map((token) => REASON_LABELS[token] || token.replaceAll("_", " ")).join(" · ");
}

function decisionExplanation(
  reason?: string | null,
  signal?: PaperSignal | null,
  fallback = "Sem bloqueios ativos"
) {
  const tokens = uniqueReasonTokens(reason);
  if (!tokens.length) return fallback;
  const hasTechnicalValidation = tokens.some((token) => TECHNICAL_VALIDATION_REASONS.has(token));
  const hasVolatility = tokens.includes("volatility_above_limit") || tokens.includes("volatility_percentile_above_80");
  const hasNegativeEv = tokens.includes("net_expected_return_not_positive");
  const hasWeakProbability = tokens.includes("probability_win_below_enter_threshold") || tokens.includes("probability_up_below_strategy_threshold");
  const hasWeakEdge = tokens.includes("win_minus_loss_edge_below_minimum") || tokens.includes("kelly_edge_below_minimum");
  const hasSizingIssue = tokens.includes("position_size_zero") || tokens.includes("rounded_to_zero_shares") || tokens.includes("exposure_caps_exhausted");
  const hasStaleData = tokens.includes("data_stale");
  const hasEventRisk = tokens.includes("event_within_48h") || tokens.includes("regime_divergence_high");

  if (hasStaleData) {
    return "Entrada vetada porque o preco de referencia esta defasado. Atualize os dados antes de confiar no sinal.";
  }
  if (hasSizingIssue) {
    return "Entrada vetada porque o controle de risco calculou lote zero ou exposicao indisponivel para este ativo.";
  }
  if (hasNegativeEv) {
    return "Entrada vetada porque o retorno esperado liquido nao paga custo, spread e risco do trade.";
  }
  if (hasVolatility && hasTechnicalValidation) {
    return "Entrada vetada: o ativo esta volátil e a estrategia nao confirmou vantagem consistente no teste recente. Melhor aguardar confirmacao.";
  }
  if (hasVolatility) {
    return "Entrada vetada pela volatilidade: o stop precisaria ficar largo demais para o risco atual.";
  }
  if (hasTechnicalValidation) {
    return "Entrada vetada porque o padrao ainda nao provou retorno liquido consistente nos testes recentes. Observar ate o edge aparecer com mais robustez.";
  }
  if (hasWeakProbability || hasWeakEdge) {
    return "Entrada vetada porque a vantagem estatistica ainda esta pequena para justificar risco real.";
  }
  if (hasEventRisk) {
    return "Entrada vetada porque contexto e preco estao divergindo; o sinal pode estar reagindo a ruido de curto prazo.";
  }
  return formatReason(reason, fallback);
}

function releaseCondition(reason?: string | null) {
  const tokens = uniqueReasonTokens(reason);
  if (!tokens.length) return null;
  if (tokens.includes("data_stale")) return "Liberar apos refresh de mercado com preco atualizado.";
  if (tokens.includes("position_size_zero") || tokens.includes("rounded_to_zero_shares")) return "Liberar quando o lote calculado ficar maior que zero dentro do limite de risco.";
  if (tokens.includes("net_expected_return_not_positive")) return "Liberar quando o EV liquido voltar a positivo depois dos custos.";
  if (tokens.includes("volatility_above_limit") || tokens.includes("volatility_percentile_above_80")) return "Liberar quando a volatilidade cair ou o alvo compensar o stop maior.";
  if (tokens.some((token) => TECHNICAL_VALIDATION_REASONS.has(token))) return "Liberar quando o proximo teste/snapshot mostrar retorno medio positivo e edge persistente.";
  if (tokens.includes("probability_win_below_enter_threshold") || tokens.includes("win_minus_loss_edge_below_minimum")) return "Liberar quando probabilidade e diferenca ganho-perda subirem acima do limiar.";
  return null;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readRefreshJob(refresh: Record<string, unknown> | null): RefreshJob | null {
  if (!refresh) return null;
  const candidate = refresh["refresh_job"];
  return candidate && typeof candidate === "object" ? (candidate as RefreshJob) : null;
}

function settledValue<T>(result: PromiseSettledResult<T>, fallback: T): T {
  return result.status === "fulfilled" ? result.value : fallback;
}

function horizonLabel(signal?: PaperSignal | null, alertMeta?: Record<string, any> | null) {
  const rawHorizon = signal?.horizon;
  if (rawHorizon) return rawHorizon;
  const days = Number(alertMeta?.horizon_days);
  return Number.isFinite(days) && days > 0 ? `${days}d` : "7d";
}

function signalIntent(signal?: PaperSignal | null): { label: string; tone: Tone } {
  const action = String(signal?.operational_action || signal?.decision || "").toUpperCase();
  if (action.includes("ENTER") || action.includes("SIMULATE_LONG")) return { label: "Comprar", tone: "good" };
  if (action.includes("WATCH")) return { label: "Observar", tone: "warn" };
  if (action.includes("NO_TRADE") || action.includes("NO_OPERATE")) return { label: "Nao entrar", tone: "bad" };
  return { label: actionLabel(signal?.operational_action || signal?.decision), tone: badgeTone(signal?.operational_action || signal?.decision) };
}

function positionIntent(alert?: RiskAlert | null): { label: string; tone: Tone } {
  const action = String(alert?.action || "").toUpperCase();
  if (action.includes("EXIT_TARGET")) return { label: "Realizar no alvo", tone: "good" };
  if (action.includes("EXIT") || action.includes("CLOSE")) return { label: "Vender", tone: "bad" };
  if (action.includes("REDUCE") || action.includes("PARTIAL")) return { label: "Realizar parcial", tone: "warn" };
  if (action.includes("MANAGE") || action.includes("ADJUST")) return { label: "Manter com ajuste", tone: "warn" };
  if (action.includes("HOLD")) return { label: "Manter", tone: "good" };
  return { label: actionLabel(alert?.action), tone: badgeTone(alert?.action) };
}

function classifyIntentLabel(label?: string | null): IntentFilter | null {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("comprar") || normalized.includes("entrada")) return "BUY";
  if (normalized.includes("vender") || normalized.includes("realizar") || normalized.includes("parcial")) return "SELL";
  if (normalized.includes("nao") || normalized.includes("não") || normalized.includes("bloque")) return "NO_OPERATE";
  return null;
}

function finiteCount(value: unknown) {
  const count = Number(value);
  return Number.isFinite(count) ? count : null;
}

function normalizeSearchText(value?: string | null) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

function assetMatchesQuery(asset: Pick<Asset, "ticker" | "name"> | null | undefined, query: string) {
  if (!asset) return false;
  const normalizedQuery = normalizeSearchText(query);
  if (!normalizedQuery) return true;
  const haystack = `${asset.ticker} ${asset.name}`;
  return normalizeSearchText(haystack).includes(normalizedQuery);
}

function modelWindowLabel(targetName?: string | null) {
  const rawWindow = String(targetName || "").match(/(\d+[dmy])$/i)?.[1]?.toLowerCase();
  if (!rawWindow) return "7d";
  const days = HORIZON_TO_DAYS[rawWindow];
  if (days === 63) return "3m";
  if (days === 252) return "1y";
  return rawWindow;
}

function modelDirectionLabel(direction?: string | null) {
  const normalized = String(direction || "").toUpperCase();
  if (normalized === "UP") return "Alta";
  if (normalized === "DOWN") return "Baixa";
  if (normalized === "SIDEWAYS") return "Lateral";
  return direction || "-";
}

function positionState(position?: Position, signal?: PaperSignal | null): { label: string; tone: Tone } {
  if (position?.status === "open") return { label: "Em carteira", tone: "good" };
  if (position?.status?.startsWith("closed")) return { label: "Posicao anterior encerrada", tone: "neutral" };
  const action = String(signal?.operational_action || signal?.decision || "").toUpperCase();
  if (action.includes("ENTER")) return { label: "Entrada permitida", tone: "info" };
  if (action.includes("WATCH")) return { label: "Em observacao", tone: "warn" };
  return { label: "Sem posicao", tone: "neutral" };
}

function nextReviewLabel(position: Position | undefined, signal: PaperSignal | null | undefined, alertMeta: Record<string, any> | null) {
  if (position?.status === "open") {
    const daysRemaining = Number(alertMeta?.days_remaining);
    if (Number.isFinite(daysRemaining)) {
      return daysRemaining > 0 ? `${daysRemaining} dia(s) restantes` : "Encerrar hoje";
    }
    return "Reavaliar no fechamento";
  }
  const action = String(signal?.operational_action || signal?.decision || "").toUpperCase();
  if (action.includes("ENTER")) return signal?.signal_date ? `Entrada desde ${formatShortDate(signal.signal_date)}` : "Entrada no proximo fechamento";
  if (action.includes("WATCH")) return "Observar no proximo fechamento";
  if (signal?.block_reason) return "Aguardar melhora dos filtros";
  return "Sem gatilho ativo";
}

function buildWhyLines(
  signal?: PaperSignal | null,
  thesis?: Thesis | null,
  alert?: RiskAlert | null,
  alertMeta?: Record<string, any> | null
) {
  const lines: string[] = [];
  if (alert?.reason) {
    lines.push(decisionExplanation(alert.reason, signal));
    const condition = releaseCondition(alert.reason);
    if (condition) lines.push(condition);
  } else if (signal?.block_reason) {
    lines.push(decisionExplanation(signal.block_reason, signal));
    const condition = releaseCondition(signal.block_reason);
    if (condition) lines.push(condition);
  } else if (String(signal?.operational_action || signal?.decision || "").toUpperCase().includes("ENTER")) {
    lines.push("Edge positivo, sem bloqueio operacional no snapshot atual.");
  }
  if (signal?.probability_win !== undefined) {
    lines.push(`Prob. de ganho: ${fmtPercent(signal.probability_win, 1)}`);
  } else if (signal?.probability_up !== undefined) {
    lines.push(`Prob. tecnica de alta: ${fmtPercent(signal.probability_up, 1)}`);
  }
  if (signal?.net_expected_return !== undefined) {
    lines.push(`Retorno esperado liquido: ${fmtPercent(signal.net_expected_return, 2)}`);
  }
  const daysRemaining = Number(alertMeta?.days_remaining);
  if (Number.isFinite(daysRemaining)) {
    lines.push(`Dias restantes no horizonte: ${daysRemaining}`);
  }
  if (alertMeta?.trailing_stop !== undefined) {
    lines.push(`Trailing stop atual: ${fmtMoney(Number(alertMeta.trailing_stop))}`);
  }
  const tags = thesis?.qualitative_context?.event_tags;
  if (Array.isArray(tags) && tags.length) {
    lines.push(`Eventos monitorados: ${tags.join(", ")}`);
  }
  return lines.slice(0, 5);
}

function PriceDelta({ rows }: { rows: PriceRow[] }) {
  if (rows.length < 2) return <span>-</span>;
  const first = rows[0]?.close || 0;
  const last = rows[rows.length - 1]?.close || 0;
  const delta = first > 0 ? last / first - 1 : 0;
  return <span className={delta >= 0 ? "text-emerald-300" : "text-rose-300"}>{fmtPercent(delta)}</span>;
}

function LoadingPanel({ message }: { message: string }) {
  return <div className="glass flex min-h-[280px] items-center justify-center rounded-2xl text-sm text-muted-foreground">{message}</div>;
}

export default function Page() {
  const [state, setState] = useState<DashboardState>({
    assets: [],
    predictions: {},
    prices: [],
    signals: [],
    positions: [],
    realPositions: [],
    alerts: [],
    alpha: null,
    gate: null,
    refresh: null
  });
  const [activeTab, setActiveTab] = useState<DashboardTab>("trending");
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [horizon, setHorizon] = useState<Horizon>("3m");
  const [investmentIntentFilter, setInvestmentIntentFilter] = useState<IntentFilter>("ALL");
  const [loading, setLoading] = useState(true);
  const [busyRefresh, setBusyRefresh] = useState(false);
  const [busyAudit, setBusyAudit] = useState(false);
  const [busyRealSave, setBusyRealSave] = useState(false);
  const [busyRealRefresh, setBusyRealRefresh] = useState(false);
  const [busyRealDeleteId, setBusyRealDeleteId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [assetSearch, setAssetSearch] = useState("");
  const [realForm, setRealForm] = useState<RealPositionFormState>(() => createEmptyRealForm());
  const [realFormAssetSearch, setRealFormAssetSearch] = useState("");
  const [realPortfolioModalOpen, setRealPortfolioModalOpen] = useState(false);
  const [realDrafts, setRealDrafts] = useState<Record<string, RealPositionFormState>>({});

  function resetRealForm(preferredTicker?: string) {
    setRealForm(createEmptyRealForm(preferredTicker || state.assets[0]?.ticker || ""));
  }

  function openRealPositionModal(position?: RealPosition) {
    setError(null);
    setRealFormAssetSearch("");
    if (position) {
      setSelectedTicker(position.ticker);
      setRealForm(createRealFormFromPosition(position));
    } else {
      resetRealForm(selectedTicker || realForm.ticker || state.assets[0]?.ticker || "");
    }
    setRealPortfolioModalOpen(true);
  }

  function closeRealPositionModal() {
    if (busyRealSave) return;
    setRealFormAssetSearch("");
    setRealPortfolioModalOpen(false);
    resetRealForm(selectedTicker || realForm.ticker || state.assets[0]?.ticker || "");
  }

  async function loadDashboard(preferredTicker?: string) {
    setLoading(true);
    setError(null);
    try {
      const assetsPayload = await api.assets();
      const refresh = await api.refreshStatus();
      const refreshJob = readRefreshJob(refresh);
      const refreshRunning = refreshJob?.status === "running";
      const firstTicker = preferredTicker || selectedTicker || assetsPayload.assets[0]?.ticker || "";
      const [signals, positions, realPositions, alerts, alpha, gate, predictionsPayload] = await Promise.allSettled([
        api.paperSignals(),
        api.positions(),
        api.realPositions(),
        api.alerts(),
        refreshRunning ? Promise.resolve(state.alpha || null) : api.alphaMetrics(),
        refreshRunning ? Promise.resolve(state.gate || null) : api.paperGate(),
        api.predictions()
      ]);

      const signalsPayload = settledValue(signals, { signals: state.signals });
      const positionsPayload = settledValue(positions, { positions: state.positions });
      const realPositionsPayload = settledValue(realPositions, { positions: state.realPositions });
      const alertsPayload = settledValue(alerts, { alerts: state.alerts });
      const alphaPayload = settledValue(alpha, state.alpha);
      const gatePayload = settledValue(gate, state.gate);
      const predictionsResult = settledValue(predictionsPayload, { predictions: Object.values(state.predictions) });
      const predictions = Object.fromEntries(
        predictionsResult.predictions.map((prediction) => [prediction.ticker, prediction] as const)
      ) as Record<string, PredictionPayload>;
      const prices = firstTicker ? (await api.prices(firstTicker, horizonLimits[horizon]).catch(() => ({ rows: [] as PriceRow[] }))).rows : [];
      setSelectedTicker(firstTicker);
      setRealForm((current) => (
        current.ticker || current.positionId || !assetsPayload.assets[0]?.ticker
          ? current
          : { ...current, ticker: assetsPayload.assets[0].ticker }
      ));
      setState({
        assets: assetsPayload.assets,
        predictions,
        prices,
        signals: signalsPayload.signals,
        positions: positionsPayload.positions,
        realPositions: realPositionsPayload.positions,
        alerts: alertsPayload.alerts,
        alpha: alphaPayload,
        gate: gatePayload,
        refresh
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar dados");
    } finally {
      setLoading(false);
    }
  }

  async function refreshMarketData() {
    setBusyRefresh(true);
    setError(null);
    try {
      await api.refreshRun({
        maxStalenessDays: 0,
        refitWindowDays: 180,
        asyncMode: true,
        forceRefresh: true,
      });

      let refreshCompleted = false;
      for (let attempt = 0; attempt < 180; attempt += 1) {
        const refresh = await api.refreshStatus();
        setState((current) => ({ ...current, refresh }));
        const job = readRefreshJob(refresh);
        const isStale = Boolean(refresh.is_stale);

        if (job?.status === "failed") {
          throw new Error(job.error || "Falha ao sincronizar mercado");
        }
        if ((job?.status === "completed" || job?.status === "idle" || !job?.status) && !isStale) {
          refreshCompleted = true;
          break;
        }

        await sleep(3000);
      }
      if (!refreshCompleted) {
        throw new Error("A sincronizacao completa ainda esta em andamento. Tente novamente em instantes.");
      }

      await api.markToMarket({ refreshPrices: true, refreshPeriod: "7d" });
      await loadDashboard(selectedTicker);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao atualizar mercado");
    } finally {
      setBusyRefresh(false);
    }
  }

  useEffect(() => {
    void loadDashboard();
  }, []);

  useEffect(() => {
    if (!selectedTicker) return;
    api.prices(selectedTicker, horizonLimits[horizon])
      .then((payload) => setState((current) => ({ ...current, prices: payload.rows })))
      .catch(() => setState((current) => ({ ...current, prices: [] })));
  }, [selectedTicker, horizon]);

  useEffect(() => {
    if (!realForm.ticker && state.assets[0]?.ticker) {
      setRealForm((current) => ({ ...current, ticker: state.assets[0]?.ticker || "" }));
    }
  }, [realForm.ticker, state.assets]);

  useEffect(() => {
    setRealDrafts((current) => {
      const next: Record<string, RealPositionFormState> = {};
      state.realPositions.forEach((position) => {
        next[position.position_id] = current[position.position_id] || createRealFormFromPosition(position);
      });
      return next;
    });
  }, [state.realPositions]);

  const selectedPrediction = selectedTicker ? state.predictions[selectedTicker] : undefined;
  const selectedSignal = selectedPrediction?.paper_signal || signalForTicker(state.signals, selectedTicker);
  const selectedTechnical = selectedPrediction?.technical_prediction;
  const selectedThesis = thesisFor(selectedSignal);
  const gateStatus = String(state.gate?.status || "unknown");
  const paperSignals = state.alpha?.paper_signals || {};
  const blockedReasons = Object.entries(paperSignals.blocked_by_reason || {}).slice(0, 6);
  const assetByTicker = useMemo(
    () => Object.fromEntries(state.assets.map((asset) => [asset.ticker, asset] as const)),
    [state.assets]
  );
  const filteredAssets = useMemo(
    () => state.assets.filter((asset) => assetMatchesQuery(asset, assetSearch)),
    [state.assets, assetSearch]
  );

  const investmentRows = useMemo<PortfolioIntent[]>(() => {
    return state.assets.map((asset) => {
      const prediction = state.predictions[asset.ticker];
      const signal = prediction?.paper_signal || signalForTicker(state.signals, asset.ticker);
      const thesis = thesisFor(signal);
      const position = latestPosition(state.positions, asset.ticker);
      const alert = latestAlert(state.alerts, asset.ticker);
      const alertMeta = parseJson<Record<string, any>>(alert?.metadata_json || null);
      const currentPositionOpen = position?.status === "open";
      const intent = currentPositionOpen ? positionIntent(alert) : signalIntent(signal);
      const status = positionState(position, signal);
      const reasonLabel = currentPositionOpen
        ? decisionExplanation(alert?.reason, signal, "Sem alerta ativo")
        : decisionExplanation(
            signal?.block_reason,
            signal,
            intent.label === "Comprar" ? "Sem bloqueios e com edge positivo" : "Sem bloqueio explicito"
          );

      return {
        asset,
        ticker: asset.ticker,
        name: asset.name,
        signal,
        position,
        alert,
        thesis,
        alertMeta,
        intentLabel: intent.label,
        intentTone: intent.tone,
        statusLabel: status.label,
        statusTone: status.tone,
        whenLabel: nextReviewLabel(position, signal, alertMeta),
        timeLabel: currentPositionOpen && alert?.evaluated_at ? formatDateTime(alert.evaluated_at) : REVIEW_TIME_LABEL,
        reasonLabel,
        reviewLabel: `${formatShortDate(signal?.signal_date)} · janela ${horizonLabel(signal, alertMeta)}`,
        whyLines: buildWhyLines(signal, thesis, alert, alertMeta),
        entryPrice: currentPositionOpen ? (position?.entry_price ?? signal?.suggested_entry) : (signal?.suggested_entry ?? position?.entry_price),
        currentPrice: currentPositionOpen ? (position?.current_price ?? alert?.current_price ?? signal?.reference_price) : (signal?.reference_price ?? position?.current_price ?? alert?.current_price),
        stopLoss: currentPositionOpen ? (position?.stop_loss ?? signal?.stop_loss) : (signal?.stop_loss ?? position?.stop_loss),
        partialTarget: currentPositionOpen ? (position?.partial_target ?? signal?.partial_target) : (signal?.partial_target ?? position?.partial_target),
        targetPrice: currentPositionOpen ? (position?.target_price ?? signal?.target_price) : (signal?.target_price ?? position?.target_price),
        trailingStop: alertMeta?.trailing_stop !== undefined ? Number(alertMeta.trailing_stop) : undefined,
        daysRemaining: Number.isFinite(Number(alertMeta?.days_remaining)) ? Number(alertMeta?.days_remaining) : null
      };
    });
  }, [state.assets, state.predictions, state.signals, state.positions, state.alerts]);

  const searchedInvestmentRows = useMemo(
    () => investmentRows.filter((row) => assetMatchesQuery(row.asset, assetSearch)),
    [investmentRows, assetSearch]
  );
  const filteredInvestmentRows = useMemo(() => {
    if (investmentIntentFilter === "ALL") return searchedInvestmentRows;
    return searchedInvestmentRows.filter((row) => classifyIntentLabel(row.intentLabel) === investmentIntentFilter);
  }, [searchedInvestmentRows, investmentIntentFilter]);
  const selectedInvestment = filteredInvestmentRows.find((row) => row.ticker === selectedTicker)
    || searchedInvestmentRows.find((row) => row.ticker === selectedTicker)
    || investmentRows.find((row) => row.ticker === selectedTicker)
    || filteredInvestmentRows[0]
    || searchedInvestmentRows[0]
    || investmentRows[0];
  const investmentFilterCounts = useMemo(() => {
    return INTENT_FILTERS.reduce<Record<IntentFilter, number>>((counts, filter) => {
      counts[filter.value] = filter.value === "ALL"
        ? investmentRows.length
        : investmentRows.filter((row) => classifyIntentLabel(row.intentLabel) === filter.value).length;
      return counts;
    }, { ALL: 0, BUY: 0, SELL: 0, NO_OPERATE: 0 });
  }, [investmentRows]);
  const selectedAsset = selectedInvestment?.asset || assetByTicker[selectedTicker] || filteredAssets[0] || state.assets[0] || null;
  const openInvestmentCount = investmentRows.filter((row) => row.position?.status === "open").length;
  const realPortfolioRows = useMemo<RealPortfolioRow[]>(() => {
    return state.realPositions.map((position) => {
      const intent = investmentRows.find((row) => row.ticker === position.ticker);
      const asset = state.assets.find((entry) => entry.ticker === position.ticker) || null;
      const assetName = asset?.name || position.ticker;
      return {
        asset,
        position,
        assetName,
        intentLabel: intent?.intentLabel || "Sem sinal",
        intentTone: intent?.intentTone || "neutral",
        whenLabel: intent?.whenLabel || "Reavaliar no fechamento",
        timeLabel: intent?.timeLabel || REVIEW_TIME_LABEL,
        reasonLabel: intent?.reasonLabel || "Sem racional ativo para este ticker.",
        updatedLabel: position.market_price_date ? `Fechamento ${formatShortDate(position.market_price_date)}` : formatDateTime(position.last_updated_at)
      };
    });
  }, [investmentRows, state.assets, state.realPositions]);
  const filteredRealPortfolioRows = useMemo(() => {
    const searchedRows = realPortfolioRows.filter((row) => assetMatchesQuery({ ticker: row.position.ticker, name: row.assetName }, assetSearch));
    if (investmentIntentFilter === "ALL") return searchedRows;
    return searchedRows.filter((row) => classifyIntentLabel(row.intentLabel) === investmentIntentFilter);
  }, [realPortfolioRows, investmentIntentFilter, assetSearch]);
  const realPortfolioSummary = useMemo(() => {
    const invested = state.realPositions.reduce((sum, position) => sum + Number(position.cost_basis || 0), 0);
    const marketValue = state.realPositions.reduce((sum, position) => sum + Number(position.market_value || 0), 0);
    const unrealizedPnl = state.realPositions.reduce((sum, position) => sum + Number(position.unrealized_pnl || 0), 0);
    return {
      count: state.realPositions.length,
      invested,
      marketValue,
      unrealizedPnl,
      unrealizedReturn: invested > 0 ? marketValue / invested - 1.0 : null
    };
  }, [state.realPositions]);
  const traderCockpit = useMemo(() => {
    const intentByTicker = new Map(investmentRows.map((row) => [row.ticker, row] as const));
    let openRiskBrl = 0;
    const divergentPositions: Array<{ ticker: string; intentLabel: string; intentTone: Tone; pnl: number }> = [];
    state.realPositions.forEach((position) => {
      const intent = intentByTicker.get(position.ticker);
      const atrPct = Number(intent?.thesis?.sizing?.atr_14);
      const referencePrice = Number(position.current_price || position.entry_price || 0);
      if (Number.isFinite(atrPct) && atrPct > 0 && referencePrice > 0) {
        // Risk in BRL = position notional × 1 ATR move (one-sigma move proxy).
        openRiskBrl += Math.abs(Number(position.quantity || 0)) * referencePrice * atrPct;
      }
      const intentClass = classifyIntentLabel(intent?.intentLabel || "");
      // Divergence: model now wants out / NO_OPERATE on a position we still hold.
      if (intent && (intentClass === "SELL" || intentClass === "NO_OPERATE")) {
        divergentPositions.push({
          ticker: position.ticker,
          intentLabel: intent.intentLabel,
          intentTone: intent.intentTone,
          pnl: Number(position.unrealized_pnl || 0),
        });
      }
    });
    const drawdownPeak = Number(state.gate?.metrics?.max_drawdown ?? NaN);
    const gateLabel = gateStatus === "approved"
      ? "Aprovado"
      : gateStatus === "blocked"
        ? "Bloqueado"
        : gateStatus === "pending"
          ? "Pendente"
          : gateStatus.replaceAll("_", " ");
    const gateTone: Tone = gateStatus === "approved" ? "good" : gateStatus === "blocked" ? "bad" : "warn";
    return {
      openPnl: realPortfolioSummary.unrealizedPnl,
      openPnlReturn: realPortfolioSummary.unrealizedReturn,
      exposure: realPortfolioSummary.marketValue,
      openRisk: openRiskBrl,
      drawdown: Number.isFinite(drawdownPeak) ? drawdownPeak : null,
      gateLabel,
      gateTone,
      divergentPositions,
    };
  }, [investmentRows, state.realPositions, state.gate, gateStatus, realPortfolioSummary]);
  const opportunityRows = useMemo(() => {
    const ranked = [...investmentRows].sort((left, right) => {
      const leftIntentBoost = left.signal?.operational_action === "ENTER_LONG" ? 1 : 0;
      const rightIntentBoost = right.signal?.operational_action === "ENTER_LONG" ? 1 : 0;
      if (leftIntentBoost !== rightIntentBoost) return rightIntentBoost - leftIntentBoost;
      const leftScore = Number(left.signal?.net_expected_return ?? left.signal?.probability_win ?? left.signal?.probability_up ?? -1);
      const rightScore = Number(right.signal?.net_expected_return ?? right.signal?.probability_win ?? right.signal?.probability_up ?? -1);
      return rightScore - leftScore;
    });
    return ranked.slice(0, 4);
  }, [investmentRows]);
  const topInvestedRows = useMemo(() => {
    return [...state.realPositions]
      .sort((left, right) => Number(right.cost_basis || 0) - Number(left.cost_basis || 0))
      .slice(0, 3);
  }, [state.realPositions]);
  const marketPanelStats = useMemo(() => {
    const buyCount = finiteCount(paperSignals.operational_actions?.ENTER_LONG)
      ?? investmentRows.filter((row) => row.signal?.operational_action === "ENTER_LONG").length;
    const noOperateCount = finiteCount(paperSignals.operational_actions?.NO_TRADE)
      ?? finiteCount(paperSignals.no_operate_count)
      ?? investmentRows.filter((row) => classifyIntentLabel(row.intentLabel) === "NO_OPERATE").length;
    const winningPositions = state.realPositions.filter((position) => Number(position.unrealized_pnl || 0) > 0).length;
    const losingPositions = state.realPositions.filter((position) => Number(position.unrealized_pnl || 0) < 0).length;
    const bestRealPosition = [...state.realPositions].sort((left, right) => Number(right.unrealized_pnl || 0) - Number(left.unrealized_pnl || 0))[0];
    const worstRealPosition = [...state.realPositions].sort((left, right) => Number(left.unrealized_pnl || 0) - Number(right.unrealized_pnl || 0))[0];
    return {
      monitored: investmentRows.length,
      buyCount,
      noOperateCount,
      realCount: state.realPositions.length,
      winningPositions,
      losingPositions,
      bestRealPosition,
      worstRealPosition,
      gateLabel: gateStatus.replaceAll("_", " ")
    };
  }, [investmentRows, paperSignals, state.realPositions, gateStatus]);
  const movingPortfolioItems = useMemo(() => {
    const topInvestedText = topInvestedRows.length
      ? topInvestedRows.map((position) => `${position.ticker} ${fmtMoney(position.cost_basis)}`).join(" / ")
      : "Sem entradas";
    const gains = state.realPositions
      .filter((position) => Number(position.unrealized_pnl || 0) > 0)
      .reduce((sum, position) => sum + Number(position.unrealized_pnl || 0), 0);
    const losses = state.realPositions
      .filter((position) => Number(position.unrealized_pnl || 0) < 0)
      .reduce((sum, position) => sum + Number(position.unrealized_pnl || 0), 0);
    return [
      { label: "Entradas", value: fmtMoney(realPortfolioSummary.invested) },
      { label: "Lucros", value: fmtMoney(gains), tone: "text-emerald-300" },
      { label: "Perdas", value: fmtMoney(losses), tone: "text-rose-300" },
      { label: "Maiores", value: topInvestedText },
      { label: "Mercado", value: fmtMoney(realPortfolioSummary.marketValue) },
      { label: "Retorno", value: fmtPercent(realPortfolioSummary.unrealizedReturn, 2) },
    ];
  }, [realPortfolioSummary, state.realPositions, topInvestedRows]);
  const isEditingRealPosition = Boolean(realForm.positionId);
  const selectedRealFormAsset = assetByTicker[realForm.ticker] || null;
  const realFormAssetOptions = useMemo(() => {
    const matches = state.assets.filter((asset) => assetMatchesQuery(asset, realFormAssetSearch));
    if (!realForm.ticker || matches.some((asset) => asset.ticker === realForm.ticker)) {
      return matches;
    }
    const selectedAssetOption = state.assets.find((asset) => asset.ticker === realForm.ticker);
    return selectedAssetOption ? [selectedAssetOption, ...matches] : matches;
  }, [state.assets, realForm.ticker, realFormAssetSearch]);

  async function auditConselheiro() {
    setBusyAudit(true);
    try {
      await api.auditConselheiro();
      await loadDashboard();
    } finally {
      setBusyAudit(false);
    }
  }

  async function submitRealPosition(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const editingPositionId = realForm.positionId;
    const quantity = Number(realForm.quantity);
    const entryPrice = Number(realForm.entryPrice);
    if (!realForm.ticker || !Number.isFinite(quantity) || quantity <= 0 || !Number.isFinite(entryPrice) || entryPrice <= 0 || !realForm.entryAt) {
      setError("Preencha ticker, quantidade, preco e data/hora da compra.");
      return;
    }

    setBusyRealSave(true);
    setError(null);
    try {
      const payload = {
        ticker: realForm.ticker,
        quantity,
        entry_price: entryPrice,
        entry_at: new Date(realForm.entryAt).toISOString(),
        notes: realForm.notes.trim() || null
      };
      if (editingPositionId) {
        await api.updateRealPosition(editingPositionId, payload);
      } else {
        await api.registerRealPosition(payload);
      }
      setSelectedTicker(realForm.ticker);
      await loadDashboard(realForm.ticker);
      setRealPortfolioModalOpen(false);
      resetRealForm(realForm.ticker);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : editingPositionId
            ? "Falha ao atualizar posicao real"
            : "Falha ao cadastrar posicao real"
      );
    } finally {
      setBusyRealSave(false);
    }
  }

  async function refreshRealPortfolio() {
    setBusyRealRefresh(true);
    setError(null);
    try {
      await api.markToMarket();
      await loadDashboard();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao atualizar carteira real");
    } finally {
      setBusyRealRefresh(false);
    }
  }

  async function removeRealPortfolioPosition(positionId: string) {
    setBusyRealDeleteId(positionId);
    setError(null);
    try {
      await api.deleteRealPosition(positionId);
      await loadDashboard(selectedTicker);
      setRealDrafts((current) => {
        const next = { ...current };
        delete next[positionId];
        return next;
      });
      if (realForm.positionId === positionId) {
        resetRealForm(realForm.ticker);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao remover posicao real");
    } finally {
      setBusyRealDeleteId(null);
    }
  }

  function updateRealDraft(positionId: string, patch: Partial<RealPositionFormState>) {
    setRealDrafts((current) => ({
      ...current,
      [positionId]: {
        ...(current[positionId] || createEmptyRealForm()),
        ...patch
      }
    }));
  }

  async function saveRealDraft(positionId: string) {
    const draft = realDrafts[positionId];
    if (!draft) return;
    const quantity = Number(draft.quantity);
    const entryPrice = Number(draft.entryPrice);
    if (!draft.ticker || !Number.isFinite(quantity) || quantity <= 0 || !Number.isFinite(entryPrice) || entryPrice <= 0 || !draft.entryAt) {
      setError("Preencha ticker, quantidade, preco e data/hora antes de salvar a linha.");
      return;
    }

    setBusyRealSave(true);
    setError(null);
    try {
      await api.updateRealPosition(positionId, {
        ticker: draft.ticker,
        quantity,
        entry_price: entryPrice,
        entry_at: new Date(draft.entryAt).toISOString(),
        notes: draft.notes.trim() || null
      });
      setSelectedTicker(draft.ticker);
      await loadDashboard(draft.ticker);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao salvar alteracoes da linha");
    } finally {
      setBusyRealSave(false);
    }
  }

  if (loading) return <main className="min-h-screen p-4 md:p-6"><LoadingPanel message="Carregando mesa operacional..." /></main>;

  return (
    <main className="min-h-screen p-4 md:p-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-sky-300/90">
              <CircleDollarSign className="h-4 w-4" /> Profit App Alpha
            </div>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight md:text-4xl">
              <span className="text-gradient">Mesa B3 + Global</span> em paper trading
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">Validacao honesta antes de dinheiro real</p>
          </div>
          <div className="flex w-full flex-wrap items-center gap-2 lg:w-auto lg:justify-end">
            <Badge tone={state.refresh?.is_stale ? "warn" : "good"}>
              {state.refresh?.is_stale ? "Dados Atrasados" : "Dados Atualizados"}
            </Badge>
            <Button onClick={() => void refreshMarketData()} variant="secondary" disabled={busyRefresh || loading}>
              <RefreshCw className={cn("h-4 w-4", busyRefresh && "animate-spin")} />
              {busyRefresh ? "Sincronizando" : "Atualizar"}
            </Button>
            <Button onClick={() => void auditConselheiro()} disabled={busyAudit}><ShieldCheck className="h-4 w-4" /> Conselheiro</Button>
          </div>
        </header>

        {error && <div className="glass rounded-2xl border-rose-400/30 bg-rose-400/5 px-4 py-3 text-sm text-rose-200">{error}</div>}

        <section className="grid gap-3 xl:grid-cols-[1fr_1.35fr_1fr]">
          <div className="glass-strong min-h-[178px] overflow-hidden rounded-2xl p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-emerald-200/90">Oportunidade</div>
                <div className="mt-1 text-xs text-muted-foreground">Melhores janelas de compra do snapshot atual</div>
              </div>
              <Badge tone="good">{marketPanelStats.buyCount} compra</Badge>
            </div>
            <div className="mt-4 space-y-2">
              {opportunityRows.map((row) => (
                <button
                  key={row.ticker}
                  type="button"
                  onClick={() => setSelectedTicker(row.ticker)}
                  className="flex w-full flex-col items-start gap-3 rounded-xl border border-white/10 bg-white/[0.035] px-3 py-3 text-left transition-colors hover:border-emerald-300/30 hover:bg-emerald-300/[0.06] sm:flex-row sm:items-center sm:justify-between sm:py-2"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className={cn("h-2.5 w-2.5 shrink-0 rounded-full", row.signal?.operational_action === "ENTER_LONG" ? "animate-pulse bg-emerald-300 shadow-[0_0_14px_rgba(52,211,153,0.9)]" : "bg-amber-200/80")} />
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold">{row.ticker}</div>
                      <div className="truncate text-xs text-muted-foreground">{actionLabel(row.signal?.operational_action || row.signal?.decision)} · {fmtPercent(row.signal?.net_expected_return, 2)}</div>
                    </div>
                  </div>
                  <div className="w-full text-left text-xs sm:w-auto sm:text-right">
                    <div className="font-semibold text-foreground">{fmtMoney(row.entryPrice)}</div>
                    <div className="text-muted-foreground">alvo {fmtMoney(row.targetPrice)}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="glass-strong relative min-h-[178px] overflow-hidden rounded-2xl p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-sky-200/90">Carteira em movimento</div>
                <div className="mt-2 text-3xl font-semibold tracking-tight">{fmtMoney(realPortfolioSummary.marketValue)}</div>
                <div className={cn("mt-1 text-sm font-medium", (realPortfolioSummary.unrealizedPnl || 0) >= 0 ? "text-emerald-300" : "text-rose-300")}>
                  {fmtMoney(realPortfolioSummary.unrealizedPnl)} · {fmtPercent(realPortfolioSummary.unrealizedReturn, 2)}
                </div>
              </div>
              <Badge tone={realPortfolioSummary.unrealizedPnl >= 0 ? "good" : "bad"}>{marketPanelStats.realCount} entradas</Badge>
            </div>
            <div className="mt-5 grid gap-2 border-y border-white/10 py-3 sm:grid-cols-2 xl:grid-cols-3">
              {movingPortfolioItems.map((item) => (
                <div key={item.label} className="min-w-0 rounded-lg border border-white/10 bg-white/[0.025] px-3 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-sky-100/60">{item.label}</div>
                  <div className={cn("mt-1 truncate text-sm font-semibold text-sky-50", item.tone)} title={item.value}>
                    {item.value}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-muted-foreground">Investido</div><div className="mt-1 font-semibold">{fmtMoney(realPortfolioSummary.invested)}</div></div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-muted-foreground">Ganhos</div><div className="mt-1 font-semibold text-emerald-300">{marketPanelStats.winningPositions}</div></div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-muted-foreground">Perdas</div><div className="mt-1 font-semibold text-rose-300">{marketPanelStats.losingPositions}</div></div>
            </div>
          </div>

          <div className="glass-strong min-h-[178px] rounded-2xl p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-violet-100/90">Estatisticas</div>
                <div className="mt-1 text-xs text-muted-foreground">Sinais, filtros e risco em tempo real</div>
              </div>
              <Badge tone={badgeTone(gateStatus)}>{marketPanelStats.gateLabel}</Badge>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Monitorados</div><div className="mt-1 text-xl font-semibold">{marketPanelStats.monitored}</div></div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Nao operar</div><div className="mt-1 text-xl font-semibold">{marketPanelStats.noOperateCount}</div></div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Melhor P/L</div><div className="mt-1 font-semibold text-emerald-300">{marketPanelStats.bestRealPosition ? `${marketPanelStats.bestRealPosition.ticker} ${fmtMoney(marketPanelStats.bestRealPosition.unrealized_pnl)}` : "-"}</div></div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Pior P/L</div><div className="mt-1 font-semibold text-rose-300">{marketPanelStats.worstRealPosition ? `${marketPanelStats.worstRealPosition.ticker} ${fmtMoney(marketPanelStats.worstRealPosition.unrealized_pnl)}` : "-"}</div></div>
            </div>
          </div>
        </section>

        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as DashboardTab)} className="w-full">
          <TabsList>
            <TabsTrigger value="trending"><TrendingUp className="h-4 w-4" /> Trending</TabsTrigger>
            <TabsTrigger value="predictions"><BrainCircuit className="h-4 w-4" /> Previsoes</TabsTrigger>
            <TabsTrigger value="investments"><WalletCards className="h-4 w-4" /> Meus Investimentos</TabsTrigger>
          </TabsList>

          <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="relative w-full max-w-xl">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="search"
                value={assetSearch}
                onChange={(event) => setAssetSearch(event.target.value)}
                placeholder="Pesquisar empresa ou ticker"
                className="h-11 w-full rounded-2xl border border-white/10 bg-white/5 pl-10 pr-4 text-sm text-foreground outline-none transition-colors focus:border-sky-400/50"
              />
            </div>
            <div className="text-xs text-muted-foreground">
              {filteredAssets.length} de {state.assets.length} ativos visiveis
            </div>
          </div>

          <TabsContent value="trending">
            <section className="grid gap-4 xl:grid-cols-[1.4fr_0.8fr]">
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {filteredAssets.map((asset) => {
                  const prediction = state.predictions[asset.ticker];
                  const paper = prediction?.paper_signal;
                  const fusion = prediction?.fusion_prediction;
                  const technical = prediction?.technical_prediction;
                  const active = selectedTicker === asset.ticker;
                  return (
                    <button key={asset.ticker} onClick={() => setSelectedTicker(asset.ticker)} className={cn("glass glass-hover rounded-2xl p-5 text-left", active && "ring-glow border-sky-400/40")}>
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <AssetLogo asset={asset} size="md" />
                          <div>
                            <div className="text-base font-semibold tracking-tight">{asset.ticker}</div>
                            <div className="mt-1 line-clamp-1 text-xs text-muted-foreground">{asset.name}</div>
                          </div>
                        </div>
                        <Badge tone={badgeTone(fusion?.fused_direction || technical?.predicted_direction)}>{fusion?.fused_direction || modelDirectionLabel(technical?.predicted_direction)}</Badge>
                      </div>
                      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                        <div><div className="text-[10px] uppercase tracking-wider text-muted-foreground">Preco</div><div className="font-semibold text-foreground">{fmtMoney(paper?.reference_price)}</div></div>
                        <div><div className="text-[10px] uppercase tracking-wider text-muted-foreground">Prob. alta</div><div className="font-semibold text-foreground">{fmtPercent(technical?.probability_up ?? fusion?.fused_score, 1)}</div></div>
                      </div>
                      <div className="mt-4 flex items-center justify-between gap-2">
                        <Badge tone={badgeTone(paper?.operational_action || paper?.decision)}>{actionLabel(paper?.operational_action || paper?.decision)}</Badge>
                        <span className="text-[11px] text-muted-foreground">{modelWindowLabel(technical?.target_name)} · {paper?.signal_date || technical?.date || "-"}</span>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="flex flex-col gap-4">
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <AssetLogo asset={selectedAsset} size="lg" />
                      <div>
                        <CardTitle>{selectedAsset?.ticker || selectedTicker || "-"}</CardTitle>
                        <div className="mt-1 text-xs text-muted-foreground">{selectedAsset?.name || "Ativo monitorado"}</div>
                        <div className="mt-1 text-xs text-muted-foreground">Retorno no intervalo: <PriceDelta rows={state.prices} /></div>
                      </div>
                    </div>
                    <div className="flex self-start rounded-xl border border-white/10 bg-white/5 p-1 backdrop-blur sm:self-auto">
                      {(["7d", "3m", "1y"] as Horizon[]).map((value) => (
                        <Button key={value} size="sm" variant={horizon === value ? "default" : "quiet"} onClick={() => setHorizon(value)}>{value}</Button>
                      ))}
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <PriceChart rows={state.prices} />
                    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 text-sm backdrop-blur">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <div className="text-xs uppercase tracking-wider text-muted-foreground">Modelo tecnico atual</div>
                          <div className="mt-1 font-medium text-foreground">Run operacional {selectedPrediction?.model_run_id || "-"}</div>
                        </div>
                        <Badge tone={modelWindowLabel(selectedTechnical?.target_name) === horizon ? "good" : "warn"}>
                          {modelWindowLabel(selectedTechnical?.target_name) === horizon ? "Janela treinada ativa" : `Sem modelo treinado em ${horizon}`}
                        </Badge>
                      </div>
                      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                        <div><div className="text-xs text-muted-foreground">Janela treinada</div><div className="font-semibold">{modelWindowLabel(selectedTechnical?.target_name)}</div></div>
                        <div><div className="text-xs text-muted-foreground">Direcao</div><div className="font-semibold">{modelDirectionLabel(selectedTechnical?.predicted_direction)}</div></div>
                        <div><div className="text-xs text-muted-foreground">Prob. alta</div><div className="font-semibold">{fmtPercent(selectedTechnical?.probability_up, 1)}</div></div>
                        <div><div className="text-xs text-muted-foreground">Retorno esperado</div><div className="font-semibold">{fmtPercent(selectedTechnical?.expected_return, 2)}</div></div>
                      </div>
                    </div>
                    {selectedAsset && (
                      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm backdrop-blur">
                        <div>
                          <div className="font-medium text-foreground">Referencia oficial do TradingView</div>
                          <div className="text-xs text-muted-foreground">Abertura do ativo selecionado na ficha oficial e no Supercharts.</div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <a
                            href={tradingViewSymbolPageUrl(selectedAsset.ticker)}
                            target="_blank"
                            rel="noopener noreferrer nofollow"
                            className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-foreground transition-colors hover:border-white/20 hover:bg-white/10"
                          >
                            Ficha do ativo <ExternalLink className="h-3.5 w-3.5" />
                          </a>
                          <a
                            href={tradingViewChartUrl(selectedAsset.ticker)}
                            target="_blank"
                            rel="noopener noreferrer nofollow"
                            className="inline-flex items-center gap-2 rounded-xl border border-sky-400/30 bg-sky-400/10 px-3 py-2 text-xs font-medium text-sky-100 transition-colors hover:border-sky-300/40 hover:bg-sky-400/15"
                          >
                            Abrir no Supercharts <ExternalLink className="h-3.5 w-3.5" />
                          </a>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {selectedAsset && (
                  <Card>
                    <CardHeader>
                      <div>
                        <CardTitle>TradingView oficial</CardTitle>
                        <div className="mt-1 text-xs text-muted-foreground">Logo e resumo do ativo via widget oficial do TradingView para {selectedAsset.ticker}.</div>
                      </div>
                      <Badge tone="info">{selectedAsset.ticker}</Badge>
                    </CardHeader>
                    <CardContent>
                      <TradingViewSymbolInfo ticker={selectedAsset.ticker} name={selectedAsset.name} />
                    </CardContent>
                  </Card>
                )}
              </div>
            </section>
          </TabsContent>

          <TabsContent value="predictions">
            <section className="grid gap-4 lg:grid-cols-[1fr_360px]">
              <div className="grid gap-3 xl:grid-cols-2">
                {filteredAssets.map((asset) => {
                  const prediction = state.predictions[asset.ticker];
                  const signal = prediction?.paper_signal;
                  const technical = prediction?.technical_prediction;
                  const thesis = thesisFor(signal);
                  const regime = thesis?.regime_gate || parseJson<Record<string, any>>(prediction?.fusion_prediction?.explanation_json || null)?.regime || {};
                  return (
                    <Card key={asset.ticker}>
                      <CardHeader>
                        <div className="flex items-center gap-3">
                          <AssetLogo asset={asset} size="sm" />
                          <div>
                            <CardTitle>{asset.ticker}</CardTitle>
                            <div className="mt-1 text-xs text-muted-foreground">{asset.name} · {signal?.signal_date || "sem sinal"}</div>
                          </div>
                        </div>
                        <Badge tone={badgeTone(signal?.operational_action || signal?.decision)}>{actionLabel(signal?.operational_action || signal?.decision)}</Badge>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
                          <div><div className="text-xs text-muted-foreground">Entrada</div><div className="font-semibold">{fmtMoney(signal?.suggested_entry)}</div></div>
                          <div><div className="text-xs text-muted-foreground">Stop</div><div className="font-semibold">{fmtMoney(signal?.stop_loss)}</div></div>
                          <div><div className="text-xs text-muted-foreground">Alvo</div><div className="font-semibold">{fmtMoney(signal?.target_price)}</div></div>
                          <div><div className="text-xs text-muted-foreground">Qtd</div><div className="font-semibold">{signal?.max_shares ?? "-"}</div></div>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
                          <div><div className="text-xs text-muted-foreground">Modelo</div><div className="font-semibold">{modelWindowLabel(technical?.target_name)}</div></div>
                          <div><div className="text-xs text-muted-foreground">Direcao tecnica</div><div className="font-semibold">{modelDirectionLabel(technical?.predicted_direction)}</div></div>
                          <div><div className="text-xs text-muted-foreground">Prob. alta</div><div className="font-semibold">{fmtPercent(technical?.probability_up, 1)}</div></div>
                          <div><div className="text-xs text-muted-foreground">Retorno esp.</div><div className="font-semibold">{fmtPercent(technical?.expected_return, 2)}</div></div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <Badge tone={badgeTone(regime.regime)}>{regime.regime || "regime"}</Badge>
                          <Badge tone={regime.override_qualitative ? "warn" : "neutral"}>override {String(Boolean(regime.override_qualitative))}</Badge>
                          <Badge tone="info">vol {fmtPercent(regime.volatility_percentile, 0)}</Badge>
                          <Badge tone={badgeTone(signal?.block_reason)}>{signal?.block_reason || "sem bloqueio"}</Badge>
                        </div>
                        <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3 text-xs leading-5 text-muted-foreground backdrop-blur">
                          <div>Kelly: {fmtPercent(thesis?.sizing?.kelly_fraction_used, 2)} · ATR: {fmtNumber(thesis?.sizing?.atr_14, 3)} · Custo pre-IR: {fmtPercent(thesis?.b3_costs?.total_pre_ir, 3)}</div>
                          <div>Tags: {(thesis?.qualitative_context?.event_tags || []).join(", ") || "-"}</div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Gate 90 dias</CardTitle>
                  <Badge tone={badgeTone(gateStatus)}>{gateStatus.replaceAll("_", " ")}</Badge>
                </CardHeader>
                <CardContent className="space-y-4">
                  <GateChart gates={state.gate?.gates || {}} />
                    <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                    <div><div className="text-xs text-muted-foreground">Trades fechados</div><div className="font-semibold">{state.gate?.metrics?.closed_trades ?? 0}</div></div>
                    <div><div className="text-xs text-muted-foreground">Sharpe</div><div className="font-semibold">{fmtNumber(state.gate?.metrics?.sharpe_net, 2)}</div></div>
                    <div><div className="text-xs text-muted-foreground">Drawdown</div><div className="font-semibold">{fmtPercent(state.gate?.metrics?.max_drawdown, 1)}</div></div>
                    <div><div className="text-xs text-muted-foreground">Profit factor</div><div className="font-semibold">{fmtNumber(state.gate?.metrics?.profit_factor, 2)}</div></div>
                  </div>
                  <div className="space-y-2">
                    {blockedReasons.map(([reason, count]) => (
                      <div key={reason} className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-xs backdrop-blur">
                        <span className="truncate pr-2">{reason}</span><Badge tone="warn">{String(count)}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </section>
          </TabsContent>

          <TabsContent value="investments">
            <div className="flex flex-col gap-4">
              <Card>
                <CardHeader>
                  <div>
                    <CardTitle>Cockpit do trader</CardTitle>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Visão sempre visível: P/L em aberto, exposição, risco em ATR, drawdown desde pico e situação do gate v2 hoje.
                    </div>
                  </div>
                  <Badge tone={traderCockpit.gateTone}>Gate hoje: {traderCockpit.gateLabel}</Badge>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-5">
                    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                      <div className="text-xs text-muted-foreground">P/L em aberto</div>
                      <div className={cn("mt-1 text-lg font-semibold", traderCockpit.openPnl >= 0 ? "text-emerald-300" : "text-rose-300")}>
                        {fmtMoney(traderCockpit.openPnl)}
                      </div>
                      <div className="text-xs text-muted-foreground">{fmtPercent(traderCockpit.openPnlReturn, 2)}</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                      <div className="text-xs text-muted-foreground">Exposição total</div>
                      <div className="mt-1 text-lg font-semibold text-foreground">{fmtMoney(traderCockpit.exposure)}</div>
                      <div className="text-xs text-muted-foreground">{realPortfolioSummary.count} posições abertas</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                      <div className="text-xs text-muted-foreground">Risco em aberto (Σ |Δ| × ATR)</div>
                      <div className="mt-1 text-lg font-semibold text-foreground">{fmtMoney(traderCockpit.openRisk)}</div>
                      <div className="text-xs text-muted-foreground">Σ qty × preço × ATR%</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                      <div className="text-xs text-muted-foreground">Drawdown desde pico</div>
                      <div className={cn("mt-1 text-lg font-semibold", (traderCockpit.drawdown ?? 0) <= -0.05 ? "text-rose-300" : "text-foreground")}>
                        {traderCockpit.drawdown === null ? "—" : fmtPercent(traderCockpit.drawdown, 1)}
                      </div>
                      <div className="text-xs text-muted-foreground">{state.gate?.metrics?.closed_trades ?? 0} trades fechados</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                      <div className="text-xs text-muted-foreground">Gate v2 permite operar?</div>
                      <div className="mt-1 flex items-center gap-2">
                        <Badge tone={traderCockpit.gateTone}>{traderCockpit.gateTone === "good" ? "SIM" : "NÃO"}</Badge>
                        <span className="text-xs text-muted-foreground">{traderCockpit.gateLabel}</span>
                      </div>
                      <div className="text-xs text-muted-foreground">Sharpe {fmtNumber(state.gate?.metrics?.sharpe_net, 2)} · PF {fmtNumber(state.gate?.metrics?.profit_factor, 2)}</div>
                    </div>
                  </div>
                  {traderCockpit.divergentPositions.length > 0 && (
                    <div className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-3">
                      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-amber-200">
                        <ShieldCheck className="h-3.5 w-3.5" /> Divergência modelo × posição
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2 text-xs">
                        {traderCockpit.divergentPositions.map((entry) => (
                          <span key={entry.ticker} className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.05] px-2 py-1">
                            <span className="font-medium text-foreground">{entry.ticker}</span>
                            <Badge tone={entry.intentTone} className="text-[10px]">{entry.intentLabel}</Badge>
                            <span className={cn(entry.pnl >= 0 ? "text-emerald-300" : "text-rose-300")}>{fmtMoney(entry.pnl)}</span>
                          </span>
                        ))}
                      </div>
                      <div className="mt-2 text-[11px] text-muted-foreground">
                        Modelo recomenda sair ou não operar nesses tickers que ainda estão em carteira. Reavalie a tese antes do próximo pregão.
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
              <section className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.2fr)_360px]">
                <Card>
                  <CardHeader>
                    <div>
                      <CardTitle>Carteira e intencoes</CardTitle>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {filteredInvestmentRows.length} de {investmentRows.length} ativos monitorados · {openInvestmentCount} em carteira
                      </div>
                    </div>
                    <div className="flex w-full flex-col items-start gap-2 sm:w-auto sm:items-end">
                      <div className="flex items-center gap-1.5 text-xs uppercase tracking-wider text-muted-foreground">
                        <Filter className="h-3.5 w-3.5" /> Filtro
                      </div>
                      <div className="flex w-full gap-1.5 overflow-x-auto pb-1 scrollbar-thin sm:w-auto sm:flex-wrap sm:overflow-visible sm:pb-0">
                        {INTENT_FILTERS.map((filter) => (
                          <button
                            key={filter.value}
                            type="button"
                            onClick={() => setInvestmentIntentFilter(filter.value)}
                            className={cn(
                              "inline-flex h-8 items-center gap-2 rounded-lg border px-3 text-xs font-medium transition-colors",
                              investmentIntentFilter === filter.value
                                ? "border-sky-400/45 bg-sky-400/15 text-sky-100"
                                : "border-white/10 bg-white/[0.03] text-muted-foreground hover:border-white/20 hover:bg-white/[0.06] hover:text-foreground"
                            )}
                          >
                            {filter.label}
                            <Badge tone={filter.tone} className="min-h-5 px-1.5 text-[10px]">{investmentFilterCounts[filter.value]}</Badge>
                          </button>
                        ))}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0">
                    {filteredInvestmentRows.length === 0 && <div className="p-6 text-sm text-muted-foreground">Nenhum ativo neste filtro.</div>}
                    {filteredInvestmentRows.length > 0 && (
                      <div className="grid gap-3 p-4 xl:grid-cols-2">
                        {filteredInvestmentRows.map((row) => (
                          <button
                            key={row.ticker}
                            type="button"
                            onClick={() => setSelectedTicker(row.ticker)}
                            className={cn(
                              "w-full rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-left transition-colors hover:border-white/20 hover:bg-white/[0.05]",
                              selectedTicker === row.ticker && "ring-glow border-sky-400/35"
                            )}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="flex min-w-0 items-center gap-3">
                                <AssetLogo asset={row.asset} size="sm" />
                                <div className="min-w-0">
                                  <div className="font-medium">{row.ticker}</div>
                                  <div className="truncate text-xs text-muted-foreground">{row.name}</div>
                                </div>
                              </div>
                              <Badge tone={row.intentTone}>{row.intentLabel}</Badge>
                            </div>
                            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <Badge tone={row.statusTone}>{row.statusLabel}</Badge>
                                <span className="text-xs text-muted-foreground">{row.reviewLabel}</span>
                              </div>
                              <span className="text-xs text-muted-foreground">{row.timeLabel}</span>
                            </div>
                            <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
                              <div>
                                <div className="text-xs text-muted-foreground">Quando</div>
                                <div className="mt-1 font-medium text-foreground">{row.whenLabel}</div>
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Atual</div>
                                <div className="mt-1 font-medium text-foreground">{fmtMoney(row.currentPrice)}</div>
                              </div>
                            </div>
                            <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                              <div className="text-xs uppercase tracking-wider text-muted-foreground">Motivo</div>
                              <div className="mt-2 line-clamp-3 text-sm leading-6 text-foreground">{row.reasonLabel}</div>
                            </div>
                            <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-2">
                                <div>Entrada</div>
                                <div className="mt-1 font-medium text-foreground">{fmtMoney(row.entryPrice)}</div>
                              </div>
                              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-2">
                                <div>Stop</div>
                                <div className="mt-1 font-medium text-foreground">{fmtMoney(row.trailingStop ?? row.stopLoss)}</div>
                              </div>
                              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-2">
                                <div>Alvo</div>
                                <div className="mt-1 font-medium text-foreground">{fmtMoney(row.targetPrice)}</div>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                <div className="flex flex-col gap-4 lg:sticky lg:top-6">
                  <Card>
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <AssetLogo asset={selectedInvestment?.asset || selectedAsset} size="md" />
                        <CardTitle>Plano operacional</CardTitle>
                      </div>
                      <Badge tone={selectedInvestment?.intentTone || "neutral"}>{selectedInvestment?.ticker || "-"}</Badge>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {selectedInvestment ? (
                        <>
                          <div>
                            <div className="text-lg font-semibold">{selectedInvestment.name}</div>
                            <div className="mt-1 text-xs text-muted-foreground">{selectedInvestment.reviewLabel}</div>
                          </div>
                          <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                            <div><div className="text-xs text-muted-foreground">Acao</div><div className="font-semibold">{selectedInvestment.intentLabel}</div></div>
                            <div><div className="text-xs text-muted-foreground">Estado</div><div className="font-semibold">{selectedInvestment.statusLabel}</div></div>
                            <div><div className="text-xs text-muted-foreground">Quando</div><div className="font-semibold">{selectedInvestment.whenLabel}</div></div>
                            <div><div className="text-xs text-muted-foreground">Horario</div><div className="font-semibold">{selectedInvestment.timeLabel}</div></div>
                            <div><div className="text-xs text-muted-foreground">Entrada</div><div className="font-semibold">{fmtMoney(selectedInvestment.entryPrice)}</div></div>
                            <div><div className="text-xs text-muted-foreground">Atual</div><div className="font-semibold">{fmtMoney(selectedInvestment.currentPrice)}</div></div>
                            <div><div className="text-xs text-muted-foreground">Stop</div><div className="font-semibold">{fmtMoney(selectedInvestment.trailingStop ?? selectedInvestment.stopLoss)}</div></div>
                            <div><div className="text-xs text-muted-foreground">Alvo</div><div className="font-semibold">{fmtMoney(selectedInvestment.targetPrice)}</div></div>
                          </div>
                          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-sm backdrop-blur">
                            <div className="text-xs uppercase tracking-wider text-muted-foreground">Motivo principal</div>
                            <div className="mt-2 font-medium text-foreground">{selectedInvestment.reasonLabel}</div>
                          </div>
                          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-sm backdrop-blur">
                            <div className="text-xs uppercase tracking-wider text-muted-foreground">Por que</div>
                            <div className="mt-2 space-y-2 text-muted-foreground">
                              {selectedInvestment.whyLines.map((line) => (
                                <div key={line}>{line}</div>
                              ))}
                              {selectedInvestment.whyLines.length === 0 && <div>Sem contexto adicional para este ativo.</div>}
                            </div>
                          </div>
                        </>
                      ) : (
                        <div className="text-sm text-muted-foreground">Selecione um ativo para ver o plano.</div>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <AssetLogo asset={selectedAsset} size="sm" />
                        <CardTitle>Racional do sinal</CardTitle>
                      </div>
                      <Badge tone={badgeTone(selectedSignal?.operational_action || selectedSignal?.decision)}>{selectedTicker || "-"}</Badge>
                    </CardHeader>
                    <CardContent className="space-y-4 text-sm">
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        <div><div className="text-xs text-muted-foreground">Prob. ganho</div><div className="font-semibold">{fmtPercent(selectedSignal?.probability_win, 1)}</div></div>
                        <div><div className="text-xs text-muted-foreground">Retorno esperado</div><div className="font-semibold">{fmtPercent(selectedSignal?.net_expected_return, 2)}</div></div>
                        <div><div className="text-xs text-muted-foreground">Qtd sugerida</div><div className="font-semibold">{selectedSignal?.max_shares ?? "-"}</div></div>
                        <div><div className="text-xs text-muted-foreground">Horizonte</div><div className="font-semibold">{selectedSignal?.horizon || "-"}</div></div>
                      </div>
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-xs leading-5 text-muted-foreground backdrop-blur">
                        <div>Regime: {selectedThesis?.regime_gate?.regime || "-"} · Kelly: {fmtPercent(selectedThesis?.sizing?.kelly_fraction_used, 2)} · ATR: {fmtNumber(selectedThesis?.sizing?.atr_14, 3)}</div>
                        <div>Setor: {selectedThesis?.sizing?.sector || "-"} · Custo pre-IR: {fmtPercent(selectedThesis?.b3_costs?.total_pre_ir, 3)}</div>
                        <div>Tags: {(selectedThesis?.qualitative_context?.event_tags || []).join(", ") || "-"}</div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </section>

              <section>
                <Card>
                  <CardHeader>
                    <div>
                      <CardTitle>Carteira real cadastrada</CardTitle>
                      <div className="mt-1 text-xs text-muted-foreground">
                        Posicoes manuais com edicao direta na linha e atualizacao pelo ultimo fechamento disponivel.
                      </div>
                    </div>
                    <div className="flex w-full flex-col items-start gap-2 sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:justify-end">
                      <Badge tone="info">{realPortfolioSummary.count}</Badge>
                      <Button size="sm" onClick={() => openRealPositionModal()} disabled={busyRealSave}>
                        <Plus className="h-3.5 w-3.5" /> Cadastrar Entrada
                      </Button>
                      <Button size="sm" variant="secondary" onClick={() => void refreshRealPortfolio()} disabled={busyRealRefresh}>
                        <RefreshCw className="h-3.5 w-3.5" /> Atualizar carteira real
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0">
                    {filteredRealPortfolioRows.length === 0 && <div className="p-6 text-sm text-muted-foreground">Nenhuma posicao real neste filtro.</div>}
                    {filteredRealPortfolioRows.length > 0 && (
                      <div className="space-y-3 p-4 lg:hidden">
                        {filteredRealPortfolioRows.map((row) => {
                          const intent = investmentRows.find((entry) => entry.ticker === row.position.ticker);
                          return (
                            <div
                              key={row.position.position_id}
                              onClick={() => setSelectedTicker(row.position.ticker)}
                              className={cn(
                                "rounded-2xl border border-white/10 bg-white/[0.03] p-4 transition-colors hover:border-white/20 hover:bg-white/[0.05]",
                                selectedTicker === row.position.ticker && "ring-glow border-sky-400/35"
                              )}
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="flex min-w-0 items-center gap-3">
                                  <AssetLogo asset={row.asset} size="sm" />
                                  <div className="min-w-0">
                                    <div className="font-medium">{row.position.ticker}</div>
                                    <div className="truncate text-xs text-muted-foreground">{row.assetName}</div>
                                  </div>
                                </div>
                                <Badge tone={intent?.intentTone || row.intentTone}>{intent?.intentLabel || row.intentLabel}</Badge>
                              </div>
                              <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                                <div>
                                  <div className="text-xs text-muted-foreground">Quantidade</div>
                                  <div className="mt-1 font-medium">{row.position.quantity}</div>
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">Compra</div>
                                  <div className="mt-1 font-medium">{fmtMoney(row.position.entry_price)}</div>
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">Atual</div>
                                  <div className="mt-1 font-medium">{fmtMoney(row.position.current_price)}</div>
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">P/L</div>
                                  <div className={cn("mt-1 font-medium", Number(row.position.unrealized_pnl || 0) >= 0 ? "text-emerald-300" : "text-rose-300")}>{fmtMoney(row.position.unrealized_pnl)}</div>
                                </div>
                              </div>
                              <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.03] p-3 text-sm">
                                <div className="text-xs uppercase tracking-wider text-muted-foreground">Quando revisar</div>
                                <div className="mt-2 font-medium text-foreground">{intent?.whenLabel || row.whenLabel}</div>
                                <div className="mt-1 text-xs text-muted-foreground">{intent?.timeLabel || row.timeLabel}</div>
                              </div>
                              <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.03] p-3 text-sm">
                                <div className="text-xs uppercase tracking-wider text-muted-foreground">Observacao</div>
                                <div className="mt-2 leading-6 text-foreground">{intent?.reasonLabel || row.reasonLabel}</div>
                              </div>
                              <div className="mt-3 flex flex-wrap gap-2">
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  disabled={busyRealSave}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    openRealPositionModal(row.position);
                                  }}
                                >
                                  <Edit3 className="h-3.5 w-3.5" /> Editar
                                </Button>
                                <Button
                                  size="sm"
                                  variant="danger"
                                  disabled={busyRealDeleteId === row.position.position_id}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void removeRealPortfolioPosition(row.position.position_id);
                                  }}
                                >
                                  <Trash2 className="h-3.5 w-3.5" /> Excluir
                                </Button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="hidden overflow-x-auto scrollbar-thin lg:block">
                    <table className="w-full min-w-[1260px] border-collapse text-sm">
                      <thead className="bg-white/[0.04] text-[11px] uppercase tracking-wider text-muted-foreground">
                        <tr>
                          <th className="w-[230px] px-4 py-3 text-left font-medium">Ativo</th>
                          <th className="w-[110px] px-4 py-3 text-right font-medium">Qtd</th>
                          <th className="w-[150px] px-4 py-3 text-right font-medium">Compra</th>
                          <th className="w-[190px] px-4 py-3 text-left font-medium">Data/hora</th>
                          <th className="px-4 py-3 text-right font-medium">Atual</th>
                          <th className="px-4 py-3 text-right font-medium">P/L</th>
                          <th className="px-4 py-3 text-left font-medium">Intencao atual</th>
                          <th className="px-4 py-3 text-left font-medium">Quando revisar</th>
                          <th className="w-[260px] px-4 py-3 text-left font-medium">Observacao</th>
                          <th className="w-[132px] px-4 py-3 text-right font-medium">Acoes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredRealPortfolioRows.map((row) => {
                          const draft = realDrafts[row.position.position_id] || createRealFormFromPosition(row.position);
                          const draftAsset = assetByTicker[draft.ticker] || row.asset;
                          const draftIntent = investmentRows.find((entry) => entry.ticker === draft.ticker);
                          const draftQuantity = Number(draft.quantity);
                          const draftEntryPrice = Number(draft.entryPrice);
                          const draftCurrentPrice = Number(draftIntent?.currentPrice ?? row.position.current_price ?? 0);
                          const liveCostBasis = Number.isFinite(draftQuantity) && Number.isFinite(draftEntryPrice) ? draftQuantity * draftEntryPrice : row.position.cost_basis;
                          const liveMarketValue = Number.isFinite(draftQuantity) && Number.isFinite(draftCurrentPrice) ? draftQuantity * draftCurrentPrice : row.position.market_value;
                          const livePnl = liveMarketValue - liveCostBasis;
                          const liveReturn = liveCostBasis > 0 ? liveMarketValue / liveCostBasis - 1.0 : null;
                          return (
                            <tr
                              key={row.position.position_id}
                              onClick={() => setSelectedTicker(row.position.ticker)}
                              className={cn(
                                "border-t border-white/5 transition-colors hover:bg-white/[0.03]",
                                selectedTicker === row.position.ticker && "bg-white/[0.04]"
                              )}
                            >
                              <td className="px-4 py-3">
                                <div className="flex items-center gap-3">
                                  <AssetLogo asset={draftAsset} size="sm" />
                                  <div className="min-w-0 flex-1">
                                    <select
                                      value={draft.ticker}
                                      onClick={(event) => event.stopPropagation()}
                                      onChange={(event) => updateRealDraft(row.position.position_id, { ticker: event.target.value })}
                                      className="h-9 w-full rounded-lg border border-white/10 bg-white/5 px-2.5 text-sm font-medium text-foreground outline-none focus:border-sky-400/50"
                                    >
                                      {state.assets.map((asset) => (
                                        <option key={asset.ticker} value={asset.ticker}>{asset.ticker} · {asset.name}</option>
                                      ))}
                                    </select>
                                    <div className="mt-1 truncate text-xs text-muted-foreground">{draftAsset?.name || row.assetName}</div>
                                  </div>
                                </div>
                              </td>
                              <td className="px-4 py-3 text-right font-medium">
                                <input
                                  type="number"
                                  min="1"
                                  step="1"
                                  value={draft.quantity}
                                  onClick={(event) => event.stopPropagation()}
                                  onChange={(event) => updateRealDraft(row.position.position_id, { quantity: event.target.value })}
                                  className="h-9 w-24 rounded-lg border border-white/10 bg-white/5 px-2.5 text-right text-sm text-foreground outline-none focus:border-sky-400/50"
                                />
                              </td>
                              <td className="px-4 py-3 text-right">
                                <input
                                  type="number"
                                  min="0"
                                  step="0.01"
                                  value={draft.entryPrice}
                                  onClick={(event) => event.stopPropagation()}
                                  onChange={(event) => updateRealDraft(row.position.position_id, { entryPrice: event.target.value })}
                                  className="h-9 w-32 rounded-lg border border-white/10 bg-white/5 px-2.5 text-right text-sm text-foreground outline-none focus:border-sky-400/50"
                                />
                                <div className="text-xs text-muted-foreground">Investido {fmtMoney(liveCostBasis)}</div>
                              </td>
                              <td className="px-4 py-3 text-xs text-muted-foreground">
                                <input
                                  type="datetime-local"
                                  value={draft.entryAt}
                                  onClick={(event) => event.stopPropagation()}
                                  onChange={(event) => updateRealDraft(row.position.position_id, { entryAt: event.target.value })}
                                  className="h-9 w-44 rounded-lg border border-white/10 bg-white/5 px-2.5 text-xs text-foreground outline-none focus:border-sky-400/50"
                                />
                                <div>{row.updatedLabel}</div>
                              </td>
                              <td className="px-4 py-3 text-right">
                                <div className="font-medium">{fmtMoney(draftCurrentPrice)}</div>
                                <div className="text-xs text-muted-foreground">Mercado {fmtMoney(liveMarketValue)}</div>
                              </td>
                              <td className={cn("px-4 py-3 text-right", livePnl >= 0 ? "text-emerald-300" : "text-rose-300")}>
                                <div className="font-medium">{fmtMoney(livePnl)}</div>
                                <div className="text-xs">{fmtPercent(liveReturn, 2)}</div>
                              </td>
                              <td className="px-4 py-3"><Badge tone={draftIntent?.intentTone || row.intentTone}>{draftIntent?.intentLabel || row.intentLabel}</Badge></td>
                              <td className="px-4 py-3 text-sm">
                                <div>{draftIntent?.whenLabel || row.whenLabel}</div>
                                <div className="text-xs text-muted-foreground">{draftIntent?.timeLabel || row.timeLabel}</div>
                              </td>
                              <td className="px-4 py-3">
                                <textarea
                                  rows={2}
                                  value={draft.notes}
                                  onClick={(event) => event.stopPropagation()}
                                  onChange={(event) => updateRealDraft(row.position.position_id, { notes: event.target.value })}
                                  className="w-full resize-none rounded-lg border border-white/10 bg-white/5 px-2.5 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                                  placeholder={draftIntent?.reasonLabel || row.reasonLabel}
                                />
                              </td>
                              <td className="px-4 py-3 text-right">
                                <div className="flex justify-end gap-2">
                                  <Button
                                    size="icon"
                                    variant="secondary"
                                    title="Salvar linha"
                                    disabled={busyRealSave}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      void saveRealDraft(row.position.position_id);
                                    }}
                                  >
                                    <Save className="h-4 w-4" />
                                  </Button>
                                  <Button
                                    size="icon"
                                    variant="secondary"
                                    title="Abrir edicao completa"
                                    disabled={busyRealSave}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      openRealPositionModal(row.position);
                                    }}
                                  >
                                    <Edit3 className="h-4 w-4" />
                                  </Button>
                                  <Button
                                    size="icon"
                                    variant="danger"
                                    title="Remover entrada"
                                    disabled={busyRealDeleteId === row.position.position_id}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      void removeRealPortfolioPosition(row.position.position_id);
                                    }}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                    </div>
                  </CardContent>
                </Card>

              </section>

              {realPortfolioModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-sm" role="dialog" aria-modal="true">
                  <div className="glass-strong max-h-[92vh] w-full max-w-2xl overflow-y-auto rounded-2xl p-5 shadow-2xl">
                    <div className="mb-5 flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2">
                          <CardTitle>{isEditingRealPosition ? "Editar entrada" : "Cadastrar Entrada"}</CardTitle>
                          <Badge tone={isEditingRealPosition ? "info" : "good"}>{isEditingRealPosition ? "Edicao" : "Novo"}</Badge>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {isEditingRealPosition ? "Ajuste os dados da posicao e salve para atualizar a tabela." : "Inclua um ativo manualmente; ao salvar, a tabela ja recarrega com a nova entrada."}
                        </div>
                      </div>
                      <Button type="button" size="icon" variant="quiet" onClick={closeRealPositionModal} disabled={busyRealSave} title="Fechar">
                        <X className="h-4 w-4" />
                      </Button>
                    </div>

                    <div className="mb-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Posicoes</div><div className="font-semibold">{realPortfolioSummary.count}</div></div>
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Investido</div><div className="font-semibold">{fmtMoney(realPortfolioSummary.invested)}</div></div>
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">Mercado</div><div className="font-semibold">{fmtMoney(realPortfolioSummary.marketValue)}</div></div>
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><div className="text-xs text-muted-foreground">P/L total</div><div className={cn("font-semibold", (realPortfolioSummary.unrealizedPnl || 0) >= 0 ? "text-emerald-300" : "text-rose-300")}>{fmtMoney(realPortfolioSummary.unrealizedPnl)}</div></div>
                    </div>

                    <form className="space-y-4" onSubmit={(event) => void submitRealPosition(event)}>
                      {selectedRealFormAsset && (
                        <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-3 text-sm backdrop-blur">
                          <AssetLogo asset={selectedRealFormAsset} size="md" />
                          <div>
                            <div className="font-medium">{selectedRealFormAsset.ticker}</div>
                            <div className="text-xs text-muted-foreground">{selectedRealFormAsset.name}</div>
                          </div>
                        </div>
                      )}
                      <div>
                        <label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">Buscar ativo</label>
                        <div className="relative">
                          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                          <input
                            type="search"
                            value={realFormAssetSearch}
                            onChange={(event) => setRealFormAssetSearch(event.target.value)}
                            placeholder="Digite ticker ou empresa"
                            className="w-full rounded-xl border border-white/10 bg-white/5 pl-10 pr-3 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                          />
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">{realFormAssetOptions.length} resultado(s)</div>
                      </div>
                      <div>
                        <label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">Empresa</label>
                        <select
                          value={realForm.ticker}
                          onChange={(event) => setRealForm((current) => ({ ...current, ticker: event.target.value }))}
                          className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                        >
                          {realFormAssetOptions.map((asset) => (
                            <option key={asset.ticker} value={asset.ticker}>{asset.ticker} · {asset.name}</option>
                          ))}
                        </select>
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div>
                          <label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">Quantidade</label>
                          <input
                            type="number"
                            min="1"
                            step="1"
                            value={realForm.quantity}
                            onChange={(event) => setRealForm((current) => ({ ...current, quantity: event.target.value }))}
                            className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">Preco de compra</label>
                          <input
                            type="number"
                            min="0"
                            step="0.01"
                            value={realForm.entryPrice}
                            onChange={(event) => setRealForm((current) => ({ ...current, entryPrice: event.target.value }))}
                            className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">Data e hora da compra</label>
                        <input
                          type="datetime-local"
                          value={realForm.entryAt}
                          onChange={(event) => setRealForm((current) => ({ ...current, entryAt: event.target.value }))}
                          className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">Observacoes</label>
                        <textarea
                          rows={3}
                          value={realForm.notes}
                          onChange={(event) => setRealForm((current) => ({ ...current, notes: event.target.value }))}
                          className="w-full resize-none rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-foreground outline-none focus:border-sky-400/50"
                          placeholder="Ex.: compra de longo prazo, aporte mensal..."
                        />
                      </div>
                      <div className="flex flex-wrap items-center justify-end gap-2 border-t border-white/10 pt-4">
                        <Button type="button" variant="secondary" disabled={busyRealSave} onClick={closeRealPositionModal}>Cancelar</Button>
                        <Button type="submit" disabled={busyRealSave}>
                          {busyRealSave ? "Salvando..." : isEditingRealPosition ? "Salvar alteracoes" : "Cadastrar na carteira"}
                        </Button>
                      </div>
                    </form>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}
