from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from app.backtesting.strategy import run_walk_forward_backtest
from app.data.database import (
    get_qualitative_features,
    get_paper_positions,
    get_paper_trading_signals,
    initialize_database,
    get_trade_outcome_runs,
    read_latest_operational_trade_outcomes,
    read_ohlcv_prices,
    read_model_predictions,
    read_operational_predictions,
    read_technical_features,
    save_paper_trading_signals,
)
from app.features.technical import build_current_technical_features
from app.models.fusion import calculate_fused_score, choose_context_for_signal
from app.models.registry import get_best_current_schema_model_run_id
from app.trading.costs import apply_costs_to_gross_return, compute_cost_breakdown
from app.trading.regime import assess_regime, estimate_volatility_percentile
from app.trading.sizing import decisive_win_probability, size_position, sector_for


PAPER_SIGNAL_VERSION = "v8_b3_kelly_regime_gate"
STRICT_NO_OPERATE_VOL_PERCENTILE = 0.80
STRICT_NO_OPERATE_DIVERGENCE = 0.35
EVENT_RECENCY_DAYS = 2
ATR_WINDOW = 14
REWARD_RISK_TOLERANCE = 1e-9

OPERATIONAL_ACTION_ENTER_LONG = "ENTER_LONG"
OPERATIONAL_ACTION_WATCHLIST = "WATCHLIST"
OPERATIONAL_ACTION_NO_TRADE = "NO_TRADE"
OPERATIONAL_ACTION_LEGACY_SIMULATE = "LEGACY_SIMULATE_LONG"
OPERATIONAL_ACTION_LEGACY_BLOCK = "LEGACY_NO_TRADE"

MIN_TRADE_OUTCOME_TEST_TRADES = 20
MIN_TRADE_OUTCOME_TEST_WIN_RATE = 0.45
MIN_TECHNICAL_FALLBACK_TRADES = 20
MIN_TECHNICAL_FALLBACK_PROFITABLE_TICKERS = 2
TECHNICAL_FALLBACK_THRESHOLD_GRID = tuple(round(value, 3) for value in np.arange(0.50, 0.951, 0.05))


@dataclass(frozen=True)
class PaperTradingPolicy:
    portfolio_value: float = 10000.0
    max_risk_per_trade: float = 0.01
    min_confidence: float = 0.48
    min_reward_risk_ratio: float = 1.5
    cost_per_trade: float = 0.002
    spread: float = 0.001
    slippage: float = 0.001
    max_volatility_21d: float = 0.045
    horizon: str = "7d"
    require_strategy_edge: bool = True

    @property
    def total_execution_drag(self) -> float:
        return self.cost_per_trade + self.spread + self.slippage


def calculate_position_size(
    portfolio_value: float,
    max_risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
) -> tuple[float, int, float]:
    risk_budget = portfolio_value * max_risk_per_trade
    per_share_risk = max(entry_price - stop_loss, 0.0)
    if per_share_risk <= 0:
        return 0.0, 0, risk_budget

    max_shares = math.floor(risk_budget / per_share_risk)
    max_position_value = max_shares * entry_price
    return float(max_position_value), int(max_shares), float(risk_budget)


def build_book_correlation_lookup(
    prices: pd.DataFrame,
    book_tickers: list[str],
    *,
    window_days: int = 126,
) -> dict[str, float]:
    """Compute average pairwise correlation of each candidate's daily returns vs the book.

    Returns a dict mapping ticker -> avg_correlation. Used by `size_position` to
    apply the correlation-aware penalty (DEFAULT_MAX_AVG_BOOK_CORRELATION).
    """
    if prices.empty or not book_tickers:
        return {}
    book_set = {t for t in book_tickers if isinstance(t, str)}
    if not book_set:
        return {}
    df = prices[["ticker", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    pivot = pivot.sort_index().tail(window_days + 5)
    if pivot.shape[0] < 30:
        return {}
    returns = pivot.pct_change().dropna(how="all")
    correlations = returns.corr(min_periods=20)
    book_in_corr = [t for t in book_set if t in correlations.columns]
    if not book_in_corr:
        return {}
    avg_corr = correlations[book_in_corr].mean(axis=1, skipna=True)
    return {str(k): float(v) for k, v in avg_corr.dropna().items() if k not in book_set}


def calculate_atr_for_signal(
    prices: pd.DataFrame,
    ticker: str,
    signal_date: str,
    fallback_price: float,
    fallback_volatility: float,
    window: int = ATR_WINDOW,
) -> float:
    if prices.empty:
        return float(fallback_price * max(fallback_volatility, 0.015))
    ticker_prices = prices[prices["ticker"] == ticker].copy()
    if ticker_prices.empty:
        return float(fallback_price * max(fallback_volatility, 0.015))

    ticker_prices["date"] = pd.to_datetime(ticker_prices["date"])
    ticker_prices = ticker_prices[ticker_prices["date"] <= pd.Timestamp(signal_date)].sort_values("date")
    if len(ticker_prices) < window + 1:
        return float(fallback_price * max(fallback_volatility, 0.015))

    high = ticker_prices["high"].astype(float)
    low = ticker_prices["low"].astype(float)
    close = ticker_prices["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean().dropna()
    if atr.empty or not math.isfinite(float(atr.iloc[-1])):
        return float(fallback_price * max(fallback_volatility, 0.015))
    return float(max(atr.iloc[-1], fallback_price * 0.0025))


def estimate_adtv_for_signal(
    prices: pd.DataFrame,
    ticker: str,
    signal_date: str,
    window: int = 21,
) -> float | None:
    if prices.empty:
        return None
    ticker_prices = prices[prices["ticker"] == ticker].copy()
    if ticker_prices.empty:
        return None
    ticker_prices["date"] = pd.to_datetime(ticker_prices["date"])
    ticker_prices = ticker_prices[ticker_prices["date"] <= pd.Timestamp(signal_date)].sort_values("date")
    if ticker_prices.empty:
        return None
    traded_value = ticker_prices["close"].astype(float) * ticker_prices["volume"].astype(float)
    latest_window = traded_value.tail(window)
    if latest_window.empty:
        return None
    return float(latest_window.mean())


def volatility_percentile_for_signal(
    prices: pd.DataFrame,
    ticker: str,
    signal_date: str,
    fallback_volatility: float,
) -> float:
    if prices.empty:
        return 0.5
    ticker_prices = prices[prices["ticker"] == ticker].copy()
    if ticker_prices.empty:
        return 0.5
    ticker_prices["date"] = pd.to_datetime(ticker_prices["date"])
    ticker_prices = ticker_prices[ticker_prices["date"] <= pd.Timestamp(signal_date)].sort_values("date")
    if len(ticker_prices) < 30:
        return 0.5
    returns = ticker_prices["close"].astype(float).pct_change(fill_method=None)
    history = returns.rolling(21, min_periods=21).std().dropna().tolist()
    if not history:
        return 0.5
    return estimate_volatility_percentile(float(fallback_volatility), history)


def current_open_exposures(portfolio_value: float) -> tuple[dict[str, float], dict[str, float]]:
    positions = get_paper_positions()
    ticker_exposure: dict[str, float] = {}
    sector_exposure: dict[str, float] = {}
    if positions.empty:
        return ticker_exposure, sector_exposure

    open_positions = positions[positions["status"] == "open"].copy()
    for row in open_positions.itertuples(index=False):
        ticker = str(row.ticker)
        notional = float(row.quantity) * float(row.current_price)
        ticker_exposure[ticker] = ticker_exposure.get(ticker, 0.0) + notional
        sector = sector_for(ticker)
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + notional

    max_portfolio = max(float(portfolio_value), 1.0)
    return (
        {ticker: min(value, max_portfolio) for ticker, value in ticker_exposure.items()},
        {sector: min(value, max_portfolio) for sector, value in sector_exposure.items()},
    )


def event_is_recent(context: dict, signal_date: str) -> bool:
    aligned = context.get("aligned_trading_date")
    if not aligned or int(context.get("event_count") or 0) <= 0:
        return False
    try:
        delta_days = (pd.Timestamp(signal_date) - pd.Timestamp(str(aligned))).days
    except (TypeError, ValueError):
        return False
    return 0 <= delta_days <= EVENT_RECENCY_DAYS


def build_regime_gate(
    *,
    prices: pd.DataFrame,
    ticker: str,
    signal_date: str,
    technical_probability_up: float,
    technical_confidence: float,
    context: dict,
    volatility_21d: float,
    data_staleness: dict | None,
) -> dict:
    volatility_percentile = volatility_percentile_for_signal(
        prices,
        ticker,
        signal_date,
        fallback_volatility=volatility_21d,
    )
    regime = assess_regime(
        technical_probability_up=technical_probability_up,
        technical_confidence=technical_confidence,
        sentiment_score=float(context.get("sentiment_score", 0.0)),
        event_count=int(context.get("event_count", 0)),
        event_severity=float(context.get("event_severity", 0.0)),
        event_magnitude=float(context.get("event_magnitude", 0.0)),
        volatility_percentile=volatility_percentile,
    )
    block_reasons: list[str] = []
    if event_is_recent(context, signal_date):
        block_reasons.append("event_within_48h")
    if regime.volatility_percentile >= STRICT_NO_OPERATE_VOL_PERCENTILE:
        block_reasons.append("volatility_percentile_above_80")
    if regime.divergence >= STRICT_NO_OPERATE_DIVERGENCE:
        block_reasons.append("regime_divergence_high")
    trading_days_behind = (data_staleness or {}).get("trading_days_behind")
    blocking_staleness = bool(data_staleness and data_staleness.get("is_stale")) and (
        trading_days_behind is None or int(trading_days_behind) > 1
    )
    if blocking_staleness:
        block_reasons.append("data_stale")

    return {
        "regime": regime.regime,
        "technical_weight": regime.technical_weight,
        "qualitative_weight": regime.qualitative_weight,
        "override_qualitative": regime.override_qualitative,
        "volatility_percentile": regime.volatility_percentile,
        "event_magnitude": regime.event_magnitude,
        "divergence": regime.divergence,
        "notes": regime.notes,
        "strict_block_reasons": block_reasons,
        "data_staleness": data_staleness or {},
    }


def build_signal_id(run_id: str, ticker: str, signal_date: str, horizon: str) -> str:
    payload = f"{PAPER_SIGNAL_VERSION}|{run_id}|{ticker}|{signal_date}|{horizon}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"paper_{digest}"


def choose_decision(
    probability_up: float,
    confidence: float,
    net_expected_return: float,
    reward_risk_ratio: float,
    volatility_21d: float,
    max_shares: int,
    policy: PaperTradingPolicy,
    strategy_gate_passed: bool = True,
    strategy_gate_reason: str | None = None,
    strategy_probability_threshold: float | None = None,
    strict_block_reasons: list[str] | None = None,
) -> tuple[str, str | None]:
    block_reasons: list[str] = list(strict_block_reasons or [])
    if policy.require_strategy_edge and not strategy_gate_passed:
        block_reasons.append(strategy_gate_reason or "strategy_gate_failed")
    if strategy_probability_threshold is not None and probability_up < strategy_probability_threshold:
        block_reasons.append("probability_up_below_strategy_threshold")
    if confidence < policy.min_confidence:
        block_reasons.append("confidence_below_minimum")
    if net_expected_return <= 0:
        block_reasons.append("net_expected_return_not_positive")
    if reward_risk_ratio < (policy.min_reward_risk_ratio - REWARD_RISK_TOLERANCE):
        block_reasons.append("reward_risk_below_minimum")
    if volatility_21d > policy.max_volatility_21d:
        block_reasons.append("volatility_above_limit")
    if max_shares <= 0:
        block_reasons.append("position_size_zero")

    if block_reasons:
        return "no_operate", ",".join(block_reasons)
    return "simulate_long", None


def build_paper_trading_signals(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    run_id: str,
    policy: PaperTradingPolicy,
    strategy_gate: dict | None = None,
    signal_source: str = "historical_test_predictions",
    prices: pd.DataFrame | None = None,
    data_staleness: dict | None = None,
    ticker_exposure: dict[str, float] | None = None,
    sector_exposure: dict[str, float] | None = None,
    book_correlation: dict[str, float] | None = None,
) -> pd.DataFrame:
    if predictions.empty or features.empty:
        return pd.DataFrame()

    latest_predictions = predictions.sort_values("date").groupby("ticker", as_index=False).tail(1)
    qualitative_features = get_qualitative_features()
    prices = prices if prices is not None else read_ohlcv_prices()
    ticker_exposure = ticker_exposure or {}
    sector_exposure = sector_exposure or {}
    feature_columns = ["ticker", "date", "close", "volatility_21d"]
    merged = latest_predictions.merge(
        features[feature_columns],
        on=["ticker", "date"],
        how="inner",
    )

    signal_records: list[dict] = []
    strategy_gate = strategy_gate or {
        "passed": True,
        "reason": None,
        "backtest_id": None,
        "cumulative_return": None,
        "buy_hold_return_avg": None,
    }
    for row in merged.itertuples(index=False):
        reference_price = float(row.close)
        expected_return = float(getattr(row, "target_return", getattr(row, "expected_return", 0.0)))
        technical_probability_up = float(row.probability_up)
        probability_down = float(row.probability_down)
        probability_sideways = float(row.probability_sideways)
        context = choose_context_for_signal(qualitative_features, str(row.ticker), str(row.date))
        technical_confidence = max(technical_probability_up, probability_down, probability_sideways)
        volatility_21d = float(row.volatility_21d)
        regime_gate = build_regime_gate(
            prices=prices,
            ticker=str(row.ticker),
            signal_date=str(row.date),
            technical_probability_up=technical_probability_up,
            technical_confidence=technical_confidence,
            context=context,
            volatility_21d=volatility_21d,
            data_staleness=data_staleness,
        )
        probability_up = calculate_fused_score(
            technical_probability_up,
            sentiment_score=float(context["sentiment_score"]),
            event_count=int(context["event_count"]),
            technical_weight=float(regime_gate["technical_weight"]),
            qualitative_weight=float(regime_gate["qualitative_weight"]),
            override_qualitative=bool(regime_gate["override_qualitative"]),
            event_severity=float(context.get("event_severity", 0.0)),
        )
        confidence = max(probability_up, probability_down, probability_sideways)

        atr = calculate_atr_for_signal(
            prices=prices,
            ticker=str(row.ticker),
            signal_date=str(row.date),
            fallback_price=reference_price,
            fallback_volatility=volatility_21d,
        )
        preliminary_stop_distance = max((2.0 * atr) / max(reference_price, 1e-9), 1e-6)
        target_return = max(expected_return, preliminary_stop_distance * policy.min_reward_risk_ratio)
        sector = sector_for(str(row.ticker))
        sizing = size_position(
            portfolio_value=policy.portfolio_value,
            entry_price=reference_price,
            atr=atr,
            probability_win=probability_up,
            expected_payoff=target_return,
            expected_loss=preliminary_stop_distance,
            ticker=str(row.ticker),
            current_ticker_exposure_brl=ticker_exposure.get(str(row.ticker), 0.0),
            current_sector_exposure_brl=sector_exposure.get(sector, 0.0),
            max_risk_per_trade=policy.max_risk_per_trade,
            avg_book_correlation=book_correlation.get(str(row.ticker), 0.0),
        )
        stop_loss = sizing.stop_price
        stop_distance = sizing.stop_distance_pct
        target_return = max(expected_return, stop_distance * policy.min_reward_risk_ratio)
        target_price = reference_price * (1 + target_return)
        partial_target = reference_price * (1 + target_return * 0.5)
        reward_risk_ratio = (target_price - reference_price) / max(reference_price - stop_loss, 1e-9)
        cost_breakdown = compute_cost_breakdown(
            notional_brl=max(sizing.notional_brl, reference_price),
            adtv_brl=estimate_adtv_for_signal(prices, str(row.ticker), str(row.date)),
        )
        net_expected_return = apply_costs_to_gross_return(expected_return, cost_breakdown)
        strict_block_reasons = list(regime_gate["strict_block_reasons"])
        if sizing.block_reason:
            strict_block_reasons.append(sizing.block_reason)
        decision, block_reason = choose_decision(
            probability_up=probability_up,
            confidence=confidence,
            net_expected_return=net_expected_return,
            reward_risk_ratio=reward_risk_ratio,
            volatility_21d=volatility_21d,
            max_shares=sizing.shares,
            policy=policy,
            strategy_gate_passed=bool(strategy_gate["passed"]),
            strategy_gate_reason=strategy_gate["reason"],
            strategy_probability_threshold=strategy_gate.get("threshold"),
            strict_block_reasons=strict_block_reasons,
        )
        thesis = {
            "ticker": row.ticker,
            "signal_date": row.date,
            "horizon": policy.horizon,
            "decision": decision,
            "block_reason": block_reason,
            "probabilities": {
                "down": probability_down,
                "sideways": probability_sideways,
                "up": probability_up,
                "technical_up": technical_probability_up,
            },
            "signal_source": signal_source,
            "qualitative_context": context,
            "policy": {
                "portfolio_value": policy.portfolio_value,
                "max_risk_per_trade": policy.max_risk_per_trade,
                "min_confidence": policy.min_confidence,
                "min_reward_risk_ratio": policy.min_reward_risk_ratio,
                "cost_per_trade": policy.cost_per_trade,
                "spread": policy.spread,
                "slippage": policy.slippage,
                "strategy_probability_threshold": strategy_gate.get("threshold"),
            },
            "b3_costs": {
                "cost_model": "b3_realistic_round_trip_v1",
                "emolumentos": cost_breakdown.emolumentos,
                "liquidacao": cost_breakdown.liquidacao,
                "corretagem": cost_breakdown.corretagem,
                "iss": cost_breakdown.iss,
                "spread": cost_breakdown.spread,
                "slippage": cost_breakdown.slippage,
                "ir_on_profit_rate": cost_breakdown.ir_on_profit_rate,
                "total_pre_ir": cost_breakdown.total_pre_ir,
            },
            "sizing": {
                "method": "quarter_kelly_atr_stop_sector_caps",
                "atr_14": atr,
                "kelly_fraction_used": sizing.kelly_fraction_used,
                "capped_by": sizing.capped_by,
                "block_reason": sizing.block_reason,
                "sector": sector,
            },
            "technical_context": {
                "volatility_21d": volatility_21d,
                "volatility_percentile": regime_gate["volatility_percentile"],
                "reference_price": reference_price,
            },
            "regime_gate": regime_gate,
            "strategy_gate": strategy_gate,
            "signal_version": PAPER_SIGNAL_VERSION,
        }
        operational_action = (
            OPERATIONAL_ACTION_LEGACY_SIMULATE
            if decision == "simulate_long"
            else OPERATIONAL_ACTION_LEGACY_BLOCK
        )
        signal_records.append(
            {
                "signal_id": build_signal_id(run_id, row.ticker, row.date, policy.horizon),
                "run_id": run_id,
                "ticker": row.ticker,
                "signal_date": row.date,
                "horizon": policy.horizon,
                "decision": decision,
                "block_reason": block_reason,
                "confidence": confidence,
                "probability_up": probability_up,
                "expected_return": expected_return,
                "net_expected_return": net_expected_return,
                "reference_price": reference_price,
                "suggested_entry": reference_price,
                "stop_loss": stop_loss,
                "partial_target": partial_target,
                "target_price": target_price,
                "max_position_value": sizing.notional_brl,
                "max_shares": sizing.shares,
                "risk_amount": sizing.risk_brl,
                "reward_risk_ratio": reward_risk_ratio,
                "model_run_id": run_id,
                "thesis_json": json.dumps(thesis, ensure_ascii=True),
                "operational_action": operational_action,
                "trade_outcome_run_id": None,
                "probability_win": None,
                "probability_loss": None,
                "probability_timeout": None,
            }
        )

    return pd.DataFrame(signal_records)


def choose_operational_action(
    probability_win: float,
    probability_loss: float,
    net_expected_return: float,
    reward_risk_ratio: float,
    volatility_21d: float,
    max_shares: int,
    policy: PaperTradingPolicy,
    strategy_gate_passed: bool,
    strategy_gate_reason: str | None,
    enter_long_min_probability_win: float = 0.50,
    enter_long_min_edge: float = 0.05,
    watchlist_min_probability_win: float = 0.40,
    strict_block_reasons: list[str] | None = None,
) -> tuple[str, str, str | None]:
    """Map a trade-outcome prediction into an explicit operational action.

    Returns ``(operational_action, decision, block_reason)`` where ``decision``
    is the legacy column constrained to ``simulate_long`` / ``no_operate`` so
    downstream consumers (paper_positions, risk_advisor) keep working.
    """
    block_reasons: list[str] = list(strict_block_reasons or [])
    effective_probability_win = decisive_win_probability(probability_win, probability_loss)
    if policy.require_strategy_edge and not strategy_gate_passed:
        block_reasons.append(strategy_gate_reason or "strategy_gate_failed")
    if max_shares <= 0:
        block_reasons.append("position_size_zero")
    if volatility_21d > policy.max_volatility_21d:
        block_reasons.append("volatility_above_limit")
    if reward_risk_ratio < (policy.min_reward_risk_ratio - REWARD_RISK_TOLERANCE):
        block_reasons.append("reward_risk_below_minimum")

    win_minus_loss = probability_win - probability_loss
    enter_long_ok = (
        not block_reasons
        and effective_probability_win >= enter_long_min_probability_win
        and net_expected_return > 0.0
        and win_minus_loss >= enter_long_min_edge
    )
    if enter_long_ok:
        return OPERATIONAL_ACTION_ENTER_LONG, "simulate_long", None

    if not block_reasons:
        if effective_probability_win < enter_long_min_probability_win:
            block_reasons.append("probability_win_below_enter_threshold")
        if net_expected_return <= 0.0:
            block_reasons.append("net_expected_return_not_positive")
        if win_minus_loss < enter_long_min_edge:
            block_reasons.append("win_minus_loss_edge_below_minimum")

    watchlist_ok = (
        effective_probability_win >= watchlist_min_probability_win
        and net_expected_return > -policy.total_execution_drag
    )
    operational_action = (
        OPERATIONAL_ACTION_WATCHLIST if watchlist_ok else OPERATIONAL_ACTION_NO_TRADE
    )
    return operational_action, "no_operate", ",".join(block_reasons) or None


def build_trade_outcome_strategy_gate(trade_outcome_predictions: pd.DataFrame) -> dict:
    if trade_outcome_predictions.empty:
        return {
            "passed": False,
            "reason": "trade_outcome_predictions_empty",
            "gate_type": "trade_outcome_model_gate",
        }
    run_id = str(trade_outcome_predictions["run_id"].iloc[0])
    runs = get_trade_outcome_runs()
    selected = runs[runs["run_id"].astype(str).eq(run_id)] if not runs.empty else pd.DataFrame()
    if selected.empty:
        return {
            "passed": False,
            "reason": "trade_outcome_run_metadata_missing",
            "gate_type": "trade_outcome_model_gate",
            "run_id": run_id,
        }
    latest = selected.iloc[0]
    simulated_trades = int(latest.get("simulated_test_trades") or 0)
    simulated_avg_return = float(latest.get("simulated_test_avg_return") or 0.0)
    simulated_win_rate = float(latest.get("simulated_test_win_rate") or 0.0)
    gate_checks = {
        "trade_outcome_min_test_trades": simulated_trades >= MIN_TRADE_OUTCOME_TEST_TRADES,
        "trade_outcome_avg_return_positive": simulated_avg_return > 0.0,
        "trade_outcome_win_rate_acceptable": simulated_win_rate >= MIN_TRADE_OUTCOME_TEST_WIN_RATE,
    }
    failed_checks = [name for name, passed in gate_checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "reason": None if not failed_checks else ",".join(f"{name}_failed" for name in failed_checks),
        "gate_type": "trade_outcome_model_gate",
        "run_id": run_id,
        "threshold": 0.50,
        "min_test_win_rate": MIN_TRADE_OUTCOME_TEST_WIN_RATE,
        "simulated_test_trades": simulated_trades,
        "simulated_test_avg_return": simulated_avg_return,
        "simulated_test_win_rate": simulated_win_rate,
        "gate_checks": gate_checks,
    }


def _score_technical_fallback_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    scored = features.copy()
    reference_price = scored["close"].astype(float)
    trend_21 = reference_price / scored["ma_21"].astype(float).clip(lower=1e-9) - 1.0
    trend_63 = reference_price / scored["ma_63"].astype(float).clip(lower=1e-9) - 1.0
    trend_252 = reference_price / scored["ma_252"].astype(float).clip(lower=1e-9) - 1.0
    return_21d = scored["return_21d"].astype(float).fillna(0.0)
    rsi_14 = scored["rsi_14"].astype(float).fillna(50.0)
    volatility_21d = scored["volatility_21d"].astype(float)
    mean_reversion = ((50.0 - rsi_14) / 100.0).clip(-0.25, 0.25)
    raw_probability_win = (
        0.50
        + 1.20 * return_21d
        + 0.75 * trend_63
        + 0.35 * trend_252
        + 0.20 * mean_reversion
        - 1.50 * np.maximum(volatility_21d - 0.035, 0.0)
    )
    probability_win = raw_probability_win.clip(0.05, 0.95)
    probability_loss = (0.72 - probability_win + np.maximum(volatility_21d - 0.025, 0.0) * 2.0).clip(0.05, 0.90)
    probability_timeout = (1.0 - probability_win - probability_loss).clip(0.0, 0.90)
    probability_total = probability_win + probability_loss + probability_timeout
    scored["technical_fallback_probability_win"] = probability_win / probability_total
    scored["technical_fallback_probability_loss"] = probability_loss / probability_total
    scored["technical_fallback_expected_return"] = (
        0.45 * return_21d + 0.25 * trend_63 + 0.10 * trend_21 + 0.05 * mean_reversion
    ).clip(-0.12, 0.12)
    scored["technical_fallback_edge"] = (
        scored["technical_fallback_probability_win"] - scored["technical_fallback_probability_loss"]
    )
    return scored


def _select_technical_fallback_rows(
    scored: pd.DataFrame,
    threshold: float,
    policy: PaperTradingPolicy,
    holding_days: int = 7,
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    ordered_scores = scored.sort_values(["ticker", "date"]).reset_index(drop=True)
    for _ticker, ticker_scores in ordered_scores.groupby("ticker", sort=False):
        ticker_scores = ticker_scores.reset_index(drop=True)
        index = 0
        while index < len(ticker_scores):
            row = ticker_scores.iloc[index]
            if (
                float(row["technical_fallback_probability_win"]) >= threshold
                and float(row["technical_fallback_edge"]) >= 0.05
                and float(row["technical_fallback_expected_return"]) > 0.0
                and float(row["volatility_21d"]) <= policy.max_volatility_21d
            ):
                selected_rows.append(row)
                index += holding_days
            else:
                index += 1
    return pd.DataFrame(selected_rows)


def _summarize_technical_fallback_rows(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "cumulative_return": 0.0,
            "average_trade_return": 0.0,
            "traded_tickers": 0,
            "profitable_tickers": 0,
        }
    returns = trades["technical_fallback_net_return"].astype(float)
    equity_curve = (1.0 + returns).cumprod()
    per_ticker = trades.groupby("ticker")["technical_fallback_net_return"].sum()
    return {
        "trades": int(len(trades)),
        "win_rate": float((returns > 0.0).mean()),
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "average_trade_return": float(returns.mean()),
        "traded_tickers": int(trades["ticker"].nunique()),
        "profitable_tickers": int((per_ticker > 0.0).sum()),
    }


def build_technical_fallback_strategy_gate(policy: PaperTradingPolicy) -> dict:
    historical_features = read_technical_features()
    if historical_features.empty or "target_return_7d" not in historical_features.columns:
        return {
            "passed": False,
            "reason": "technical_fallback_history_missing",
            "gate_type": "technical_fallback_historical_gate",
        }
    scored = _score_technical_fallback_features(historical_features)
    scored = scored.dropna(
        subset=[
            "technical_fallback_probability_win",
            "technical_fallback_probability_loss",
            "technical_fallback_expected_return",
            "technical_fallback_edge",
            "target_return_7d",
            "volatility_21d",
        ]
    ).copy()
    if scored.empty:
        return {
            "passed": False,
            "reason": "technical_fallback_scores_empty",
            "gate_type": "technical_fallback_historical_gate",
        }
    cost_breakdown = compute_cost_breakdown(notional_brl=policy.portfolio_value)
    gross_net_return = scored["target_return_7d"].astype(float) - cost_breakdown.total_pre_ir
    scored["technical_fallback_net_return"] = np.where(
        gross_net_return > 0.0,
        gross_net_return * (1.0 - cost_breakdown.ir_on_profit_rate),
        gross_net_return,
    )
    validation_scores = scored[scored["time_split"].eq("validation")].copy()
    test_scores = scored[scored["time_split"].eq("test")].copy()
    threshold_records: list[dict] = []
    for threshold in TECHNICAL_FALLBACK_THRESHOLD_GRID:
        validation_trades = _select_technical_fallback_rows(validation_scores, threshold, policy)
        metrics = _summarize_technical_fallback_rows(validation_trades)
        metrics["threshold"] = float(threshold)
        threshold_records.append(metrics)
    threshold_grid = pd.DataFrame(threshold_records)
    candidates = threshold_grid[
        (threshold_grid["trades"] >= MIN_TECHNICAL_FALLBACK_TRADES)
        & (threshold_grid["average_trade_return"] > 0.0)
        & (threshold_grid["cumulative_return"] > 0.0)
    ].copy()
    if candidates.empty:
        return {
            "passed": False,
            "reason": "technical_fallback_validation_edge_failed",
            "gate_type": "technical_fallback_historical_gate",
            "threshold_grid": threshold_records,
        }
    selected = candidates.sort_values(
        ["average_trade_return", "cumulative_return", "trades"],
        ascending=[False, False, False],
    ).iloc[0]
    selected_threshold = float(selected["threshold"])
    test_trades = _select_technical_fallback_rows(test_scores, selected_threshold, policy)
    test_metrics = _summarize_technical_fallback_rows(test_trades)
    gate_checks = {
        "technical_fallback_test_min_trades": test_metrics["trades"] >= MIN_TECHNICAL_FALLBACK_TRADES,
        "technical_fallback_test_return_positive": test_metrics["cumulative_return"] > 0.0,
        "technical_fallback_test_avg_return_positive": test_metrics["average_trade_return"] > 0.0,
        "technical_fallback_profitable_tickers_acceptable": (
            test_metrics["profitable_tickers"] >= MIN_TECHNICAL_FALLBACK_PROFITABLE_TICKERS
        ),
    }
    failed_checks = [name for name, passed in gate_checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "reason": None if not failed_checks else ",".join(f"{name}_failed" for name in failed_checks),
        "gate_type": "technical_fallback_historical_gate",
        "threshold": selected_threshold,
        "validation_selection": selected.to_dict(),
        "test_metrics": test_metrics,
        "gate_checks": gate_checks,
    }


def build_directional_strategy_gate(
    *,
    run_id: str,
    holding_days: int,
    cost_per_trade: float,
    portfolio_value: float,
) -> dict:
    walk_forward = run_walk_forward_backtest(
        run_id=run_id,
        holding_days=holding_days,
        cost_per_trade=cost_per_trade,
        use_b3_costs=True,
        notional_brl=portfolio_value,
    )
    passed = bool(walk_forward["strategy_gate_passed"])
    failed_checks = [
        f"{name}_failed"
        for name, check_passed in walk_forward.get("gate_checks", {}).items()
        if not check_passed
    ]
    reason = None if passed else ",".join(failed_checks) or walk_forward["strategy_gate_reason"]
    return {
        "passed": passed,
        "reason": reason,
        "gate_type": "directional_walk_forward_gate",
        "backtest_id": walk_forward["backtest_id"],
        "threshold": walk_forward["threshold"],
        "trades": walk_forward["trades"],
        "cumulative_return": walk_forward["cumulative_return"],
        "buy_hold_return_avg": walk_forward["buy_hold_return_avg"],
        "max_drawdown": walk_forward["max_drawdown"],
        "passing_windows": walk_forward["passing_windows"],
        "total_windows": walk_forward["total_windows"],
        "passing_window_ratio": walk_forward["passing_window_ratio"],
        "traded_tickers": walk_forward["traded_tickers"],
        "profitable_tickers": walk_forward["profitable_tickers"],
        "gate_checks": walk_forward["gate_checks"],
    }


def build_trade_outcome_paper_signals(
    trade_outcome_predictions: pd.DataFrame,
    features: pd.DataFrame,
    run_id: str,
    policy: PaperTradingPolicy,
    strategy_gate: dict | None = None,
    enter_long_min_probability_win: float = 0.50,
    enter_long_min_edge: float = 0.05,
    watchlist_min_probability_win: float = 0.40,
    prices: pd.DataFrame | None = None,
    data_staleness: dict | None = None,
    ticker_exposure: dict[str, float] | None = None,
    sector_exposure: dict[str, float] | None = None,
    book_correlation: dict[str, float] | None = None,
) -> pd.DataFrame:
    if trade_outcome_predictions.empty or features.empty:
        return pd.DataFrame()

    qualitative_features = get_qualitative_features()
    prices = prices if prices is not None else read_ohlcv_prices()
    ticker_exposure = ticker_exposure or {}
    sector_exposure = sector_exposure or {}
    book_correlation = book_correlation or {}
    feature_columns = ["ticker", "date", "close", "volatility_21d"]
    merged = trade_outcome_predictions.merge(
        features[feature_columns],
        on=["ticker"],
        how="inner",
        suffixes=("", "_feat"),
    )
    if merged.empty:
        return pd.DataFrame()

    strategy_gate = strategy_gate or {
        "passed": True,
        "reason": None,
        "backtest_id": None,
        "cumulative_return": None,
        "buy_hold_return_avg": None,
    }
    signal_records: list[dict] = []
    for record in merged.to_dict(orient="records"):
        ticker = str(record["ticker"])
        signal_date = str(record.get("date_feat") or record["date"])
        reference_price = float(record["close"])
        volatility_21d = float(record["volatility_21d"])
        model_stop_distance = float(record["stop_distance"])
        model_target_distance = float(record["target_distance"])
        legacy_execution_drag = float(record["execution_drag"])
        probability_win = float(record["probability_win"])
        probability_loss = float(record["probability_loss"])
        probability_timeout = float(record["probability_timeout"])
        expected_return_pred = float(record["expected_return"])

        context = choose_context_for_signal(qualitative_features, ticker, signal_date)
        confidence = max(probability_win, probability_loss, probability_timeout)
        regime_gate = build_regime_gate(
            prices=prices,
            ticker=ticker,
            signal_date=signal_date,
            technical_probability_up=probability_win,
            technical_confidence=confidence,
            context=context,
            volatility_21d=volatility_21d,
            data_staleness=data_staleness,
        )
        atr = calculate_atr_for_signal(
            prices=prices,
            ticker=ticker,
            signal_date=signal_date,
            fallback_price=reference_price,
            fallback_volatility=volatility_21d,
        )
        preliminary_stop_distance = max((2.0 * atr) / max(reference_price, 1e-9), 1e-6)
        target_distance = max(model_target_distance, preliminary_stop_distance * policy.min_reward_risk_ratio)
        sector = sector_for(ticker)
        sizing = size_position(
            portfolio_value=policy.portfolio_value,
            entry_price=reference_price,
            atr=atr,
            probability_win=probability_win,
            probability_loss=probability_loss,
            expected_payoff=target_distance,
            expected_loss=preliminary_stop_distance,
            ticker=ticker,
            current_ticker_exposure_brl=ticker_exposure.get(ticker, 0.0),
            current_sector_exposure_brl=sector_exposure.get(sector, 0.0),
            max_risk_per_trade=policy.max_risk_per_trade,
            avg_book_correlation=book_correlation.get(ticker, 0.0),
        )
        stop_distance = sizing.stop_distance_pct
        target_distance = max(target_distance, stop_distance * policy.min_reward_risk_ratio)

        stop_loss = reference_price * (1.0 - stop_distance)
        target_price = reference_price * (1.0 + target_distance)
        partial_target = reference_price * (1.0 + target_distance * 0.5)
        reward_risk_ratio = (target_price - reference_price) / max(
            reference_price - stop_loss, 1e-9
        )
        gross_expected_return = expected_return_pred + legacy_execution_drag
        cost_breakdown = compute_cost_breakdown(
            notional_brl=max(sizing.notional_brl, reference_price),
            adtv_brl=estimate_adtv_for_signal(prices, ticker, signal_date),
        )
        net_expected_return = apply_costs_to_gross_return(gross_expected_return, cost_breakdown)
        strict_block_reasons = list(regime_gate["strict_block_reasons"])
        if sizing.block_reason:
            strict_block_reasons.append(sizing.block_reason)

        operational_action, decision, block_reason = choose_operational_action(
            probability_win=probability_win,
            probability_loss=probability_loss,
            net_expected_return=net_expected_return,
            reward_risk_ratio=reward_risk_ratio,
            volatility_21d=volatility_21d,
            max_shares=sizing.shares,
            policy=policy,
            strategy_gate_passed=bool(strategy_gate["passed"]),
            strategy_gate_reason=strategy_gate["reason"],
            enter_long_min_probability_win=enter_long_min_probability_win,
            enter_long_min_edge=enter_long_min_edge,
            watchlist_min_probability_win=watchlist_min_probability_win,
            strict_block_reasons=strict_block_reasons,
        )

        thesis = {
            "ticker": ticker,
            "signal_date": signal_date,
            "horizon": policy.horizon,
            "decision": decision,
            "operational_action": operational_action,
            "block_reason": block_reason,
            "trade_outcome": {
                "probability_win": probability_win,
                "probability_loss": probability_loss,
                "probability_timeout": probability_timeout,
                "expected_return": expected_return_pred,
                "gross_expected_return": gross_expected_return,
                "model_stop_distance": model_stop_distance,
                "applied_stop_distance": stop_distance,
                "target_distance": target_distance,
                "legacy_execution_drag": legacy_execution_drag,
                "execution_drag": cost_breakdown.total_pre_ir,
                "trade_outcome_run_id": str(record["run_id"]),
                "horizon_days": int(record.get("horizon_days") or 0),
                "inference_version": str(record.get("inference_version", "")),
            },
            "signal_source": "operational_trade_outcomes",
            "qualitative_context": context,
            "policy": {
                "portfolio_value": policy.portfolio_value,
                "max_risk_per_trade": policy.max_risk_per_trade,
                "min_reward_risk_ratio": policy.min_reward_risk_ratio,
                "cost_per_trade": policy.cost_per_trade,
                "spread": policy.spread,
                "slippage": policy.slippage,
                "enter_long_min_probability_win": enter_long_min_probability_win,
                "enter_long_min_edge": enter_long_min_edge,
                "watchlist_min_probability_win": watchlist_min_probability_win,
            },
            "b3_costs": {
                "cost_model": "b3_realistic_round_trip_v1",
                "emolumentos": cost_breakdown.emolumentos,
                "liquidacao": cost_breakdown.liquidacao,
                "corretagem": cost_breakdown.corretagem,
                "iss": cost_breakdown.iss,
                "spread": cost_breakdown.spread,
                "slippage": cost_breakdown.slippage,
                "ir_on_profit_rate": cost_breakdown.ir_on_profit_rate,
                "total_pre_ir": cost_breakdown.total_pre_ir,
            },
            "sizing": {
                "method": "quarter_kelly_atr_stop_sector_caps",
                "atr_14": atr,
                "kelly_fraction_used": sizing.kelly_fraction_used,
                "capped_by": sizing.capped_by,
                "block_reason": sizing.block_reason,
                "sector": sector,
            },
            "technical_context": {
                "volatility_21d": volatility_21d,
                "volatility_percentile": regime_gate["volatility_percentile"],
                "reference_price": reference_price,
            },
            "regime_gate": regime_gate,
            "strategy_gate": strategy_gate,
            "signal_version": PAPER_SIGNAL_VERSION,
        }
        signal_records.append(
            {
                "signal_id": build_signal_id(run_id, ticker, signal_date, policy.horizon),
                "run_id": run_id,
                "ticker": ticker,
                "signal_date": signal_date,
                "horizon": policy.horizon,
                "decision": decision,
                "block_reason": block_reason,
                "confidence": confidence,
                "probability_up": probability_win,
                "expected_return": gross_expected_return,
                "net_expected_return": net_expected_return,
                "reference_price": reference_price,
                "suggested_entry": reference_price,
                "stop_loss": stop_loss,
                "partial_target": partial_target,
                "target_price": target_price,
                "max_position_value": sizing.notional_brl,
                "max_shares": sizing.shares,
                "risk_amount": sizing.risk_brl,
                "reward_risk_ratio": reward_risk_ratio,
                "model_run_id": run_id,
                "thesis_json": json.dumps(thesis, ensure_ascii=True),
                "operational_action": operational_action,
                "trade_outcome_run_id": str(record["run_id"]),
                "probability_win": probability_win,
                "probability_loss": probability_loss,
                "probability_timeout": probability_timeout,
            }
        )
    return pd.DataFrame(signal_records)


def _bounded(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def build_technical_fallback_paper_signals(
    features: pd.DataFrame,
    run_id: str,
    policy: PaperTradingPolicy,
    strategy_gate: dict | None = None,
    prices: pd.DataFrame | None = None,
    data_staleness: dict | None = None,
    ticker_exposure: dict[str, float] | None = None,
    sector_exposure: dict[str, float] | None = None,
    book_correlation: dict[str, float] | None = None,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    prices = prices if prices is not None else read_ohlcv_prices()
    ticker_exposure = ticker_exposure or {}
    sector_exposure = sector_exposure or {}
    book_correlation = book_correlation or {}
    qualitative_features = get_qualitative_features()
    strategy_gate = strategy_gate or {
        "passed": True,
        "reason": None,
        "backtest_id": None,
        "cumulative_return": None,
        "buy_hold_return_avg": None,
    }

    signal_records: list[dict] = []
    for row in features.to_dict(orient="records"):
        ticker = str(row["ticker"])
        signal_date = str(row["date"])
        reference_price = float(row["close"])
        volatility_21d = float(row["volatility_21d"])
        ma_21 = float(row["ma_21"])
        ma_63 = float(row["ma_63"])
        ma_252 = float(row["ma_252"])
        return_21d = float(row.get("return_21d") or 0.0)
        rsi_14 = float(row.get("rsi_14") or 50.0)

        trend_21 = reference_price / max(ma_21, 1e-9) - 1.0
        trend_63 = reference_price / max(ma_63, 1e-9) - 1.0
        trend_252 = reference_price / max(ma_252, 1e-9) - 1.0
        mean_reversion = _bounded((50.0 - rsi_14) / 100.0, -0.25, 0.25)
        raw_probability_win = (
            0.50
            + 1.20 * return_21d
            + 0.75 * trend_63
            + 0.35 * trend_252
            + 0.20 * mean_reversion
            - 1.50 * max(volatility_21d - 0.035, 0.0)
        )
        probability_win = _bounded(raw_probability_win, 0.05, 0.95)
        probability_loss = _bounded(0.72 - probability_win + max(volatility_21d - 0.025, 0.0) * 2.0, 0.05, 0.90)
        probability_timeout = _bounded(1.0 - probability_win - probability_loss, 0.0, 0.90)
        probability_total = probability_win + probability_loss + probability_timeout
        probability_win /= probability_total
        probability_loss /= probability_total
        probability_timeout /= probability_total

        expected_return = _bounded(
            0.45 * return_21d + 0.25 * trend_63 + 0.10 * trend_21 + 0.05 * mean_reversion,
            -0.12,
            0.12,
        )
        confidence = max(probability_win, probability_loss, probability_timeout)
        context = choose_context_for_signal(qualitative_features, ticker, signal_date)
        regime_gate = build_regime_gate(
            prices=prices,
            ticker=ticker,
            signal_date=signal_date,
            technical_probability_up=probability_win,
            technical_confidence=confidence,
            context=context,
            volatility_21d=volatility_21d,
            data_staleness=data_staleness,
        )
        atr = calculate_atr_for_signal(
            prices=prices,
            ticker=ticker,
            signal_date=signal_date,
            fallback_price=reference_price,
            fallback_volatility=volatility_21d,
        )
        preliminary_stop_distance = max((2.0 * atr) / max(reference_price, 1e-9), 1e-6)
        target_distance = max(abs(expected_return), preliminary_stop_distance * policy.min_reward_risk_ratio)
        sector = sector_for(ticker)
        sizing = size_position(
            portfolio_value=policy.portfolio_value,
            entry_price=reference_price,
            atr=atr,
            probability_win=probability_win,
            probability_loss=probability_loss,
            expected_payoff=target_distance,
            expected_loss=preliminary_stop_distance,
            ticker=ticker,
            current_ticker_exposure_brl=ticker_exposure.get(ticker, 0.0),
            current_sector_exposure_brl=sector_exposure.get(sector, 0.0),
            max_risk_per_trade=policy.max_risk_per_trade,
            avg_book_correlation=book_correlation.get(ticker, 0.0),
        )
        stop_distance = sizing.stop_distance_pct
        target_distance = max(target_distance, stop_distance * policy.min_reward_risk_ratio)
        stop_loss = reference_price * (1.0 - stop_distance)
        target_price = reference_price * (1.0 + target_distance)
        partial_target = reference_price * (1.0 + target_distance * 0.5)
        reward_risk_ratio = (target_price - reference_price) / max(reference_price - stop_loss, 1e-9)
        cost_breakdown = compute_cost_breakdown(
            notional_brl=max(sizing.notional_brl, reference_price),
            adtv_brl=estimate_adtv_for_signal(prices, ticker, signal_date),
        )
        net_expected_return = apply_costs_to_gross_return(expected_return, cost_breakdown)
        strict_block_reasons = list(regime_gate["strict_block_reasons"])
        if sizing.block_reason:
            strict_block_reasons.append(sizing.block_reason)
        operational_action, decision, block_reason = choose_operational_action(
            probability_win=probability_win,
            probability_loss=probability_loss,
            net_expected_return=net_expected_return,
            reward_risk_ratio=reward_risk_ratio,
            volatility_21d=volatility_21d,
            max_shares=sizing.shares,
            policy=policy,
            strategy_gate_passed=bool(strategy_gate["passed"]),
            strategy_gate_reason=strategy_gate["reason"],
            enter_long_min_probability_win=float(strategy_gate.get("threshold") or 0.50),
            strict_block_reasons=strict_block_reasons,
        )
        thesis = {
            "ticker": ticker,
            "signal_date": signal_date,
            "horizon": policy.horizon,
            "decision": decision,
            "operational_action": operational_action,
            "block_reason": block_reason,
            "signal_source": "technical_fallback_current_features",
            "probabilities": {
                "win": probability_win,
                "loss": probability_loss,
                "timeout": probability_timeout,
            },
            "technical_fallback": {
                "return_21d": return_21d,
                "trend_21": trend_21,
                "trend_63": trend_63,
                "trend_252": trend_252,
                "rsi_14": rsi_14,
                "expected_return": expected_return,
            },
            "qualitative_context": context,
            "policy": {
                "portfolio_value": policy.portfolio_value,
                "max_risk_per_trade": policy.max_risk_per_trade,
                "min_reward_risk_ratio": policy.min_reward_risk_ratio,
                "cost_per_trade": policy.cost_per_trade,
                "spread": policy.spread,
                "slippage": policy.slippage,
            },
            "b3_costs": {
                "cost_model": "b3_realistic_round_trip_v1",
                "emolumentos": cost_breakdown.emolumentos,
                "liquidacao": cost_breakdown.liquidacao,
                "corretagem": cost_breakdown.corretagem,
                "iss": cost_breakdown.iss,
                "spread": cost_breakdown.spread,
                "slippage": cost_breakdown.slippage,
                "ir_on_profit_rate": cost_breakdown.ir_on_profit_rate,
                "total_pre_ir": cost_breakdown.total_pre_ir,
            },
            "sizing": {
                "method": "quarter_kelly_atr_stop_sector_caps",
                "atr_14": atr,
                "kelly_fraction_used": sizing.kelly_fraction_used,
                "capped_by": sizing.capped_by,
                "block_reason": sizing.block_reason,
                "sector": sector,
            },
            "technical_context": {
                "volatility_21d": volatility_21d,
                "volatility_percentile": regime_gate["volatility_percentile"],
                "reference_price": reference_price,
            },
            "regime_gate": regime_gate,
            "strategy_gate": strategy_gate,
            "signal_version": PAPER_SIGNAL_VERSION,
        }
        signal_records.append(
            {
                "signal_id": build_signal_id(run_id, ticker, signal_date, policy.horizon),
                "run_id": run_id,
                "ticker": ticker,
                "signal_date": signal_date,
                "horizon": policy.horizon,
                "decision": decision,
                "block_reason": block_reason,
                "confidence": confidence,
                "probability_up": probability_win,
                "expected_return": expected_return,
                "net_expected_return": net_expected_return,
                "reference_price": reference_price,
                "suggested_entry": reference_price,
                "stop_loss": stop_loss,
                "partial_target": partial_target,
                "target_price": target_price,
                "max_position_value": sizing.notional_brl,
                "max_shares": sizing.shares,
                "risk_amount": sizing.risk_brl,
                "reward_risk_ratio": reward_risk_ratio,
                "model_run_id": run_id,
                "thesis_json": json.dumps(thesis, ensure_ascii=True),
                "operational_action": operational_action,
                "trade_outcome_run_id": None,
                "probability_win": probability_win,
                "probability_loss": probability_loss,
                "probability_timeout": probability_timeout,
            }
        )
    return pd.DataFrame(signal_records)


def generate_paper_trading_signals(
    run_id: str | None = None,
    portfolio_value: float = 10000.0,
    max_risk_per_trade: float = 0.01,
    min_confidence: float = 0.48,
    min_reward_risk_ratio: float = 1.5,
    cost_per_trade: float = 0.002,
    spread: float = 0.001,
    slippage: float = 0.001,
    max_volatility_21d: float = 0.045,
    require_strategy_edge: bool = True,
) -> dict:
    initialize_database()
    selected_run_id = run_id or get_best_current_schema_model_run_id()
    policy = PaperTradingPolicy(
        portfolio_value=portfolio_value,
        max_risk_per_trade=max_risk_per_trade,
        min_confidence=min_confidence,
        min_reward_risk_ratio=min_reward_risk_ratio,
        cost_per_trade=cost_per_trade,
        spread=spread,
        slippage=slippage,
        max_volatility_21d=max_volatility_21d,
        require_strategy_edge=require_strategy_edge,
    )
    directional_strategy_gate = {
        "passed": True,
        "reason": None,
        "gate_type": "not_required",
        "backtest_id": None,
        "cumulative_return": None,
        "buy_hold_return_avg": None,
    }
    prices = read_ohlcv_prices()
    ticker_exposure, sector_exposure = current_open_exposures(portfolio_value)
    book_correlation = build_book_correlation_lookup(
        prices=prices,
        book_tickers=list(ticker_exposure.keys()),
    )
    try:
        from app.pipelines.refresh import detect_staleness

        data_staleness = detect_staleness()
    except Exception as exc:  # pragma: no cover - defensive operational guard
        data_staleness = {"is_stale": True, "reason": f"staleness_check_failed:{exc}"}
    trade_outcome_predictions = read_latest_operational_trade_outcomes()
    current_features = build_current_technical_features(prices)
    if not trade_outcome_predictions.empty:
        strategy_gate = (
            build_trade_outcome_strategy_gate(trade_outcome_predictions)
            if require_strategy_edge
            else directional_strategy_gate
        )
        signals = build_trade_outcome_paper_signals(
            trade_outcome_predictions,
            current_features,
            selected_run_id,
            policy,
            strategy_gate,
            prices=prices,
            data_staleness=data_staleness,
            ticker_exposure=ticker_exposure,
            sector_exposure=sector_exposure,
            book_correlation=book_correlation,
        )
        signal_source = "operational_trade_outcomes"
    else:
        operational_predictions = read_operational_predictions(selected_run_id)
        if operational_predictions.empty:
            predictions = read_model_predictions(selected_run_id, split="test")
            features = read_technical_features()
            signal_source = "historical_test_predictions"
        else:
            predictions = operational_predictions.copy()
            features = current_features
            signal_source = "operational_predictions"
        strategy_gate = (
            build_directional_strategy_gate(
                run_id=selected_run_id,
                holding_days=7,
                cost_per_trade=cost_per_trade,
                portfolio_value=portfolio_value,
            )
            if require_strategy_edge
            else directional_strategy_gate
        )
        signals = build_paper_trading_signals(
            predictions,
            features,
            selected_run_id,
            policy,
            strategy_gate,
            signal_source=signal_source,
            prices=prices,
            data_staleness=data_staleness,
            ticker_exposure=ticker_exposure,
            sector_exposure=sector_exposure,
            book_correlation=book_correlation,
        )
    covered_tickers = set() if signals.empty else set(signals["ticker"].astype(str))
    uncovered_features = current_features[~current_features["ticker"].astype(str).isin(covered_tickers)].copy()
    fallback_strategy_gate = (
        build_technical_fallback_strategy_gate(policy)
        if require_strategy_edge
        else directional_strategy_gate
    )
    fallback_signals = build_technical_fallback_paper_signals(
        uncovered_features,
        selected_run_id,
        policy,
        fallback_strategy_gate,
        prices=prices,
        data_staleness=data_staleness,
        ticker_exposure=ticker_exposure,
        sector_exposure=sector_exposure,
        book_correlation=book_correlation,
    )
    if not fallback_signals.empty:
        signals = fallback_signals if signals.empty else pd.concat([signals, fallback_signals], ignore_index=True)
    inserted_rows = save_paper_trading_signals(signals)
    if signals.empty:
        simulated = 0
        blocked = 0
        operational_breakdown: dict[str, int] = {}
    else:
        simulated = int((signals["decision"] == "simulate_long").sum())
        blocked = int((signals["decision"] == "no_operate").sum())
        if "operational_action" in signals.columns:
            operational_breakdown = (
                signals["operational_action"].fillna("UNKNOWN").value_counts().to_dict()
            )
        else:
            operational_breakdown = {}
    return {
        "run_id": selected_run_id,
        "generated": int(len(signals)),
        "inserted": int(inserted_rows),
        "simulate_long": simulated,
        "no_operate": blocked,
        "strategy_gate": strategy_gate,
        "fallback_strategy_gate": fallback_strategy_gate,
        "signal_source": signal_source,
        "operational_actions": operational_breakdown,
    }


def get_paper_trading_summary() -> pd.DataFrame:
    signals = get_paper_trading_signals()
    if signals.empty:
        return pd.DataFrame()
    return signals[
        [
            "signal_id",
            "ticker",
            "signal_date",
            "horizon",
            "decision",
            "block_reason",
            "confidence",
            "expected_return",
            "net_expected_return",
            "reference_price",
            "stop_loss",
            "target_price",
            "max_position_value",
            "max_shares",
            "reward_risk_ratio",
        ]
    ]
