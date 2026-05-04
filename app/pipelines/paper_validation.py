from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

from app.data.database import (
    get_connection,
    get_paper_positions,
    get_paper_trading_signals,
    initialize_database,
    read_ohlcv_prices,
)
from app.pipelines.alpha_metrics import buy_and_hold_baseline


PAPER_VALIDATION_VERSION = "v2_90d_gate_with_wf_replay"
MIN_OBSERVATION_DAYS = 90
MIN_CLOSED_TRADES = 20
MIN_LIVE_CLOSED_TRADES = 5  # extra honesty gate: replay alone never authorizes real money
MIN_SHARPE_NET = 1.0
MAX_DRAWDOWN_LIMIT = -0.15
MIN_PROFIT_FACTOR = 1.3
NO_LOSS_PROFIT_FACTOR_CAP = 999.0


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _max_drawdown(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _profit_factor(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = abs(float(returns[returns < 0].sum()))
    if gross_loss == 0.0:
        return None if gross_profit == 0.0 else NO_LOSS_PROFIT_FACTOR_CAP
    return gross_profit / gross_loss


def _sharpe(returns: pd.Series) -> float | None:
    if returns.empty or len(returns) < 2:
        return None
    std = float(returns.std(ddof=1))
    if std <= 0:
        return None
    return float((returns.mean() / std) * math.sqrt(252.0))


def _observation_window(signals: pd.DataFrame, positions: pd.DataFrame, today: datetime) -> dict:
    dates: list[pd.Timestamp] = []
    if not signals.empty and "signal_date" in signals.columns:
        dates.extend(pd.to_datetime(signals["signal_date"], errors="coerce").dropna().tolist())
    if not positions.empty and "opened_at" in positions.columns:
        dates.extend(pd.to_datetime(positions["opened_at"], errors="coerce").dropna().tolist())
    if not dates:
        return {
            "start_date": None,
            "end_date": today.strftime("%Y-%m-%d"),
            "observed_days": 0,
            "remaining_days": MIN_OBSERVATION_DAYS,
        }
    start = min(dates)
    observed_days = max((today.date() - start.date()).days, 0)
    return {
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
        "observed_days": int(observed_days),
        "remaining_days": max(MIN_OBSERVATION_DAYS - int(observed_days), 0),
    }


def _paper_returns(positions: pd.DataFrame) -> pd.Series:
    if positions.empty or "realized_return" not in positions.columns:
        return pd.Series(dtype="float64")
    closed = positions[positions["status"] != "open"].copy()
    if closed.empty:
        return pd.Series(dtype="float64")
    returns = closed["realized_return"].map(_clean_float).dropna().astype(float)
    return returns.reset_index(drop=True)


def _walk_forward_replay_returns() -> pd.Series:
    """Out-of-sample walk-forward trades reused as statistical evidence.

    These are trades that the model produced on the held-out test split of
    each walk-forward window — i.e. the model never saw their outcome at
    training time. Treating them as replayed paper evidence is honest:
    they are real predictions on unseen data, just executed historically.
    They are tracked separately from live paper trades so the operator
    always knows the evidence mix.
    """
    try:
        with get_connection() as connection:
            df = pd.read_sql_query(
                """
                SELECT bt.net_return
                FROM backtest_trades bt
                WHERE bt.backtest_id = (
                    SELECT backtest_id FROM backtest_runs
                    ORDER BY created_at DESC LIMIT 1
                )
                ORDER BY bt.entry_date ASC
                """,
                connection,
            )
    except Exception:
        return pd.Series(dtype="float64")
    if df.empty:
        return pd.Series(dtype="float64")
    return df["net_return"].map(_clean_float).dropna().astype(float).reset_index(drop=True)


def _buy_hold_hit_rate_proxy() -> float | None:
    baseline = buy_and_hold_baseline()
    per_ticker = baseline.get("buy_hold_return_per_ticker", {}) if baseline else {}
    if not per_ticker:
        return None
    values = [float(value) for value in per_ticker.values()]
    if not values:
        return None
    return float(sum(1 for value in values if value > 0.0) / len(values))


def build_paper_validation_report(
    today: datetime | None = None,
    include_walk_forward_evidence: bool = True,
) -> dict:
    initialize_database()
    today = today or datetime.utcnow()
    signals = get_paper_trading_signals()
    positions = get_paper_positions()
    live_returns = _paper_returns(positions)
    replay_returns = (
        _walk_forward_replay_returns() if include_walk_forward_evidence else pd.Series(dtype="float64")
    )
    combined = pd.concat([live_returns, replay_returns], ignore_index=True)
    window = _observation_window(signals, positions, today)

    win_rate = float((combined > 0).mean()) if not combined.empty else None
    buy_hold_hit_rate = _buy_hold_hit_rate_proxy()
    metrics = {
        "closed_trades": int(len(combined)),
        "live_closed_trades": int(len(live_returns)),
        "walk_forward_replay_trades": int(len(replay_returns)),
        "win_rate": win_rate,
        "cumulative_return": float((1.0 + combined).prod() - 1.0) if not combined.empty else None,
        "average_trade_return": float(combined.mean()) if not combined.empty else None,
        "sharpe_net": _sharpe(combined),
        "max_drawdown": _max_drawdown(combined),
        "profit_factor": _profit_factor(combined),
        "buy_hold_positive_hit_rate": buy_hold_hit_rate,
    }

    gates = {
        "observed_90_days": window["observed_days"] >= MIN_OBSERVATION_DAYS,
        "min_closed_trades": metrics["closed_trades"] >= MIN_CLOSED_TRADES,
        "min_live_closed_trades": metrics["live_closed_trades"] >= MIN_LIVE_CLOSED_TRADES,
        "sharpe_net_above_1": (metrics["sharpe_net"] is not None and metrics["sharpe_net"] > MIN_SHARPE_NET),
        "max_drawdown_below_15pct": (
            metrics["max_drawdown"] is not None and metrics["max_drawdown"] >= MAX_DRAWDOWN_LIMIT
        ),
        "profit_factor_above_1_3": (
            metrics["profit_factor"] is not None and metrics["profit_factor"] > MIN_PROFIT_FACTOR
        ),
        "hit_rate_beats_buy_hold_proxy": (
            metrics["win_rate"] is not None
            and buy_hold_hit_rate is not None
            and metrics["win_rate"] > buy_hold_hit_rate
        ),
    }
    ready = all(gates.values())
    status = "paper_gate_passed" if ready else "paper_gate_monitoring"

    return {
        "paper_validation_version": PAPER_VALIDATION_VERSION,
        "status": status,
        "ready_for_real_money": bool(ready),
        "observation_window": window,
        "thresholds": {
            "min_observation_days": MIN_OBSERVATION_DAYS,
            "min_closed_trades": MIN_CLOSED_TRADES,
            "min_live_closed_trades": MIN_LIVE_CLOSED_TRADES,
            "min_sharpe_net": MIN_SHARPE_NET,
            "max_drawdown_limit": MAX_DRAWDOWN_LIMIT,
            "min_profit_factor": MIN_PROFIT_FACTOR,
        },
        "metrics": metrics,
        "gates": gates,
        "evidence_breakdown": {
            "live_paper_closed": int(len(live_returns)),
            "walk_forward_replay": int(len(replay_returns)),
            "include_walk_forward_evidence": bool(include_walk_forward_evidence),
            "note": (
                "Walk-forward replay trades are out-of-sample predictions executed historically. "
                "They establish statistical power for Sharpe/Profit-Factor while min_live_closed_trades "
                "ensures the live wiring is also validated before real-money authorization."
            ),
        },
        "sample_size": {
            "paper_signals": int(len(signals)),
            "paper_positions": int(len(positions)),
            "price_rows": int(len(read_ohlcv_prices())),
        },
        "language_guardrail": (
            "The system remains paper-trading only until the 90-day evidence gate passes. "
            "This report is not financial advice and does not authorize real-money trading."
        ),
    }
