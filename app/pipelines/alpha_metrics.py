"""Aggregate alpha metrics for the dashboard / API.

Pulls together the metrics the methodology document requires for the alpha
layer: directional accuracy, cumulative return, drawdown, win rate, gain/loss
averages, risk/reward ratio, comparison vs buy & hold, count of "no operate"
signals.  All numbers come from artefacts already stored in SQLite, so the
endpoint is fast and read-only.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.data.database import (
    get_backtest_runs,
    get_operational_predictions,
    get_operational_trade_outcomes,
    get_paper_positions,
    get_paper_trading_signals,
    get_trade_outcome_runs,
    initialize_database,
    read_ohlcv_prices,
    read_technical_features,
)


ALPHA_METRICS_VERSION = "v1_alpha_metrics_summary"


def _to_python(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _latest_signal_snapshot(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "ticker" not in signals.columns:
        return signals

    snapshot = signals.copy()
    if "signal_date" in snapshot.columns:
        snapshot["signal_date"] = pd.to_datetime(snapshot["signal_date"], errors="coerce")
    if "created_at" in snapshot.columns:
        snapshot["created_at"] = pd.to_datetime(snapshot["created_at"], errors="coerce")

    sort_columns = [column for column in ["ticker", "signal_date", "created_at"] if column in snapshot.columns]
    if len(sort_columns) == 1:
        return snapshot.drop_duplicates(subset=["ticker"], keep="last")
    return snapshot.sort_values(sort_columns).groupby("ticker", as_index=False).tail(1)


def directional_accuracy_summary() -> dict:
    features = read_technical_features()
    operational = get_operational_predictions()
    if features.empty or operational.empty:
        return {"directional_predictions": 0, "directional_accuracy": None}
    actuals = features[["ticker", "date", "target_direction_7d"]].dropna()
    actuals["date"] = actuals["date"].astype(str)
    operational["date"] = operational["date"].astype(str)
    merged = operational.merge(
        actuals,
        on=["ticker", "date"],
        how="inner",
    )
    if merged.empty:
        return {"directional_predictions": int(len(operational)), "directional_accuracy": None}
    correct = (merged["predicted_direction"] == merged["target_direction_7d"]).sum()
    return {
        "directional_predictions": int(len(merged)),
        "directional_accuracy": float(correct) / float(len(merged)),
    }


def backtest_metrics_summary() -> dict:
    runs = get_backtest_runs()
    if runs.empty:
        return {}
    last = runs.sort_values("created_at").iloc[-1]
    return {
        "backtest_id": str(last["backtest_id"]),
        "threshold": _to_python(last.get("threshold")),
        "trades": int(last.get("trades") or 0),
        "win_rate": _to_python(last.get("win_rate")),
        "cumulative_return": _to_python(last.get("cumulative_return")),
        "average_trade_return": _to_python(last.get("average_trade_return")),
        "max_drawdown": _to_python(last.get("max_drawdown")),
        "buy_hold_return_avg": _to_python(last.get("buy_hold_return_avg")),
    }


def trade_outcome_run_summary() -> dict:
    runs = get_trade_outcome_runs()
    if runs.empty:
        return {}
    last = runs.iloc[0]
    return {
        "run_id": str(last["run_id"]),
        "horizon_days": int(last["horizon_days"]),
        "validation_accuracy": _to_python(last.get("validation_accuracy")),
        "test_accuracy": _to_python(last.get("test_accuracy")),
        "simulated_test_trades": _to_python(last.get("simulated_test_trades")),
        "simulated_test_avg_return": _to_python(last.get("simulated_test_avg_return")),
        "simulated_test_win_rate": _to_python(last.get("simulated_test_win_rate")),
    }


def trade_outcome_inference_summary() -> dict:
    predictions = get_operational_trade_outcomes()
    if predictions.empty:
        return {"trade_outcome_predictions": 0}
    latest_per_ticker = (
        predictions.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    )
    return {
        "trade_outcome_predictions": int(len(latest_per_ticker)),
        "average_probability_win": float(latest_per_ticker["probability_win"].mean()),
        "average_expected_return": float(latest_per_ticker["expected_return"].mean()),
        "tickers_with_positive_expected_return": int(
            (latest_per_ticker["expected_return"] > 0.0).sum()
        ),
    }


def paper_signal_summary() -> dict:
    signals = get_paper_trading_signals()
    if signals.empty:
        return {
            "signals": 0,
            "operational_actions": {},
            "blocked_by_reason": {},
            "no_operate_count": 0,
        }
    signals = _latest_signal_snapshot(signals)
    blocked_by_reason: dict[str, int] = {}
    for reason in signals["block_reason"].dropna().tolist():
        for token in str(reason).split(","):
            token = token.strip()
            if not token:
                continue
            blocked_by_reason[token] = blocked_by_reason.get(token, 0) + 1
    if "operational_action" in signals.columns:
        operational_actions = (
            signals["operational_action"].fillna("UNKNOWN").value_counts().to_dict()
        )
    else:
        operational_actions = {}
    return {
        "signals": int(len(signals)),
        "simulate_long": int((signals["decision"] == "simulate_long").sum()),
        "no_operate_count": int((signals["decision"] == "no_operate").sum()),
        "operational_actions": operational_actions,
        "blocked_by_reason": blocked_by_reason,
    }


def paper_portfolio_summary() -> dict:
    positions = get_paper_positions()
    if positions.empty:
        return {"positions": 0}
    closed = positions[positions["status"] != "open"]
    open_positions = positions[positions["status"] == "open"]
    realised = closed["realized_return"].dropna()
    wins = realised[realised > 0]
    losses = realised[realised <= 0]
    return {
        "positions": int(len(positions)),
        "open_positions": int(len(open_positions)),
        "closed_positions": int(len(closed)),
        "win_rate_realised": float(len(wins) / len(realised)) if not realised.empty else None,
        "average_realised_gain": float(wins.mean()) if not wins.empty else None,
        "average_realised_loss": float(losses.mean()) if not losses.empty else None,
        "cumulative_realised_return": float(realised.sum()) if not realised.empty else 0.0,
    }


def buy_and_hold_baseline() -> dict:
    prices = read_ohlcv_prices()
    if prices.empty:
        return {}
    returns = {}
    for ticker, ticker_prices in prices.groupby("ticker"):
        ordered = ticker_prices.sort_values("date")
        if len(ordered) < 2:
            continue
        first = float(ordered["close"].iloc[0])
        last = float(ordered["close"].iloc[-1])
        if first <= 0:
            continue
        returns[ticker] = last / first - 1.0
    if not returns:
        return {}
    series = pd.Series(returns)
    return {
        "buy_hold_return_avg": float(series.mean()),
        "buy_hold_return_per_ticker": {
            ticker: float(value) for ticker, value in returns.items()
        },
    }


def build_alpha_metrics() -> dict:
    initialize_database()
    from app.pipelines.paper_validation import build_paper_validation_report

    return {
        "alpha_metrics_version": ALPHA_METRICS_VERSION,
        "directional": directional_accuracy_summary(),
        "walk_forward": backtest_metrics_summary(),
        "trade_outcome_model": trade_outcome_run_summary(),
        "trade_outcome_inference": trade_outcome_inference_summary(),
        "paper_signals": paper_signal_summary(),
        "paper_portfolio": paper_portfolio_summary(),
        "paper_validation_gate": build_paper_validation_report(),
        "buy_and_hold": buy_and_hold_baseline(),
        "language_guardrail": (
            "Experimental alpha metrics for paper-trading study only. "
            "Past performance is not indicative of future results and these "
            "numbers are not a recommendation."
        ),
    }
