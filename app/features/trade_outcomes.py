"""Operational trade-outcome dataset.

Builds labels that reflect whether a long trade entered at the close would have
hit a volatility-based target before its volatility-based stop within a fixed
holding horizon, after deducting execution drag (cost + spread + slippage).

The labels are aligned with the operational gate the paper trader applies, so
training a model on them produces signals that can be consumed directly by the
gate without redefining direction as a proxy for "good trade".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.database import read_ohlcv_prices, read_technical_features


OUTCOME_LABELS = ["loss", "timeout", "win"]
OUTCOME_TO_ID = {label: index for index, label in enumerate(OUTCOME_LABELS)}
ID_TO_OUTCOME = {index: label for label, index in OUTCOME_TO_ID.items()}

BASE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "volatility_21d",
    "volatility_63d",
    "volume_ratio_21d",
    "drawdown_252d",
    "rsi_14",
]

DERIVED_FEATURE_COLUMNS = [
    "close_to_ma_7",
    "close_to_ma_21",
    "close_to_ma_63",
    "close_to_ma_252",
    "ma_7_to_ma_21",
    "ma_21_to_ma_63",
    "ma_63_to_ma_252",
    "rsi_14_scaled",
    "volatility_21_to_63",
    "return_21d_to_volatility",
]

REGIME_FEATURE_COLUMNS = [
    "market_return_21d_p50",
    "market_volatility_21d_p50",
    "relative_return_21d",
    "relative_volatility_21d",
]

TRADE_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_close = df["close"].astype("float64")
    df["rsi_14_scaled"] = (df["rsi_14"].astype("float64") - 50.0) / 50.0
    df["close_to_ma_7"] = safe_close / df["ma_7"].astype("float64") - 1.0
    df["close_to_ma_21"] = safe_close / df["ma_21"].astype("float64") - 1.0
    df["close_to_ma_63"] = safe_close / df["ma_63"].astype("float64") - 1.0
    df["close_to_ma_252"] = safe_close / df["ma_252"].astype("float64") - 1.0
    df["ma_7_to_ma_21"] = df["ma_7"].astype("float64") / df["ma_21"].astype("float64") - 1.0
    df["ma_21_to_ma_63"] = df["ma_21"].astype("float64") / df["ma_63"].astype("float64") - 1.0
    df["ma_63_to_ma_252"] = df["ma_63"].astype("float64") / df["ma_252"].astype("float64") - 1.0
    vol_63 = df["volatility_63d"].astype("float64").clip(lower=1e-9)
    df["volatility_21_to_63"] = df["volatility_21d"].astype("float64") / vol_63
    df["return_21d_to_volatility"] = df["return_21d"].astype("float64") / vol_63
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    daily_market_ret = df.groupby("date")["return_21d"].transform("median")
    daily_market_vol = df.groupby("date")["volatility_21d"].transform("median")
    df["market_return_21d_p50"] = daily_market_ret
    df["market_volatility_21d_p50"] = daily_market_vol
    df["relative_return_21d"] = df["return_21d"].astype("float64") - daily_market_ret
    df["relative_volatility_21d"] = df["volatility_21d"].astype("float64") - daily_market_vol
    return df


def calculate_stop_distance(volatility_21d: float) -> float:
    return float(np.clip(float(volatility_21d) * 2.0, 0.03, 0.12))


def simulate_trade_outcome(
    entry_close: float,
    future_bars: list[dict],
    stop_distance: float,
    target_distance: float,
    holding_days: int,
    execution_drag: float,
):
    """Walk forward bar-by-bar; conservatively assume stop fills before target
    when both prices are touched in the same bar."""
    if not future_bars:
        return None, None, None
    horizon_bars = future_bars[:holding_days]
    if not horizon_bars:
        return None, None, None
    stop_price = entry_close * (1.0 - stop_distance)
    target_price = entry_close * (1.0 + target_distance)
    for offset, bar in enumerate(horizon_bars, start=1):
        low = float(bar["low"])
        high = float(bar["high"])
        hit_stop = low <= stop_price
        hit_target = high >= target_price
        if hit_stop:
            return "loss", -stop_distance - execution_drag, offset
        if hit_target:
            return "win", target_distance - execution_drag, offset
    last_close = float(horizon_bars[-1]["close"])
    raw_return = last_close / entry_close - 1.0
    return "timeout", raw_return - execution_drag, len(horizon_bars)


def build_trade_outcome_dataset(
    holding_days: int = 7,
    min_reward_risk: float = 1.5,
    cost_per_trade: float = 0.002,
    spread: float = 0.001,
    slippage: float = 0.001,
) -> pd.DataFrame:
    features = read_technical_features()
    if features.empty:
        return pd.DataFrame()
    prices = read_ohlcv_prices()
    if prices.empty:
        return pd.DataFrame()

    features = add_derived_features(features)
    features = add_regime_features(features)

    execution_drag = float(cost_per_trade) + float(spread) + float(slippage)

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")

    rows: list[dict] = []
    for ticker, ticker_prices in prices.groupby("ticker", sort=False):
        ticker_prices = ticker_prices.reset_index(drop=True)
        date_to_index = {d: i for i, d in enumerate(ticker_prices["date"].tolist())}
        highs = ticker_prices["high"].to_numpy(dtype="float64")
        lows = ticker_prices["low"].to_numpy(dtype="float64")
        closes = ticker_prices["close"].to_numpy(dtype="float64")

        ticker_features = features[features["ticker"] == ticker]
        for _, feature_row in ticker_features.iterrows():
            date_str = (
                feature_row["date"]
                if isinstance(feature_row["date"], str)
                else pd.Timestamp(feature_row["date"]).strftime("%Y-%m-%d")
            )
            idx = date_to_index.get(date_str)
            if idx is None:
                continue
            entry_close = float(closes[idx])
            stop_distance = calculate_stop_distance(feature_row["volatility_21d"])
            target_distance = stop_distance * float(min_reward_risk)
            future_slice: list[dict] = []
            stop_at = min(idx + 1 + holding_days, len(closes))
            for j in range(idx + 1, stop_at):
                future_slice.append(
                    {
                        "high": float(highs[j]),
                        "low": float(lows[j]),
                        "close": float(closes[j]),
                    }
                )
            outcome, trade_return, exit_offset = simulate_trade_outcome(
                entry_close,
                future_slice,
                stop_distance,
                target_distance,
                holding_days,
                execution_drag,
            )
            if outcome is None:
                continue
            record = {
                "ticker": ticker,
                "date": date_str,
                "time_split": feature_row["time_split"],
                "entry_close": entry_close,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "execution_drag": execution_drag,
                "trade_outcome": outcome,
                "trade_return": float(trade_return),
                "exit_offset_days": int(exit_offset),
            }
            for column in TRADE_FEATURE_COLUMNS:
                if column in feature_row.index:
                    record[column] = float(feature_row[column])
            rows.append(record)

    if not rows:
        return pd.DataFrame()
    dataset = pd.DataFrame(rows)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(
        subset=TRADE_FEATURE_COLUMNS + ["trade_outcome", "trade_return"]
    )
    return dataset.reset_index(drop=True)


def build_current_trade_outcome_features() -> pd.DataFrame:
    """Latest feature row per ticker, ready for trade-outcome inference.

    Uses the price-derived features fed into the supervised dataset, which
    guarantees the same engineering as training.  The latest row is the most
    recent trading session for which technical indicators (252d window etc.) are
    fully defined.
    """
    from app.features.technical import build_current_technical_features

    prices = read_ohlcv_prices()
    if prices.empty:
        return pd.DataFrame()
    current = build_current_technical_features(prices)
    if current.empty:
        return pd.DataFrame()
    current = add_derived_features(current)
    current = add_regime_features(current)
    current = current.replace([np.inf, -np.inf], np.nan)
    current = current.dropna(subset=TRADE_FEATURE_COLUMNS)
    return current.reset_index(drop=True)
