from __future__ import annotations

import pandas as pd

from app.data.database import (
    initialize_database,
    read_ohlcv_prices,
    replace_technical_features,
)


SHORT_HORIZON_DAYS = 7
MEDIUM_HORIZON_DAYS = 63
LONG_HORIZON_DAYS = 252
MINIMUM_HISTORY_DAYS = 252


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    price_delta = close_prices.diff()
    gains = price_delta.clip(lower=0)
    losses = -price_delta.clip(upper=0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + relative_strength))


def classify_direction(target_return: pd.Series, threshold: float = 0.005) -> pd.Series:
    direction = pd.Series("sideways", index=target_return.index, dtype="object")
    direction = direction.mask(target_return > threshold, "up")
    direction = direction.mask(target_return < -threshold, "down")
    direction = direction.mask(target_return.isna(), pd.NA)
    return direction


def assign_time_split(features: pd.DataFrame) -> pd.DataFrame:
    split_frames: list[pd.DataFrame] = []

    for _ticker, ticker_features in features.groupby("ticker", sort=False):
        ordered_features = ticker_features.sort_values("date").copy()
        row_count = len(ordered_features)
        train_end = int(row_count * 0.70)
        validation_end = int(row_count * 0.85)

        ordered_features["time_split"] = "test"
        ordered_features.iloc[:train_end, ordered_features.columns.get_loc("time_split")] = "train"
        ordered_features.iloc[
            train_end:validation_end,
            ordered_features.columns.get_loc("time_split"),
        ] = "validation"
        split_frames.append(ordered_features)

    return pd.concat(split_frames, ignore_index=True) if split_frames else features


def build_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prepared_prices = prices.copy()
    prepared_prices["date"] = pd.to_datetime(prepared_prices["date"])
    prepared_prices = prepared_prices.sort_values(["ticker", "date"])

    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_prices in prepared_prices.groupby("ticker", sort=False):
        ticker_features = ticker_prices.sort_values("date").copy()
        close_prices = ticker_features["close"]
        volume = ticker_features["volume"]

        ticker_features["return_1d"] = close_prices.pct_change(fill_method=None)
        ticker_features["return_5d"] = close_prices.pct_change(periods=5, fill_method=None)
        ticker_features["return_21d"] = close_prices.pct_change(periods=21, fill_method=None)
        ticker_features["ma_7"] = close_prices.rolling(window=7, min_periods=7).mean()
        ticker_features["ma_21"] = close_prices.rolling(window=21, min_periods=21).mean()
        ticker_features["ma_63"] = close_prices.rolling(window=63, min_periods=63).mean()
        ticker_features["ma_252"] = close_prices.rolling(window=252, min_periods=252).mean()
        ticker_features["volatility_21d"] = ticker_features["return_1d"].rolling(
            window=21,
            min_periods=21,
        ).std()
        ticker_features["volatility_63d"] = ticker_features["return_1d"].rolling(
            window=63,
            min_periods=63,
        ).std()
        ticker_features["volume_ratio_21d"] = volume / volume.rolling(window=21, min_periods=21).mean()
        rolling_peak = close_prices.rolling(window=252, min_periods=252).max()
        ticker_features["drawdown_252d"] = close_prices / rolling_peak - 1
        ticker_features["rsi_14"] = calculate_rsi(close_prices)

        ticker_features["target_return_7d"] = close_prices.shift(-SHORT_HORIZON_DAYS) / close_prices - 1
        ticker_features["target_return_3m"] = close_prices.shift(-MEDIUM_HORIZON_DAYS) / close_prices - 1
        ticker_features["target_return_1y"] = close_prices.shift(-LONG_HORIZON_DAYS) / close_prices - 1
        ticker_features["target_direction_7d"] = classify_direction(ticker_features["target_return_7d"])
        ticker_features["target_direction_3m"] = classify_direction(ticker_features["target_return_3m"])
        ticker_features["target_direction_1y"] = classify_direction(ticker_features["target_return_1y"])
        ticker_features["ticker"] = ticker
        feature_frames.append(ticker_features)

    features = pd.concat(feature_frames, ignore_index=True)
    features = features.dropna(
        subset=[
            "ma_252",
            "volatility_63d",
            "drawdown_252d",
            "rsi_14",
            "target_return_7d",
            "target_return_3m",
            "target_return_1y",
        ]
    )
    features = assign_time_split(features)
    features["date"] = features["date"].dt.strftime("%Y-%m-%d")

    return features[
        [
            "ticker",
            "date",
            "close",
            "volume",
            "return_1d",
            "return_5d",
            "return_21d",
            "ma_7",
            "ma_21",
            "ma_63",
            "ma_252",
            "volatility_21d",
            "volatility_63d",
            "volume_ratio_21d",
            "drawdown_252d",
            "rsi_14",
            "target_return_7d",
            "target_return_3m",
            "target_return_1y",
            "target_direction_7d",
            "target_direction_3m",
            "target_direction_1y",
            "time_split",
        ]
    ]


def build_current_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prepared_prices = prices.copy()
    prepared_prices["date"] = pd.to_datetime(prepared_prices["date"])
    prepared_prices = prepared_prices.sort_values(["ticker", "date"])

    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_prices in prepared_prices.groupby("ticker", sort=False):
        ticker_features = ticker_prices.sort_values("date").copy()
        close_prices = ticker_features["close"]
        volume = ticker_features["volume"]

        ticker_features["return_1d"] = close_prices.pct_change(fill_method=None)
        ticker_features["return_5d"] = close_prices.pct_change(periods=5, fill_method=None)
        ticker_features["return_21d"] = close_prices.pct_change(periods=21, fill_method=None)
        ticker_features["ma_7"] = close_prices.rolling(window=7, min_periods=7).mean()
        ticker_features["ma_21"] = close_prices.rolling(window=21, min_periods=21).mean()
        ticker_features["ma_63"] = close_prices.rolling(window=63, min_periods=63).mean()
        ticker_features["ma_252"] = close_prices.rolling(window=252, min_periods=252).mean()
        ticker_features["volatility_21d"] = ticker_features["return_1d"].rolling(
            window=21,
            min_periods=21,
        ).std()
        ticker_features["volatility_63d"] = ticker_features["return_1d"].rolling(
            window=63,
            min_periods=63,
        ).std()
        ticker_features["volume_ratio_21d"] = volume / volume.rolling(window=21, min_periods=21).mean()
        rolling_peak = close_prices.rolling(window=252, min_periods=252).max()
        ticker_features["drawdown_252d"] = close_prices / rolling_peak - 1
        ticker_features["rsi_14"] = calculate_rsi(close_prices)
        ticker_features["ticker"] = ticker
        feature_frames.append(ticker_features)

    features = pd.concat(feature_frames, ignore_index=True)
    features = features.dropna(
        subset=[
            "ma_252",
            "volatility_63d",
            "drawdown_252d",
            "rsi_14",
        ]
    )
    if features.empty:
        return pd.DataFrame()

    features = features.sort_values("date").groupby("ticker", as_index=False).tail(1)
    features["date"] = features["date"].dt.strftime("%Y-%m-%d")

    return features[
        [
            "ticker",
            "date",
            "close",
            "volume",
            "return_1d",
            "return_5d",
            "return_21d",
            "ma_7",
            "ma_21",
            "ma_63",
            "ma_252",
            "volatility_21d",
            "volatility_63d",
            "volume_ratio_21d",
            "drawdown_252d",
            "rsi_14",
        ]
    ]


def generate_technical_features() -> int:
    initialize_database()
    prices = read_ohlcv_prices()
    features = build_technical_features(prices)
    return replace_technical_features(features)