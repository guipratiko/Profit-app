from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import CONTEXT_INDEX_TICKERS, INITIAL_ASSETS
from app.data.database import (
    initialize_database,
    read_ohlcv_prices,
    replace_technical_features,
)


SHORT_HORIZON_DAYS = 7
MEDIUM_HORIZON_DAYS = 63
LONG_HORIZON_DAYS = 252
MINIMUM_HISTORY_DAYS = 252

# Heuristic mapping: BR tickers benchmark vs Ibovespa, US tickers vs SPY.
# Used to compute relative_strength_5d (return - benchmark return).
BR_BENCHMARK = "^BVSP"
US_BENCHMARK = "SPY"


def _benchmark_for(ticker: str) -> str:
    return BR_BENCHMARK if ticker.endswith(".SA") else US_BENCHMARK


def _build_index_context(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot context indices into a per-date frame with their 5d returns.

    Returns columns: date, index_spy_return_5d, index_qqq_return_5d, index_bvsp_return_5d.
    Missing indices come back as NaN (handled by downstream dropna).
    """
    if prices.empty:
        return pd.DataFrame(columns=["date"])
    index_tickers = list(CONTEXT_INDEX_TICKERS.keys())
    index_prices = prices[prices["ticker"].isin(index_tickers)].copy()
    if index_prices.empty:
        return pd.DataFrame(columns=["date"])
    column_map = {
        "SPY": "index_spy_return_5d",
        "QQQ": "index_qqq_return_5d",
        "^BVSP": "index_bvsp_return_5d",
    }
    frames: list[pd.DataFrame] = []
    for ticker, group in index_prices.groupby("ticker", sort=False):
        col = column_map.get(ticker)
        if col is None:
            continue
        ordered = group.sort_values("date")[["date", "close"]].copy()
        ordered[col] = ordered["close"].pct_change(periods=5, fill_method=None)
        frames.append(ordered[["date", col]])
    if not frames:
        return pd.DataFrame(columns=["date"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date")


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    price_delta = close_prices.diff()
    gains = price_delta.clip(lower=0)
    losses = -price_delta.clip(upper=0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + relative_strength))


def classify_direction(target_return: pd.Series, threshold: float = 0.005) -> pd.Series:
    """Static-threshold classifier kept for backwards compatibility / analytics."""
    direction = pd.Series("sideways", index=target_return.index, dtype="object")
    direction = direction.mask(target_return > threshold, "up")
    direction = direction.mask(target_return < -threshold, "down")
    direction = direction.mask(target_return.isna(), pd.NA)
    return direction


def classify_direction_vol_aware(
    target_return: pd.Series,
    volatility_21d: pd.Series,
    horizon_days: int = SHORT_HORIZON_DAYS,
    cost_buffer: float = 0.004,
    vol_multiplier: float = 0.5,
) -> pd.Series:
    """Label direction using a per-row threshold: cost + vol * sqrt(horizon) * mult.

    The fixed-0.5% threshold treats noise and signal identically across very
    different volatility regimes; this version scales the band with realised
    daily volatility annualised to the horizon, plus a buffer covering the
    minimum execution drag (spread+slippage+fees).
    """
    horizon_scale = float(horizon_days) ** 0.5
    band = volatility_21d.astype("float64").fillna(0.0).abs() * horizon_scale * vol_multiplier
    band = band + float(cost_buffer)
    direction = pd.Series("sideways", index=target_return.index, dtype="object")
    direction = direction.mask(target_return > band, "up")
    direction = direction.mask(target_return < -band, "down")
    direction = direction.mask(target_return.isna(), pd.NA)
    return direction


def compute_enter_long_label(
    target_return: pd.Series,
    volatility_21d: pd.Series,
    horizon_days: int = SHORT_HORIZON_DAYS,
    cost_buffer: float = 0.006,
    vol_multiplier: float = 0.6,
) -> pd.Series:
    """Binary label: 1 when forward return clears costs + volatility-scaled edge band.

    Threshold = cost_buffer + vol_21d * sqrt(horizon) * vol_multiplier.
    cost_buffer (0.6%) approximates round-trip fee + half-spread for liquid BR/US names.
    vol_multiplier (0.6) requires the move to exceed roughly 0.6 sigma of horizon noise,
    so that the trained classifier targets edges robust to typical price oscillation
    rather than memorising directional noise (root cause of the 3-class 0.37 val_acc).
    """
    horizon_scale = float(horizon_days) ** 0.5
    band = volatility_21d.astype("float64").fillna(0.0).abs() * horizon_scale * vol_multiplier
    threshold = band + float(cost_buffer)
    label = (target_return.astype("float64") > threshold).astype("Int64")
    label = label.mask(target_return.isna(), pd.NA)
    return label


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

    if not split_frames:
        # Sem linhas (histórico curto → dropna remove tudo) ou sem grupos: garantir coluna time_split
        # para o caller poder reindexar / persistir sem KeyError.
        out = features.copy()
        if "time_split" not in out.columns:
            out["time_split"] = pd.Series(dtype="string")
        return out

    return pd.concat(split_frames, ignore_index=True)


def build_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prepared_prices = prices.copy()
    prepared_prices["date"] = pd.to_datetime(prepared_prices["date"])
    prepared_prices = prepared_prices.sort_values(["ticker", "date"])

    # Build per-date cross-asset context (index returns) before filtering them out.
    index_context = _build_index_context(prepared_prices)
    if not index_context.empty:
        index_context["date"] = pd.to_datetime(index_context["date"])

    # Trading universe: drop context indices so they receive no targets/predictions.
    context_set = set(CONTEXT_INDEX_TICKERS.keys())
    universe_prices = prepared_prices[~prepared_prices["ticker"].isin(context_set)].copy()

    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_prices in universe_prices.groupby("ticker", sort=False):
        ticker_features = ticker_prices.sort_values("date").copy()
        close_prices = ticker_features["close"]
        open_prices = ticker_features["open"]
        high_prices = ticker_features["high"]
        low_prices = ticker_features["low"]
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

        prev_close = close_prices.shift(1)
        true_range_components = pd.concat(
            [
                (high_prices - low_prices),
                (high_prices - prev_close).abs(),
                (low_prices - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range_components.rolling(window=14, min_periods=14).mean()
        ticker_features["atr_pct_14"] = atr_14 / close_prices
        ticker_features["gap_pct"] = (open_prices - prev_close) / prev_close
        ticker_features["candle_body_pct"] = (close_prices - open_prices) / open_prices.replace(0, pd.NA)
        daily_range_pct = (high_prices - low_prices) / close_prices.replace(0, pd.NA)
        ticker_features["range_pct_21d"] = daily_range_pct.rolling(window=21, min_periods=21).mean()

        ticker_features["target_return_7d"] = close_prices.shift(-SHORT_HORIZON_DAYS) / close_prices - 1
        ticker_features["target_return_3m"] = close_prices.shift(-MEDIUM_HORIZON_DAYS) / close_prices - 1
        ticker_features["target_return_1y"] = close_prices.shift(-LONG_HORIZON_DAYS) / close_prices - 1
        ticker_features["target_direction_7d"] = classify_direction_vol_aware(
            ticker_features["target_return_7d"],
            ticker_features["volatility_21d"],
            horizon_days=SHORT_HORIZON_DAYS,
        )
        ticker_features["target_direction_3m"] = classify_direction_vol_aware(
            ticker_features["target_return_3m"],
            ticker_features["volatility_21d"],
            horizon_days=MEDIUM_HORIZON_DAYS,
        )
        ticker_features["target_direction_1y"] = classify_direction_vol_aware(
            ticker_features["target_return_1y"],
            ticker_features["volatility_21d"],
            horizon_days=LONG_HORIZON_DAYS,
        )
        ticker_features["target_enter_long_7d"] = compute_enter_long_label(
            ticker_features["target_return_7d"],
            ticker_features["volatility_21d"],
            horizon_days=SHORT_HORIZON_DAYS,
        )

        # --- Regime feature: volatility-of-volatility (21d std of vol_21d) ---
        ticker_features["vol_of_vol_21d"] = ticker_features["volatility_21d"].rolling(
            window=21, min_periods=21
        ).std()

        # --- Volume-profile feature: OBV slope (21d OLS slope of normalised OBV) ---
        # Vectorised: with x_centered summing to zero, slope = sum(x_centered * y) / x_var.
        # Implemented via numpy.convolve over the whole series for speed.
        signed_volume = volume.astype("float64") * np.sign(ticker_features["return_1d"].astype("float64")).fillna(0.0)
        obv = signed_volume.cumsum()
        obv_scale = obv.abs().rolling(window=21, min_periods=21).mean()
        obv_scale = obv_scale.where(obv_scale > 0, np.nan)
        obv_normalized = (obv / obv_scale).astype("float64")
        window_len = 21
        x_centered = np.arange(window_len, dtype="float64") - (window_len - 1) / 2.0
        x_var = float((x_centered ** 2).sum())
        y_values = obv_normalized.to_numpy(dtype="float64")
        if len(y_values) >= window_len and x_var > 0:
            kernel = x_centered[::-1]  # convolve reverses the kernel
            convolved = np.convolve(np.nan_to_num(y_values, nan=0.0), kernel, mode="valid") / x_var
            nan_mask = (
                pd.Series(np.isnan(y_values).astype("float64"))
                .rolling(window=window_len, min_periods=window_len)
                .sum()
                .to_numpy()[window_len - 1 :]
            )
            convolved = np.where(nan_mask > 0, np.nan, convolved)
            slope_series = np.concatenate([np.full(window_len - 1, np.nan), convolved])
        else:
            slope_series = np.full(len(y_values), np.nan)
        ticker_features["obv_slope_21d"] = slope_series

        ticker_features["ticker"] = ticker
        feature_frames.append(ticker_features)

    features = pd.concat(feature_frames, ignore_index=True)

    # --- Cross-asset join: index 5d returns + relative strength vs benchmark ---
    if not index_context.empty:
        features = features.merge(index_context, on="date", how="left")
        bench_ret = np.where(
            features["ticker"].astype(str).str.endswith(".SA"),
            features.get("index_bvsp_return_5d", pd.Series([np.nan] * len(features))).astype("float64"),
            features.get("index_spy_return_5d", pd.Series([np.nan] * len(features))).astype("float64"),
        )
        features["relative_strength_5d"] = features["return_5d"].astype("float64") - bench_ret
    else:
        features["index_spy_return_5d"] = np.nan
        features["index_qqq_return_5d"] = np.nan
        features["index_bvsp_return_5d"] = np.nan
        features["relative_strength_5d"] = np.nan

    # --- Breadth feature: % of universe trading above its 252d MA per date ---
    above_ma = (features["close"] > features["ma_252"]).astype("float64")
    breadth = features.assign(_above=above_ma).groupby("date")["_above"].transform("mean")
    features["breadth_above_ma200"] = breadth

    features = features.dropna(
        subset=[
            "ma_252",
            "volatility_63d",
            "drawdown_252d",
            "rsi_14",
            "atr_pct_14",
            "range_pct_21d",
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
            "atr_pct_14",
            "gap_pct",
            "candle_body_pct",
            "range_pct_21d",
            "target_return_7d",
            "target_return_3m",
            "target_return_1y",
            "target_direction_7d",
            "target_direction_3m",
            "target_direction_1y",
            "target_enter_long_7d",
            "index_spy_return_5d",
            "index_qqq_return_5d",
            "index_bvsp_return_5d",
            "relative_strength_5d",
            "vol_of_vol_21d",
            "obv_slope_21d",
            "breadth_above_ma200",
            "time_split",
        ]
    ]


def build_current_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prepared_prices = prices.copy()
    prepared_prices["date"] = pd.to_datetime(prepared_prices["date"])
    prepared_prices = prepared_prices.sort_values(["ticker", "date"])

    # Cross-asset context computed before stripping context indices from universe.
    index_context = _build_index_context(prepared_prices)
    if not index_context.empty:
        index_context["date"] = pd.to_datetime(index_context["date"])

    context_set = set(CONTEXT_INDEX_TICKERS.keys())
    universe_prices = prepared_prices[~prepared_prices["ticker"].isin(context_set)].copy()

    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_prices in universe_prices.groupby("ticker", sort=False):
        ticker_features = ticker_prices.sort_values("date").copy()
        close_prices = ticker_features["close"]
        open_prices = ticker_features["open"]
        high_prices = ticker_features["high"]
        low_prices = ticker_features["low"]
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
        prev_close = close_prices.shift(1)
        true_range_components = pd.concat(
            [
                (high_prices - low_prices),
                (high_prices - prev_close).abs(),
                (low_prices - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range_components.rolling(window=14, min_periods=14).mean()
        ticker_features["atr_pct_14"] = atr_14 / close_prices
        ticker_features["gap_pct"] = (open_prices - prev_close) / prev_close
        ticker_features["candle_body_pct"] = (close_prices - open_prices) / open_prices.replace(0, pd.NA)
        daily_range_pct = (high_prices - low_prices) / close_prices.replace(0, pd.NA)
        ticker_features["range_pct_21d"] = daily_range_pct.rolling(window=21, min_periods=21).mean()

        # Regime feature: vol-of-vol (21d std of vol_21d).
        ticker_features["vol_of_vol_21d"] = ticker_features["volatility_21d"].rolling(
            window=21, min_periods=21
        ).std()

        # OBV slope (21d OLS slope of normalised OBV).
        signed_volume = volume.astype("float64") * np.sign(
            ticker_features["return_1d"].astype("float64")
        ).fillna(0.0)
        obv = signed_volume.cumsum()
        obv_scale = obv.abs().rolling(window=21, min_periods=21).mean()
        obv_scale = obv_scale.where(obv_scale > 0, np.nan)
        obv_normalized = (obv / obv_scale).astype("float64")
        window_len = 21
        x_centered = np.arange(window_len, dtype="float64") - (window_len - 1) / 2.0
        x_var = float((x_centered ** 2).sum())
        y_values = obv_normalized.to_numpy(dtype="float64")
        if len(y_values) >= window_len and x_var > 0:
            kernel = x_centered[::-1]
            convolved = np.convolve(np.nan_to_num(y_values, nan=0.0), kernel, mode="valid") / x_var
            nan_mask = (
                pd.Series(np.isnan(y_values).astype("float64"))
                .rolling(window=window_len, min_periods=window_len)
                .sum()
                .to_numpy()[window_len - 1 :]
            )
            convolved = np.where(nan_mask > 0, np.nan, convolved)
            slope_series = np.concatenate([np.full(window_len - 1, np.nan), convolved])
        else:
            slope_series = np.full(len(y_values), np.nan)
        ticker_features["obv_slope_21d"] = slope_series

        ticker_features["ticker"] = ticker
        feature_frames.append(ticker_features)

    features = pd.concat(feature_frames, ignore_index=True)

    # Cross-asset join: index 5d returns + relative strength vs benchmark.
    if not index_context.empty:
        features = features.merge(index_context, on="date", how="left")
        bench_ret = np.where(
            features["ticker"].astype(str).str.endswith(".SA"),
            features.get("index_bvsp_return_5d", pd.Series([np.nan] * len(features))).astype("float64"),
            features.get("index_spy_return_5d", pd.Series([np.nan] * len(features))).astype("float64"),
        )
        features["relative_strength_5d"] = features["return_5d"].astype("float64") - bench_ret
    else:
        features["index_spy_return_5d"] = np.nan
        features["index_qqq_return_5d"] = np.nan
        features["index_bvsp_return_5d"] = np.nan
        features["relative_strength_5d"] = np.nan

    # Breadth feature: % of universe trading above its 252d MA per date.
    above_ma = (features["close"] > features["ma_252"]).astype("float64")
    breadth = features.assign(_above=above_ma).groupby("date")["_above"].transform("mean")
    features["breadth_above_ma200"] = breadth

    features = features.dropna(
        subset=[
            "ma_252",
            "volatility_63d",
            "drawdown_252d",
            "rsi_14",
            "atr_pct_14",
            "range_pct_21d",
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
            "atr_pct_14",
            "gap_pct",
            "candle_body_pct",
            "range_pct_21d",
            "index_spy_return_5d",
            "index_qqq_return_5d",
            "index_bvsp_return_5d",
            "relative_strength_5d",
            "vol_of_vol_21d",
            "obv_slope_21d",
            "breadth_above_ma200",
        ]
    ]


def generate_technical_features() -> int:
    initialize_database()
    prices = read_ohlcv_prices()
    features = build_technical_features(prices)
    return replace_technical_features(features)