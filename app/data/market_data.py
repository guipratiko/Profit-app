from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from app.config import (
    CONTEXT_INDEX_TICKERS,
    DEFAULT_PRICE_INTERVAL,
    DEFAULT_PRICE_PERIOD,
    INITIAL_ASSETS,
)
from app.data.database import (
    initialize_database,
    register_context_index_assets,
    upsert_ohlcv_prices,
)


YFINANCE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

MARKET_SNAPSHOT_TIMEZONE = ZoneInfo("America/Sao_Paulo")


def _coerce_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _coerce_int(value) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number


def _fast_info_value(fast_info, *keys: str):
    for key in keys:
        try:
            value = fast_info.get(key)
        except AttributeError:
            try:
                value = fast_info[key]
            except Exception:
                value = None
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def _latest_intraday_snapshot(ticker: str, snapshot_date: str) -> pd.DataFrame:
    try:
        intraday = yf.download(
            ticker,
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if intraday.empty:
        return pd.DataFrame()

    if isinstance(intraday.columns, pd.MultiIndex):
        intraday.columns = intraday.columns.get_level_values(0)
    intraday = intraday.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    return _snapshot_from_intraday_frame(ticker, intraday, snapshot_date)


def _snapshot_from_intraday_frame(ticker: str, intraday: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    if intraday.empty:
        return pd.DataFrame()
    intraday = intraday.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    intraday = intraday.dropna(subset=["close"])
    if intraday.empty:
        return pd.DataFrame()

    row = intraday.iloc[-1]
    close_price = _coerce_float(row.get("close"))
    if close_price is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": snapshot_date,
                "open": _coerce_float(row.get("open")) or close_price,
                "high": max(_coerce_float(row.get("high")) or close_price, close_price),
                "low": min(_coerce_float(row.get("low")) or close_price, close_price),
                "close": close_price,
                "adj_close": _coerce_float(row.get("adj_close")) or close_price,
                "volume": _coerce_int(row.get("volume")) or 0,
                "source": "yfinance_intraday_snapshot",
            }
        ]
    )


def _batch_latest_intraday_snapshots(tickers: list[str], snapshot_date: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    try:
        intraday = yf.download(
            tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}
    if intraday.empty:
        return {}

    snapshots: dict[str, pd.DataFrame] = {}
    if isinstance(intraday.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in intraday.columns.get_level_values(0):
                continue
            snapshots[ticker] = _snapshot_from_intraday_frame(ticker, intraday[ticker], snapshot_date)
    elif len(tickers) == 1:
        snapshots[tickers[0]] = _snapshot_from_intraday_frame(tickers[0], intraday, snapshot_date)
    return snapshots


def fetch_ohlcv(
    ticker: str,
    period: str = DEFAULT_PRICE_PERIOD,
    interval: str = DEFAULT_PRICE_INTERVAL,
) -> pd.DataFrame:
    raw_prices = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw_prices.empty:
        return pd.DataFrame()

    if isinstance(raw_prices.columns, pd.MultiIndex):
        raw_prices.columns = raw_prices.columns.get_level_values(0)

    prices = raw_prices.reset_index().rename(columns=YFINANCE_COLUMNS)
    prices.columns = [str(column).lower().replace(" ", "_") for column in prices.columns]

    if "date" not in prices.columns:
        raise ValueError(f"Missing date column for {ticker}")

    prices["ticker"] = ticker
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    prices["source"] = "yfinance"

    expected_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
    ]
    return prices[expected_columns].dropna(subset=["date", "close"])


def fetch_latest_quote_snapshot(ticker: str, snapshot_at: datetime | None = None) -> pd.DataFrame:
    """Return a one-row daily OHLCV snapshot using the latest available quote.

    Daily yfinance bars only settle after the market session. The cockpit needs
    the value at the instant the user presses Atualizar, so we persist the quote
    as today's close while keeping open/high/low/volume populated when yfinance
    exposes them.
    """
    snapshot_at = snapshot_at or datetime.now(tz=MARKET_SNAPSHOT_TIMEZONE)
    snapshot_date = snapshot_at.astimezone(MARKET_SNAPSHOT_TIMEZONE).strftime("%Y-%m-%d")
    intraday_snapshot = _latest_intraday_snapshot(ticker=ticker, snapshot_date=snapshot_date)
    if not intraday_snapshot.empty:
        return intraday_snapshot

    try:
        quote = yf.Ticker(ticker)
        fast_info = getattr(quote, "fast_info", {}) or {}
    except Exception:
        fast_info = {}

    latest_price = (
        _coerce_float(_fast_info_value(fast_info, "last_price", "lastPrice"))
        or _coerce_float(_fast_info_value(fast_info, "regular_market_price", "regularMarketPrice"))
        or _coerce_float(_fast_info_value(fast_info, "previous_close", "previousClose"))
    )
    if latest_price is None:
        return pd.DataFrame()

    open_price = _coerce_float(_fast_info_value(fast_info, "open")) or latest_price
    high_price = max(_coerce_float(_fast_info_value(fast_info, "day_high", "dayHigh")) or latest_price, latest_price)
    low_price = min(_coerce_float(_fast_info_value(fast_info, "day_low", "dayLow")) or latest_price, latest_price)
    volume = (
        _coerce_int(_fast_info_value(fast_info, "last_volume", "lastVolume"))
        or _coerce_int(_fast_info_value(fast_info, "regular_market_volume", "regularMarketVolume"))
        or 0
    )

    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": snapshot_date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": latest_price,
                "adj_close": latest_price,
                "volume": volume,
                "source": "yfinance_live_quote",
            }
        ]
    )


def update_all_prices(
    tickers: list[str] | None = None,
    period: str = DEFAULT_PRICE_PERIOD,
    interval: str = DEFAULT_PRICE_INTERVAL,
    include_context_indices: bool = True,
) -> dict[str, int]:
    initialize_database()
    if include_context_indices:
        register_context_index_assets()
    selected_tickers = tickers or list(INITIAL_ASSETS)
    if include_context_indices and tickers is None:
        # Append context indices for download but they are filtered out of
        # downstream training/prediction by the feature pipeline.
        for index_ticker in CONTEXT_INDEX_TICKERS:
            if index_ticker not in selected_tickers:
                selected_tickers.append(index_ticker)
    updated_rows: dict[str, int] = {}

    for ticker in selected_tickers:
        prices = fetch_ohlcv(ticker=ticker, period=period, interval=interval)
        updated_rows[ticker] = upsert_ohlcv_prices(prices)

    return updated_rows


def update_latest_quote_snapshots(tickers: list[str] | None = None) -> dict[str, int]:
    initialize_database()
    selected_tickers = tickers or list(INITIAL_ASSETS)
    snapshot_at = datetime.now(tz=MARKET_SNAPSHOT_TIMEZONE)
    snapshot_date = snapshot_at.astimezone(MARKET_SNAPSHOT_TIMEZONE).strftime("%Y-%m-%d")
    batch_snapshots = _batch_latest_intraday_snapshots(selected_tickers, snapshot_date)
    updated_rows: dict[str, int] = {}
    for ticker in selected_tickers:
        try:
            snapshot = batch_snapshots.get(ticker)
            if snapshot is None or snapshot.empty:
                snapshot = fetch_latest_quote_snapshot(ticker=ticker, snapshot_at=snapshot_at)
            updated_rows[ticker] = upsert_ohlcv_prices(snapshot)
        except Exception:
            updated_rows[ticker] = 0
    return updated_rows
