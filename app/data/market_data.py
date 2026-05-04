from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.config import DEFAULT_PRICE_INTERVAL, DEFAULT_PRICE_PERIOD, INITIAL_ASSETS
from app.data.database import initialize_database, upsert_ohlcv_prices


YFINANCE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


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


def update_all_prices(
    tickers: list[str] | None = None,
    period: str = DEFAULT_PRICE_PERIOD,
    interval: str = DEFAULT_PRICE_INTERVAL,
) -> dict[str, int]:
    initialize_database()
    selected_tickers = tickers or list(INITIAL_ASSETS)
    updated_rows: dict[str, int] = {}

    for ticker in selected_tickers:
        prices = fetch_ohlcv(ticker=ticker, period=period, interval=interval)
        updated_rows[ticker] = upsert_ohlcv_prices(prices)

    return updated_rows
