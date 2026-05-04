from __future__ import annotations

import hashlib
import json
import re
from datetime import time
from typing import Iterable

import pandas as pd

from app.config import INITIAL_ASSETS
from app.data.database import initialize_database, read_ohlcv_prices, save_news_events


ENTITY_REPLACEMENTS = {
    "banco central": "BACEN",
    "bacen": "BACEN",
    "copom": "COPOM",
    "comite de politica monetaria": "COPOM",
    "comitê de política monetária": "COPOM",
    "petroleo": "PETROLEO",
    "petróleo": "PETROLEO",
    "minerio de ferro": "MINERIO_DE_FERRO",
    "minério de ferro": "MINERIO_DE_FERRO",
}
MARKET_CLOSE = time(18, 0)


def normalize_text(value: str | None) -> str:
    text = value or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.lower()
    for source, target in ENTITY_REPLACEMENTS.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered, flags=re.IGNORECASE)
    return lowered


def get_trading_calendar(prices: pd.DataFrame | None = None) -> list[pd.Timestamp]:
    if prices is None:
        prices = read_ohlcv_prices()
    if prices.empty:
        return []
    dates = pd.to_datetime(prices["date"]).drop_duplicates().sort_values()
    return [pd.Timestamp(date).normalize() for date in dates]


def align_to_next_trading_session(
    published_at: str | pd.Timestamp,
    trading_calendar: Iterable[pd.Timestamp],
    market_close: time = MARKET_CLOSE,
) -> str:
    timestamp = pd.Timestamp(published_at)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    publication_day = timestamp.normalize()
    calendar = sorted(pd.Timestamp(date).normalize() for date in trading_calendar)
    if not calendar:
        raise ValueError("Trading calendar is empty. Load OHLCV prices before aligning news.")

    for trading_day in calendar:
        if trading_day < publication_day:
            continue
        if trading_day == publication_day and timestamp.time() >= market_close:
            continue
        return trading_day.strftime("%Y-%m-%d")

    return calendar[-1].strftime("%Y-%m-%d")


def build_event_id(ticker: str, title: str, published_at: str, source: str) -> str:
    payload = f"{ticker}|{title}|{published_at}|{source}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"news_{digest}"


def build_news_event(
    ticker: str,
    title: str,
    published_at: str,
    trading_calendar: Iterable[pd.Timestamp],
    source: str = "manual",
    body: str | None = None,
    url: str | None = None,
    raw: dict | None = None,
) -> dict:
    normalized_title = normalize_text(title)
    normalized_body = normalize_text(body)
    normalized_text = " ".join(part for part in [normalized_title, normalized_body] if part).strip()
    raw_payload = raw or {
        "ticker": ticker,
        "title": title,
        "body": body,
        "published_at": published_at,
        "source": source,
        "url": url,
    }
    return {
        "event_id": build_event_id(ticker, title, published_at, source),
        "ticker": ticker,
        "source": source,
        "title": title,
        "body": body,
        "normalized_text": normalized_text,
        "published_at": pd.Timestamp(published_at).isoformat(),
        "aligned_trading_date": align_to_next_trading_session(published_at, trading_calendar),
        "url": url,
        "raw_json": json.dumps(raw_payload, ensure_ascii=True),
    }


def build_sample_news_events() -> list[dict]:
    initialize_database()
    calendar = get_trading_calendar()
    if not calendar:
        raise ValueError("No OHLCV prices found. Run update-prices before creating sample news events.")

    last_session = calendar[-2] if len(calendar) > 1 else calendar[-1]
    after_close = last_session.replace(hour=20, minute=0, second=0)
    tickers = list(INITIAL_ASSETS.keys())[:3]
    samples = [
        {
            "ticker": tickers[0],
            "title": "Petrobras divulga fato relevante apos fechamento do mercado",
            "body": "Comunicado cita PETROLEO, dividendos e contexto de Banco Central.",
        },
        {
            "ticker": tickers[1],
            "title": "Vale acompanha variacao do minerio de ferro no exterior",
            "body": "Evento associado a commodities e demanda internacional.",
        },
        {
            "ticker": tickers[2],
            "title": "Itau avalia cenario de juros antes do Copom",
            "body": "Texto normaliza Copom, Bacen e Banco Central para entidades coerentes.",
        },
    ]
    return [
        build_news_event(
            ticker=sample["ticker"],
            title=sample["title"],
            body=sample["body"],
            published_at=after_close.isoformat(),
            trading_calendar=calendar,
            source="sample",
        )
        for sample in samples
    ]


def save_sample_news_events() -> dict:
    events = build_sample_news_events()
    inserted_rows = save_news_events(events)
    aligned_dates = sorted({event["aligned_trading_date"] for event in events})
    return {
        "generated": len(events),
        "inserted": inserted_rows,
        "aligned_dates": aligned_dates,
    }
