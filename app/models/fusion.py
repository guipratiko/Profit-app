"""Adaptive fusion of technical and qualitative signals.

Phase 1 rewrite. Replaces the prior "soma simples com ajuste pequeno" rule
with a regime-adaptive policy:

* In a **technical regime** (calm vol, no events) the TF probability dominates
  with a small sentiment nudge.
* In a **narrative regime** (event load and/or vol tail) the qualitative score
  carries 60-85% of the weight.
* When the qualitative evidence is *both* directional and severe (Copom +
  clearly negative tone, strong fato-relevante, etc.), the fusion **overrides**
  the technical direction entirely. This is the "the market doesn't care
  what the chart says today" case.

Public surface is preserved so paper.py and the CLI keep working:
- ``calculate_fused_score(probability_up, sentiment_score, event_count)`` -
  same name and inputs; internally uses the adaptive policy with default
  regime assumptions when no extra metadata is supplied.
- ``choose_context_for_signal(...)`` - same name, augmented to also expose
  event tags / severity / magnitude when available in the qualitative row.
- ``run_fusion_predictions(run_id=None)`` - same name, returns the same dict
  shape; new keys are additive.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from app.data.database import (
    get_fusion_predictions,
    get_latest_model_run_id,
    get_qualitative_features,
    initialize_database,
    read_model_predictions,
    read_ohlcv_prices,
    save_fusion_predictions,
)
from app.trading.regime import (
    SENTIMENT_OVERRIDE_THRESHOLD,
    assess_regime,
    estimate_volatility_percentile,
)


FUSION_VERSION = "v2_regime_adaptive_policy"
HORIZON = "7d"

# Default fallback weights when called with the legacy signature (no regime info).
LEGACY_FALLBACK_WEIGHTS = (0.85, 0.15)  # technical, qualitative


def build_fusion_id(run_id: str, ticker: str, signal_date: str, horizon: str) -> str:
    payload = f"{FUSION_VERSION}|{run_id}|{ticker}|{signal_date}|{horizon}"
    return "fusion_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def confidence_from_prediction(row: pd.Series) -> float:
    return float(max(row["probability_down"], row["probability_sideways"], row["probability_up"]))


def _parse_metadata(metadata_text) -> dict:
    if not metadata_text:
        return {}
    try:
        payload = json.loads(str(metadata_text))
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        return {}


def choose_context_for_signal(
    qualitative_features: pd.DataFrame,
    ticker: str,
    signal_date: str,
) -> dict:
    """Return the latest qualitative context known *up to* signal_date.

    Backwards-compatible: always returns the legacy keys (sentiment_score,
    event_count, sentiment_label, aligned_trading_date). Adds event_tags,
    event_severity, event_magnitude when the qualitative row has them in
    metadata_json (rows produced by the v2 sentiment model).
    """
    empty_context = {
        "sentiment_score": 0.0,
        "event_count": 0,
        "sentiment_label": "neutral",
        "aligned_trading_date": None,
        "event_tags": [],
        "event_severity": 0.0,
        "event_magnitude": 0.0,
    }
    if qualitative_features.empty:
        return empty_context

    ticker_context = qualitative_features[qualitative_features["ticker"] == ticker].copy()
    if ticker_context.empty:
        return empty_context

    ticker_context["aligned_trading_date"] = pd.to_datetime(ticker_context["aligned_trading_date"])
    signal_timestamp = pd.Timestamp(signal_date)
    ticker_context = ticker_context[ticker_context["aligned_trading_date"] <= signal_timestamp]
    if ticker_context.empty:
        return empty_context

    latest = ticker_context.sort_values("aligned_trading_date").iloc[-1]
    metadata = _parse_metadata(latest.get("metadata_json"))
    return {
        "sentiment_score": float(latest["sentiment_score"]),
        "event_count": int(latest["event_count"]),
        "sentiment_label": str(latest["sentiment_label"]),
        "aligned_trading_date": latest["aligned_trading_date"].strftime("%Y-%m-%d"),
        "event_tags": list(metadata.get("event_tags", [])),
        "event_severity": float(metadata.get("event_severity_max", 0.0)),
        "event_magnitude": float(metadata.get("event_magnitude_max", 0.0)),
    }


def _ticker_volatility_context(prices: pd.DataFrame, ticker: str, signal_date: str) -> tuple[float, float]:
    """Return (current_21d_vol, percentile_within_history) for the ticker."""
    if prices.empty:
        return 0.0, 0.5
    sub = prices[prices["ticker"] == ticker].copy()
    if sub.empty:
        return 0.0, 0.5
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub[sub["date"] <= pd.Timestamp(signal_date)].sort_values("date")
    if len(sub) < 30:
        return 0.0, 0.5
    sub["ret"] = sub["close"].astype(float).pct_change()
    sub["vol_21d"] = sub["ret"].rolling(21).std()
    history = sub["vol_21d"].dropna().tolist()
    if not history:
        return 0.0, 0.5
    current = float(history[-1])
    pct = estimate_volatility_percentile(current, history)
    return current, pct


def calculate_fused_score(
    probability_up: float,
    sentiment_score: float,
    event_count: int,
    *,
    technical_weight: float | None = None,
    qualitative_weight: float | None = None,
    override_qualitative: bool = False,
    event_severity: float = 0.0,
) -> float:
    """Adaptive blend with backwards-compatible fallback.

    Legacy signature still works (paper.py calls it with the first three args
    only). When called with extra weight kwargs, applies the regime policy.
    Returns a probability-like score in [0, 1].
    """
    p_up = float(probability_up)
    sentiment = float(sentiment_score)

    # Override path: qualitative direction substitutes the technical signal.
    if override_qualitative and event_count > 0:
        magnitude = min(1.0, abs(sentiment) * max(event_severity, 0.5))
        if sentiment >= 0:
            return float(0.5 + 0.45 * magnitude)
        return float(0.5 - 0.45 * magnitude)

    # Adaptive blend.
    if technical_weight is None or qualitative_weight is None:
        if event_count <= 0:
            return float(max(0.0, min(1.0, p_up)))
        technical_weight, qualitative_weight = LEGACY_FALLBACK_WEIGHTS
        sentiment_contrib = 0.5 + sentiment * 0.5
        fused = technical_weight * p_up + qualitative_weight * sentiment_contrib
        return float(max(0.0, min(1.0, fused)))

    sentiment_contrib = 0.5 + sentiment * 0.5
    fused = float(technical_weight) * p_up + float(qualitative_weight) * sentiment_contrib
    return float(max(0.0, min(1.0, fused)))


def direction_from_fused_score(fused_score: float) -> str:
    if fused_score >= 0.55:
        return "up"
    if fused_score <= 0.30:
        return "down"
    return "sideways"


def build_fusion_predictions(
    technical_predictions: pd.DataFrame,
    qualitative_features: pd.DataFrame,
    run_id: str,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if technical_predictions.empty:
        return pd.DataFrame()

    if prices is None:
        prices = read_ohlcv_prices()

    latest_predictions = (
        technical_predictions.sort_values("date").groupby("ticker", as_index=False).tail(1)
    )

    records: list[dict] = []
    for _index, row in latest_predictions.iterrows():
        ticker = str(row["ticker"])
        signal_date = str(row["date"])
        technical_probability_up = float(row["probability_up"])
        technical_confidence = confidence_from_prediction(row)
        context = choose_context_for_signal(qualitative_features, ticker, signal_date)
        sentiment_score = float(context["sentiment_score"])
        event_count = int(context["event_count"])
        event_severity = float(context["event_severity"])
        event_magnitude = float(context["event_magnitude"])
        event_tags = list(context["event_tags"])

        current_vol, vol_pct = _ticker_volatility_context(prices, ticker, signal_date)
        regime = assess_regime(
            technical_probability_up=technical_probability_up,
            technical_confidence=technical_confidence,
            sentiment_score=sentiment_score,
            event_count=event_count,
            event_severity=event_severity,
            event_magnitude=event_magnitude,
            volatility_percentile=vol_pct,
        )

        fused_score = calculate_fused_score(
            technical_probability_up,
            sentiment_score,
            event_count,
            technical_weight=regime.technical_weight,
            qualitative_weight=regime.qualitative_weight,
            override_qualitative=regime.override_qualitative,
            event_severity=event_severity,
        )
        fused_direction = direction_from_fused_score(fused_score)

        explanation = {
            "language_guardrail": (
                "Experimental fused prediction only. Not financial advice and not a real order. "
                "Hypothesis-based regime-adaptive policy."
            ),
            "technical": {
                "probability_up": technical_probability_up,
                "confidence": technical_confidence,
                "model_run_id": run_id,
            },
            "contextual": {
                "sentiment_score": sentiment_score,
                "event_count": event_count,
                "sentiment_label": context["sentiment_label"],
                "event_tags": event_tags,
                "event_severity": event_severity,
                "event_magnitude": event_magnitude,
                "aligned_trading_date": context["aligned_trading_date"],
            },
            "regime": {
                "regime": regime.regime,
                "technical_weight": regime.technical_weight,
                "qualitative_weight": regime.qualitative_weight,
                "override_qualitative": regime.override_qualitative,
                "volatility_21d": current_vol,
                "volatility_percentile": regime.volatility_percentile,
                "divergence": regime.divergence,
                "notes": regime.notes,
                "override_threshold": SENTIMENT_OVERRIDE_THRESHOLD,
            },
            "fusion": {
                "version": FUSION_VERSION,
                "rule": (
                    "regime-adaptive blend; qualitative override when severity*|sentiment| >= threshold"
                ),
                "fused_score": fused_score,
                "fused_direction": fused_direction,
            },
        }
        records.append(
            {
                "fusion_id": build_fusion_id(run_id, ticker, signal_date, HORIZON),
                "run_id": run_id,
                "ticker": ticker,
                "signal_date": signal_date,
                "horizon": HORIZON,
                "fusion_version": FUSION_VERSION,
                "technical_probability_up": technical_probability_up,
                "technical_confidence": technical_confidence,
                "sentiment_score": sentiment_score,
                "qualitative_event_count": event_count,
                "fused_score": fused_score,
                "fused_direction": fused_direction,
                "explanation_json": json.dumps(explanation, ensure_ascii=True),
            }
        )
    return pd.DataFrame(records)


def run_fusion_predictions(run_id: str | None = None) -> dict:
    initialize_database()
    selected_run_id = run_id or get_latest_model_run_id()
    technical_predictions = read_model_predictions(selected_run_id, split="test")
    qualitative_features = get_qualitative_features()
    prices = read_ohlcv_prices()
    predictions = build_fusion_predictions(
        technical_predictions, qualitative_features, selected_run_id, prices=prices
    )
    inserted = save_fusion_predictions(predictions)
    directions = {} if predictions.empty else predictions["fused_direction"].value_counts().to_dict()
    regimes: dict[str, int] = {}
    overrides = 0
    if not predictions.empty:
        for raw in predictions["explanation_json"].tolist():
            try:
                payload = json.loads(raw).get("regime", {})
            except (TypeError, ValueError):
                continue
            regime = payload.get("regime", "unknown")
            regimes[regime] = regimes.get(regime, 0) + 1
            if payload.get("override_qualitative"):
                overrides += 1
    return {
        "run_id": selected_run_id,
        "generated": int(len(predictions)),
        "inserted": int(inserted),
        "directions": directions,
        "regimes": regimes,
        "qualitative_overrides": overrides,
        "fusion_version": FUSION_VERSION,
    }


def get_fusion_summary() -> pd.DataFrame:
    predictions = get_fusion_predictions()
    if predictions.empty:
        return pd.DataFrame()
    return predictions[
        [
            "ticker",
            "signal_date",
            "horizon",
            "technical_probability_up",
            "technical_confidence",
            "sentiment_score",
            "qualitative_event_count",
            "fused_score",
            "fused_direction",
            "fusion_version",
        ]
    ]
