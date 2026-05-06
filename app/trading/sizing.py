"""Position sizing: fractional Kelly + ATR stop + exposure caps.

Replaces the prior "1% risk per trade with arbitrary stop_distance" approach
with a more rigorous policy:

* Stop distance is anchored to **ATR(14)** (Average True Range) so the stop
  reflects the asset's actual recent noise, not a hard-coded 3%.
* Position size is computed via **¼ Kelly** (fractional Kelly to control
  bankroll volatility), bounded by:
    - max risk per trade as fraction of equity (default 1%);
    - max exposure per ticker (default 25% of equity);
    - max exposure per sector (default 40% of equity).
* If Kelly is negative or below a floor, sizing returns zero shares — a
  natural NÃO OPERAR signal.

All B3 top-7 sector mappings are centralised here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Sector map covering the current trading universe (config.INITIAL_ASSETS).
# Used for sector-level exposure caps in size_position(). New tickers default
# to "uncategorized" (treated as its own sector bucket) — extend this map
# instead of letting them land in the same default bucket.
TICKER_SECTOR: dict[str, str] = {
    # --- Brazil ---
    "PETR4.SA": "energy_oil",
    "VALE3.SA": "materials_mining",
    "ITUB4.SA": "financials_banks",
    "BBDC4.SA": "financials_banks",
    "BBAS3.SA": "financials_banks",
    "SANB11.SA": "financials_banks",
    "B3SA3.SA": "financials_exchange",
    "BPAC11.SA": "financials_banks",
    "ABEV3.SA": "consumer_staples",
    "MGLU3.SA": "consumer_discretionary",
    "LREN3.SA": "consumer_discretionary",
    "RENT3.SA": "consumer_discretionary",
    "RAIL3.SA": "industrials_transport",
    "WEGE3.SA": "industrials",
    "EQTL3.SA": "utilities_electric",
    "SBSP3.SA": "utilities_water",
    "RDOR3.SA": "healthcare_services",
    "HAPV3.SA": "healthcare_services",
    "SUZB3.SA": "materials_paper",
    "KLBN11.SA": "materials_paper",
    "CSNA3.SA": "materials_steel",
    "GGBR4.SA": "materials_steel",
    "USIM5.SA": "materials_steel",
    # --- US ---
    "AAPL": "technology_hardware",
    "MSFT": "technology_software",
    "GOOGL": "communication_services",
    "AMZN": "consumer_discretionary",
    "NVDA": "semiconductors",
    "META": "communication_services",
    "TSLA": "consumer_discretionary_ev",
    "AMD": "semiconductors",
    "AVGO": "semiconductors",
    "ORCL": "technology_software",
    "CRM": "technology_software",
    "JPM": "financials_banks",
    "BAC": "financials_banks",
    "GS": "financials_banks",
    "WMT": "consumer_staples",
    "COST": "consumer_staples",
    "HD": "consumer_discretionary",
    "XOM": "energy_oil",
    "CVX": "energy_oil",
    "JNJ": "healthcare_pharma",
    "UNH": "healthcare_services",
}

DEFAULT_KELLY_FRACTION = 0.25            # ¼ Kelly
DEFAULT_MAX_RISK_PER_TRADE = 0.01        # 1% equity at risk per trade
DEFAULT_MAX_EXPOSURE_PER_TICKER = 0.25   # 25% of equity in a single ticker
DEFAULT_MAX_EXPOSURE_PER_SECTOR = 0.40   # 40% of equity in a single sector
DEFAULT_MIN_KELLY_EDGE = 0.005           # below 0.5% Kelly edge → no trade
DEFAULT_ATR_STOP_MULTIPLIER = 2.0        # stop = entry - 2 * ATR
DEFAULT_MAX_AVG_BOOK_CORRELATION = 0.60  # avg corr with existing book above this triggers penalty
DEFAULT_CORRELATION_PENALTY_FLOOR = 0.25  # never reduce notional below 25% of pre-penalty value


@dataclass(frozen=True)
class SizingDecision:
    shares: int
    notional_brl: float
    risk_brl: float
    stop_price: float
    stop_distance_pct: float
    kelly_fraction_used: float
    capped_by: list[str]
    block_reason: str | None
    correlation_penalty: float = 0.0
    avg_book_correlation: float = 0.0


def correlation_scale(
    avg_book_correlation: float,
    *,
    threshold: float = DEFAULT_MAX_AVG_BOOK_CORRELATION,
    floor: float = DEFAULT_CORRELATION_PENALTY_FLOOR,
) -> float:
    """Linear penalty: notional × max(floor, 1 - (corr-threshold)/(1-threshold)).

    avg_book_correlation ∈ [-1, 1]. Below threshold → no penalty (returns 1.0).
    At corr=1.0 → notional scaled to ``floor`` (default 25%).
    Strongly negatively-correlated picks get a small uplift up to 1.10 to reward
    diversification (capped to keep behaviour predictable).
    """
    corr = float(max(-1.0, min(1.0, avg_book_correlation)))
    if corr <= threshold:
        if corr <= -0.3:
            return 1.10
        return 1.0
    span = max(1.0 - threshold, 1e-9)
    raw = 1.0 - (corr - threshold) / span
    return float(max(floor, raw))


def clamp_probability(probability: float) -> float:
    return float(max(0.0, min(1.0, float(probability))))


def decisive_win_probability(probability_win: float, probability_loss: float | None = None) -> float:
    p_win = clamp_probability(probability_win)
    if probability_loss is None:
        return p_win

    p_loss = clamp_probability(probability_loss)
    decisive_mass = p_win + p_loss
    if decisive_mass <= 1e-9:
        return p_win
    return float(p_win / decisive_mass)


def kelly_fraction(probability_win: float, win_loss_ratio: float) -> float:
    """Standard Kelly: f* = p - (1-p)/b, where b = win/loss ratio."""
    if win_loss_ratio <= 0:
        return 0.0
    p = clamp_probability(probability_win)
    return float(p - (1.0 - p) / float(win_loss_ratio))


def timeout_adjusted_kelly_fraction(
    probability_win: float,
    probability_loss: float | None,
    win_loss_ratio: float,
) -> float:
    if probability_loss is None:
        return kelly_fraction(probability_win, win_loss_ratio)

    p_win = clamp_probability(probability_win)
    p_loss = clamp_probability(probability_loss)
    decisive_mass = p_win + p_loss
    if decisive_mass <= 1e-9:
        return 0.0

    # Timeout is a neutral outcome in the trade-outcome model, so size on the
    # decisive win/loss odds and discount by the decisive mass.
    return float(kelly_fraction(decisive_win_probability(p_win, p_loss), win_loss_ratio) * decisive_mass)


def atr_stop_price(entry_price: float, atr: float, multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER) -> float:
    """Symmetric long stop: entry - multiplier * ATR. Floors at 50% of entry to avoid absurd stops."""
    raw_stop = entry_price - multiplier * max(atr, 0.0)
    return float(max(raw_stop, entry_price * 0.5))


def size_position(
    *,
    portfolio_value: float,
    entry_price: float,
    atr: float,
    probability_win: float,
    probability_loss: float | None = None,
    expected_payoff: float,        # expected upside if win (e.g. target_distance)
    expected_loss: float,          # expected downside if loss (positive number, e.g. stop_distance)
    ticker: str,
    current_ticker_exposure_brl: float = 0.0,
    current_sector_exposure_brl: float = 0.0,
    kelly_fraction_cap: float = DEFAULT_KELLY_FRACTION,
    max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
    max_exposure_per_ticker: float = DEFAULT_MAX_EXPOSURE_PER_TICKER,
    max_exposure_per_sector: float = DEFAULT_MAX_EXPOSURE_PER_SECTOR,
    min_kelly_edge: float = DEFAULT_MIN_KELLY_EDGE,
    atr_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
    avg_book_correlation: float = 0.0,
    correlation_threshold: float = DEFAULT_MAX_AVG_BOOK_CORRELATION,
    correlation_floor: float = DEFAULT_CORRELATION_PENALTY_FLOOR,
) -> SizingDecision:
    capped_by: list[str] = []
    block_reason: str | None = None

    if entry_price <= 0 or portfolio_value <= 0:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, 0.0, [], "invalid_inputs")

    stop_price = atr_stop_price(entry_price, atr, atr_multiplier)
    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0:
        return SizingDecision(0, 0.0, 0.0, stop_price, 0.0, 0.0, [], "stop_price_invalid")

    # Kelly edge — when the model has win/loss/timeout outcomes, treat timeout
    # as neutral rather than as an implicit loss.
    win_loss_ratio = float(expected_payoff) / max(float(expected_loss), 1e-9)
    raw_kelly = timeout_adjusted_kelly_fraction(probability_win, probability_loss, win_loss_ratio)
    if raw_kelly < min_kelly_edge:
        return SizingDecision(
            0, 0.0, 0.0, stop_price,
            per_share_risk / entry_price,
            raw_kelly, [], "kelly_edge_below_minimum",
        )
    fractional_kelly = min(raw_kelly, 1.0) * kelly_fraction_cap

    kelly_notional = portfolio_value * fractional_kelly
    risk_budget = portfolio_value * max_risk_per_trade
    risk_capped_notional = (risk_budget / per_share_risk) * entry_price

    notional = min(kelly_notional, risk_capped_notional)
    if notional == risk_capped_notional and risk_capped_notional < kelly_notional:
        capped_by.append("max_risk_per_trade")
    elif notional == kelly_notional:
        capped_by.append("fractional_kelly")

    # Per-ticker cap.
    per_ticker_cap = portfolio_value * max_exposure_per_ticker
    ticker_room = max(per_ticker_cap - current_ticker_exposure_brl, 0.0)
    if notional > ticker_room:
        notional = ticker_room
        capped_by.append("max_exposure_per_ticker")

    # Per-sector cap.
    per_sector_cap = portfolio_value * max_exposure_per_sector
    sector_room = max(per_sector_cap - current_sector_exposure_brl, 0.0)
    if notional > sector_room:
        notional = sector_room
        capped_by.append("max_exposure_per_sector")

    # Correlation-aware cap: penalise picks that are highly correlated with the
    # existing book (or boost slightly when negatively correlated).
    correlation_penalty_factor = correlation_scale(
        avg_book_correlation,
        threshold=correlation_threshold,
        floor=correlation_floor,
    )
    if correlation_penalty_factor < 1.0:
        notional = notional * correlation_penalty_factor
        capped_by.append("avg_book_correlation_above_threshold")
    elif correlation_penalty_factor > 1.0:
        # uplift is informational, do not exceed per-ticker cap that was just applied
        notional = min(notional * correlation_penalty_factor, ticker_room, sector_room)

    if notional <= 0:
        return SizingDecision(
            0, 0.0, 0.0, stop_price,
            per_share_risk / entry_price, fractional_kelly,
            capped_by, "exposure_caps_exhausted",
            correlation_penalty=1.0 - correlation_penalty_factor,
            avg_book_correlation=float(avg_book_correlation),
        )

    shares = int(math.floor(notional / entry_price))
    if shares <= 0:
        return SizingDecision(
            0, 0.0, 0.0, stop_price,
            per_share_risk / entry_price, fractional_kelly,
            capped_by, "rounded_to_zero_shares",
            correlation_penalty=1.0 - correlation_penalty_factor,
            avg_book_correlation=float(avg_book_correlation),
        )

    final_notional = shares * entry_price
    final_risk = shares * per_share_risk

    return SizingDecision(
        shares=shares,
        notional_brl=float(final_notional),
        risk_brl=float(final_risk),
        stop_price=float(stop_price),
        stop_distance_pct=float(per_share_risk / entry_price),
        kelly_fraction_used=float(fractional_kelly),
        capped_by=capped_by,
        block_reason=block_reason,
        correlation_penalty=float(1.0 - correlation_penalty_factor),
        avg_book_correlation=float(avg_book_correlation),
    )


def sector_for(ticker: str) -> str:
    return TICKER_SECTOR.get(ticker, "unknown")
