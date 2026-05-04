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


# Sector map for the 7 monitored B3 names — used for sector-level exposure caps.
TICKER_SECTOR: dict[str, str] = {
    "PETR4.SA": "energy_oil",
    "VALE3.SA": "materials_mining",
    "ITUB4.SA": "financials_banks",
    "BBDC4.SA": "financials_banks",
    "BBAS3.SA": "financials_banks",
    "ABEV3.SA": "consumer_staples",
    "WEGE3.SA": "industrials",
    "AAPL": "technology_hardware",
    "MSFT": "technology_software",
    "GOOGL": "communication_services",
    "AMZN": "consumer_discretionary",
    "NVDA": "semiconductors",
    "META": "communication_services",
    "TSLA": "consumer_discretionary_ev",
}

DEFAULT_KELLY_FRACTION = 0.25            # ¼ Kelly
DEFAULT_MAX_RISK_PER_TRADE = 0.01        # 1% equity at risk per trade
DEFAULT_MAX_EXPOSURE_PER_TICKER = 0.25   # 25% of equity in a single ticker
DEFAULT_MAX_EXPOSURE_PER_SECTOR = 0.40   # 40% of equity in a single sector
DEFAULT_MIN_KELLY_EDGE = 0.005           # below 0.5% Kelly edge → no trade
DEFAULT_ATR_STOP_MULTIPLIER = 2.0        # stop = entry - 2 * ATR


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


def kelly_fraction(probability_win: float, win_loss_ratio: float) -> float:
    """Standard Kelly: f* = p - (1-p)/b, where b = win/loss ratio."""
    if win_loss_ratio <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(probability_win)))
    return float(p - (1.0 - p) / float(win_loss_ratio))


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
) -> SizingDecision:
    capped_by: list[str] = []
    block_reason: str | None = None

    if entry_price <= 0 or portfolio_value <= 0:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, 0.0, [], "invalid_inputs")

    stop_price = atr_stop_price(entry_price, atr, atr_multiplier)
    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0:
        return SizingDecision(0, 0.0, 0.0, stop_price, 0.0, 0.0, [], "stop_price_invalid")

    # Kelly edge — uses payoff/loss ratio derived from the trade-outcome model.
    win_loss_ratio = float(expected_payoff) / max(float(expected_loss), 1e-9)
    raw_kelly = kelly_fraction(probability_win, win_loss_ratio)
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

    if notional <= 0:
        return SizingDecision(
            0, 0.0, 0.0, stop_price,
            per_share_risk / entry_price, fractional_kelly,
            capped_by, "exposure_caps_exhausted",
        )

    shares = int(math.floor(notional / entry_price))
    if shares <= 0:
        return SizingDecision(
            0, 0.0, 0.0, stop_price,
            per_share_risk / entry_price, fractional_kelly,
            capped_by, "rounded_to_zero_shares",
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
    )


def sector_for(ticker: str) -> str:
    return TICKER_SECTOR.get(ticker, "unknown")
