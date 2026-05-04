"""B3 real-world transaction costs.

Replaces the simplistic flat ``cost_per_trade`` parameter used in the early
backtests. Models actual Brazilian equity market frictions so the walk-forward
gate stops being optimistic about strategy edge.

Components per round-trip (entry + exit):

* **Emolumentos B3**: 0.0325% per side for swing trade equities (cash market).
* **Liquidação CBLC**: ~0.0275% per side (inside emolumentos for current B3
  fee schedule, kept here as separate constant for traceability).
* **Corretagem**: many home brokers are R$0 for stocks; we keep a
  configurable per-leg fee defaulting to R$0 for transparency.
* **ISS** (Imposto sobre Serviços) on corretagem: 5% of corretagem (zero when
  corretagem = 0).
* **IR** (Imposto de Renda):
  - Swing trade: 15% on net monthly profit, with monthly exemption of
    R$ 20.000 in vendas (handled at portfolio level — here we apply it per
    trade as a worst-case 15% on positive net result).
  - Day trade: 20% with no exemption + IRRF 1% (not used in this 7-day model).
* **Spread** (bid-ask): half-spread per side in proportion to ADTV.
* **Slippage**: function of trade size vs ADTV — bigger trades pay more.

All values are returned as fractional return drag, so they slot into the
existing ``net_return = gross_return - drag`` math.
"""

from __future__ import annotations

from dataclasses import dataclass


B3_EMOLUMENTOS_PER_SIDE = 0.0000325   # 0.00325%
B3_LIQUIDACAO_PER_SIDE = 0.000275      # 0.0275%  (kept separate from emolumentos)
DEFAULT_CORRETAGEM_PER_LEG_BRL = 0.0
ISS_RATE = 0.05
IR_SWING_RATE = 0.15
IR_DAYTRADE_RATE = 0.20

# Spread baseline for the top-7 B3 names (PETR4/VALE3/ITUB4/etc are all very
# liquid, so half-spread is small in normal conditions).
DEFAULT_HALF_SPREAD_BPS = 2.0  # 0.02% per side

# Slippage scales with order participation in ADTV (Average Daily Traded Value).
# Empirical rule of thumb on B3 large caps: 1% of ADTV ≈ 5 bps slippage.
SLIPPAGE_BPS_PER_PERCENT_OF_ADTV = 5.0


@dataclass(frozen=True)
class CostBreakdown:
    """All costs expressed as a *fraction* of position notional, per round-trip."""
    emolumentos: float
    liquidacao: float
    corretagem: float
    iss: float
    spread: float
    slippage: float
    ir_on_profit_rate: float   # NOT yet applied — caller multiplies by max(0, gross_return)
    total_pre_ir: float        # sum of all execution drag, before IR
    is_daytrade: bool


def compute_cost_breakdown(
    notional_brl: float,
    corretagem_per_leg_brl: float = DEFAULT_CORRETAGEM_PER_LEG_BRL,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    adtv_brl: float | None = None,
    is_daytrade: bool = False,
) -> CostBreakdown:
    """Round-trip costs as fractions of notional.

    ``adtv_brl`` (Average Daily Traded Value in BRL) is optional. If supplied,
    slippage scales with ``notional_brl / adtv_brl``. If not, we assume a
    conservative 0.1% participation (5 bps slippage per side).
    """
    notional_brl = max(float(notional_brl), 1.0)

    emolumentos = 2.0 * B3_EMOLUMENTOS_PER_SIDE
    liquidacao = 2.0 * B3_LIQUIDACAO_PER_SIDE
    corretagem_total = 2.0 * float(corretagem_per_leg_brl)
    corretagem_frac = corretagem_total / notional_brl
    iss_frac = ISS_RATE * corretagem_frac
    spread = 2.0 * (float(half_spread_bps) / 10_000.0)

    if adtv_brl and adtv_brl > 0:
        participation = notional_brl / float(adtv_brl)
    else:
        participation = 0.001  # 0.1% baseline
    slippage_per_side = (participation * 100.0) * (SLIPPAGE_BPS_PER_PERCENT_OF_ADTV / 10_000.0)
    slippage = 2.0 * slippage_per_side

    ir_rate = IR_DAYTRADE_RATE if is_daytrade else IR_SWING_RATE

    total_pre_ir = emolumentos + liquidacao + corretagem_frac + iss_frac + spread + slippage

    return CostBreakdown(
        emolumentos=emolumentos,
        liquidacao=liquidacao,
        corretagem=corretagem_frac,
        iss=iss_frac,
        spread=spread,
        slippage=slippage,
        ir_on_profit_rate=ir_rate,
        total_pre_ir=total_pre_ir,
        is_daytrade=is_daytrade,
    )


def apply_costs_to_gross_return(
    gross_return: float,
    breakdown: CostBreakdown,
) -> float:
    """Net return after execution drag and IR on positive result."""
    after_execution = gross_return - breakdown.total_pre_ir
    if after_execution > 0:
        after_execution *= (1.0 - breakdown.ir_on_profit_rate)
    return float(after_execution)


def estimate_round_trip_drag(
    notional_brl: float,
    adtv_brl: float | None = None,
    corretagem_per_leg_brl: float = DEFAULT_CORRETAGEM_PER_LEG_BRL,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    is_daytrade: bool = False,
) -> float:
    """Convenience: just the pre-IR fractional drag (matches old `cost_per_trade`)."""
    return compute_cost_breakdown(
        notional_brl=notional_brl,
        corretagem_per_leg_brl=corretagem_per_leg_brl,
        half_spread_bps=half_spread_bps,
        adtv_brl=adtv_brl,
        is_daytrade=is_daytrade,
    ).total_pre_ir
