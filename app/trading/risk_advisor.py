from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import pandas as pd

from app.data.database import (
    get_paper_positions,
    get_paper_trading_signals,
    get_risk_alerts,
    initialize_database,
    read_ohlcv_prices,
    save_paper_positions,
    save_risk_alerts,
)


RISK_ADVISOR_VERSION = "v3_conselheiro_alpha_profit_capture"
ENTRY_OPERATIONAL_ACTIONS = {"ENTER_LONG", "LEGACY_SIMULATE_LONG"}

# Map textual horizons used by paper signals to trading-day counts.
HORIZON_TO_DAYS = {
    "7d": 7,
    "21d": 21,
    "3m": 63,
    "63d": 63,
    "1y": 252,
    "252d": 252,
}


def build_position_id(signal_id: str) -> str:
    digest = hashlib.sha256(f"{RISK_ADVISOR_VERSION}|{signal_id}".encode("utf-8")).hexdigest()[:16]
    return f"pos_{digest}"


def build_alert_id(position_id: str, evaluated_at: str, action: str) -> str:
    payload = f"{RISK_ADVISOR_VERSION}|{position_id}|{evaluated_at}|{action}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"alert_{digest}"


def latest_prices_by_ticker(prices: pd.DataFrame | None = None) -> dict[str, float]:
    if prices is None:
        prices = read_ohlcv_prices()
    if prices.empty:
        return {}
    latest = prices.sort_values("date").groupby("ticker", as_index=False).tail(1)
    return {str(row.ticker): float(row.close) for row in latest.itertuples(index=False)}


def latest_signal_snapshot(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "ticker" not in signals.columns:
        return signals

    snapshot = signals.copy()
    if "signal_date" in snapshot.columns:
        snapshot["signal_date"] = pd.to_datetime(snapshot["signal_date"], errors="coerce")
    if "created_at" in snapshot.columns:
        snapshot["created_at"] = pd.to_datetime(snapshot["created_at"], errors="coerce")

    sort_columns = [column for column in ["ticker", "signal_date", "created_at"] if column in snapshot.columns]
    if len(sort_columns) == 1:
        return snapshot.drop_duplicates(subset=["ticker"], keep="last")
    return snapshot.sort_values(sort_columns).groupby("ticker", as_index=False).tail(1)


def build_positions_from_signals(signals: pd.DataFrame | None = None) -> pd.DataFrame:
    if signals is None:
        signals = get_paper_trading_signals()
    if signals.empty:
        return pd.DataFrame()

    signals = latest_signal_snapshot(signals)

    if "operational_action" not in signals.columns:
        signals["operational_action"] = None
    simulated = signals[
        signals["decision"].eq("simulate_long")
        | signals["operational_action"].isin(ENTRY_OPERATIONAL_ACTIONS)
    ].copy()
    if simulated.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for row in simulated.itertuples(index=False):
        quantity = int(row.max_shares)
        if quantity <= 0:
            continue
        entry_price = float(row.suggested_entry)
        metadata = {
            "risk_advisor_version": RISK_ADVISOR_VERSION,
            "source": "paper_trading_signal",
            "operational_action": getattr(row, "operational_action", None),
            "decision": getattr(row, "decision", None),
            "block_reason": getattr(row, "block_reason", None),
        }
        records.append(
            {
                "position_id": build_position_id(str(row.signal_id)),
                "signal_id": row.signal_id,
                "run_id": row.run_id,
                "ticker": row.ticker,
                "opened_at": row.signal_date,
                "horizon": row.horizon,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": float(row.stop_loss),
                "partial_target": float(row.partial_target),
                "target_price": float(row.target_price),
                "current_price": entry_price,
                "status": "open",
                "unrealized_return": 0.0,
                "realized_return": None,
                "last_evaluated_at": None,
                "metadata_json": json.dumps(metadata, ensure_ascii=True),
            }
        )
    return pd.DataFrame(records)


def evaluate_position(position: pd.Series | dict, current_price: float, evaluated_at: str | None = None) -> dict:
    data = dict(position)
    evaluated_at = evaluated_at or datetime.utcnow().isoformat(timespec="seconds")
    entry_price = float(data["entry_price"])
    stop_loss = float(data["stop_loss"])
    partial_target = float(data["partial_target"])
    target_price = float(data["target_price"])
    unrealized_return = current_price / entry_price - 1 if entry_price > 0 else 0.0

    if current_price <= stop_loss:
        action = "close_position"
        severity = "critical"
        reason = "stop_loss_reached"
        status = "closed_stop"
        realized_return = unrealized_return
    elif current_price >= target_price:
        action = "close_position"
        severity = "high"
        reason = "target_price_reached"
        status = "closed_target"
        realized_return = unrealized_return
    elif current_price >= partial_target:
        action = "take_partial_profit"
        severity = "medium"
        reason = "partial_target_reached"
        status = "open"
        realized_return = None
    elif unrealized_return > 0.03 and current_price > entry_price:
        action = "adjust_stop"
        severity = "low"
        reason = "profit_buffer_available"
        status = "open"
        realized_return = None
    else:
        action = "hold_position"
        severity = "info"
        reason = "risk_within_policy"
        status = "open"
        realized_return = None

    metadata = {
        "risk_advisor_version": RISK_ADVISOR_VERSION,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "partial_target": partial_target,
        "target_price": target_price,
    }
    return {
        "position_id": data["position_id"],
        "ticker": data["ticker"],
        "evaluated_at": evaluated_at,
        "action": action,
        "severity": severity,
        "reason": reason,
        "current_price": float(current_price),
        "unrealized_return": float(unrealized_return),
        "status": status,
        "realized_return": realized_return,
        "metadata_json": json.dumps(metadata, ensure_ascii=True),
    }


def audit_paper_portfolio() -> dict:
    initialize_database()
    candidate_positions = build_positions_from_signals()
    opened = save_paper_positions(candidate_positions)

    positions = get_paper_positions()
    open_positions = positions[positions["status"] == "open"].copy() if not positions.empty else pd.DataFrame()
    latest_prices = latest_prices_by_ticker()
    evaluated_at = datetime.utcnow().isoformat(timespec="seconds")
    alerts: list[dict] = []
    updated_positions: list[dict] = []

    for _index, position in open_positions.iterrows():
        ticker = str(position["ticker"])
        current_price = latest_prices.get(ticker, float(position["current_price"]))
        evaluation = evaluate_position(position, current_price=current_price, evaluated_at=evaluated_at)
        alerts.append(
            {
                "alert_id": build_alert_id(str(position["position_id"]), evaluated_at, evaluation["action"]),
                "position_id": position["position_id"],
                "ticker": ticker,
                "evaluated_at": evaluated_at,
                "action": evaluation["action"],
                "severity": evaluation["severity"],
                "reason": evaluation["reason"],
                "current_price": evaluation["current_price"],
                "unrealized_return": evaluation["unrealized_return"],
                "metadata_json": evaluation["metadata_json"],
            }
        )
        updated = position.to_dict()
        updated["current_price"] = evaluation["current_price"]
        updated["status"] = evaluation["status"]
        updated["unrealized_return"] = evaluation["unrealized_return"]
        updated["realized_return"] = evaluation["realized_return"]
        updated["last_evaluated_at"] = evaluated_at
        updated_positions.append(updated)

    updated_frame = pd.DataFrame(updated_positions)
    saved_updates = save_paper_positions(updated_frame) if not updated_frame.empty else 0
    alert_frame = pd.DataFrame(alerts)
    inserted_alerts = save_risk_alerts(alert_frame) if not alert_frame.empty else 0
    refreshed_positions = get_paper_positions()
    open_count = int((refreshed_positions["status"] == "open").sum()) if not refreshed_positions.empty else 0
    return {
        "opened_positions": int(opened),
        "evaluated_positions": int(len(open_positions)),
        "updated_positions": int(saved_updates),
        "alerts": int(inserted_alerts),
        "open_positions": open_count,
        "risk_advisor_version": RISK_ADVISOR_VERSION,
    }


def horizon_in_days(horizon: str | None) -> int:
    if not horizon:
        return 7
    key = str(horizon).strip().lower()
    return HORIZON_TO_DAYS.get(key, 7)


def parse_iso_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def _safe_signal_thesis(signal_row: pd.Series | None) -> dict:
    if signal_row is None:
        return {}
    raw = signal_row.get("thesis_json")
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return {}


def compute_residual_expected_value(
    probability_win: float,
    probability_loss: float,
    probability_timeout: float,
    target_distance: float,
    stop_distance: float,
    execution_drag: float,
    days_remaining: int,
    horizon_days: int,
    realised_return: float,
) -> float:
    """Estimate residual EV of holding the position to horizon end.

    The intuition: as time decays, the probability of timing out at the
    realised return grows; the upside is capped at ``target_distance`` and the
    downside at ``-stop_distance``.  We blend the original win/loss/timeout
    probabilities with a time-decay factor.
    """
    horizon_days = max(int(horizon_days), 1)
    days_remaining = max(int(days_remaining), 0)
    decay = days_remaining / horizon_days  # 1.0 fresh, 0.0 expired
    p_win_remaining = max(probability_win * decay, 0.0)
    p_loss_remaining = max(probability_loss * decay, 0.0)
    p_path = p_win_remaining + p_loss_remaining
    p_timeout_remaining = max(1.0 - p_path, 0.0)
    target_payoff = target_distance - execution_drag
    stop_payoff = -stop_distance - execution_drag
    timeout_payoff = realised_return
    return (
        p_win_remaining * target_payoff
        + p_loss_remaining * stop_payoff
        + p_timeout_remaining * timeout_payoff
    )


def compute_trailing_stop(
    entry_price: float,
    base_stop_loss: float,
    current_price: float,
    high_watermark: float,
    trailing_floor_ratio: float = 0.5,
) -> float:
    """Move the stop up once the position is in profit so trailing protects gains.

    The trailing stop never moves below the original stop loss and tightens to
    ``trailing_floor_ratio`` of the favourable excursion as profit expands.
    """
    if entry_price <= 0:
        return base_stop_loss
    favourable_excursion = max(high_watermark - entry_price, 0.0)
    trailing_candidate = entry_price + favourable_excursion * trailing_floor_ratio
    if current_price <= entry_price:
        return base_stop_loss
    return max(base_stop_loss, trailing_candidate)


def conselheiro_evaluate_position(
    position: pd.Series | dict,
    current_price: float,
    high_watermark: float,
    today: datetime,
    signal_row: pd.Series | None,
    evaluated_at: str,
) -> dict:
    data = dict(position)
    entry_price = float(data["entry_price"])
    base_stop = float(data["stop_loss"])
    partial_target = float(data["partial_target"])
    target_price = float(data["target_price"])
    horizon_days = horizon_in_days(str(data.get("horizon", "7d")))
    opened_at = parse_iso_date(data.get("opened_at"))
    days_elapsed = max((today - opened_at).days, 0)
    days_remaining = max(horizon_days - days_elapsed, 0)
    unrealised_return = current_price / entry_price - 1.0 if entry_price > 0 else 0.0

    thesis = _safe_signal_thesis(signal_row)
    trade_outcome = thesis.get("trade_outcome", {}) if isinstance(thesis, dict) else {}
    probability_win = float(trade_outcome.get("probability_win", 0.0))
    probability_loss = float(trade_outcome.get("probability_loss", 0.0))
    probability_timeout = float(trade_outcome.get("probability_timeout", 1.0))
    stop_distance = float(
        trade_outcome.get(
            "stop_distance",
            max(1.0 - base_stop / entry_price, 1e-6) if entry_price > 0 else 0.03,
        )
    )
    target_distance = float(
        trade_outcome.get(
            "target_distance",
            max(target_price / entry_price - 1.0, 1e-6) if entry_price > 0 else 0.045,
        )
    )
    execution_drag = float(trade_outcome.get("execution_drag", 0.004))

    trailing_stop = compute_trailing_stop(
        entry_price=entry_price,
        base_stop_loss=base_stop,
        current_price=current_price,
        high_watermark=high_watermark,
    )
    residual_ev = compute_residual_expected_value(
        probability_win=probability_win,
        probability_loss=probability_loss,
        probability_timeout=probability_timeout,
        target_distance=target_distance,
        stop_distance=stop_distance,
        execution_drag=execution_drag,
        days_remaining=days_remaining,
        horizon_days=horizon_days,
        realised_return=unrealised_return,
    )

    profit_capture_return = max(0.018, target_distance * 0.35)
    residual_ev_floor = max(execution_drag, 0.003)

    if current_price <= base_stop:
        action = "EXIT_STOP"
        severity = "critical"
        reason = "hard_stop_breached"
        status = "closed_stop"
        realized_return = unrealised_return
    elif current_price >= target_price:
        action = "EXIT_TARGET"
        severity = "high"
        reason = "target_price_reached"
        status = "closed_target"
        realized_return = unrealised_return
    elif current_price <= trailing_stop and current_price > entry_price:
        action = "EXIT_TRAILING_STOP"
        severity = "high"
        reason = "trailing_stop_protected_profit"
        status = "closed_trailing_stop"
        realized_return = unrealised_return
    elif days_remaining <= 0:
        action = "EXIT_TIMEOUT"
        severity = "medium"
        reason = "horizon_elapsed"
        status = "closed_timeout"
        realized_return = unrealised_return
    elif residual_ev <= 0.0 and unrealised_return > 0.0:
        action = "EXIT_EV_NEGATIVE"
        severity = "high"
        reason = "residual_expected_value_non_positive"
        status = "closed_ev_negative"
        realized_return = unrealised_return
    elif unrealised_return >= profit_capture_return and residual_ev <= residual_ev_floor:
        action = "EXIT_PROFIT_CAPTURE"
        severity = "high"
        reason = "profit_available_residual_edge_compressed"
        status = "closed_profit_capture"
        realized_return = unrealised_return
    elif current_price >= partial_target:
        action = "REDUCE_PARTIAL"
        severity = "medium"
        reason = "partial_target_reached"
        status = "open"
        realized_return = None
    elif unrealised_return > 0.03:
        action = "MANAGE_POSITION"
        severity = "low"
        reason = "profit_buffer_available_trail_stop"
        status = "open"
        realized_return = None
    else:
        action = "HOLD_POSITION"
        severity = "info"
        reason = "risk_within_policy"
        status = "open"
        realized_return = None

    metadata = {
        "risk_advisor_version": RISK_ADVISOR_VERSION,
        "entry_price": entry_price,
        "base_stop_loss": base_stop,
        "trailing_stop": trailing_stop,
        "partial_target": partial_target,
        "target_price": target_price,
        "horizon_days": horizon_days,
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
        "residual_expected_value": residual_ev,
        "high_watermark": high_watermark,
        "trade_outcome_probabilities": {
            "win": probability_win,
            "loss": probability_loss,
            "timeout": probability_timeout,
        },
    }
    return {
        "position_id": data["position_id"],
        "ticker": data["ticker"],
        "evaluated_at": evaluated_at,
        "action": action,
        "severity": severity,
        "reason": reason,
        "current_price": float(current_price),
        "unrealized_return": float(unrealised_return),
        "status": status,
        "realized_return": realized_return,
        "metadata_json": json.dumps(metadata, ensure_ascii=True),
    }


def compute_high_watermarks(
    open_positions: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, float]:
    if open_positions.empty or prices.empty:
        return {}
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    watermarks: dict[str, float] = {}
    for _, position in open_positions.iterrows():
        ticker = str(position["ticker"])
        opened_at = str(position["opened_at"])[:10]
        ticker_prices = prices[(prices["ticker"] == ticker) & (prices["date"] >= opened_at)]
        if ticker_prices.empty:
            watermarks[str(position["position_id"])] = float(position["current_price"])
        else:
            watermarks[str(position["position_id"])] = float(ticker_prices["high"].max())
    return watermarks


def audit_paper_portfolio_with_conselheiro() -> dict:
    """Conselheiro-driven audit using EV decay and trailing stops.

    Coexists with :func:`audit_paper_portfolio` for backwards compatibility.
    """
    initialize_database()
    candidate_positions = build_positions_from_signals()
    opened = save_paper_positions(candidate_positions)

    positions = get_paper_positions()
    open_positions = (
        positions[positions["status"] == "open"].copy() if not positions.empty else pd.DataFrame()
    )
    if open_positions.empty:
        return {
            "opened_positions": int(opened),
            "evaluated_positions": 0,
            "updated_positions": 0,
            "alerts": 0,
            "open_positions": 0,
            "risk_advisor_version": RISK_ADVISOR_VERSION,
            "actions": {},
        }

    prices = read_ohlcv_prices()
    latest_prices = latest_prices_by_ticker(prices)
    watermarks = compute_high_watermarks(open_positions, prices)

    signals = get_paper_trading_signals()
    signals_by_id = (
        signals.set_index("signal_id") if not signals.empty else pd.DataFrame()
    )

    evaluated_at = datetime.utcnow().isoformat(timespec="seconds")
    today = datetime.utcnow()
    alerts: list[dict] = []
    updated_positions: list[dict] = []
    actions_breakdown: dict[str, int] = {}

    for _index, position in open_positions.iterrows():
        ticker = str(position["ticker"])
        position_id = str(position["position_id"])
        current_price = latest_prices.get(ticker, float(position["current_price"]))
        high_watermark = max(watermarks.get(position_id, current_price), current_price)
        signal_row = (
            signals_by_id.loc[str(position["signal_id"])]
            if not signals_by_id.empty and str(position["signal_id"]) in signals_by_id.index
            else None
        )
        evaluation = conselheiro_evaluate_position(
            position=position,
            current_price=current_price,
            high_watermark=high_watermark,
            today=today,
            signal_row=signal_row,
            evaluated_at=evaluated_at,
        )
        actions_breakdown[evaluation["action"]] = (
            actions_breakdown.get(evaluation["action"], 0) + 1
        )
        alerts.append(
            {
                "alert_id": build_alert_id(position_id, evaluated_at, evaluation["action"]),
                "position_id": position_id,
                "ticker": ticker,
                "evaluated_at": evaluated_at,
                "action": evaluation["action"],
                "severity": evaluation["severity"],
                "reason": evaluation["reason"],
                "current_price": evaluation["current_price"],
                "unrealized_return": evaluation["unrealized_return"],
                "metadata_json": evaluation["metadata_json"],
            }
        )
        updated = position.to_dict()
        updated["current_price"] = evaluation["current_price"]
        updated["status"] = evaluation["status"]
        updated["unrealized_return"] = evaluation["unrealized_return"]
        updated["realized_return"] = evaluation["realized_return"]
        updated["last_evaluated_at"] = evaluated_at
        updated_positions.append(updated)

    updated_frame = pd.DataFrame(updated_positions)
    saved_updates = save_paper_positions(updated_frame) if not updated_frame.empty else 0
    alert_frame = pd.DataFrame(alerts)
    inserted_alerts = save_risk_alerts(alert_frame) if not alert_frame.empty else 0
    refreshed_positions = get_paper_positions()
    open_count = (
        int((refreshed_positions["status"] == "open").sum()) if not refreshed_positions.empty else 0
    )
    return {
        "opened_positions": int(opened),
        "evaluated_positions": int(len(open_positions)),
        "updated_positions": int(saved_updates),
        "alerts": int(inserted_alerts),
        "open_positions": open_count,
        "risk_advisor_version": RISK_ADVISOR_VERSION,
        "actions": actions_breakdown,
    }


def get_portfolio_summary() -> pd.DataFrame:
    positions = get_paper_positions()
    if positions.empty:
        return pd.DataFrame()
    return positions[
        [
            "position_id",
            "ticker",
            "opened_at",
            "horizon",
            "quantity",
            "entry_price",
            "current_price",
            "stop_loss",
            "partial_target",
            "target_price",
            "status",
            "unrealized_return",
            "realized_return",
            "last_evaluated_at",
        ]
    ]


def get_risk_alert_summary() -> pd.DataFrame:
    alerts = get_risk_alerts()
    if alerts.empty:
        return pd.DataFrame()
    return alerts[
        [
            "alert_id",
            "position_id",
            "ticker",
            "evaluated_at",
            "action",
            "severity",
            "reason",
            "current_price",
            "unrealized_return",
        ]
    ]
