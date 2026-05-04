from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pandas as pd

from app.data.database import (
    get_latest_model_run_id,
    read_technical_features,
    read_model_predictions,
    save_backtest_run,
)
from app.trading.costs import apply_costs_to_gross_return, compute_cost_breakdown


DEFAULT_BACKTEST_NOTIONAL_BRL = 10000.0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    equity_curve = pd.concat([pd.Series([1.0]), equity_curve], ignore_index=True)
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "cumulative_return": 0.0,
            "average_trade_return": 0.0,
            "max_drawdown": 0.0,
        }

    equity_curve = (1 + trades["net_return"]).cumprod()
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades["net_return"] > 0).mean()),
        "cumulative_return": float(equity_curve.iloc[-1] - 1),
        "average_trade_return": float(trades["net_return"].mean()),
        "max_drawdown": calculate_max_drawdown(equity_curve),
    }


def select_non_overlapping_trades(
    predictions: pd.DataFrame,
    threshold: float,
    holding_days: int,
    cost_per_trade: float,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
) -> pd.DataFrame:
    selected_trades: list[dict] = []

    for ticker, ticker_predictions in predictions.groupby("ticker", sort=False):
        ordered = ticker_predictions.sort_values("date").reset_index(drop=True)
        index = 0
        while index < len(ordered):
            row = ordered.iloc[index]
            if float(row["probability_up"]) >= threshold:
                exit_index = min(index + holding_days, len(ordered) - 1)
                exit_row = ordered.iloc[exit_index]
                gross_return = float(row["target_return"])
                if use_b3_costs:
                    cost_breakdown = compute_cost_breakdown(notional_brl=notional_brl)
                    net_return = apply_costs_to_gross_return(gross_return, cost_breakdown)
                    execution_drag = cost_breakdown.total_pre_ir
                    cost_model = "b3_realistic_round_trip_v1"
                else:
                    net_return = gross_return - cost_per_trade
                    execution_drag = cost_per_trade
                    cost_model = "fixed_round_trip_cost"
                selected_trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": row["date"],
                        "exit_date": exit_row["date"],
                        "probability_up": float(row["probability_up"]),
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "execution_drag": execution_drag,
                        "cost_model": cost_model,
                    }
                )
                index += holding_days
            else:
                index += 1

    return pd.DataFrame(selected_trades)


def calculate_buy_hold_average(split: str = "test") -> float:
    features = read_technical_features()
    features = features[features["time_split"] == split].copy()
    ticker_returns: list[float] = []
    for _ticker, ticker_features in features.groupby("ticker", sort=False):
        ordered = ticker_features.sort_values("date")
        if ordered.empty:
            continue
        first_close = float(ordered.iloc[0]["close"])
        last_close = float(ordered.iloc[-1]["close"])
        ticker_returns.append(last_close / first_close - 1)
    if not ticker_returns:
        return 0.0
    return float(sum(ticker_returns) / len(ticker_returns))


def build_threshold_grid(
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> list[float]:
    thresholds: list[float] = []
    current = Decimal(str(min_threshold))
    maximum = Decimal(str(max_threshold))
    increment = Decimal(str(step))
    while current <= maximum:
        thresholds.append(float(current))
        current += increment
    return thresholds


def evaluate_thresholds(
    predictions: pd.DataFrame,
    thresholds: list[float],
    holding_days: int,
    cost_per_trade: float,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
) -> pd.DataFrame:
    records: list[dict] = []
    for threshold in thresholds:
        trades = select_non_overlapping_trades(
            predictions=predictions,
            threshold=threshold,
            holding_days=holding_days,
            cost_per_trade=cost_per_trade,
            use_b3_costs=use_b3_costs,
            notional_brl=notional_brl,
        )
        metrics = summarize_trades(trades)
        metrics["threshold"] = threshold
        records.append(metrics)
    return pd.DataFrame(records)


def select_threshold_from_predictions(
    predictions: pd.DataFrame,
    holding_days: int = 7,
    cost_per_trade: float = 0.002,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
    min_threshold: float = 0.35,
    max_threshold: float = 0.85,
    step: float = 0.025,
    min_trades: int = 10,
    max_drawdown: float = 0.20,
    require_positive_return: bool = True,
) -> dict:
    if predictions.empty:
        return {
            "selected_threshold": 1.01,
            "strategy_gate_passed": False,
            "strategy_gate_reason": "empty_calibration_window",
            "threshold_grid": [],
            "threshold_selection": None,
        }

    thresholds = build_threshold_grid(min_threshold, max_threshold, step)
    grid = evaluate_thresholds(
        predictions=predictions,
        thresholds=thresholds,
        holding_days=holding_days,
        cost_per_trade=cost_per_trade,
        use_b3_costs=use_b3_costs,
        notional_brl=notional_brl,
    )
    candidates = grid[grid["trades"] >= min_trades].copy()
    if require_positive_return:
        candidates = candidates[candidates["cumulative_return"] > 0].copy()
    candidates = candidates[candidates["average_trade_return"] > 0].copy()
    candidates = candidates[candidates["max_drawdown"] >= -abs(max_drawdown)].copy()

    if candidates.empty:
        return {
            "selected_threshold": 1.01,
            "strategy_gate_passed": False,
            "strategy_gate_reason": "no_threshold_with_positive_edge_and_acceptable_drawdown",
            "threshold_grid": grid.to_dict(orient="records"),
            "threshold_selection": None,
        }

    selected = candidates.sort_values(
        ["average_trade_return", "cumulative_return", "max_drawdown"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "selected_threshold": float(selected["threshold"]),
        "strategy_gate_passed": True,
        "strategy_gate_reason": "threshold_has_positive_edge",
        "threshold_grid": grid.to_dict(orient="records"),
        "threshold_selection": selected.to_dict(),
    }


def select_threshold_from_validation(
    run_id: str,
    holding_days: int = 7,
    cost_per_trade: float = 0.002,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
    min_threshold: float = 0.35,
    max_threshold: float = 0.85,
    step: float = 0.025,
    min_trades: int = 10,
    max_validation_drawdown: float = 0.20,
    require_positive_return: bool = True,
) -> dict:
    validation_predictions = read_model_predictions(run_id, split="validation")
    if validation_predictions.empty:
        raise ValueError("No validation predictions found for threshold selection.")

    selection = select_threshold_from_predictions(
        predictions=validation_predictions,
        holding_days=holding_days,
        cost_per_trade=cost_per_trade,
        use_b3_costs=use_b3_costs,
        notional_brl=notional_brl,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
        min_trades=min_trades,
        max_drawdown=max_validation_drawdown,
        require_positive_return=require_positive_return,
    )
    return {
        "selected_threshold": selection["selected_threshold"],
        "strategy_gate_passed": selection["strategy_gate_passed"],
        "strategy_gate_reason": selection["strategy_gate_reason"].replace("threshold", "validation_threshold"),
        "validation_grid": selection["threshold_grid"],
        "validation_selection": selection["threshold_selection"],
    }


def calculate_buy_hold_by_ticker(split: str = "test") -> dict:
    features = read_technical_features()
    features = features[features["time_split"] == split].copy()
    returns: dict[str, float] = {}
    for ticker, ticker_features in features.groupby("ticker", sort=False):
        ordered = ticker_features.sort_values("date")
        if ordered.empty:
            continue
        first_close = float(ordered.iloc[0]["close"])
        last_close = float(ordered.iloc[-1]["close"])
        returns[str(ticker)] = float(last_close / first_close - 1)
    return returns


def summarize_trades_by_ticker(trades: pd.DataFrame, split: str = "test") -> list[dict]:
    buy_hold_by_ticker = calculate_buy_hold_by_ticker(split=split)
    records: list[dict] = []
    for ticker in sorted(buy_hold_by_ticker):
        ticker_trades = trades[trades["ticker"] == ticker] if not trades.empty else pd.DataFrame()
        metrics = summarize_trades(ticker_trades)
        records.append(
            {
                "ticker": ticker,
                "trades": metrics["trades"],
                "win_rate": metrics["win_rate"],
                "cumulative_return": metrics["cumulative_return"],
                "average_trade_return": metrics["average_trade_return"],
                "max_drawdown": metrics["max_drawdown"],
                "buy_hold_return": buy_hold_by_ticker[ticker],
                "beats_buy_hold": metrics["cumulative_return"] > buy_hold_by_ticker[ticker],
            }
        )
    return records


def build_walk_forward_windows(predictions: pd.DataFrame, window_size: int) -> list[tuple[str, str]]:
    dates = sorted(predictions["date"].drop_duplicates().tolist())
    windows: list[tuple[str, str]] = []
    for start_index in range(0, len(dates), window_size):
        window_dates = dates[start_index : start_index + window_size]
        if not window_dates:
            continue
        windows.append((str(window_dates[0]), str(window_dates[-1])))
    return windows


def trim_calibration_lookback(predictions: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if predictions.empty or lookback_days <= 0:
        return predictions
    dates = sorted(predictions["date"].drop_duplicates().tolist())
    selected_dates = set(dates[-lookback_days:])
    return predictions[predictions["date"].isin(selected_dates)].copy()


def run_probability_backtest(
    run_id: str | None = None,
    threshold: float = 0.45,
    holding_days: int = 7,
    cost_per_trade: float = 0.002,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
    split: str = "test",
    metadata_extra: dict | None = None,
) -> dict:
    selected_run_id = run_id or get_latest_model_run_id()
    predictions = read_model_predictions(selected_run_id, split=split)
    if predictions.empty:
        raise ValueError(f"No {split} predictions found for backtest.")

    trades = select_non_overlapping_trades(
        predictions=predictions,
        threshold=threshold,
        holding_days=holding_days,
        cost_per_trade=cost_per_trade,
        use_b3_costs=use_b3_costs,
        notional_brl=notional_brl,
    )

    metrics = summarize_trades(trades)

    buy_hold_return_avg = calculate_buy_hold_average(split=split)
    backtest_id = f"bt_7d_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    if not trades.empty:
        trades = trades.copy()
        trades["backtest_id"] = backtest_id

    metadata = {
        "strategy": "probability_up_threshold_non_overlapping",
        "split": split,
        "notes": "Long-only paper backtest. No real orders. Net return uses realistic B3 costs when enabled.",
        "cost_model": "b3_realistic_round_trip_v1" if use_b3_costs else "fixed_round_trip_cost",
        "use_b3_costs": bool(use_b3_costs),
        "backtest_notional_brl": float(notional_brl),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    backtest = {
        "backtest_id": backtest_id,
        "run_id": selected_run_id,
        "threshold": threshold,
        "holding_days": holding_days,
        "cost_per_trade": cost_per_trade,
        "trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "cumulative_return": metrics["cumulative_return"],
        "average_trade_return": metrics["average_trade_return"],
        "max_drawdown": metrics["max_drawdown"],
        "buy_hold_return_avg": buy_hold_return_avg,
        "metadata_json": json.dumps(metadata),
    }
    save_backtest_run(backtest, trades)
    return backtest


def run_validation_selected_backtest(
    run_id: str | None = None,
    holding_days: int = 7,
    cost_per_trade: float = 0.002,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
    min_threshold: float = 0.35,
    max_threshold: float = 0.85,
    step: float = 0.025,
    min_trades: int = 10,
    max_validation_drawdown: float = 0.20,
) -> dict:
    selected_run_id = run_id or get_latest_model_run_id()
    selection = select_threshold_from_validation(
        run_id=selected_run_id,
        holding_days=holding_days,
        cost_per_trade=cost_per_trade,
        use_b3_costs=use_b3_costs,
        notional_brl=notional_brl,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
        min_trades=min_trades,
        max_validation_drawdown=max_validation_drawdown,
    )
    metadata_extra = {
        "threshold_source": "validation_grid_search",
        "strategy_gate_passed_on_validation": selection["strategy_gate_passed"],
        "strategy_gate_reason": selection["strategy_gate_reason"],
        "validation_selection": selection["validation_selection"],
        "validation_grid": selection["validation_grid"],
        "max_validation_drawdown": max_validation_drawdown,
    }
    backtest = run_probability_backtest(
        run_id=selected_run_id,
        threshold=selection["selected_threshold"],
        holding_days=holding_days,
        cost_per_trade=cost_per_trade,
        use_b3_costs=use_b3_costs,
        notional_brl=notional_brl,
        split="test",
        metadata_extra=metadata_extra,
    )
    backtest["strategy_gate_passed_on_validation"] = selection["strategy_gate_passed"]
    backtest["strategy_gate_reason"] = selection["strategy_gate_reason"]
    backtest["validation_selection"] = selection["validation_selection"]
    return backtest


def run_walk_forward_backtest(
    run_id: str | None = None,
    holding_days: int = 7,
    cost_per_trade: float = 0.002,
    use_b3_costs: bool = True,
    notional_brl: float = DEFAULT_BACKTEST_NOTIONAL_BRL,
    window_size: int = 63,
    calibration_lookback_days: int = 504,
    min_threshold: float = 0.35,
    max_threshold: float = 0.85,
    step: float = 0.025,
    min_calibration_trades: int = 10,
    max_calibration_drawdown: float = 0.20,
    max_test_drawdown: float = 0.25,
    min_passing_window_ratio: float = 0.50,
    min_profitable_tickers: int = 2,
) -> dict:
    selected_run_id = run_id or get_latest_model_run_id()
    prediction_frames = [
        read_model_predictions(selected_run_id, split="train"),
        read_model_predictions(selected_run_id, split="validation"),
        read_model_predictions(selected_run_id, split="test"),
    ]
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    test_predictions = all_predictions[all_predictions["time_split"] == "test"].copy()
    if test_predictions.empty:
        raise ValueError("No test predictions found for walk-forward validation.")

    all_predictions = all_predictions.sort_values(["date", "ticker"]).reset_index(drop=True)
    windows = build_walk_forward_windows(test_predictions, window_size=window_size)
    walk_records: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    for window_index, (window_start, window_end) in enumerate(windows, start=1):
        calibration_predictions = all_predictions[all_predictions["date"] < window_start].copy()
        calibration_predictions = trim_calibration_lookback(
            calibration_predictions,
            lookback_days=calibration_lookback_days,
        )
        evaluation_predictions = test_predictions[
            (test_predictions["date"] >= window_start) & (test_predictions["date"] <= window_end)
        ].copy()
        selection = select_threshold_from_predictions(
            predictions=calibration_predictions,
            holding_days=holding_days,
            cost_per_trade=cost_per_trade,
            use_b3_costs=use_b3_costs,
            notional_brl=notional_brl,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            step=step,
            min_trades=min_calibration_trades,
            max_drawdown=max_calibration_drawdown,
        )
        threshold = selection["selected_threshold"]
        trades = select_non_overlapping_trades(
            predictions=evaluation_predictions,
            threshold=threshold,
            holding_days=holding_days,
            cost_per_trade=cost_per_trade,
            use_b3_costs=use_b3_costs,
            notional_brl=notional_brl,
        )
        if not trades.empty:
            trades = trades.copy()
            trades["walk_forward_window"] = window_index
            trade_frames.append(trades)
        metrics = summarize_trades(trades)
        walk_records.append(
            {
                "window": window_index,
                "start_date": window_start,
                "end_date": window_end,
                "threshold": threshold,
                "calibration_gate_passed": selection["strategy_gate_passed"],
                "calibration_gate_reason": selection["strategy_gate_reason"],
                "trades": metrics["trades"],
                "win_rate": metrics["win_rate"],
                "cumulative_return": metrics["cumulative_return"],
                "average_trade_return": metrics["average_trade_return"],
                "max_drawdown": metrics["max_drawdown"],
                "window_passed": metrics["cumulative_return"] > 0 and metrics["average_trade_return"] > 0,
                "threshold_selection": selection["threshold_selection"],
            }
        )

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    metrics = summarize_trades(trades)
    per_ticker = summarize_trades_by_ticker(trades, split="test")
    buy_hold_return_avg = calculate_buy_hold_average(split="test")
    passing_windows = sum(1 for record in walk_records if record["window_passed"])
    passing_window_ratio = passing_windows / len(walk_records) if walk_records else 0.0
    traded_tickers = sum(1 for record in per_ticker if record["trades"] > 0)
    profitable_tickers = sum(1 for record in per_ticker if record["cumulative_return"] > 0 and record["trades"] > 0)
    average_threshold = (
        sum(record["threshold"] for record in walk_records) / len(walk_records)
        if walk_records
        else 1.01
    )
    gate_checks = {
        "positive_return": metrics["cumulative_return"] > 0,
        "beats_buy_hold_average": metrics["cumulative_return"] > buy_hold_return_avg,
        "drawdown_acceptable": metrics["max_drawdown"] >= -abs(max_test_drawdown),
        "passing_window_ratio_acceptable": passing_window_ratio >= min_passing_window_ratio,
        "profitable_tickers_acceptable": profitable_tickers >= min_profitable_tickers,
    }
    strategy_gate_passed = all(gate_checks.values())
    failed_checks = [name for name, passed in gate_checks.items() if not passed]
    strategy_gate_reason = "walk_forward_gate_passed" if strategy_gate_passed else ",".join(failed_checks)

    backtest_id = f"wf_7d_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    if not trades.empty:
        trades = trades.copy()
        trades["backtest_id"] = backtest_id
    metadata = {
        "strategy": "walk_forward_threshold_policy",
        "split": "test",
        "notes": "Policy-level walk-forward. Threshold is recalibrated using only predictions before each evaluation window.",
        "cost_model": "b3_realistic_round_trip_v1" if use_b3_costs else "fixed_round_trip_cost",
        "use_b3_costs": bool(use_b3_costs),
        "backtest_notional_brl": float(notional_brl),
        "window_size": window_size,
        "calibration_lookback_days": calibration_lookback_days,
        "holding_days": holding_days,
        "min_calibration_trades": min_calibration_trades,
        "max_calibration_drawdown": max_calibration_drawdown,
        "max_test_drawdown": max_test_drawdown,
        "min_passing_window_ratio": min_passing_window_ratio,
        "min_profitable_tickers": min_profitable_tickers,
        "walk_forward_windows": walk_records,
        "per_ticker": per_ticker,
        "gate_checks": gate_checks,
        "strategy_gate_passed": strategy_gate_passed,
        "strategy_gate_reason": strategy_gate_reason,
    }
    backtest = {
        "backtest_id": backtest_id,
        "run_id": selected_run_id,
        "threshold": average_threshold,
        "holding_days": holding_days,
        "cost_per_trade": cost_per_trade,
        "trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "cumulative_return": metrics["cumulative_return"],
        "average_trade_return": metrics["average_trade_return"],
        "max_drawdown": metrics["max_drawdown"],
        "buy_hold_return_avg": buy_hold_return_avg,
        "metadata_json": json.dumps(metadata),
    }
    save_backtest_run(backtest, trades)
    backtest.update(
        {
            "strategy_gate_passed": strategy_gate_passed,
            "strategy_gate_reason": strategy_gate_reason,
            "passing_windows": passing_windows,
            "total_windows": len(walk_records),
            "passing_window_ratio": passing_window_ratio,
            "traded_tickers": traded_tickers,
            "profitable_tickers": profitable_tickers,
            "per_ticker": per_ticker,
            "walk_forward_windows": walk_records,
            "gate_checks": gate_checks,
        }
    )
    return backtest