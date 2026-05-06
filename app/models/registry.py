from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from app.config import resolve_artifact_dir
from app.data.database import get_backtest_runs, get_latest_model_run_id, get_model_runs
from app.models.tensorflow_direction import FEATURE_COLUMNS


logger = logging.getLogger(__name__)


def read_scaler_feature_columns(artifact_path: str) -> list[str]:
    scaler_path = resolve_artifact_dir(artifact_path) / "scaler.json"
    if not scaler_path.exists():
        return []
    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    return list(payload.get("feature_columns", []))


def is_current_feature_schema(artifact_path: str) -> bool:
    return read_scaler_feature_columns(artifact_path) == FEATURE_COLUMNS


def get_current_schema_model_runs() -> pd.DataFrame:
    model_runs = get_model_runs()
    if model_runs.empty:
        return model_runs
    compatible = model_runs.copy()
    compatible["schema_compatible"] = compatible["artifact_path"].map(is_current_feature_schema)
    return compatible[compatible["schema_compatible"]].copy()


def _emit_decision(decision: dict, verbose: bool, level: str = "info") -> None:
    if not verbose:
        return
    payload = json.dumps(decision)
    if level == "warning":
        logger.warning("[model-selection] %s", payload)
    else:
        logger.info("[model-selection] %s", payload)
    print(f"[model-selection] {payload}")


def _parse_metadata_json(value) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _metadata_passing_window_ratio(metadata: dict) -> float:
    if "passing_window_ratio" in metadata:
        return float(metadata.get("passing_window_ratio") or 0.0)
    windows = metadata.get("walk_forward_windows")
    if not isinstance(windows, list) or not windows:
        return 0.0
    passed = sum(1 for window in windows if isinstance(window, dict) and bool(window.get("window_passed")))
    return float(passed / len(windows))


def _metadata_profitable_tickers(metadata: dict) -> int:
    if "profitable_tickers" in metadata:
        return int(metadata.get("profitable_tickers") or 0)
    per_ticker = metadata.get("per_ticker")
    if not isinstance(per_ticker, list):
        return 0
    return sum(
        1
        for ticker in per_ticker
        if isinstance(ticker, dict)
        and int(ticker.get("trades") or 0) > 0
        and float(ticker.get("cumulative_return") or 0.0) > 0.0
    )


def select_best_current_schema_model(verbose: bool = True) -> dict:
    """Return the chosen run_id together with the reason it was picked."""
    compatible_runs = get_current_schema_model_runs()
    if compatible_runs.empty:
        run_id = get_latest_model_run_id()
        decision = {
            "run_id": run_id,
            "reason": "no_schema_compatible_run_found_falling_back_to_latest",
            "compatible_runs": 0,
            "backtest_evaluated": False,
        }
        _emit_decision(decision, verbose, level="warning")
        return decision

    backtests = get_backtest_runs()
    if backtests.empty:
        run_id = str(compatible_runs.sort_values("trained_at", ascending=False).iloc[0]["run_id"])
        decision = {
            "run_id": run_id,
            "reason": "no_backtest_runs_picked_latest_compatible_by_trained_at",
            "compatible_runs": int(len(compatible_runs)),
            "backtest_evaluated": False,
        }
        _emit_decision(decision, verbose)
        return decision

    ranked = backtests.merge(
        compatible_runs[["run_id", "trained_at"]],
        on="run_id",
        how="inner",
    )
    if ranked.empty:
        run_id = str(compatible_runs.sort_values("trained_at", ascending=False).iloc[0]["run_id"])
        decision = {
            "run_id": run_id,
            "reason": "no_backtests_for_compatible_runs_picked_latest_trained",
            "compatible_runs": int(len(compatible_runs)),
            "backtest_evaluated": False,
        }
        _emit_decision(decision, verbose)
        return decision

    ranked = ranked.copy()
    metadata = ranked["metadata_json"].map(_parse_metadata_json) if "metadata_json" in ranked.columns else pd.Series([{}] * len(ranked))
    ranked["strategy_gate_passed"] = metadata.map(lambda item: bool(item.get("strategy_gate_passed", False)))
    ranked["passing_window_ratio"] = metadata.map(_metadata_passing_window_ratio)
    ranked["profitable_tickers"] = metadata.map(_metadata_profitable_tickers)
    ranked["beats_buy_hold"] = ranked["cumulative_return"] > ranked["buy_hold_return_avg"]
    ranked = ranked.sort_values(
        [
            "strategy_gate_passed",
            "beats_buy_hold",
            "passing_window_ratio",
            "profitable_tickers",
            "cumulative_return",
            "max_drawdown",
            "trades",
            "created_at",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )
    top = ranked.iloc[0]
    decision = {
        "run_id": str(top["run_id"]),
        "reason": (
            "selected_approved_walk_forward_run"
            if bool(top["strategy_gate_passed"])
            else "no_approved_walk_forward_run_selected_best_available"
        ),
        "compatible_runs": int(len(compatible_runs)),
        "backtest_evaluated": True,
        "strategy_gate_passed": bool(top["strategy_gate_passed"]),
        "beats_buy_hold": bool(top["beats_buy_hold"]),
        "passing_window_ratio": float(top["passing_window_ratio"]),
        "profitable_tickers": int(top["profitable_tickers"]),
        "cumulative_return": float(top["cumulative_return"]),
        "max_drawdown": float(top["max_drawdown"]),
        "trades": int(top["trades"]),
    }
    _emit_decision(decision, verbose)
    return decision


def get_best_current_schema_model_run_id() -> str:
    return select_best_current_schema_model(verbose=True)["run_id"]
