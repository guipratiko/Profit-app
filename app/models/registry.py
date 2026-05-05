from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import resolve_artifact_dir
from app.data.database import get_backtest_runs, get_latest_model_run_id, get_model_runs
from app.models.tensorflow_direction import FEATURE_COLUMNS


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


def get_best_current_schema_model_run_id() -> str:
    compatible_runs = get_current_schema_model_runs()
    if compatible_runs.empty:
        return get_latest_model_run_id()

    backtests = get_backtest_runs()
    if backtests.empty:
        return str(compatible_runs.sort_values("trained_at", ascending=False).iloc[0]["run_id"])

    ranked = backtests.merge(
        compatible_runs[["run_id", "trained_at"]],
        on="run_id",
        how="inner",
    )
    if ranked.empty:
        return str(compatible_runs.sort_values("trained_at", ascending=False).iloc[0]["run_id"])

    ranked = ranked.copy()
    ranked["beats_buy_hold"] = ranked["cumulative_return"] > ranked["buy_hold_return_avg"]
    ranked = ranked.sort_values(
        ["beats_buy_hold", "cumulative_return", "max_drawdown", "trades", "created_at"],
        ascending=[False, False, False, False, False],
    )
    return str(ranked.iloc[0]["run_id"])