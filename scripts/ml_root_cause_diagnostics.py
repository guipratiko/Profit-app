from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data.database import (
    get_backtest_runs,
    get_model_runs,
    get_news_events,
    get_paper_trading_signals,
    get_price_counts,
    get_qualitative_features,
    read_model_predictions,
    read_operational_predictions,
    read_technical_features,
)
from app.models.tensorflow_direction import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, add_model_features


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n## {title}")
    if frame.empty:
        print("empty")
        return
    print(frame.to_string(index=False))


def _safe_auc(y_true: pd.Series, scores: pd.Series) -> float | None:
    if len(y_true) == 0 or y_true.nunique(dropna=True) < 2:
        return None
    try:
        return float(roc_auc_score(y_true.astype(int), scores.astype(float)))
    except ValueError:
        return None


def _describe(series: pd.Series) -> dict[str, float]:
    stats = series.astype(float).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {str(key): round(float(value), 4) for key, value in stats.items()}


def _recent_binary_run_ids(limit: int = 6) -> list[str]:
    runs = get_model_runs()
    if runs.empty:
        return []
    binary = runs[runs["target_name"].astype(str).eq("target_enter_long_7d")].copy()
    if binary.empty:
        binary = runs[runs["run_id"].astype(str).str.startswith("tf_binary_")].copy()
    if "trained_at" in binary.columns:
        binary = binary.sort_values("trained_at")
    elif "created_at" in binary.columns:
        binary = binary.sort_values("created_at")
    return binary["run_id"].astype(str).tail(limit).tolist()


def diagnose_dataset() -> None:
    features = read_technical_features()
    print("## Dataset")
    if features.empty:
        print("No technical features found.")
        return
    features = add_model_features(features.copy())
    features["date"] = pd.to_datetime(features["date"])
    required = RAW_FEATURE_COLUMNS + FEATURE_COLUMNS + [
        "target_enter_long_7d",
        "target_return_7d",
        "volatility_21d",
    ]
    valid = features.dropna(subset=required).copy()
    print(
        f"rows={len(features)} valid_model_rows={len(valid)} "
        f"tickers={valid['ticker'].nunique()} "
        f"dates={valid['date'].min().date()}..{valid['date'].max().date()}"
    )
    split = (
        valid.groupby("time_split")
        .agg(
            rows=("ticker", "size"),
            tickers=("ticker", "nunique"),
            pos_rate=("target_enter_long_7d", "mean"),
            median_7d_return=("target_return_7d", "median"),
            mean_7d_return=("target_return_7d", "mean"),
        )
        .round(4)
        .reset_index()
    )
    _print_frame("Split label distribution", split)
    by_ticker = (
        valid.groupby("ticker")["target_enter_long_7d"]
        .agg(rows="count", pos_rate="mean")
        .sort_values("pos_rate")
        .round(4)
    )
    _print_frame("Lowest positive-rate tickers", by_ticker.head(8).reset_index())
    _print_frame("Highest positive-rate tickers", by_ticker.tail(8).reset_index())
    threshold = valid["volatility_21d"].astype(float).abs() * math.sqrt(7) * 0.6 + 0.006
    print("\n## Label strictness")
    print("threshold_7d_stats", _describe(threshold))
    print("target_return_7d_stats", _describe(valid["target_return_7d"]))
    validation = valid[valid["time_split"].eq("validation")].copy()
    auc_rows: list[dict[str, float | str]] = []
    y = validation["target_enter_long_7d"].astype(int)
    for column in FEATURE_COLUMNS:
        x = pd.to_numeric(validation[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = x.notna()
        if mask.sum() < 50 or y[mask].nunique() < 2:
            continue
        auc = _safe_auc(y[mask], x[mask])
        if auc is None:
            continue
        auc_rows.append({"feature": column, "auc": auc, "abs_auc": max(auc, 1.0 - auc)})
    univariate = pd.DataFrame(auc_rows).sort_values("abs_auc", ascending=False).head(12).round(4)
    _print_frame("Top univariate validation AUC", univariate)


def diagnose_runs(run_ids: list[str]) -> None:
    runs = get_model_runs()
    selected = runs[runs["run_id"].astype(str).isin(run_ids)].copy()
    run_cols = [
        column
        for column in [
            "run_id",
            "target_name",
            "validation_accuracy",
            "validation_loss",
            "test_accuracy",
            "created_at",
        ]
        if column in selected.columns
    ]
    _print_frame("Recent binary model runs", selected[run_cols])
    records: list[dict[str, object]] = []
    for run_id in run_ids:
        split_predictions = {
            split: read_model_predictions(run_id, split=split)
            for split in ["validation", "test"]
        }
        if all(frame.empty for frame in split_predictions.values()):
            records.append({"run_id": run_id, "split": "all", "rows": 0})
            continue
        for split in ["validation", "test"]:
            subset = split_predictions[split].copy()
            probabilities = subset["probability_up"].astype(float)
            y_true = subset["actual_direction"].astype(str).eq("up").astype(int)
            records.append(
                {
                    "run_id": run_id,
                    "split": split,
                    "rows": len(subset),
                    "auc": None if (auc := _safe_auc(y_true, probabilities)) is None else round(auc, 4),
                    "p_min": round(float(probabilities.min()), 4) if len(probabilities) else None,
                    "p50": round(float(probabilities.median()), 4) if len(probabilities) else None,
                    "p90": round(float(probabilities.quantile(0.9)), 4) if len(probabilities) else None,
                    "p_max": round(float(probabilities.max()), 4) if len(probabilities) else None,
                    "p_ge_060": int((probabilities >= 0.60).sum()),
                }
            )
    _print_frame("Prediction probability diagnostics", pd.DataFrame(records))
    operational_rows: list[dict[str, object]] = []
    for run_id in run_ids:
        operational = read_operational_predictions(run_id)
        if operational.empty:
            continue
        latest = operational.sort_values("date").groupby("ticker", as_index=False).tail(1)
        probabilities = latest["probability_up"].astype(float)
        operational_rows.append(
            {
                "run_id": run_id,
                "latest_date": str(latest["date"].max()),
                "tickers": len(latest),
                "directions": dict(latest["predicted_direction"].value_counts()),
                "p50": round(float(probabilities.median()), 4),
                "p90": round(float(probabilities.quantile(0.9)), 4),
                "p_max": round(float(probabilities.max()), 4),
            }
        )
    _print_frame("Current operational predictions", pd.DataFrame(operational_rows))


def diagnose_operational(run_ids: list[str]) -> None:
    backtests = get_backtest_runs()
    if not backtests.empty:
        recent = backtests[backtests["run_id"].astype(str).isin(run_ids)].tail(12).copy()
        columns = [
            column
            for column in [
                "run_id",
                "threshold",
                "trades",
                "win_rate",
                "cumulative_return",
                "average_trade_return",
                "max_drawdown",
                "buy_hold_return_avg",
                "created_at",
            ]
            if column in recent.columns
        ]
        _print_frame("Recent backtests for binary runs", recent[columns])
    news = get_news_events()
    qualitative = get_qualitative_features()
    paper = get_paper_trading_signals()
    prices = get_price_counts()
    print("\n## Operational coverage")
    print(f"news_events={len(news)} news_tickers={0 if news.empty else news['ticker'].nunique()}")
    print(f"qual_features={len(qualitative)} qual_tickers={0 if qualitative.empty else qualitative['ticker'].nunique()}")
    if not paper.empty:
        latest_paper = paper.sort_values("created_at").tail(40)
        print("paper_actions", dict(latest_paper["operational_action"].fillna("UNKNOWN").value_counts()))
        print("paper_decisions", dict(latest_paper["decision"].fillna("UNKNOWN").value_counts()))
    _print_frame("Price counts sample", prices.head(12))


def main() -> None:
    run_ids = _recent_binary_run_ids(limit=6)
    diagnose_dataset()
    print("\n## Run ids inspected")
    print(run_ids)
    diagnose_runs(run_ids)
    diagnose_operational(run_ids)


if __name__ == "__main__":
    main()