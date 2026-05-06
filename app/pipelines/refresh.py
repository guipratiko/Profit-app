"""Online refresh pipeline.

Implements the lightweight online-learning trigger described in the project
brief: when the user has been away (the latest OHLCV in the database is more
than ``max_staleness_days`` business days behind today) the system pulls the
missing bars, regenerates technical features, refits the operational head of
the trade-outcome model on the most recent window without retraining the
direction classifier, and re-runs operational inference.

This mirrors the "freeze the deep base, fine-tune the predictive head" pattern
recommended in the methodology document.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import INITIAL_ASSETS, resolve_artifact_dir
from app.data.database import (
    get_price_counts,
    get_trade_outcome_runs,
    initialize_database,
)
from app.features.technical import generate_technical_features
from app.features.trade_outcomes import TRADE_FEATURE_COLUMNS, build_trade_outcome_dataset


REFRESH_PIPELINE_VERSION = "v1_online_head_refit"
B3_TIMEZONE = ZoneInfo("America/Sao_Paulo")
B3_DAILY_CLOSE_HOUR = 18


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _b3_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        date(year, 1, 25),
        easter - timedelta(days=48),
        easter - timedelta(days=47),
        easter - timedelta(days=2),
        date(year, 4, 21),
        date(year, 5, 1),
        easter + timedelta(days=60),
        date(year, 7, 9),
        date(year, 9, 7),
        date(year, 10, 12),
        date(year, 11, 2),
        date(year, 11, 15),
        date(year, 11, 20),
        date(year, 12, 24),
        date(year, 12, 25),
        date(year, 12, 31),
    }


def _to_b3_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).astimezone(B3_TIMEZONE)
    return value.astimezone(B3_TIMEZONE)


def _is_b3_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _b3_holidays(day.year)


def _previous_b3_trading_day(day: date) -> date:
    current = day - timedelta(days=1)
    while not _is_b3_trading_day(current):
        current -= timedelta(days=1)
    return current


def _latest_completed_b3_session(reference: datetime) -> date:
    market_now = _to_b3_datetime(reference)
    current_day = market_now.date()
    if _is_b3_trading_day(current_day) and market_now.hour >= B3_DAILY_CLOSE_HOUR:
        return current_day
    return _previous_b3_trading_day(current_day)


def _trading_days_between(start_day: date, end_day: date) -> int:
    if end_day <= start_day:
        return 0
    days_behind = 0
    cursor = start_day + timedelta(days=1)
    while cursor <= end_day:
        if _is_b3_trading_day(cursor):
            days_behind += 1
        cursor += timedelta(days=1)
    return days_behind


def detect_staleness(today: datetime | None = None) -> dict:
    reference = today or datetime.utcnow()
    market_now = _to_b3_datetime(reference)
    expected_latest_date = _latest_completed_b3_session(reference)
    counts = get_price_counts()
    if counts.empty:
        return {
            "is_stale": True,
            "reason": "no_prices_stored",
            "latest_date": None,
            "today": market_now.strftime("%Y-%m-%d"),
            "market_reference_date": expected_latest_date.strftime("%Y-%m-%d"),
            "calendar_days_behind": None,
            "trading_days_behind": None,
        }
    latest_date_text = str(counts["last_date"].max())
    latest_date = datetime.strptime(latest_date_text, "%Y-%m-%d").date()
    calendar_days_behind = max((expected_latest_date - latest_date).days, 0)
    trading_days_behind = _trading_days_between(latest_date, expected_latest_date)
    return {
        "is_stale": trading_days_behind > 0,
        "reason": None if trading_days_behind <= 0 else "trading_session_lag",
        "latest_date": latest_date_text,
        "today": market_now.strftime("%Y-%m-%d"),
        "market_reference_date": expected_latest_date.strftime("%Y-%m-%d"),
        "calendar_days_behind": int(calendar_days_behind),
        "trading_days_behind": int(trading_days_behind),
    }


def refit_trade_outcome_head(
    run_id: str | None = None,
    window_days: int = 180,
    max_iter: int = 80,
    learning_rate: float = 0.05,
) -> dict:
    """Quick head re-fit on the most recent supervised slice.

    Loads the existing classifier/regressor, fits a fresh pair on rows whose
    ``date`` lies within ``window_days`` of the latest available supervised
    label (effectively the recent window the trader missed), and stores them
    next to the original artefacts as ``classifier_recent.joblib`` /
    ``regressor_recent.joblib``.  Inference falls back to the base artefacts
    when the recent ones are absent.
    """
    try:
        import joblib
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "scikit-learn required for refit. Use the Python 3.11 environment."
        ) from exc

    runs = get_trade_outcome_runs()
    if runs.empty:
        return {"status": "skipped", "reason": "no_trade_outcome_runs"}
    target_run_id = run_id or str(runs.iloc[0]["run_id"])
    sub = runs[runs["run_id"] == target_run_id]
    if sub.empty:
        return {"status": "skipped", "reason": f"unknown_run_id:{target_run_id}"}
    artifact_dir = resolve_artifact_dir(str(sub.iloc[0]["artifact_path"]))
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    dataset = build_trade_outcome_dataset(
        holding_days=int(metadata.get("holding_days", 7)),
        min_reward_risk=float(metadata.get("min_reward_risk", 1.5)),
        cost_per_trade=float(metadata.get("cost_per_trade", 0.002)),
        spread=float(metadata.get("spread", 0.001)),
        slippage=float(metadata.get("slippage", 0.001)),
    )
    if dataset.empty:
        return {"status": "skipped", "reason": "trade_outcome_dataset_empty"}

    cutoff = pd.to_datetime(dataset["date"]).max() - pd.Timedelta(days=window_days)
    recent = dataset[pd.to_datetime(dataset["date"]) >= cutoff]
    if recent.empty or recent.shape[0] < 50:
        return {
            "status": "skipped",
            "reason": "insufficient_recent_rows",
            "rows": int(recent.shape[0]),
            "window_days": window_days,
        }

    from app.features.trade_outcomes import OUTCOME_TO_ID

    feature_columns = list(metadata.get("feature_columns", TRADE_FEATURE_COLUMNS))
    x_recent = recent[feature_columns].astype("float32").to_numpy()
    y_recent = recent["trade_outcome"].map(OUTCOME_TO_ID).astype("int32").to_numpy()
    r_recent = recent["trade_return"].astype("float32").to_numpy()

    classifier = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=int(metadata.get("random_state", 42)),
    )
    classifier.fit(x_recent, y_recent)
    regressor = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=int(metadata.get("random_state", 42)),
    )
    regressor.fit(x_recent, r_recent)
    joblib.dump(classifier, artifact_dir / "classifier_recent.joblib")
    joblib.dump(regressor, artifact_dir / "regressor_recent.joblib")

    metadata.setdefault("head_refits", []).append(
        {
            "performed_at": datetime.utcnow().isoformat(timespec="seconds"),
            "window_days": int(window_days),
            "rows": int(recent.shape[0]),
            "max_iter": int(max_iter),
            "learning_rate": float(learning_rate),
            "refresh_version": REFRESH_PIPELINE_VERSION,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "status": "refit",
        "run_id": target_run_id,
        "rows": int(recent.shape[0]),
        "window_days": int(window_days),
    }


def run_refresh_pipeline(
    tickers: Iterable[str] | None = None,
    max_staleness_days: int = 1,
    refit_window_days: int = 180,
    skip_price_update: bool = False,
    force_refresh: bool = False,
) -> dict:
    """Detect staleness and re-execute the operational pipeline end-to-end."""
    initialize_database()
    staleness = detect_staleness()

    actions: dict = {
        "refresh_pipeline_version": REFRESH_PIPELINE_VERSION,
        "staleness": staleness,
    }
    if not skip_price_update and not force_refresh:
        lag_sessions = staleness.get("trading_days_behind")
        if lag_sessions is not None and lag_sessions <= max_staleness_days:
            actions["status"] = "fresh_no_action"
            return actions
        if lag_sessions is None and not staleness["is_stale"]:
            actions["status"] = "fresh_no_action"
            return actions

    tickers_list = list(tickers) if tickers else list(INITIAL_ASSETS)

    # Local import keeps yfinance optional when only running diagnostics.
    if not skip_price_update:
        from app.data.market_data import update_all_prices, update_latest_quote_snapshots

        if force_refresh:
            updated_rows = update_all_prices(tickers=tickers_list, period="7d")
        else:
            updated_rows = update_all_prices(tickers=tickers_list)
        actions["updated_prices"] = updated_rows
        actions["updated_live_quotes"] = update_latest_quote_snapshots(tickers=tickers_list)

    actions["technical_feature_rows"] = generate_technical_features()

    head_refit = refit_trade_outcome_head(window_days=refit_window_days)
    actions["trade_outcome_head_refit"] = head_refit

    try:
        from app.models.inference import run_current_inference

        actions["direction_inference"] = run_current_inference()
    except Exception as exc:  # pragma: no cover - direction inference optional
        actions["direction_inference_error"] = str(exc)

    try:
        from app.models.fusion import run_fusion_predictions

        actions["fusion_predictions"] = run_fusion_predictions()
    except Exception as exc:  # pragma: no cover
        actions["fusion_predictions_error"] = str(exc)

    try:
        from app.models.trade_outcome import run_trade_outcome_inference

        actions["trade_outcome_inference"] = run_trade_outcome_inference()
    except Exception as exc:  # pragma: no cover
        actions["trade_outcome_inference_error"] = str(exc)

    from app.trading.paper import generate_paper_trading_signals

    actions["paper_trading"] = generate_paper_trading_signals()
    actions["status"] = "refreshed"
    actions["completed_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return actions
