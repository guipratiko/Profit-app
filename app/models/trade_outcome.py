"""Operational trade-outcome model.

Trains a sklearn HistGradientBoostingClassifier (win/loss/timeout) and a
HistGradientBoostingRegressor (expected post-cost return) on top of the trade
outcome dataset built from real OHLCV.  Inference produces probabilities and an
expected return per ticker for the latest trading session, which are persisted
into ``operational_trade_outcomes`` and consumed by the paper-trading gate to
emit operational instructions (ENTER_LONG / WATCHLIST / NO_TRADE).

This model answers the operational question directly — "does this trade have
positive expectancy after stop, target and cost?" — instead of forcing the
paper trader to translate a direction probability into a trading decision.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional heavy deps live in the py3.11 env
    import joblib
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.metrics import accuracy_score, log_loss

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time fallback
    joblib = None  # type: ignore[assignment]
    HistGradientBoostingClassifier = None  # type: ignore[assignment]
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    log_loss = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

from app.config import STORAGE_DIR, resolve_artifact_dir
from app.data.database import (
    get_trade_outcome_runs,
    initialize_database,
    read_latest_operational_trade_outcomes,
    save_operational_trade_outcomes,
    save_trade_outcome_run,
)
from app.features.trade_outcomes import (
    ID_TO_OUTCOME,
    OUTCOME_LABELS,
    OUTCOME_TO_ID,
    TRADE_FEATURE_COLUMNS,
    build_current_trade_outcome_features,
    build_trade_outcome_dataset,
    calculate_stop_distance,
)


MODEL_NAME = "sklearn_trade_outcome_classifier_v1"
INFERENCE_VERSION = "v1_trade_outcome_win_loss_timeout"


def _require_sklearn() -> None:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn and joblib are required for the trade outcome model. "
            "Use the Python 3.11 environment (py -3.11) configured for ML."
        )


def _split_arrays(dataset: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    sub = dataset[dataset["time_split"] == split].copy()
    if sub.empty:
        empty = np.zeros((0,), dtype="float32")
        return sub, np.zeros((0, len(TRADE_FEATURE_COLUMNS)), dtype="float32"), empty.astype("int32"), empty
    feature_matrix = sub[TRADE_FEATURE_COLUMNS].astype("float32").to_numpy()
    labels = sub["trade_outcome"].map(OUTCOME_TO_ID).astype("int32").to_numpy()
    returns = sub["trade_return"].astype("float32").to_numpy()
    return sub, feature_matrix, labels, returns


def train_trade_outcome_model(
    holding_days: int = 7,
    min_reward_risk: float = 1.5,
    cost_per_trade: float = 0.002,
    spread: float = 0.001,
    slippage: float = 0.001,
    max_iter: int = 250,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict:
    _require_sklearn()
    initialize_database()
    dataset = build_trade_outcome_dataset(
        holding_days=holding_days,
        min_reward_risk=min_reward_risk,
        cost_per_trade=cost_per_trade,
        spread=spread,
        slippage=slippage,
    )
    if dataset.empty:
        raise ValueError("Trade outcome dataset is empty; populate OHLCV/features first.")

    train_df, x_tr, y_tr, r_tr = _split_arrays(dataset, "train")
    val_df, x_va, y_va, r_va = _split_arrays(dataset, "validation")
    test_df, x_te, y_te, r_te = _split_arrays(dataset, "test")
    if x_tr.shape[0] == 0 or x_va.shape[0] == 0 or x_te.shape[0] == 0:
        raise ValueError("Trade outcome dataset is missing one of train/validation/test splits.")

    classifier = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    classifier.fit(x_tr, y_tr)

    regressor = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    regressor.fit(x_tr, r_tr)

    label_index = list(range(len(OUTCOME_LABELS)))
    val_proba = classifier.predict_proba(x_va)
    val_pred = val_proba.argmax(axis=1)
    val_accuracy = float(accuracy_score(y_va, val_pred))
    val_log_loss = float(log_loss(y_va, val_proba, labels=label_index))

    test_proba = classifier.predict_proba(x_te)
    test_pred = test_proba.argmax(axis=1)
    test_accuracy = float(accuracy_score(y_te, test_pred))
    test_log_loss = float(log_loss(y_te, test_proba, labels=label_index))

    win_index = OUTCOME_TO_ID["win"]
    test_p_win = test_proba[:, win_index]
    test_predicted_return = regressor.predict(x_te)
    enter_mask = (test_p_win >= 0.50) & (test_predicted_return > 0.0)
    if enter_mask.any():
        sim_trades = int(enter_mask.sum())
        sim_avg_return = float(r_te[enter_mask].mean())
        sim_win_rate = float((r_te[enter_mask] > 0.0).mean())
    else:
        sim_trades = 0
        sim_avg_return = 0.0
        sim_win_rate = 0.0

    run_id = (
        f"to_{holding_days}d_"
        f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    )
    artifact_dir = STORAGE_DIR / "models" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, artifact_dir / "classifier.joblib")
    joblib.dump(regressor, artifact_dir / "regressor.joblib")

    metadata = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "labels": OUTCOME_LABELS,
        "feature_columns": TRADE_FEATURE_COLUMNS,
        "holding_days": int(holding_days),
        "min_reward_risk": float(min_reward_risk),
        "cost_per_trade": float(cost_per_trade),
        "spread": float(spread),
        "slippage": float(slippage),
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "random_state": int(random_state),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "validation_accuracy": val_accuracy,
        "validation_log_loss": val_log_loss,
        "test_accuracy": test_accuracy,
        "test_log_loss": test_log_loss,
        "simulated_test_trades": sim_trades,
        "simulated_test_avg_return": sim_avg_return,
        "simulated_test_win_rate": sim_win_rate,
        "outcome_distribution": {
            label: int((dataset["trade_outcome"] == label).sum())
            for label in OUTCOME_LABELS
        },
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    save_trade_outcome_run(
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "horizon_days": holding_days,
            "min_reward_risk": min_reward_risk,
            "cost_per_trade": cost_per_trade,
            "spread": spread,
            "slippage": slippage,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "validation_accuracy": val_accuracy,
            "validation_log_loss": val_log_loss,
            "test_accuracy": test_accuracy,
            "test_log_loss": test_log_loss,
            "simulated_test_trades": sim_trades,
            "simulated_test_avg_return": sim_avg_return,
            "simulated_test_win_rate": sim_win_rate,
            "artifact_path": str(artifact_dir),
            "metadata_json": json.dumps(metadata),
        }
    )
    return metadata


def get_latest_trade_outcome_run_id() -> str:
    runs = get_trade_outcome_runs()
    if runs.empty:
        raise ValueError("No trade outcome runs available. Train the model first.")
    return str(runs.iloc[0]["run_id"])


def has_trade_outcome_runs() -> bool:
    runs = get_trade_outcome_runs()
    return not runs.empty


def load_trade_outcome_artifacts(run_id: str, prefer_recent: bool = True):
    _require_sklearn()
    runs = get_trade_outcome_runs()
    sub = runs[runs["run_id"] == run_id]
    if sub.empty:
        raise ValueError(f"Trade outcome run not found: {run_id}")
    artifact_dir = resolve_artifact_dir(str(sub.iloc[0]["artifact_path"]))
    classifier_path = artifact_dir / "classifier.joblib"
    regressor_path = artifact_dir / "regressor.joblib"
    if prefer_recent:
        recent_classifier = artifact_dir / "classifier_recent.joblib"
        recent_regressor = artifact_dir / "regressor_recent.joblib"
        if recent_classifier.exists() and recent_regressor.exists():
            classifier_path = recent_classifier
            regressor_path = recent_regressor
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["loaded_artifacts"] = {
        "classifier": classifier_path.name,
        "regressor": regressor_path.name,
    }
    return classifier, regressor, metadata


def run_trade_outcome_inference(
    run_id: str | None = None,
    min_reward_risk: float | None = None,
    cost_per_trade: float | None = None,
    spread: float | None = None,
    slippage: float | None = None,
) -> dict:
    _require_sklearn()
    initialize_database()
    selected_run_id = run_id or get_latest_trade_outcome_run_id()
    classifier, regressor, metadata = load_trade_outcome_artifacts(selected_run_id)

    holding_days = int(metadata.get("holding_days", 7))
    effective_reward_risk = float(
        metadata.get("min_reward_risk", 1.5) if min_reward_risk is None else min_reward_risk
    )
    effective_cost = float(
        metadata.get("cost_per_trade", 0.002) if cost_per_trade is None else cost_per_trade
    )
    effective_spread = float(
        metadata.get("spread", 0.001) if spread is None else spread
    )
    effective_slippage = float(
        metadata.get("slippage", 0.001) if slippage is None else slippage
    )
    execution_drag = effective_cost + effective_spread + effective_slippage

    current = build_current_trade_outcome_features()
    if current.empty:
        return {
            "run_id": selected_run_id,
            "generated": 0,
            "inserted": 0,
            "latest_date": None,
            "inference_version": INFERENCE_VERSION,
            "directions": {},
        }

    expected_columns = list(metadata.get("feature_columns", TRADE_FEATURE_COLUMNS))
    feature_matrix = current[expected_columns].astype("float32").to_numpy()
    proba = classifier.predict_proba(feature_matrix)
    expected_return = regressor.predict(feature_matrix)

    win_index = OUTCOME_TO_ID["win"]
    loss_index = OUTCOME_TO_ID["loss"]
    timeout_index = OUTCOME_TO_ID["timeout"]

    records: list[dict] = []
    for offset, (_, feature_row) in enumerate(current.iterrows()):
        stop_distance = calculate_stop_distance(feature_row["volatility_21d"])
        records.append(
            {
                "run_id": selected_run_id,
                "ticker": str(feature_row["ticker"]),
                "date": str(feature_row["date"]),
                "horizon_days": holding_days,
                "probability_win": float(proba[offset, win_index]),
                "probability_loss": float(proba[offset, loss_index]),
                "probability_timeout": float(proba[offset, timeout_index]),
                "expected_return": float(expected_return[offset]),
                "stop_distance": float(stop_distance),
                "target_distance": float(stop_distance * effective_reward_risk),
                "execution_drag": float(execution_drag),
                "inference_version": INFERENCE_VERSION,
            }
        )
    predictions_df = pd.DataFrame(records)
    inserted = save_operational_trade_outcomes(predictions_df)

    if predictions_df.empty:
        directions = {}
        latest_date = None
    else:
        labels = []
        for _, record in predictions_df.iterrows():
            argmax_index = int(
                np.argmax(
                    [
                        record["probability_loss"],
                        record["probability_timeout"],
                        record["probability_win"],
                    ]
                )
            )
            labels.append(ID_TO_OUTCOME[argmax_index])
        directions = pd.Series(labels).value_counts().to_dict()
        latest_date = str(predictions_df["date"].max())

    return {
        "run_id": selected_run_id,
        "generated": int(len(predictions_df)),
        "inserted": int(inserted),
        "latest_date": latest_date,
        "inference_version": INFERENCE_VERSION,
        "directions": directions,
    }


def latest_operational_trade_outcomes() -> pd.DataFrame:
    return read_latest_operational_trade_outcomes()
