from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on the ML environment
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time fallback
    joblib = None  # type: ignore[assignment]
    HistGradientBoostingClassifier = None  # type: ignore[assignment]
    IsotonicRegression = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

from app.config import STORAGE_DIR
from app.data.database import initialize_database, read_technical_features, save_model_run
from app.models.tensorflow_direction import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, add_model_features


MODEL_NAME = "sklearn_hist_gradient_boosting_enter_long"
TARGET_NAME = "target_enter_long_7d"
TARGET_RETURN_NAME = "target_return_7d"
HIGH_CONFIDENCE_THRESHOLD = 0.60


def _require_sklearn() -> None:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn and joblib are required. Use the Python 3.11 ML environment.")


def load_dataset() -> pd.DataFrame:
    initialize_database()
    dataset = read_technical_features()
    if dataset.empty:
        raise ValueError("No technical features found. Run generate-features first.")
    if TARGET_NAME not in dataset.columns:
        raise ValueError(f"Column {TARGET_NAME} missing. Run generate-features first.")
    dataset = add_model_features(dataset)
    dataset = dataset.dropna(
        subset=RAW_FEATURE_COLUMNS + FEATURE_COLUMNS + [TARGET_NAME, TARGET_RETURN_NAME]
    ).copy()
    dataset[TARGET_NAME] = dataset[TARGET_NAME].astype("int32")
    return dataset


def prepare_split(dataset: pd.DataFrame, split: str, scaler: "StandardScaler | None" = None):
    split_data = dataset[dataset["time_split"] == split].copy()
    x_values = split_data[FEATURE_COLUMNS].astype("float32").to_numpy()
    y_values = split_data[TARGET_NAME].astype("int32").to_numpy()
    if scaler is None:
        if StandardScaler is None:
            raise ModuleNotFoundError("scikit-learn is required to scale features.")
        scaler = StandardScaler()
        x_values = scaler.fit_transform(x_values)
    else:
        x_values = scaler.transform(x_values)
    return split_data, x_values, y_values, scaler


def fit_isotonic_calibrator(probabilities: np.ndarray, y_values: np.ndarray) -> "IsotonicRegression | None":
    if IsotonicRegression is None or len(probabilities) == 0:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probabilities.flatten(), y_values.astype("float64"))
    return calibrator


def apply_calibration(probabilities: np.ndarray, calibrator: "IsotonicRegression | None") -> np.ndarray:
    if calibrator is None:
        return probabilities.flatten()
    return calibrator.predict(probabilities.flatten())


def serialize_isotonic(calibrator: "IsotonicRegression | None") -> dict | None:
    if calibrator is None:
        return None
    return {
        "method": "isotonic",
        "x_thresholds": calibrator.X_thresholds_.tolist(),
        "y_thresholds": calibrator.y_thresholds_.tolist(),
    }


def fit_split_conformal(p_calibrated: np.ndarray, y_true: np.ndarray, alpha: float = 0.10) -> dict:
    if len(p_calibrated) == 0:
        return {
            "alpha": float(alpha),
            "quantile": None,
            "calibration_size": 0,
            "method": "split_conformal_v1",
        }
    p = np.asarray(p_calibrated, dtype="float64").flatten()
    y = np.asarray(y_true, dtype="float64").flatten()
    nonconformity = np.where(y >= 0.5, 1.0 - p, p)
    n = len(nonconformity)
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = max(1, min(rank, n))
    quantile = float(np.partition(nonconformity, rank - 1)[rank - 1])
    return {
        "alpha": float(alpha),
        "quantile": quantile,
        "calibration_size": int(n),
        "method": "split_conformal_v1",
    }


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype("int32")
    metrics: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(y_true, predicted)) if accuracy_score else None,
        "high_confidence_count": int((probabilities >= HIGH_CONFIDENCE_THRESHOLD).sum()),
        "base_rate_positive": float(y_true.mean()) if len(y_true) else None,
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, probabilities)) if roc_auc_score else None
    except ValueError:
        metrics["auc"] = None
    mask = probabilities >= HIGH_CONFIDENCE_THRESHOLD
    if mask.any() and precision_score:
        metrics["precision_at_p60"] = float(
            precision_score(y_true[mask], np.ones(int(mask.sum()), dtype="int32"), zero_division=0)
        )
    else:
        metrics["precision_at_p60"] = None
    return metrics


def predict_probability(model, x_values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x_values)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    return probabilities[:, 1]


def build_predictions(model, dataset: pd.DataFrame, scaler, run_id: str, calibrator) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        split_data, x_values, _y, _ = prepare_split(dataset, split, scaler)
        if split_data.empty:
            continue
        raw_probability = predict_probability(model, x_values)
        calibrated = apply_calibration(raw_probability, calibrator)
        frame = split_data[["ticker", "date", "time_split", TARGET_NAME, TARGET_RETURN_NAME]].copy()
        frame["run_id"] = run_id
        frame["target_name"] = TARGET_NAME
        frame["actual_direction"] = np.where(frame[TARGET_NAME] == 1, "up", "down")
        frame["predicted_direction"] = np.where(calibrated >= 0.5, "up", "down")
        frame["probability_down"] = 1.0 - calibrated
        frame["probability_sideways"] = 0.0
        frame["probability_up"] = calibrated
        frame["target_return"] = frame[TARGET_RETURN_NAME]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_sklearn_binary_model(
    max_iter: int = 300,
    learning_rate: float = 0.04,
    l2_regularization: float = 0.02,
    max_leaf_nodes: int = 15,
    min_samples_leaf: int = 40,
    seed: int = 42,
) -> dict:
    _require_sklearn()
    dataset = load_dataset()
    train_data, x_train, y_train, scaler = prepare_split(dataset, "train")
    val_data, x_val, y_val, _ = prepare_split(dataset, "validation", scaler)
    test_data, x_test, y_test, _ = prepare_split(dataset, "test", scaler)
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError("Dataset is missing one of train/validation/test splits.")

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    raw_val = predict_probability(model, x_val)
    raw_test = predict_probability(model, x_test)
    calibrator = fit_isotonic_calibrator(raw_val, y_val)
    calibrated_val = apply_calibration(raw_val, calibrator)
    calibrated_test = apply_calibration(raw_test, calibrator)
    val_metrics = compute_metrics(y_val, calibrated_val)
    test_metrics = compute_metrics(y_test, calibrated_test)
    raw_val_metrics = compute_metrics(y_val, raw_val)
    raw_test_metrics = compute_metrics(y_test, raw_test)
    conformal = fit_split_conformal(calibrated_val, y_val, alpha=0.10)

    run_id = f"sk_binary_7d_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    model_dir = STORAGE_DIR / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")

    scaler_payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "probability_calibration": serialize_isotonic(calibrator),
        "conformal_calibration": conformal,
        "artifact_kind": "sklearn_hist_gradient_boosting_binary",
    }
    (model_dir / "scaler.json").write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "target_name": TARGET_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "validation_metrics_calibrated": val_metrics,
        "validation_metrics_raw": raw_val_metrics,
        "test_metrics_calibrated": test_metrics,
        "test_metrics_raw": raw_test_metrics,
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(val_data)),
        "test_rows": int(len(test_data)),
        "train_positive_rate": float(y_train.mean()),
        "calibrator_method": "isotonic" if calibrator is not None else "none",
        "conformal_calibration": conformal,
        "max_iter": int(max_iter),
        "n_iter": int(getattr(model, "n_iter_", max_iter)),
        "learning_rate": float(learning_rate),
        "l2_regularization": float(l2_regularization),
        "max_leaf_nodes": int(max_leaf_nodes),
        "min_samples_leaf": int(min_samples_leaf),
        "seed": int(seed),
        "artifact_kind": "sklearn_hist_gradient_boosting_binary",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    predictions = build_predictions(model, dataset, scaler, run_id, calibrator)
    save_model_run(
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "target_name": TARGET_NAME,
            "train_rows": int(len(train_data)),
            "validation_rows": int(len(val_data)),
            "test_rows": int(len(test_data)),
            "validation_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "artifact_path": str(model_dir),
            "metadata_json": json.dumps(metadata),
        },
        predictions,
    )
    return metadata