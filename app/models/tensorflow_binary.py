"""Binary TensorFlow direction model: target = `enter_long_7d` (1/0).

Replaces the high-noise 3-class direction head. Uses sigmoid + binary
crossentropy and calibrates probabilities with isotonic regression on the
validation set so that downstream EV computations consume well-calibrated
probabilities of "forward return clears the cost+volatility band".

Reports val_acc, val_AUC, and precision@p>=0.60 — the metric that actually
matters for entry signals (precision of high-confidence longs), not raw accuracy.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    tf = None
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    IsotonicRegression = None
    accuracy_score = None
    roc_auc_score = None
    precision_score = None
    StandardScaler = None

from app.config import STORAGE_DIR
from app.data.database import initialize_database, read_technical_features, save_model_run
from app.models.tensorflow_direction import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    add_model_features,
)


MODEL_NAME = "tensorflow_binary_enter_long"
TARGET_NAME = "target_enter_long_7d"
TARGET_RETURN_NAME = "target_return_7d"
HIGH_CONFIDENCE_THRESHOLD = 0.60


def load_dataset() -> pd.DataFrame:
    initialize_database()
    dataset = read_technical_features()
    if dataset.empty:
        raise ValueError("No technical features found. Run generate-features first.")
    if TARGET_NAME not in dataset.columns:
        raise ValueError(
            f"Column {TARGET_NAME} missing. Re-run generate-features to populate the binary label."
        )
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


def build_model(
    input_dim: int,
    *,
    l2_strength: float = 0.0,
    dropout_scale: float = 1.0,
) -> "tf.keras.Model":
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required to train the binary model.")
    regularizer = tf.keras.regularizers.l2(l2_strength) if l2_strength and l2_strength > 0 else None
    drop_a = max(0.0, min(0.6, 0.30 * dropout_scale))
    drop_b = max(0.0, min(0.6, 0.20 * dropout_scale))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(96, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(drop_a),
            tf.keras.layers.Dense(48, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(drop_b),
            tf.keras.layers.Dense(24, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision", thresholds=HIGH_CONFIDENCE_THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=HIGH_CONFIDENCE_THRESHOLD),
        ],
    )
    return model


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


def compute_metrics(y_true: np.ndarray, p_calibrated: np.ndarray) -> dict:
    metrics: dict[str, float | int | None] = {}
    predicted = (p_calibrated >= 0.5).astype("int32")
    metrics["accuracy"] = float(accuracy_score(y_true, predicted)) if accuracy_score else None
    try:
        metrics["auc"] = float(roc_auc_score(y_true, p_calibrated)) if roc_auc_score else None
    except ValueError:
        metrics["auc"] = None
    high_conf_mask = p_calibrated >= HIGH_CONFIDENCE_THRESHOLD
    metrics["high_confidence_count"] = int(high_conf_mask.sum())
    if high_conf_mask.any() and precision_score:
        high_conf_pred = np.ones(high_conf_mask.sum(), dtype="int32")
        high_conf_true = y_true[high_conf_mask]
        metrics["precision_at_p60"] = float(precision_score(high_conf_true, high_conf_pred, zero_division=0))
    else:
        metrics["precision_at_p60"] = None
    metrics["base_rate_positive"] = float(y_true.mean()) if len(y_true) else None
    return metrics


def serialize_isotonic(calibrator: "IsotonicRegression | None") -> dict | None:
    if calibrator is None:
        return None
    return {
        "method": "isotonic",
        "x_thresholds": calibrator.X_thresholds_.tolist(),
        "y_thresholds": calibrator.y_thresholds_.tolist(),
    }


def fit_split_conformal(
    p_calibrated: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """Split-conformal calibration for the predicted positive class.

    Nonconformity score: s_i = 1 - p_i if y_i == 1 else p_i (i.e. how far the
    calibrated probability is from the realised label). The (1 - alpha)-quantile
    of those scores yields a conservative band [p - q, p + q] that covers the
    true outcome with probability >= 1 - alpha (under exchangeability).
    """
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
    # Conformal correction with finite-sample adjustment: ceil((n+1)(1-alpha))/n.
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = max(1, min(rank, n))
    quantile = float(np.partition(nonconformity, rank - 1)[rank - 1])
    return {
        "alpha": float(alpha),
        "quantile": quantile,
        "calibration_size": int(n),
        "method": "split_conformal_v1",
    }


def build_predictions(
    model,
    dataset: pd.DataFrame,
    scaler,
    run_id: str,
    calibrator,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        split_data, x_values, _y, _ = prepare_split(dataset, split, scaler)
        if len(split_data) == 0:
            continue
        raw_probability = model.predict(x_values, verbose=0).flatten()
        calibrated = apply_calibration(raw_probability, calibrator)
        # NOTE: write into the legacy 3-class predictions table by mapping sigmoid p to
        # (probability_down=1-p, probability_sideways=0, probability_up=p) so existing
        # consumers (frontend/predictions, fusion) keep working until they are migrated.
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


def train_tensorflow_binary_model(
    epochs: int = 60,
    batch_size: int = 128,
    *,
    seed: int = 42,
    l2_strength: float = 0.0,
    dropout_scale: float = 1.0,
) -> dict:
    if tf is None or accuracy_score is None:
        raise ModuleNotFoundError("TensorFlow + scikit-learn required.")
    tf.keras.utils.set_random_seed(int(seed))

    dataset = load_dataset()
    train_data, x_train, y_train, scaler = prepare_split(dataset, "train")
    val_data, x_val, y_val, _ = prepare_split(dataset, "validation", scaler)
    test_data, x_test, y_test, _ = prepare_split(dataset, "test", scaler)

    pos_rate = float(y_train.mean()) if len(y_train) else 0.5
    pos_rate = min(max(pos_rate, 1e-3), 1 - 1e-3)
    class_weight = {0: 0.5 / (1 - pos_rate), 1: 0.5 / pos_rate}

    model = build_model(
        input_dim=x_train.shape[1],
        l2_strength=l2_strength,
        dropout_scale=dropout_scale,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping],
        class_weight=class_weight,
    )

    raw_val = model.predict(x_val, verbose=0).flatten()
    raw_test = model.predict(x_test, verbose=0).flatten()
    calibrator = fit_isotonic_calibrator(raw_val, y_val)
    calibrated_val = apply_calibration(raw_val, calibrator)
    calibrated_test = apply_calibration(raw_test, calibrator)

    val_metrics = compute_metrics(y_val, calibrated_val)
    test_metrics = compute_metrics(y_test, calibrated_test)
    raw_val_metrics = compute_metrics(y_val, raw_val)
    conformal = fit_split_conformal(calibrated_val, y_val, alpha=0.10)

    run_id = f"tf_binary_7d_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    model_dir = STORAGE_DIR / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.json"
    metadata_path = model_dir / "metadata.json"

    model.save(model_path)
    scaler_payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "class_weight": {str(k): float(v) for k, v in class_weight.items()},
        "probability_calibration": serialize_isotonic(calibrator),
        "conformal_calibration": conformal,
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "target_name": TARGET_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "epochs_ran": len(history.history["loss"]),
        "validation_metrics_calibrated": val_metrics,
        "validation_metrics_raw": raw_val_metrics,
        "test_metrics_calibrated": test_metrics,
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(val_data)),
        "test_rows": int(len(test_data)),
        "train_positive_rate": pos_rate,
        "calibrator_method": "isotonic" if calibrator is not None else "none",
        "conformal_calibration": conformal,
        "seed": int(seed),
        "l2_strength": float(l2_strength),
        "dropout_scale": float(dropout_scale),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    predictions = build_predictions(model, dataset, scaler, run_id, calibrator)
    run_record = {
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
    }
    save_model_run(run_record, predictions)
    return metadata
