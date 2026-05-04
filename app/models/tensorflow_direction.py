from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - runtime environment guard
    tf = None
try:
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # pragma: no cover - runtime environment guard
    accuracy_score = None
    StandardScaler = None

from app.config import INITIAL_ASSETS, STORAGE_DIR
from app.data.database import initialize_database, read_technical_features, save_model_run


MODEL_NAME = "tensorflow_direction_classifier"
TARGET_NAME = "target_direction_7d"
TARGET_RETURN_NAME = "target_return_7d"
LABELS = ["down", "sideways", "up"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
RAW_FEATURE_COLUMNS = [
    "close",
    "return_1d",
    "return_5d",
    "return_21d",
    "ma_7",
    "ma_21",
    "ma_63",
    "ma_252",
    "volatility_21d",
    "volatility_63d",
    "volume_ratio_21d",
    "drawdown_252d",
    "rsi_14",
]
MODEL_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "volatility_21d",
    "volatility_63d",
    "volume_ratio_21d",
    "drawdown_252d",
    "rsi_14_scaled",
    "close_to_ma_7",
    "close_to_ma_21",
    "close_to_ma_63",
    "close_to_ma_252",
    "ma_7_to_ma_21",
    "ma_21_to_ma_63",
    "ma_63_to_ma_252",
    "volatility_21_to_63",
    "return_21d_to_volatility",
]


def ticker_feature_name(ticker: str) -> str:
    sanitized = ticker.replace(".", "_").replace("-", "_")
    return f"ticker_{sanitized}"


TICKER_FEATURE_COLUMNS = [ticker_feature_name(ticker) for ticker in INITIAL_ASSETS]
FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + TICKER_FEATURE_COLUMNS


def apply_temperature_scaling(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    scaled_logits = np.log(clipped) / max(float(temperature), 1e-6)
    scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def calculate_negative_log_likelihood(probabilities: np.ndarray, y_values: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    return float(-np.log(clipped[np.arange(len(y_values)), y_values]).mean())


def fit_temperature_scaler(probabilities: np.ndarray, y_values: np.ndarray) -> dict:
    if len(probabilities) == 0:
        return {
            "method": "none",
            "temperature": 1.0,
            "validation_nll_before": None,
            "validation_nll_after": None,
        }

    candidates = np.linspace(0.50, 1.00, 51)
    before = calculate_negative_log_likelihood(probabilities, y_values)
    scored = [
        (calculate_negative_log_likelihood(apply_temperature_scaling(probabilities, temperature), y_values), temperature)
        for temperature in candidates
    ]
    after, temperature = min(scored, key=lambda item: item[0])
    return {
        "method": "temperature_scaling",
        "temperature": float(temperature),
        "validation_nll_before": before,
        "validation_nll_after": float(after),
    }


def calibrate_probabilities(probabilities: np.ndarray, calibration: dict | None) -> np.ndarray:
    if not calibration or calibration.get("method") != "temperature_scaling":
        return probabilities
    return apply_temperature_scaling(probabilities, float(calibration.get("temperature", 1.0)))


def calculate_class_expected_returns(dataset: pd.DataFrame) -> dict[str, float]:
    defaults = {label: 0.0 for label in LABELS}
    grouped = dataset.groupby(TARGET_NAME)[TARGET_RETURN_NAME].mean().to_dict()
    for label in LABELS:
        if label in grouped and pd.notna(grouped[label]):
            defaults[label] = float(grouped[label])
    return defaults


def estimate_expected_returns(probabilities: np.ndarray, class_expected_returns: dict[str, float]) -> np.ndarray:
    class_returns = np.array([float(class_expected_returns.get(label, 0.0)) for label in LABELS], dtype="float32")
    return probabilities @ class_returns


def add_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    prepared = dataset.copy()
    prepared["rsi_14_scaled"] = (prepared["rsi_14"] - 50.0) / 50.0
    prepared["close_to_ma_7"] = prepared["close"] / prepared["ma_7"] - 1.0
    prepared["close_to_ma_21"] = prepared["close"] / prepared["ma_21"] - 1.0
    prepared["close_to_ma_63"] = prepared["close"] / prepared["ma_63"] - 1.0
    prepared["close_to_ma_252"] = prepared["close"] / prepared["ma_252"] - 1.0
    prepared["ma_7_to_ma_21"] = prepared["ma_7"] / prepared["ma_21"] - 1.0
    prepared["ma_21_to_ma_63"] = prepared["ma_21"] / prepared["ma_63"] - 1.0
    prepared["ma_63_to_ma_252"] = prepared["ma_63"] / prepared["ma_252"] - 1.0
    prepared["volatility_21_to_63"] = prepared["volatility_21d"] / prepared["volatility_63d"].clip(lower=1e-9)
    prepared["return_21d_to_volatility"] = prepared["return_21d"] / prepared["volatility_63d"].clip(lower=1e-9)

    for ticker in INITIAL_ASSETS:
        prepared[ticker_feature_name(ticker)] = (prepared["ticker"] == ticker).astype("float32")

    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    return prepared


def load_model_dataset() -> pd.DataFrame:
    initialize_database()
    dataset = read_technical_features()
    if dataset.empty:
        raise ValueError("No technical features found. Run generate-features first.")
    dataset = add_model_features(dataset)
    dataset = dataset.dropna(subset=RAW_FEATURE_COLUMNS + FEATURE_COLUMNS + [TARGET_NAME, TARGET_RETURN_NAME]).copy()
    dataset = dataset[dataset[TARGET_NAME].isin(LABELS)]
    return dataset


def build_model(input_dim: int) -> tf.keras.Model:
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required to train the direction model. Use Python <3.13.")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_split(dataset: pd.DataFrame, split: str, scaler: StandardScaler | None = None):
    split_data = dataset[dataset["time_split"] == split].copy()
    x_values = split_data[FEATURE_COLUMNS].astype("float32").to_numpy()
    y_values = split_data[TARGET_NAME].map(LABEL_TO_ID).astype("int32").to_numpy()

    if scaler is None:
        if StandardScaler is None:
            raise ModuleNotFoundError("scikit-learn is required to scale model features.")
        scaler = StandardScaler()
        x_values = scaler.fit_transform(x_values)
    else:
        x_values = scaler.transform(x_values)

    return split_data, x_values, y_values, scaler


def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    class_counts = pd.Series(y_train).value_counts().to_dict()
    total_rows = len(y_train)
    class_weights: dict[int, float] = {}
    for label_id in range(len(LABELS)):
        count = class_counts.get(label_id, 0)
        class_weights[label_id] = total_rows / (len(LABELS) * count) if count else 1.0
    return class_weights


def build_predictions(
    model: tf.keras.Model,
    dataset: pd.DataFrame,
    scaler: StandardScaler,
    run_id: str,
    calibration: dict | None = None,
) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        split_data, x_values, _y_values, _scaler = prepare_split(dataset, split, scaler)
        raw_probabilities = model.predict(x_values, verbose=0)
        probabilities = calibrate_probabilities(raw_probabilities, calibration)
        predicted_ids = probabilities.argmax(axis=1)
        frame = split_data[["ticker", "date", "time_split", TARGET_NAME, TARGET_RETURN_NAME]].copy()
        frame["run_id"] = run_id
        frame["target_name"] = TARGET_NAME
        frame["actual_direction"] = frame[TARGET_NAME]
        frame["predicted_direction"] = [ID_TO_LABEL[int(label_id)] for label_id in predicted_ids]
        frame["probability_down"] = probabilities[:, LABEL_TO_ID["down"]]
        frame["probability_sideways"] = probabilities[:, LABEL_TO_ID["sideways"]]
        frame["probability_up"] = probabilities[:, LABEL_TO_ID["up"]]
        frame["target_return"] = frame[TARGET_RETURN_NAME]
        prediction_frames.append(frame)
    return pd.concat(prediction_frames, ignore_index=True)


def train_tensorflow_direction_model(epochs: int = 40, batch_size: int = 64) -> dict:
    if tf is None or accuracy_score is None:
        raise ModuleNotFoundError("TensorFlow and scikit-learn are required to train the direction model. Use Python <3.13.")
    tf.keras.utils.set_random_seed(42)
    dataset = load_model_dataset()
    train_data, x_train, y_train, scaler = prepare_split(dataset, "train")
    validation_data, x_validation, y_validation, _scaler = prepare_split(dataset, "validation", scaler)
    test_data, x_test, y_test, _scaler = prepare_split(dataset, "test", scaler)

    model = build_model(input_dim=x_train.shape[1])
    class_weights = calculate_class_weights(y_train)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    raw_validation_probabilities = model.predict(x_validation, verbose=0)
    raw_test_probabilities = model.predict(x_test, verbose=0)
    probability_calibration = fit_temperature_scaler(raw_validation_probabilities, y_validation)
    validation_probabilities = calibrate_probabilities(raw_validation_probabilities, probability_calibration)
    test_probabilities = calibrate_probabilities(raw_test_probabilities, probability_calibration)
    validation_predictions = validation_probabilities.argmax(axis=1)
    test_predictions = test_probabilities.argmax(axis=1)
    validation_accuracy = float(accuracy_score(y_validation, validation_predictions))
    test_accuracy = float(accuracy_score(y_test, test_predictions))
    class_expected_returns = calculate_class_expected_returns(train_data)

    run_id = f"tf_direction_7d_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
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
        "class_weights": class_weights,
        "probability_calibration": probability_calibration,
        "class_expected_returns": class_expected_returns,
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "target_name": TARGET_NAME,
        "labels": LABELS,
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "class_weights": class_weights,
        "probability_calibration": probability_calibration,
        "class_expected_returns": class_expected_returns,
        "epochs_ran": len(history.history["loss"]),
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)),
        "test_rows": int(len(test_data)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    predictions = build_predictions(model, dataset, scaler, run_id, calibration=probability_calibration)
    run_record = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "target_name": TARGET_NAME,
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)),
        "test_rows": int(len(test_data)),
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "artifact_path": str(model_dir),
        "metadata_json": json.dumps(metadata),
    }
    save_model_run(run_record, predictions)
    return metadata