from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from app.data.database import (
    get_model_runs,
    initialize_database,
    read_ohlcv_prices,
    read_technical_features,
    save_operational_predictions,
)
from app.features.technical import build_current_technical_features
from app.models.tensorflow_direction import (
    FEATURE_COLUMNS,
    ID_TO_LABEL,
    LABELS,
    TARGET_NAME,
    TARGET_RETURN_NAME,
    add_model_features,
    calibrate_probabilities,
    calculate_class_expected_returns,
    estimate_expected_returns,
)
from app.models.registry import get_best_current_schema_model_run_id, is_current_feature_schema


INFERENCE_VERSION = "v1_current_ohlcv_no_future_targets"


def remove_unsupported_keras_config_keys(value):
    if isinstance(value, dict):
        for key in ["quantization_config", "renorm", "renorm_clipping", "renorm_momentum"]:
            value.pop(key, None)
        for nested_value in value.values():
            remove_unsupported_keras_config_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            remove_unsupported_keras_config_keys(nested_value)


def load_keras_model_compat(model_path: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path)
    except TypeError as error:
        if "quantization_config" not in str(error):
            raise

    with tempfile.TemporaryDirectory() as temp_dir:
        cleaned_path = Path(temp_dir) / model_path.name
        with zipfile.ZipFile(model_path, "r") as source_archive:
            with zipfile.ZipFile(cleaned_path, "w") as target_archive:
                for item in source_archive.infolist():
                    data = source_archive.read(item.filename)
                    if item.filename == "config.json":
                        config = json.loads(data.decode("utf-8"))
                        remove_unsupported_keras_config_keys(config)
                        data = json.dumps(config).encode("utf-8")
                    target_archive.writestr(item, data)
        return tf.keras.models.load_model(cleaned_path)


def load_scaler(scaler_payload: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_payload["mean"], dtype="float64")
    scaler.scale_ = np.array(scaler_payload["scale"], dtype="float64")
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler_payload["feature_columns"])
    return scaler


def get_model_artifact_dir(run_id: str) -> Path:
    model_runs = get_model_runs()
    selected = model_runs[model_runs["run_id"] == run_id]
    if selected.empty:
        raise ValueError(f"Model run not found: {run_id}")
    return Path(str(selected.iloc[0]["artifact_path"]))


def load_inference_artifacts(run_id: str) -> tuple[tf.keras.Model, StandardScaler, dict]:
    artifact_dir = get_model_artifact_dir(run_id)
    model_path = artifact_dir / "model.keras"
    scaler_path = artifact_dir / "scaler.json"
    if not model_path.exists() or not scaler_path.exists():
        raise ValueError(f"Missing model artifacts for run_id={run_id}: {artifact_dir}")

    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    if not is_current_feature_schema(str(artifact_dir)):
        raise ValueError(
            f"Model run {run_id} uses an older feature schema. Train a new model or choose a compatible run."
        )
    model = load_keras_model_compat(model_path)
    scaler = load_scaler(scaler_payload)
    return model, scaler, scaler_payload


def build_current_model_dataset() -> pd.DataFrame:
    prices = read_ohlcv_prices()
    current_features = build_current_technical_features(prices)
    if current_features.empty:
        return pd.DataFrame()
    dataset = add_model_features(current_features)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    return dataset.dropna(subset=FEATURE_COLUMNS).copy()


def load_class_expected_returns(scaler_payload: dict) -> dict[str, float]:
    artifact_returns = scaler_payload.get("class_expected_returns")
    if artifact_returns:
        return {label: float(artifact_returns.get(label, 0.0)) for label in LABELS}

    supervised_features = read_technical_features()
    if supervised_features.empty:
        return {label: 0.0 for label in LABELS}
    supervised_features = supervised_features.dropna(subset=[TARGET_NAME, TARGET_RETURN_NAME]).copy()
    supervised_features = supervised_features[supervised_features[TARGET_NAME].isin(LABELS)]
    if supervised_features.empty:
        return {label: 0.0 for label in LABELS}
    return calculate_class_expected_returns(supervised_features)


def build_operational_predictions(
    current_dataset: pd.DataFrame,
    model: tf.keras.Model,
    scaler: StandardScaler,
    scaler_payload: dict,
    run_id: str,
) -> pd.DataFrame:
    if current_dataset.empty:
        return pd.DataFrame()

    x_values = current_dataset[FEATURE_COLUMNS].astype("float32").to_numpy()
    x_scaled = scaler.transform(x_values)
    raw_probabilities = model.predict(x_scaled, verbose=0)
    calibration = scaler_payload.get("probability_calibration")
    probabilities = calibrate_probabilities(raw_probabilities, calibration)
    predicted_ids = probabilities.argmax(axis=1)
    class_expected_returns = load_class_expected_returns(scaler_payload)
    expected_returns = estimate_expected_returns(probabilities, class_expected_returns)

    predictions = current_dataset[["ticker", "date"]].copy()
    predictions["run_id"] = run_id
    predictions["target_name"] = TARGET_NAME
    predictions["predicted_direction"] = [ID_TO_LABEL[int(label_id)] for label_id in predicted_ids]
    predictions["probability_down"] = probabilities[:, 0]
    predictions["probability_sideways"] = probabilities[:, 1]
    predictions["probability_up"] = probabilities[:, 2]
    predictions["raw_probability_down"] = raw_probabilities[:, 0]
    predictions["raw_probability_sideways"] = raw_probabilities[:, 1]
    predictions["raw_probability_up"] = raw_probabilities[:, 2]
    predictions["expected_return"] = expected_returns
    predictions["calibration_method"] = (calibration or {}).get("method", "none")
    predictions["inference_version"] = INFERENCE_VERSION
    return predictions


def run_current_inference(run_id: str | None = None) -> dict:
    initialize_database()
    selected_run_id = run_id or get_best_current_schema_model_run_id()
    model, scaler, scaler_payload = load_inference_artifacts(selected_run_id)
    current_dataset = build_current_model_dataset()
    predictions = build_operational_predictions(
        current_dataset=current_dataset,
        model=model,
        scaler=scaler,
        scaler_payload=scaler_payload,
        run_id=selected_run_id,
    )
    inserted = save_operational_predictions(predictions)
    latest_date = None if predictions.empty else str(predictions["date"].max())
    directions = {} if predictions.empty else predictions["predicted_direction"].value_counts().to_dict()
    return {
        "run_id": selected_run_id,
        "generated": int(len(predictions)),
        "inserted": int(inserted),
        "latest_date": latest_date,
        "directions": directions,
        "inference_version": INFERENCE_VERSION,
    }