from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    joblib = None
try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    tf = None
from sklearn.preprocessing import StandardScaler

from app.config import resolve_artifact_dir
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


def load_keras_model_compat(model_path: Path):
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required to load Keras model artifacts.")
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
    return resolve_artifact_dir(str(selected.iloc[0]["artifact_path"]))


def load_inference_artifacts(run_id: str) -> tuple[object, StandardScaler, dict]:
    artifact_dir = get_model_artifact_dir(run_id)
    model_path = artifact_dir / "model.keras"
    sklearn_model_path = artifact_dir / "model.joblib"
    scaler_path = artifact_dir / "scaler.json"
    if not scaler_path.exists() or (not model_path.exists() and not sklearn_model_path.exists()):
        raise ValueError(f"Missing model artifacts for run_id={run_id}: {artifact_dir}")

    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    if not is_current_feature_schema(str(artifact_dir)):
        raise ValueError(
            f"Model run {run_id} uses an older feature schema. Train a new model or choose a compatible run."
        )
    if sklearn_model_path.exists():
        if joblib is None:
            raise ModuleNotFoundError("joblib is required to load sklearn model artifacts.")
        model = joblib.load(sklearn_model_path)
    else:
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
    model: object,
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


# --- Binary (enter_long_7d) inference path --------------------------------
#
# The 3-class direction model historically wrote three softmax columns into
# operational_predictions. The new binary model (target_enter_long_7d) emits
# a single calibrated probability of "forward 7d return clears cost+vol band".
# To keep downstream consumers (fusion, paper signals, frontend) working, we
# re-shape the binary output as (down=1-p, sideways=0, up=p) and tag the
# predictions with target_name="target_enter_long_7d" so callers that care
# about the semantic difference can branch on it.

BINARY_INFERENCE_VERSION = "v1_binary_enter_long_p60"


def _read_run_metadata(run_id: str) -> dict:
    metadata_path = get_model_artifact_dir(run_id) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _is_binary_run(run_id: str) -> bool:
    metadata = _read_run_metadata(run_id)
    target_name = str(metadata.get("target_name", ""))
    if target_name == "target_enter_long_7d":
        return True
    return bool(run_id.startswith("tf_binary_"))


def _apply_isotonic_payload(probabilities: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Recreate the isotonic mapping serialised by tensorflow_binary."""
    if not calibration or calibration.get("method") != "isotonic":
        return probabilities.flatten()
    x_thresholds = np.asarray(calibration.get("x_thresholds", []), dtype="float64")
    y_thresholds = np.asarray(calibration.get("y_thresholds", []), dtype="float64")
    if x_thresholds.size == 0:
        return probabilities.flatten()
    return np.clip(np.interp(probabilities.flatten(), x_thresholds, y_thresholds), 0.0, 1.0)


def _predict_binary_probability(model: object, x_scaled: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_scaled)
        probabilities = np.asarray(probabilities, dtype="float64")
        if probabilities.ndim == 1:
            return probabilities.flatten()
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]
    predict = getattr(model, "predict")
    return np.asarray(predict(x_scaled, verbose=0), dtype="float64").flatten()


def build_binary_operational_predictions(
    current_dataset: pd.DataFrame,
    model: object,
    scaler: StandardScaler,
    scaler_payload: dict,
    run_id: str,
) -> pd.DataFrame:
    if current_dataset.empty:
        return pd.DataFrame()

    feature_columns = list(scaler_payload.get("feature_columns") or FEATURE_COLUMNS)
    x_values = current_dataset[feature_columns].astype("float32").to_numpy()
    x_scaled = scaler.transform(x_values)
    raw_p_up = _predict_binary_probability(model, x_scaled)
    calibration = scaler_payload.get("probability_calibration")
    p_up = _apply_isotonic_payload(raw_p_up, calibration)

    # Conformal interval: payload may live under either "conformal_calibration"
    # (current trainings) or be missing for older runs. q ~ (1 - alpha)-quantile of
    # nonconformity scores on the validation set.
    conformal_payload = scaler_payload.get("conformal_calibration") or {}
    quantile_value = conformal_payload.get("quantile")
    conformal_q = float(quantile_value) if quantile_value is not None else None
    conformal_alpha = float(conformal_payload.get("alpha", 0.10))
    if conformal_q is not None:
        interval_low = np.clip(p_up - conformal_q, 0.0, 1.0)
        interval_high = np.clip(p_up + conformal_q, 0.0, 1.0)
    else:
        interval_low = p_up
        interval_high = p_up

    predictions = current_dataset[["ticker", "date"]].copy()
    predictions["run_id"] = run_id
    predictions["target_name"] = "target_enter_long_7d"
    predictions["predicted_direction"] = np.where(p_up >= 0.5, "up", "down")
    predictions["probability_down"] = 1.0 - p_up
    predictions["probability_sideways"] = 0.0
    predictions["probability_up"] = p_up
    predictions["raw_probability_down"] = 1.0 - raw_p_up
    predictions["raw_probability_sideways"] = 0.0
    predictions["raw_probability_up"] = raw_p_up
    predictions["conformal_interval_low"] = interval_low
    predictions["conformal_interval_high"] = interval_high
    predictions["conformal_alpha"] = conformal_alpha
    predictions["conformal_quantile"] = conformal_q if conformal_q is not None else np.nan
    # Binary head emits a calibrated probability, not a forecasted return.
    # Expected return is computed downstream by fusion/paper using the
    # configured payoff model; keep the column populated with 0.0 to satisfy
    # the operational_predictions schema.
    predictions["expected_return"] = 0.0
    predictions["calibration_method"] = (calibration or {}).get("method", "none")
    predictions["inference_version"] = BINARY_INFERENCE_VERSION
    return predictions


def run_current_inference(run_id: str | None = None) -> dict:
    initialize_database()
    selected_run_id = run_id or get_best_current_schema_model_run_id()
    model, scaler, scaler_payload = load_inference_artifacts(selected_run_id)
    current_dataset = build_current_model_dataset()

    if _is_binary_run(selected_run_id):
        predictions = build_binary_operational_predictions(
            current_dataset=current_dataset,
            model=model,
            scaler=scaler,
            scaler_payload=scaler_payload,
            run_id=selected_run_id,
        )
        version = BINARY_INFERENCE_VERSION
    else:
        predictions = build_operational_predictions(
            current_dataset=current_dataset,
            model=model,
            scaler=scaler,
            scaler_payload=scaler_payload,
            run_id=selected_run_id,
        )
        version = INFERENCE_VERSION

    inserted = save_operational_predictions(predictions)
    latest_date = None if predictions.empty else str(predictions["date"].max())
    directions = {} if predictions.empty else predictions["predicted_direction"].value_counts().to_dict()
    return {
        "run_id": selected_run_id,
        "generated": int(len(predictions)),
        "inserted": int(inserted),
        "latest_date": latest_date,
        "directions": directions,
        "inference_version": version,
    }