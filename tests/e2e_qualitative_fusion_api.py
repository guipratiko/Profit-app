import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.api import app
from app.data.database import (
    get_fusion_predictions,
    get_latest_model_run_id,
    get_qualitative_features,
    read_model_predictions,
    read_ohlcv_prices,
    save_news_events,
)
from app.data.news_events import build_news_event, save_sample_news_events
from app.models.fusion import run_fusion_predictions
from app.models.pytorch_sentiment import evaluate_manual_sample, generate_qualitative_features


def main() -> None:
    run_id = get_latest_model_run_id()
    predictions = read_model_predictions(run_id, split="test")
    if predictions.empty:
        raise AssertionError("Expected test predictions before Sprint 8 fusion")

    latest_prediction = predictions.sort_values("date").groupby("ticker", as_index=False).tail(1).iloc[0]
    ticker = str(latest_prediction["ticker"])
    signal_date = str(latest_prediction["date"])
    calendar = [
        date.normalize()
        for date in pd.to_datetime(read_ohlcv_prices()["date"]).drop_duplicates().sort_values()
    ]
    aligned_event = build_news_event(
        ticker=ticker,
        title="Lucro forte e dividendos positivos sustentam expectativa de alta",
        body="Evento qualitativo de teste para validar sentimento positivo no pipeline PyTorch MVP.",
        published_at=f"{signal_date}T10:00:00",
        trading_calendar=calendar,
        source="e2e_sprint_7_9",
    )
    save_news_events([aligned_event])
    save_sample_news_events()

    sentiment_result = generate_qualitative_features()
    if sentiment_result["generated"] <= 0:
        raise AssertionError("Expected qualitative sentiment features to be generated")
    manual_eval = evaluate_manual_sample()
    if not all(item["passed"] for item in manual_eval):
        raise AssertionError(f"Manual sentiment sanity check failed: {manual_eval}")

    qualitative_features = get_qualitative_features()
    if qualitative_features.empty:
        raise AssertionError("Qualitative feature table is empty")
    matching_context = qualitative_features[
        (qualitative_features["ticker"] == ticker)
        & (qualitative_features["aligned_trading_date"] == signal_date)
    ]
    if matching_context.empty:
        raise AssertionError("Expected qualitative context aligned to the latest technical signal date")

    fusion_result = run_fusion_predictions(run_id=run_id)
    if fusion_result["generated"] <= 0:
        raise AssertionError("Expected fused predictions to be generated")
    fusion_predictions = get_fusion_predictions()
    if fusion_predictions.empty:
        raise AssertionError("Fusion prediction table is empty")
    ticker_fusion = fusion_predictions[
        (fusion_predictions["ticker"] == ticker)
        & (fusion_predictions["run_id"] == run_id)
    ].head(1)
    if ticker_fusion.empty:
        raise AssertionError("Expected a fused prediction for the ticker with aligned context")
    if int(ticker_fusion.iloc[0]["qualitative_event_count"]) <= 0:
        raise AssertionError("Expected fused prediction to include qualitative event count")

    client = TestClient(app)
    health = client.get("/health")
    if health.status_code != 200 or health.json()["status"] != "ok":
        raise AssertionError("Health endpoint failed")
    assets = client.get("/assets")
    if assets.status_code != 200 or not assets.json()["assets"]:
        raise AssertionError("Assets endpoint failed")
    prediction_response = client.get(f"/predictions/{ticker}")
    if prediction_response.status_code != 200:
        raise AssertionError(f"Prediction endpoint failed: {prediction_response.text}")
    explanation_response = client.get(f"/predictions/{ticker}/explanation")
    if explanation_response.status_code != 200:
        raise AssertionError(f"Explanation endpoint failed: {explanation_response.text}")
    metrics_response = client.get("/paper/metrics")
    if metrics_response.status_code != 200:
        raise AssertionError("Paper metrics endpoint failed")

    print("E2E qualitative/fusion/API pipeline passed")
    print("Sentiment result:")
    print(sentiment_result)
    print("Fusion result:")
    print(fusion_result)
    print("API ticker:", ticker)


if __name__ == "__main__":
    main()
