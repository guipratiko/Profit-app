import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.database import get_model_runs, get_news_events, get_operational_predictions, get_paper_trading_signals
from app.data.news_events import save_sample_news_events
from app.models.inference import run_current_inference
from app.trading.paper import generate_paper_trading_signals


def main() -> None:
    model_runs = get_model_runs()
    if model_runs.empty:
        raise AssertionError("Expected at least one trained model run before Sprint 5 E2E")

    run_id = str(model_runs.iloc[0]["run_id"])
    inference_result = run_current_inference(run_id=run_id)
    if inference_result["generated"] <= 0:
        raise AssertionError("Expected current operational predictions to be generated")
    operational_predictions = get_operational_predictions()
    if operational_predictions.empty:
        raise AssertionError("Operational prediction table is empty")

    paper_result = generate_paper_trading_signals(run_id=run_id)
    if paper_result["generated"] <= 0:
        raise AssertionError("Expected paper-trading signals to be generated")
    accepted_sources = {"operational_predictions", "operational_trade_outcomes"}
    if paper_result["signal_source"] not in accepted_sources:
        raise AssertionError(
            f"Expected paper signals to use a current operational source, got {paper_result['signal_source']}"
        )
    if paper_result["simulate_long"] + paper_result["no_operate"] != paper_result["generated"]:
        raise AssertionError("Paper decisions do not add up to generated signals")
    if "strategy_gate" not in paper_result:
        raise AssertionError("Expected paper trading result to include strategy gate metadata")
    if "passing_windows" not in paper_result["strategy_gate"]:
        raise AssertionError("Expected walk-forward metadata in paper strategy gate")

    paper_signals = get_paper_trading_signals()
    if paper_signals.empty:
        raise AssertionError("Paper-trading signal table is empty")
    required_decisions = set(paper_signals["decision"].unique())
    if not required_decisions.issubset({"simulate_long", "no_operate"}):
        raise AssertionError(f"Unexpected paper decision values: {required_decisions}")
    if (paper_signals["max_shares"] < 0).any():
        raise AssertionError("Position sizing produced negative shares")

    news_result = save_sample_news_events()
    if news_result["generated"] != 3:
        raise AssertionError("Expected three sample news events")
    news_events = get_news_events()
    if news_events.empty:
        raise AssertionError("News/event table is empty")
    sample_events = news_events[news_events["source"] == "sample"]
    if sample_events.empty:
        raise AssertionError("Sample news events were not stored")
    if (sample_events["published_at"].str.slice(0, 10) >= sample_events["aligned_trading_date"]).any():
        raise AssertionError("Expected after-close sample news to align to a later trading date")
    if not sample_events["normalized_text"].str.contains("BACEN|COPOM|MINERIO_DE_FERRO", regex=True).any():
        raise AssertionError("Expected normalized entities in sample news text")

    print("E2E paper trading/news pipeline passed")
    print("Paper result:")
    print(paper_result)
    print("Operational inference result:")
    print(inference_result)
    print("Latest paper signals:")
    print(paper_signals.head(7).to_string(index=False))
    print("Sample news events:")
    print(sample_events[["ticker", "published_at", "aligned_trading_date", "title"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
