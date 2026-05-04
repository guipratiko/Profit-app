"""End-to-end check for the operational trade-outcome model.

Trains the trade-outcome classifier/regressor on the existing OHLCV +
technical features, runs current inference, generates paper signals using the
new operational gate and asserts that the paper trading layer surfaces the
expressive ``operational_action`` field (ENTER_LONG / WATCHLIST / NO_TRADE).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.database import (
    get_model_runs,
    get_operational_trade_outcomes,
    get_paper_trading_signals,
    get_trade_outcome_runs,
)
from app.features.trade_outcomes import OUTCOME_LABELS, build_trade_outcome_dataset
from app.models.trade_outcome import (
    run_trade_outcome_inference,
    train_trade_outcome_model,
)
from app.trading.paper import (
    OPERATIONAL_ACTION_ENTER_LONG,
    OPERATIONAL_ACTION_NO_TRADE,
    OPERATIONAL_ACTION_WATCHLIST,
    generate_paper_trading_signals,
)


VALID_OPERATIONAL_ACTIONS = {
    OPERATIONAL_ACTION_ENTER_LONG,
    OPERATIONAL_ACTION_WATCHLIST,
    OPERATIONAL_ACTION_NO_TRADE,
}


def main() -> None:
    model_runs = get_model_runs()
    if model_runs.empty:
        raise AssertionError("Train a direction model before running trade outcome E2E")
    direction_run_id = str(model_runs.iloc[0]["run_id"])

    dataset = build_trade_outcome_dataset()
    if dataset.empty:
        raise AssertionError("Trade outcome dataset is empty; populate OHLCV/features")
    distinct_outcomes = set(dataset["trade_outcome"].unique())
    if not distinct_outcomes.issubset(set(OUTCOME_LABELS)):
        raise AssertionError(f"Unexpected trade outcome labels: {distinct_outcomes}")

    metadata = train_trade_outcome_model(max_iter=120)
    if metadata["train_rows"] <= 0 or metadata["test_rows"] <= 0:
        raise AssertionError("Trade outcome training did not produce all splits")
    runs = get_trade_outcome_runs()
    if runs.empty:
        raise AssertionError("Trade outcome run was not persisted")

    inference = run_trade_outcome_inference()
    if inference["generated"] <= 0:
        raise AssertionError("Trade outcome inference produced no rows")
    operational = get_operational_trade_outcomes()
    if operational.empty:
        raise AssertionError("operational_trade_outcomes table is empty")

    paper_result = generate_paper_trading_signals(run_id=direction_run_id)
    if paper_result["signal_source"] != "operational_trade_outcomes":
        raise AssertionError(
            f"Expected paper signals to consume trade outcomes, got {paper_result['signal_source']}"
        )
    actions = paper_result.get("operational_actions", {})
    if not actions:
        raise AssertionError("Expected operational_actions breakdown in paper result")
    if not set(actions.keys()).issubset(VALID_OPERATIONAL_ACTIONS):
        raise AssertionError(f"Unknown operational actions emitted: {actions}")

    signals = get_paper_trading_signals()
    latest_signals = signals[signals["run_id"] == direction_run_id]
    if latest_signals.empty:
        raise AssertionError("No paper signals stored for the direction run id")
    if "operational_action" not in latest_signals.columns:
        raise AssertionError("paper_trading_signals.operational_action column missing")
    persisted_actions = set(latest_signals["operational_action"].dropna().unique())
    if not persisted_actions:
        raise AssertionError("operational_action column was not populated")
    if not persisted_actions.issubset(
        VALID_OPERATIONAL_ACTIONS
        | {"LEGACY_SIMULATE_LONG", "LEGACY_NO_TRADE"}
    ):
        raise AssertionError(f"Persisted operational actions look malformed: {persisted_actions}")

    if (latest_signals["probability_win"].dropna() < 0).any():
        raise AssertionError("Negative probability_win persisted")
    if (latest_signals["probability_win"].dropna() > 1).any():
        raise AssertionError("probability_win above 1 persisted")

    print("Trade outcome E2E passed")
    print(f"trade_outcome_run: {metadata['run_id']}")
    print(f"validation_accuracy: {metadata['validation_accuracy']:.4f}")
    print(f"test_accuracy: {metadata['test_accuracy']:.4f}")
    print(f"simulated_test_avg_return: {metadata['simulated_test_avg_return']:.4f}")
    print(f"simulated_test_win_rate: {metadata['simulated_test_win_rate']:.4f}")
    print(f"signal_source: {paper_result['signal_source']}")
    print(f"operational_actions: {actions}")


if __name__ == "__main__":
    main()
