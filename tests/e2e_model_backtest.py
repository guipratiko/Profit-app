import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backtesting.strategy import (
    run_probability_backtest,
    run_validation_selected_backtest,
    run_walk_forward_backtest,
)
from app.data.database import get_backtest_runs, get_model_runs
from app.models.tensorflow_direction import train_tensorflow_direction_model


def main() -> None:
    metadata = train_tensorflow_direction_model(epochs=12, batch_size=64)
    if metadata["test_rows"] <= 0:
        raise AssertionError("Expected test rows for trained model")

    backtest = run_probability_backtest(run_id=metadata["run_id"])
    if backtest["trades"] <= 0:
        raise AssertionError("Expected at least one backtest trade")

    optimized_backtest = run_validation_selected_backtest(run_id=metadata["run_id"])
    if optimized_backtest["threshold"] <= 0:
        raise AssertionError("Expected optimized threshold to be positive")
    if "strategy_gate_reason" not in optimized_backtest:
        raise AssertionError("Expected strategy gate metadata in optimized backtest")

    walk_forward = run_walk_forward_backtest(run_id=metadata["run_id"])
    if walk_forward["total_windows"] <= 0:
        raise AssertionError("Expected walk-forward windows")
    if not walk_forward["per_ticker"]:
        raise AssertionError("Expected per-ticker walk-forward report")
    if "strategy_gate_reason" not in walk_forward:
        raise AssertionError("Expected walk-forward strategy gate reason")

    model_runs = get_model_runs()
    backtest_runs = get_backtest_runs()
    if model_runs.empty:
        raise AssertionError("Model run table is empty")
    if backtest_runs.empty:
        raise AssertionError("Backtest run table is empty")

    print("E2E model/backtest pipeline passed")
    print("Latest model run:")
    print(model_runs.head(1).to_string(index=False))
    print("Latest backtest run:")
    print(backtest_runs.head(1).to_string(index=False))
    print("Optimized backtest:")
    print(optimized_backtest)
    print("Walk-forward backtest:")
    print(walk_forward)


if __name__ == "__main__":
    main()