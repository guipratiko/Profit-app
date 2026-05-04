import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.database import get_feature_counts, get_price_counts
from app.data.market_data import update_all_prices
from app.features.technical import generate_technical_features


def assert_not_empty(label: str, value: int) -> None:
    if value <= 0:
        raise AssertionError(f"Expected {label} to be greater than zero")


def main() -> None:
    updated_rows = update_all_prices(period="10y")
    missing_tickers = [ticker for ticker, rows in updated_rows.items() if rows == 0]
    if missing_tickers:
        raise AssertionError(f"No OHLCV rows downloaded for: {missing_tickers}")

    feature_rows = generate_technical_features()
    assert_not_empty("technical feature rows", feature_rows)

    price_summary = get_price_counts()
    feature_summary = get_feature_counts()

    if len(price_summary) != 7:
        raise AssertionError(f"Expected 7 tickers in price summary, got {len(price_summary)}")
    if feature_summary.empty:
        raise AssertionError("Feature summary is empty")

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(feature_summary["time_split"].unique())
    if not expected_splits.issubset(actual_splits):
        raise AssertionError(f"Missing splits: {expected_splits - actual_splits}")

    print("E2E alpha pipeline passed")
    print("Price summary:")
    print(price_summary.to_string(index=False))
    print("Feature summary:")
    print(feature_summary.to_string(index=False))


if __name__ == "__main__":
    main()