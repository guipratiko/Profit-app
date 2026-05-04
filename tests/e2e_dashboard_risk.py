import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.api import app
from app.trading.risk_advisor import audit_paper_portfolio, evaluate_position


def main() -> None:
    sample_position = {
        "position_id": "pos_unit_sample",
        "ticker": "PETR4.SA",
        "entry_price": 100.0,
        "stop_loss": 94.0,
        "partial_target": 108.0,
        "target_price": 115.0,
    }
    stop_eval = evaluate_position(sample_position, current_price=93.5, evaluated_at="2026-05-03T12:00:00")
    if stop_eval["action"] != "close_position" or stop_eval["reason"] != "stop_loss_reached":
        raise AssertionError(f"Expected stop loss close action, got {stop_eval}")
    target_eval = evaluate_position(sample_position, current_price=116.0, evaluated_at="2026-05-03T12:00:00")
    if target_eval["action"] != "close_position" or target_eval["reason"] != "target_price_reached":
        raise AssertionError(f"Expected target close action, got {target_eval}")
    partial_eval = evaluate_position(sample_position, current_price=109.0, evaluated_at="2026-05-03T12:00:00")
    if partial_eval["action"] != "take_partial_profit":
        raise AssertionError(f"Expected partial profit action, got {partial_eval}")

    audit_result = audit_paper_portfolio()
    if "risk_advisor_version" not in audit_result:
        raise AssertionError("Risk audit did not return version metadata")
    if audit_result["opened_positions"] < 0 or audit_result["evaluated_positions"] < 0:
        raise AssertionError("Risk audit counters must be non-negative")

    client = TestClient(app)
    dashboard = client.get("/")
    if dashboard.status_code != 200 or "Profit App Alpha" not in dashboard.text:
        raise AssertionError("Dashboard route failed")
    positions = client.get("/portfolio/positions")
    if positions.status_code != 200 or "positions" not in positions.json():
        raise AssertionError("Portfolio positions endpoint failed")
    alerts = client.get("/portfolio/alerts")
    if alerts.status_code != 200 or "alerts" not in alerts.json():
        raise AssertionError("Portfolio alerts endpoint failed")
    audit = client.post("/portfolio/audit")
    if audit.status_code != 200 or "risk_advisor_version" not in audit.json():
        raise AssertionError("Portfolio audit endpoint failed")

    print("E2E dashboard/risk advisor pipeline passed")
    print("Audit result:")
    print(audit_result)
    print("Rule checks:")
    print({"stop": stop_eval["reason"], "partial": partial_eval["reason"], "target": target_eval["reason"]})


if __name__ == "__main__":
    main()
