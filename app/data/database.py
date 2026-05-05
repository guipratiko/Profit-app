import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.config import DATABASE_PATH, INITIAL_ASSETS, STORAGE_DIR


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ohlcv_prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    source TEXT NOT NULL,
    downloaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS technical_features (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL NOT NULL,
    volume INTEGER,
    return_1d REAL,
    return_5d REAL,
    return_21d REAL,
    ma_7 REAL,
    ma_21 REAL,
    ma_63 REAL,
    ma_252 REAL,
    volatility_21d REAL,
    volatility_63d REAL,
    volume_ratio_21d REAL,
    drawdown_252d REAL,
    rsi_14 REAL,
    target_return_7d REAL,
    target_return_3m REAL,
    target_return_1y REAL,
    target_direction_7d TEXT,
    target_direction_3m TEXT,
    target_direction_1y TEXT,
    time_split TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    target_name TEXT NOT NULL,
    trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    train_rows INTEGER NOT NULL,
    validation_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    validation_accuracy REAL,
    test_accuracy REAL,
    artifact_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_predictions (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    time_split TEXT NOT NULL,
    target_name TEXT NOT NULL,
    actual_direction TEXT NOT NULL,
    predicted_direction TEXT NOT NULL,
    probability_down REAL NOT NULL,
    probability_sideways REAL NOT NULL,
    probability_up REAL NOT NULL,
    target_return REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, ticker, date),
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS operational_predictions (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    target_name TEXT NOT NULL,
    predicted_direction TEXT NOT NULL,
    probability_down REAL NOT NULL,
    probability_sideways REAL NOT NULL,
    probability_up REAL NOT NULL,
    raw_probability_down REAL NOT NULL,
    raw_probability_sideways REAL NOT NULL,
    raw_probability_up REAL NOT NULL,
    expected_return REAL NOT NULL,
    calibration_method TEXT NOT NULL,
    inference_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, ticker, date),
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    threshold REAL NOT NULL,
    holding_days INTEGER NOT NULL,
    cost_per_trade REAL NOT NULL,
    trades INTEGER NOT NULL,
    win_rate REAL,
    cumulative_return REAL,
    average_trade_return REAL,
    max_drawdown REAL,
    buy_hold_return_avg REAL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    backtest_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    exit_date TEXT NOT NULL,
    probability_up REAL NOT NULL,
    gross_return REAL NOT NULL,
    net_return REAL NOT NULL,
    PRIMARY KEY (backtest_id, ticker, entry_date),
    FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS paper_trading_signals (
    signal_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    horizon TEXT NOT NULL,
    decision TEXT NOT NULL,
    block_reason TEXT,
    confidence REAL NOT NULL,
    probability_up REAL NOT NULL,
    expected_return REAL NOT NULL,
    net_expected_return REAL NOT NULL,
    reference_price REAL NOT NULL,
    suggested_entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    partial_target REAL NOT NULL,
    target_price REAL NOT NULL,
    max_position_value REAL NOT NULL,
    max_shares INTEGER NOT NULL,
    risk_amount REAL NOT NULL,
    reward_risk_ratio REAL NOT NULL,
    model_run_id TEXT NOT NULL,
    thesis_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS news_events (
    event_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    normalized_text TEXT NOT NULL,
    published_at TEXT NOT NULL,
    aligned_trading_date TEXT NOT NULL,
    url TEXT,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS qualitative_features (
    feature_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    aligned_trading_date TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    sentiment_score REAL NOT NULL,
    sentiment_label TEXT NOT NULL,
    positive_score REAL NOT NULL,
    negative_score REAL NOT NULL,
    neutral_score REAL NOT NULL,
    embedding_json TEXT NOT NULL,
    source_event_ids_json TEXT NOT NULL,
    model_name TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, aligned_trading_date, model_name),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS fusion_predictions (
    fusion_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    horizon TEXT NOT NULL,
    fusion_version TEXT NOT NULL,
    technical_probability_up REAL NOT NULL,
    technical_confidence REAL NOT NULL,
    sentiment_score REAL NOT NULL,
    qualitative_event_count INTEGER NOT NULL,
    fused_score REAL NOT NULL,
    fused_direction TEXT NOT NULL,
    explanation_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, ticker, signal_date, horizon, fusion_version),
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    position_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    horizon TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    partial_target REAL NOT NULL,
    target_price REAL NOT NULL,
    current_price REAL NOT NULL,
    status TEXT NOT NULL,
    unrealized_return REAL NOT NULL,
    realized_return REAL,
    last_evaluated_at TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (signal_id),
    FOREIGN KEY (signal_id) REFERENCES paper_trading_signals(signal_id),
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS real_positions (
    position_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_at TEXT NOT NULL,
    cost_basis REAL NOT NULL,
    current_price REAL,
    last_updated_at TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS risk_alerts (
    alert_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    action TEXT NOT NULL,
    severity TEXT NOT NULL,
    reason TEXT NOT NULL,
    current_price REAL NOT NULL,
    unrealized_return REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES paper_positions(position_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

CREATE TABLE IF NOT EXISTS trade_outcome_runs (
    run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    min_reward_risk REAL NOT NULL,
    cost_per_trade REAL NOT NULL,
    spread REAL NOT NULL,
    slippage REAL NOT NULL,
    trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    train_rows INTEGER NOT NULL,
    validation_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    validation_accuracy REAL,
    validation_log_loss REAL,
    test_accuracy REAL,
    test_log_loss REAL,
    simulated_test_trades INTEGER,
    simulated_test_avg_return REAL,
    simulated_test_win_rate REAL,
    artifact_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operational_trade_outcomes (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    probability_win REAL NOT NULL,
    probability_loss REAL NOT NULL,
    probability_timeout REAL NOT NULL,
    expected_return REAL NOT NULL,
    stop_distance REAL NOT NULL,
    target_distance REAL NOT NULL,
    execution_drag REAL NOT NULL,
    inference_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, ticker, date),
    FOREIGN KEY (run_id) REFERENCES trade_outcome_runs(run_id),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);
"""


PAPER_SIGNAL_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("operational_action", "TEXT"),
    ("trade_outcome_run_id", "TEXT"),
    ("probability_win", "REAL"),
    ("probability_loss", "REAL"),
    ("probability_timeout", "REAL"),
)


def _apply_paper_signal_migrations(connection: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in connection.execute("PRAGMA table_info(paper_trading_signals)").fetchall()
    }
    for column, column_type in PAPER_SIGNAL_MIGRATIONS:
        if column not in existing:
            connection.execute(
                f"ALTER TABLE paper_trading_signals ADD COLUMN {column} {column_type}"
            )


def get_connection(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30.0)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        connection.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    return connection


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    with get_connection(database_path) as connection:
        connection.executescript(SCHEMA_SQL)
        _apply_paper_signal_migrations(connection)
        connection.executemany(
            "INSERT OR IGNORE INTO assets (ticker, name) VALUES (?, ?)",
            INITIAL_ASSETS.items(),
        )


def upsert_ohlcv_prices(prices: pd.DataFrame, database_path: Path = DATABASE_PATH) -> int:
    if prices.empty:
        return 0

    required_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
    ]
    records = prices[required_columns].to_records(index=False).tolist()

    with get_connection(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO ohlcv_prices (
                ticker, date, open, high, low, close, adj_close, volume, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                adj_close = excluded.adj_close,
                volume = excluded.volume,
                source = excluded.source,
                downloaded_at = CURRENT_TIMESTAMP
            """,
            records,
        )
        return len(records)


def get_price_counts(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                ticker,
                COUNT(*) AS rows,
                MIN(date) AS first_date,
                MAX(date) AS last_date
            FROM ohlcv_prices
            GROUP BY ticker
            ORDER BY ticker
            """,
            connection,
        )


def read_ohlcv_prices(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                adj_close,
                volume
            FROM ohlcv_prices
            ORDER BY ticker, date
            """,
            connection,
        )


def replace_technical_features(
    features: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if features.empty:
        return 0

    required_columns = [
        "ticker",
        "date",
        "close",
        "volume",
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
        "target_return_7d",
        "target_return_3m",
        "target_return_1y",
        "target_direction_7d",
        "target_direction_3m",
        "target_direction_1y",
        "time_split",
    ]
    records = features[required_columns].where(pd.notna(features), None).to_records(index=False).tolist()

    with get_connection(database_path) as connection:
        connection.execute("DELETE FROM technical_features")
        connection.executemany(
            """
            INSERT INTO technical_features (
                ticker,
                date,
                close,
                volume,
                return_1d,
                return_5d,
                return_21d,
                ma_7,
                ma_21,
                ma_63,
                ma_252,
                volatility_21d,
                volatility_63d,
                volume_ratio_21d,
                drawdown_252d,
                rsi_14,
                target_return_7d,
                target_return_3m,
                target_return_1y,
                target_direction_7d,
                target_direction_3m,
                target_direction_1y,
                time_split
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        return len(records)


def get_feature_counts(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                ticker,
                time_split,
                COUNT(*) AS rows,
                MIN(date) AS first_date,
                MAX(date) AS last_date
            FROM technical_features
            GROUP BY ticker, time_split
            ORDER BY ticker, time_split
            """,
            connection,
        )


def read_technical_features(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM technical_features
            ORDER BY ticker, date
            """,
            connection,
        )


def save_model_run(
    run: dict,
    predictions: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> None:
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO model_runs (
                run_id,
                model_name,
                target_name,
                train_rows,
                validation_rows,
                test_rows,
                validation_accuracy,
                test_accuracy,
                artifact_path,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run["run_id"],
                run["model_name"],
                run["target_name"],
                run["train_rows"],
                run["validation_rows"],
                run["test_rows"],
                run["validation_accuracy"],
                run["test_accuracy"],
                run["artifact_path"],
                run["metadata_json"],
            ),
        )
        connection.execute("DELETE FROM model_predictions WHERE run_id = ?", (run["run_id"],))
        prediction_records = predictions[
            [
                "run_id",
                "ticker",
                "date",
                "time_split",
                "target_name",
                "actual_direction",
                "predicted_direction",
                "probability_down",
                "probability_sideways",
                "probability_up",
                "target_return",
            ]
        ].to_records(index=False).tolist()
        connection.executemany(
            """
            INSERT INTO model_predictions (
                run_id,
                ticker,
                date,
                time_split,
                target_name,
                actual_direction,
                predicted_direction,
                probability_down,
                probability_sideways,
                probability_up,
                target_return
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            prediction_records,
        )


def get_model_runs(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                run_id,
                model_name,
                target_name,
                trained_at,
                train_rows,
                validation_rows,
                test_rows,
                validation_accuracy,
                test_accuracy,
                artifact_path
            FROM model_runs
            ORDER BY trained_at DESC
            """,
            connection,
        )


def get_latest_model_run_id(database_path: Path = DATABASE_PATH) -> str:
    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT run_id
            FROM model_runs
            ORDER BY trained_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise ValueError("No model runs found. Train a model first.")
    return str(row[0])


def get_best_model_run_id(database_path: Path = DATABASE_PATH) -> str:
    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT run_id
            FROM backtest_runs
            WHERE trades >= 1
            ORDER BY
                CASE WHEN cumulative_return > buy_hold_return_avg THEN 1 ELSE 0 END DESC,
                cumulative_return DESC,
                max_drawdown DESC,
                trades DESC,
                created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is not None:
        return str(row[0])
    return get_latest_model_run_id(database_path=database_path)


def read_model_predictions(
    run_id: str,
    split: str = "test",
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM model_predictions
            WHERE run_id = ? AND time_split = ?
            ORDER BY ticker, date
            """,
            connection,
            params=(run_id, split),
        )


def save_operational_predictions(
    predictions: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if predictions.empty:
        return 0

    required_columns = [
        "run_id",
        "ticker",
        "date",
        "target_name",
        "predicted_direction",
        "probability_down",
        "probability_sideways",
        "probability_up",
        "raw_probability_down",
        "raw_probability_sideways",
        "raw_probability_up",
        "expected_return",
        "calibration_method",
        "inference_version",
    ]
    records = predictions[required_columns].where(pd.notna(predictions), None).to_records(index=False).tolist()

    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO operational_predictions (
                run_id,
                ticker,
                date,
                target_name,
                predicted_direction,
                probability_down,
                probability_sideways,
                probability_up,
                raw_probability_down,
                raw_probability_sideways,
                raw_probability_up,
                expected_return,
                calibration_method,
                inference_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, ticker, date) DO UPDATE SET
                target_name = excluded.target_name,
                predicted_direction = excluded.predicted_direction,
                probability_down = excluded.probability_down,
                probability_sideways = excluded.probability_sideways,
                probability_up = excluded.probability_up,
                raw_probability_down = excluded.raw_probability_down,
                raw_probability_sideways = excluded.raw_probability_sideways,
                raw_probability_up = excluded.raw_probability_up,
                expected_return = excluded.expected_return,
                calibration_method = excluded.calibration_method,
                inference_version = excluded.inference_version,
                created_at = CURRENT_TIMESTAMP
            """,
            records,
        )
        return connection.total_changes - before


def read_operational_predictions(
    run_id: str,
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM operational_predictions
            WHERE run_id = ?
            ORDER BY ticker, date
            """,
            connection,
            params=(run_id,),
        )


def get_operational_predictions(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM operational_predictions
            ORDER BY created_at DESC, ticker
            """,
            connection,
        )


def save_backtest_run(
    backtest: dict,
    trades: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> None:
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO backtest_runs (
                backtest_id,
                run_id,
                threshold,
                holding_days,
                cost_per_trade,
                trades,
                win_rate,
                cumulative_return,
                average_trade_return,
                max_drawdown,
                buy_hold_return_avg,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                backtest["backtest_id"],
                backtest["run_id"],
                backtest["threshold"],
                backtest["holding_days"],
                backtest["cost_per_trade"],
                backtest["trades"],
                backtest["win_rate"],
                backtest["cumulative_return"],
                backtest["average_trade_return"],
                backtest["max_drawdown"],
                backtest["buy_hold_return_avg"],
                backtest["metadata_json"],
            ),
        )
        connection.execute("DELETE FROM backtest_trades WHERE backtest_id = ?", (backtest["backtest_id"],))
        if not trades.empty:
            trade_records = trades[
                [
                    "backtest_id",
                    "ticker",
                    "entry_date",
                    "exit_date",
                    "probability_up",
                    "gross_return",
                    "net_return",
                ]
            ].to_records(index=False).tolist()
            connection.executemany(
                """
                INSERT INTO backtest_trades (
                    backtest_id,
                    ticker,
                    entry_date,
                    exit_date,
                    probability_up,
                    gross_return,
                    net_return
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                trade_records,
            )


def get_backtest_runs(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                backtest_id,
                run_id,
                created_at,
                threshold,
                holding_days,
                cost_per_trade,
                trades,
                win_rate,
                cumulative_return,
                average_trade_return,
                max_drawdown,
                buy_hold_return_avg
            FROM backtest_runs
            ORDER BY created_at DESC
            """,
            connection,
        )


def save_paper_trading_signals(
    signals: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if signals.empty:
        return 0

    required_columns = [
        "signal_id",
        "run_id",
        "ticker",
        "signal_date",
        "horizon",
        "decision",
        "block_reason",
        "confidence",
        "probability_up",
        "expected_return",
        "net_expected_return",
        "reference_price",
        "suggested_entry",
        "stop_loss",
        "partial_target",
        "target_price",
        "max_position_value",
        "max_shares",
        "risk_amount",
        "reward_risk_ratio",
        "model_run_id",
        "thesis_json",
        "operational_action",
        "trade_outcome_run_id",
        "probability_win",
        "probability_loss",
        "probability_timeout",
    ]
    prepared = signals.copy()
    for column in required_columns:
        if column not in prepared.columns:
            prepared[column] = None
    records = prepared[required_columns].where(pd.notna(prepared), None).to_records(index=False).tolist()

    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO paper_trading_signals (
                signal_id,
                run_id,
                ticker,
                signal_date,
                horizon,
                decision,
                block_reason,
                confidence,
                probability_up,
                expected_return,
                net_expected_return,
                reference_price,
                suggested_entry,
                stop_loss,
                partial_target,
                target_price,
                max_position_value,
                max_shares,
                risk_amount,
                reward_risk_ratio,
                model_run_id,
                thesis_json,
                operational_action,
                trade_outcome_run_id,
                probability_win,
                probability_loss,
                probability_timeout
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                run_id = excluded.run_id,
                ticker = excluded.ticker,
                signal_date = excluded.signal_date,
                horizon = excluded.horizon,
                decision = excluded.decision,
                block_reason = excluded.block_reason,
                confidence = excluded.confidence,
                probability_up = excluded.probability_up,
                expected_return = excluded.expected_return,
                net_expected_return = excluded.net_expected_return,
                reference_price = excluded.reference_price,
                suggested_entry = excluded.suggested_entry,
                stop_loss = excluded.stop_loss,
                partial_target = excluded.partial_target,
                target_price = excluded.target_price,
                max_position_value = excluded.max_position_value,
                max_shares = excluded.max_shares,
                risk_amount = excluded.risk_amount,
                reward_risk_ratio = excluded.reward_risk_ratio,
                model_run_id = excluded.model_run_id,
                thesis_json = excluded.thesis_json,
                operational_action = excluded.operational_action,
                trade_outcome_run_id = excluded.trade_outcome_run_id,
                probability_win = excluded.probability_win,
                probability_loss = excluded.probability_loss,
                probability_timeout = excluded.probability_timeout
            """,
            records,
        )
        return connection.total_changes - before


def get_paper_trading_signals(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM paper_trading_signals
            ORDER BY created_at DESC, ticker
            """,
            connection,
        )


def save_news_events(
    events: Iterable[dict],
    database_path: Path = DATABASE_PATH,
) -> int:
    records = [
        (
            event["event_id"],
            event["ticker"],
            event["source"],
            event["title"],
            event.get("body"),
            event["normalized_text"],
            event["published_at"],
            event["aligned_trading_date"],
            event.get("url"),
            event["raw_json"],
        )
        for event in events
    ]
    if not records:
        return 0

    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO news_events (
                event_id,
                ticker,
                source,
                title,
                body,
                normalized_text,
                published_at,
                aligned_trading_date,
                url,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        return connection.total_changes - before


def get_news_events(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM news_events
            ORDER BY published_at DESC, ticker
            """,
            connection,
        )


def save_qualitative_features(
    features: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if features.empty:
        return 0

    required_columns = [
        "feature_id",
        "ticker",
        "aligned_trading_date",
        "event_count",
        "sentiment_score",
        "sentiment_label",
        "positive_score",
        "negative_score",
        "neutral_score",
        "embedding_json",
        "source_event_ids_json",
        "model_name",
        "metadata_json",
    ]
    records = features[required_columns].where(pd.notna(features), None).to_records(index=False).tolist()
    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO qualitative_features (
                feature_id,
                ticker,
                aligned_trading_date,
                event_count,
                sentiment_score,
                sentiment_label,
                positive_score,
                negative_score,
                neutral_score,
                embedding_json,
                source_event_ids_json,
                model_name,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, aligned_trading_date, model_name) DO UPDATE SET
                event_count = excluded.event_count,
                sentiment_score = excluded.sentiment_score,
                sentiment_label = excluded.sentiment_label,
                positive_score = excluded.positive_score,
                negative_score = excluded.negative_score,
                neutral_score = excluded.neutral_score,
                embedding_json = excluded.embedding_json,
                source_event_ids_json = excluded.source_event_ids_json,
                metadata_json = excluded.metadata_json,
                created_at = CURRENT_TIMESTAMP
            """,
            records,
        )
        return connection.total_changes - before


def get_qualitative_features(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM qualitative_features
            ORDER BY aligned_trading_date DESC, ticker
            """,
            connection,
        )


def save_fusion_predictions(
    predictions: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if predictions.empty:
        return 0

    required_columns = [
        "fusion_id",
        "run_id",
        "ticker",
        "signal_date",
        "horizon",
        "fusion_version",
        "technical_probability_up",
        "technical_confidence",
        "sentiment_score",
        "qualitative_event_count",
        "fused_score",
        "fused_direction",
        "explanation_json",
    ]
    records = predictions[required_columns].where(pd.notna(predictions), None).to_records(index=False).tolist()
    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO fusion_predictions (
                fusion_id,
                run_id,
                ticker,
                signal_date,
                horizon,
                fusion_version,
                technical_probability_up,
                technical_confidence,
                sentiment_score,
                qualitative_event_count,
                fused_score,
                fused_direction,
                explanation_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, ticker, signal_date, horizon, fusion_version) DO UPDATE SET
                technical_probability_up = excluded.technical_probability_up,
                technical_confidence = excluded.technical_confidence,
                sentiment_score = excluded.sentiment_score,
                qualitative_event_count = excluded.qualitative_event_count,
                fused_score = excluded.fused_score,
                fused_direction = excluded.fused_direction,
                explanation_json = excluded.explanation_json,
                created_at = CURRENT_TIMESTAMP
            """,
            records,
        )
        return connection.total_changes - before


def get_fusion_predictions(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM fusion_predictions
            ORDER BY created_at DESC, ticker
            """,
            connection,
        )


def save_paper_positions(
    positions: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if positions.empty:
        return 0

    required_columns = [
        "position_id",
        "signal_id",
        "run_id",
        "ticker",
        "opened_at",
        "horizon",
        "quantity",
        "entry_price",
        "stop_loss",
        "partial_target",
        "target_price",
        "current_price",
        "status",
        "unrealized_return",
        "realized_return",
        "last_evaluated_at",
        "metadata_json",
    ]
    records = positions[required_columns].where(pd.notna(positions), None).to_records(index=False).tolist()
    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO paper_positions (
                position_id,
                signal_id,
                run_id,
                ticker,
                opened_at,
                horizon,
                quantity,
                entry_price,
                stop_loss,
                partial_target,
                target_price,
                current_price,
                status,
                unrealized_return,
                realized_return,
                last_evaluated_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                current_price = excluded.current_price,
                status = excluded.status,
                unrealized_return = excluded.unrealized_return,
                realized_return = excluded.realized_return,
                last_evaluated_at = excluded.last_evaluated_at,
                metadata_json = excluded.metadata_json
            """,
            records,
        )
        return connection.total_changes - before


def get_paper_positions(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM paper_positions
            ORDER BY created_at DESC, ticker
            """,
            connection,
        )


def save_real_positions(
    positions: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if positions.empty:
        return 0

    required_columns = [
        "position_id",
        "ticker",
        "quantity",
        "entry_price",
        "entry_at",
        "cost_basis",
        "current_price",
        "last_updated_at",
        "notes",
    ]
    prepared = positions.copy()
    for column in required_columns:
        if column not in prepared.columns:
            prepared[column] = None

    prepared["cost_basis"] = prepared["cost_basis"].where(
        pd.notna(prepared["cost_basis"]),
        prepared["quantity"].astype(float) * prepared["entry_price"].astype(float),
    )
    records = prepared[required_columns].where(pd.notna(prepared), None).to_records(index=False).tolist()
    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO real_positions (
                position_id,
                ticker,
                quantity,
                entry_price,
                entry_at,
                cost_basis,
                current_price,
                last_updated_at,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_id) DO UPDATE SET
                ticker = excluded.ticker,
                quantity = excluded.quantity,
                entry_price = excluded.entry_price,
                entry_at = excluded.entry_at,
                cost_basis = excluded.cost_basis,
                current_price = COALESCE(excluded.current_price, real_positions.current_price),
                last_updated_at = COALESCE(excluded.last_updated_at, real_positions.last_updated_at),
                notes = excluded.notes
            """,
            records,
        )
        return connection.total_changes - before


def delete_real_position(
    position_id: str,
    database_path: Path = DATABASE_PATH,
) -> bool:
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM real_positions WHERE position_id = ?",
            (position_id,),
        )
        return cursor.rowcount > 0


def update_real_position(
    position_id: str,
    ticker: str,
    quantity: int,
    entry_price: float,
    entry_at: str,
    notes: str | None,
    database_path: Path = DATABASE_PATH,
) -> bool:
    updated_at = datetime.utcnow().isoformat(timespec="seconds")
    cost_basis = float(quantity) * float(entry_price)
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE real_positions
            SET ticker = ?,
                quantity = ?,
                entry_price = ?,
                entry_at = ?,
                cost_basis = ?,
                current_price = ?,
                last_updated_at = ?,
                notes = ?
            WHERE position_id = ?
            """,
            (
                ticker,
                int(quantity),
                float(entry_price),
                entry_at,
                cost_basis,
                float(entry_price),
                updated_at,
                notes,
                position_id,
            ),
        )
        return cursor.rowcount > 0


def update_real_position_prices(
    prices: pd.DataFrame | None = None,
    database_path: Path = DATABASE_PATH,
) -> int:
    with get_connection(database_path) as connection:
        positions = pd.read_sql_query(
            """
            SELECT position_id, ticker
            FROM real_positions
            """,
            connection,
        )
    if positions.empty:
        return 0

    if prices is None:
        prices = read_ohlcv_prices(database_path)
    if prices.empty:
        return 0

    latest_prices = (
        prices.sort_values(["ticker", "date"]) 
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "close"]]
    )
    merged = positions.merge(latest_prices, on="ticker", how="left")
    merged = merged[merged["close"].notna()].copy()
    if merged.empty:
        return 0

    updated_at = datetime.utcnow().isoformat(timespec="seconds")
    records = [
        (float(row.close), updated_at, str(row.position_id))
        for row in merged.itertuples(index=False)
    ]
    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            UPDATE real_positions
            SET current_price = ?,
                last_updated_at = ?
            WHERE position_id = ?
            """,
            records,
        )
        return connection.total_changes - before


def get_real_positions(
    database_path: Path = DATABASE_PATH,
    mark_to_market: bool = True,
) -> pd.DataFrame:
    prices = read_ohlcv_prices(database_path)
    if mark_to_market:
        update_real_position_prices(prices=prices, database_path=database_path)

    with get_connection(database_path) as connection:
        positions = pd.read_sql_query(
            """
            SELECT *
            FROM real_positions
            ORDER BY entry_at DESC, created_at DESC, ticker
            """,
            connection,
        )
    if positions.empty:
        return positions

    positions = positions.copy()
    positions["cost_basis"] = positions["cost_basis"].fillna(
        positions["quantity"].astype(float) * positions["entry_price"].astype(float)
    )
    positions["current_price"] = positions["current_price"].fillna(positions["entry_price"])

    if not prices.empty:
        latest_prices = (
            prices.sort_values(["ticker", "date"]) 
            .groupby("ticker", as_index=False)
            .tail(1)[["ticker", "date", "close"]]
            .rename(columns={"date": "market_price_date", "close": "market_close"})
        )
        positions = positions.merge(latest_prices, on="ticker", how="left")
        positions["current_price"] = positions["market_close"].combine_first(positions["current_price"])
        positions["market_price_date"] = positions["market_price_date"].where(
            pd.notna(positions["market_price_date"]),
            None,
        )
        positions = positions.drop(columns=["market_close"])
    else:
        positions["market_price_date"] = None

    positions["market_value"] = positions["quantity"].astype(float) * positions["current_price"].astype(float)
    positions["unrealized_pnl"] = positions["market_value"] - positions["cost_basis"].astype(float)
    positions["unrealized_return"] = positions["market_value"] / positions["cost_basis"].astype(float) - 1.0
    return positions


def save_risk_alerts(
    alerts: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if alerts.empty:
        return 0

    required_columns = [
        "alert_id",
        "position_id",
        "ticker",
        "evaluated_at",
        "action",
        "severity",
        "reason",
        "current_price",
        "unrealized_return",
        "metadata_json",
    ]
    records = alerts[required_columns].where(pd.notna(alerts), None).to_records(index=False).tolist()
    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR REPLACE INTO risk_alerts (
                alert_id,
                position_id,
                ticker,
                evaluated_at,
                action,
                severity,
                reason,
                current_price,
                unrealized_return,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        return connection.total_changes - before


def get_risk_alerts(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM risk_alerts
            ORDER BY created_at DESC, ticker
            """,
            connection,
        )


def save_trade_outcome_run(
    run: dict,
    database_path: Path = DATABASE_PATH,
) -> None:
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO trade_outcome_runs (
                run_id,
                model_name,
                horizon_days,
                min_reward_risk,
                cost_per_trade,
                spread,
                slippage,
                train_rows,
                validation_rows,
                test_rows,
                validation_accuracy,
                validation_log_loss,
                test_accuracy,
                test_log_loss,
                simulated_test_trades,
                simulated_test_avg_return,
                simulated_test_win_rate,
                artifact_path,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run["run_id"],
                run["model_name"],
                int(run["horizon_days"]),
                float(run["min_reward_risk"]),
                float(run["cost_per_trade"]),
                float(run["spread"]),
                float(run["slippage"]),
                int(run["train_rows"]),
                int(run["validation_rows"]),
                int(run["test_rows"]),
                run.get("validation_accuracy"),
                run.get("validation_log_loss"),
                run.get("test_accuracy"),
                run.get("test_log_loss"),
                run.get("simulated_test_trades"),
                run.get("simulated_test_avg_return"),
                run.get("simulated_test_win_rate"),
                run["artifact_path"],
                run["metadata_json"],
            ),
        )


def get_trade_outcome_runs(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM trade_outcome_runs
            ORDER BY trained_at DESC
            """,
            connection,
        )


def save_operational_trade_outcomes(
    predictions: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> int:
    if predictions.empty:
        return 0

    required_columns = [
        "run_id",
        "ticker",
        "date",
        "horizon_days",
        "probability_win",
        "probability_loss",
        "probability_timeout",
        "expected_return",
        "stop_distance",
        "target_distance",
        "execution_drag",
        "inference_version",
    ]
    records = (
        predictions[required_columns]
        .where(pd.notna(predictions), None)
        .to_records(index=False)
        .tolist()
    )

    with get_connection(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT INTO operational_trade_outcomes (
                run_id,
                ticker,
                date,
                horizon_days,
                probability_win,
                probability_loss,
                probability_timeout,
                expected_return,
                stop_distance,
                target_distance,
                execution_drag,
                inference_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, ticker, date) DO UPDATE SET
                horizon_days = excluded.horizon_days,
                probability_win = excluded.probability_win,
                probability_loss = excluded.probability_loss,
                probability_timeout = excluded.probability_timeout,
                expected_return = excluded.expected_return,
                stop_distance = excluded.stop_distance,
                target_distance = excluded.target_distance,
                execution_drag = excluded.execution_drag,
                inference_version = excluded.inference_version,
                created_at = CURRENT_TIMESTAMP
            """,
            records,
        )
        return connection.total_changes - before


def read_operational_trade_outcomes(
    run_id: str,
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM operational_trade_outcomes
            WHERE run_id = ?
            ORDER BY ticker, date
            """,
            connection,
            params=(run_id,),
        )


def read_latest_operational_trade_outcomes(
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Return the most recent prediction per ticker across all runs."""
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT t.*
            FROM operational_trade_outcomes AS t
            JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM operational_trade_outcomes
                GROUP BY ticker
            ) AS latest
              ON latest.ticker = t.ticker AND latest.max_date = t.date
            JOIN (
                SELECT run_id, ticker, date, MAX(created_at) AS max_created
                FROM operational_trade_outcomes
                GROUP BY ticker, date
            ) AS most_recent
              ON most_recent.ticker = t.ticker
             AND most_recent.date = t.date
             AND most_recent.max_created = t.created_at
            ORDER BY t.ticker
            """,
            connection,
        )


def get_operational_trade_outcomes(
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM operational_trade_outcomes
            ORDER BY created_at DESC, ticker
            """,
            connection,
        )
