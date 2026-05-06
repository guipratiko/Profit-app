# Project Status

## Current Status

The project is in alpha construction. Sprints 1 through 11 are implemented at code level. The current production posture is PostgreSQL-backed, API-driven, and gated by walk-forward, signal quality, sizing, and risk evidence.

Last verified after E2E for Sprints 1-11: 2026-05-03.

## Completed

- Initial Python package structure created.
- SQLite persistence created in `storage/profit_app.sqlite3`.
- Initial B3 asset universe configured with 7 tickers.
- OHLCV ingestion implemented with `yfinance`.
- CLI commands created for database initialization, price updates, price summaries, feature generation, and feature summaries.
- Technical feature engineering implemented.
- Future targets created for 7 trading days, 3 months, and 1 year.
- Chronological train/validation/test split implemented.
- E2E alpha pipeline script created.
- E2E alpha pipeline passed after full 10-year OHLCV refresh.
- TensorFlow 7-day direction baseline implemented.
- Model run persistence implemented.
- Prediction persistence implemented.
- Basic long-only probability-threshold backtest implemented.
- Validation-selected threshold backtest implemented with drawdown filter.
- Policy-level walk-forward validation implemented with rolling calibration windows.
- Per-ticker backtest diagnostics implemented.
- Backtest run and trade persistence implemented.
- E2E model/backtest pipeline passed.
- Paper trading signal persistence implemented with immutable thesis IDs.
- Paper trading policy implemented with confidence, costs, spread, slippage, reward/risk, volatility limits, stop, target, and position sizing.
- Explicit `no_operate` decision implemented for blocked signals.
- News/event persistence implemented.
- Text cleanup, entity normalization, and after-close-to-next-session event alignment implemented.
- PyTorch MVP qualitative sentiment pipeline implemented with deterministic fallback when PyTorch wheels are unavailable for the active Python version.
- Qualitative feature persistence implemented with sentiment score, label, event count, source event IDs, and text embedding JSON.
- Technical/contextual fusion pipeline implemented.
- Fusion prediction persistence implemented with explanation JSON.
- FastAPI backend implemented with assets, prices, predictions, explanations, local recalculation, paper signals, blocked signals, metrics, news, and qualitative feature endpoints.
- E2E qualitative/fusion/API script created.
- HTML dashboard MVP implemented at `/`, consuming the FastAPI endpoints directly.
- Paper position persistence implemented for simulated positions generated from `simulate_long` theses.
- Deterministic risk advisor implemented with hold, adjust stop, partial profit, close at target, and close at stop actions.
- Risk alert persistence implemented.
- E2E dashboard/risk-advisor script created.
- E2E paper trading/news pipeline passed.

## E2E Verification Log

The latest complete verification passed with these commands:

```powershell
\.\.venv\Scripts\python.exe tests\e2e_alpha_pipeline.py
.\.venv311\Scripts\python.exe tests\e2e_model_backtest.py
.\.venv\Scripts\python.exe tests\e2e_paper_news.py
```

Verified outcomes:

- Data ingestion pipeline passes.
- Technical feature pipeline passes.
- TensorFlow training pipeline passes.
- Model prediction persistence passes.
- Backtest pipeline passes.
- Risk-aware optimized backtest pipeline passes.
- Walk-forward validation pipeline passes and can explicitly pass or fail the strategy gate.
- Paper trading thesis generation passes.
- News/event sample ingestion and temporal alignment passes.
- No VS Code/Pylance errors are currently reported for `app` or `tests`.

## Verified Data State

- OHLCV source: `yfinance`.
- Database path: `storage/profit_app.sqlite3`.
- Stored tickers: 7.
- OHLCV rows per ticker: 2492.
- OHLCV date range: 2016-05-02 to 2026-04-30.
- Technical feature rows total: 13920.
- Technical feature date range: 2017-05-04 to 2025-04-28.

The technical feature dataset ends before the latest OHLCV date because `target_return_1y` requires approximately 252 future trading days. This is expected and avoids future-data leakage.

## Current Commands

```powershell
.\.venv\Scripts\python.exe -m app.cli update-prices
.\.venv\Scripts\python.exe -m app.cli price-summary
.\.venv\Scripts\python.exe -m app.cli generate-features
.\.venv\Scripts\python.exe -m app.cli feature-summary
.\.venv\Scripts\python.exe tests\e2e_alpha_pipeline.py
.\.venv311\Scripts\python.exe -m app.cli train-tf-direction --epochs 40 --batch-size 64
.\.venv311\Scripts\python.exe -m app.cli model-summary
.\.venv311\Scripts\python.exe -m app.cli run-backtest --threshold 0.45 --holding-days 7 --cost-per-trade 0.002
.\.venv\Scripts\python.exe -m app.cli run-optimized-backtest
.\.venv\Scripts\python.exe -m app.cli run-walk-forward
.\.venv311\Scripts\python.exe -m app.cli backtest-summary
.\.venv311\Scripts\python.exe tests\e2e_model_backtest.py
.\.venv\Scripts\python.exe -m app.cli generate-paper-signals
.\.venv\Scripts\python.exe -m app.cli paper-summary
.\.venv\Scripts\python.exe -m app.cli seed-sample-news
.\.venv\Scripts\python.exe -m app.cli news-summary
.\.venv\Scripts\python.exe -m app.cli analyze-news-sentiment
.\.venv\Scripts\python.exe -m app.cli sentiment-summary
.\.venv\Scripts\python.exe -m app.cli run-fusion
.\.venv\Scripts\python.exe -m app.cli fusion-summary
.\.venv\Scripts\python.exe -m app.cli audit-portfolio
.\.venv\Scripts\python.exe -m app.cli portfolio-summary
.\.venv\Scripts\python.exe -m app.cli risk-alert-summary
.\.venv\Scripts\python.exe tests\e2e_paper_news.py
.\.venv\Scripts\python.exe tests\e2e_qualitative_fusion_api.py
.\.venv\Scripts\python.exe tests\e2e_dashboard_risk.py
.\.venv\Scripts\uvicorn.exe app.api:app --reload --host 127.0.0.1 --port 8000
```

Use `.venv311` for TensorFlow. The default `.venv` uses Python 3.14, and TensorFlow is not available for that Python version in this environment.

## Verified Model State

Latest verified model run:

- Run ID: `tf_direction_7d_20260503182823_efb41eb2`
- Model: `tensorflow_direction_classifier`
- Target: `target_direction_7d`
- Train rows: 9741
- Validation rows: 2086
- Test rows: 2093
- Epochs requested: 12
- Epochs ran: 12
- Validation accuracy: 0.3571
- Test accuracy: 0.3688
- Artifact path: `storage/models/tf_direction_7d_20260503182823_efb41eb2`
- Feature policy: relative technical features, ticker one-hot encoding, class weights, batch normalization, and dropout.

Latest verified optimized backtest:

- Backtest ID: `bt_7d_20260503180005_fba25861`
- Threshold: 0.575, selected on validation with max validation drawdown 0.20
- Holding days: 7
- Cost per trade: 0.002
- Trades: 24
- Win rate: 0.5000
- Cumulative return: 0.2414
- Average trade return: 0.0102
- Max drawdown: -0.1349
- Average buy-and-hold return over the same test period: 0.0312

Latest verified walk-forward validation:

- Backtest ID: `wf_7d_20260503183206_32332ca3`
- Selected probability threshold: 0.4550
- Windows passed: 4 of 5
- Passing window ratio: 0.8000
- Trades: 20
- Win rate: 0.5500
- Cumulative return: 0.1109
- Average trade return: 0.0061
- Max drawdown: -0.0935
- Average buy-and-hold return over the same test period: 0.0312
- Traded tickers: 5
- Profitable tickers: 3
- Strategy gate: passed because return was positive, beat average buy-and-hold, drawdown was acceptable, temporal stability passed, and profitable tickers were sufficiently dispersed.

Latest verified paper trading signal generation:

- Model run ID: `tf_direction_7d_20260503182823_efb41eb2`
- Generated signals: 7
- New immutable signals inserted: 7
- `simulate_long`: 0
- `no_operate`: 7
- Strategy gate: passed under walk-forward validation.
- Signal gate: every current signal was blocked because current probabilities, confidence, expected return, or reward/risk did not satisfy the paper-trading policy.
- Paper signal version: `v4_walk_forward_threshold_gate`.
- The thesis now stores `strategy_probability_threshold` and blocks signals with `probability_up_below_strategy_threshold` when the current probability does not reach the walk-forward-selected threshold.
- Position sizing policy: R$ 10,000 portfolio, 1% max risk per operation, fixed simulated costs/spread/slippage.

Latest verified sample news/event ingestion:

- Generated sample events: 3
- Source: `sample`
- After-close publication timestamp: 2026-04-29 20:00
- Aligned trading date: 2026-04-30

Latest verified qualitative sentiment run:

- Model name: `pytorch_lexicon_sentiment_mvp`
- Events analyzed: 4
- Qualitative features generated: 4
- Labels: 3 positive, 1 neutral
- PyTorch available in active `.venv`: false, because the active Python is 3.14. The code uses PyTorch tensor operations when PyTorch is installed in a compatible interpreter and a deterministic local embedding fallback otherwise.

Latest verified fusion run:

- Model run ID: `tf_direction_7d_20260503180939_395464d8`
- Fusion version: `v1_technical_contextual_score`
- Fused predictions generated: 7
- Fused directions: 5 sideways, 2 down
- Explanation payload: technical probability, technical confidence, selected contextual sentiment, event count, fusion rule, fused score, and risk controls.

Latest verified API state:

- FastAPI app: `app.api:app`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:8000/`
- Verified endpoints in E2E: `/health`, `/assets`, `/predictions/{ticker}`, `/predictions/{ticker}/explanation`, `/paper/metrics`, `/portfolio/positions`, `/portfolio/alerts`, `/portfolio/audit`.

Latest verified dashboard/risk advisor state:

- E2E dashboard/risk advisor: passed.
- Simulated positions opened from historical `simulate_long` theses: 4.
- Positions evaluated by latest audit: 4.
- Latest rule checks validated: stop loss closes, partial target suggests partial profit, target closes.
- Current latest model paper signals remain all `no_operate`; the simulated positions currently shown in the portfolio come from earlier historical paper theses that had been allowed before the stricter current gate.

Interpretation: the original fixed-threshold TensorFlow baseline was weak, but the improved model and policy-level walk-forward gate now produce a candidate alpha for controlled paper trading. The correct operational result for the latest current-day signals is still `no_operate`, because passing the historical strategy gate is necessary but not sufficient: each live paper signal must also pass the walk-forward probability threshold, confidence, net expected return, and reward/risk checks.

Production status: the system has an alpha candidate with walk-forward evidence and an operational gate that blocks entries when today's signal quality is insufficient.

## True Project State

- The project can collect real OHLCV data.
- The project can persist market data locally.
- The project can generate technical features and supervised learning targets.
- The project has a passing E2E validation script for ingestion and feature generation.
- The project can train a real TensorFlow model in `.venv311`.
- The project can persist model runs and predictions.
- The project can run a basic backtest on the model's test-split predictions.
- The project can select a probability threshold on validation with a drawdown filter and test it out-of-sample.
- The project can run policy-level walk-forward validation with rolling calibration windows.
- The project can produce per-ticker performance diagnostics for the strategy.
- The project can generate immutable paper-trading theses from model predictions.
- The project can explicitly block weak signals with `no_operate` reasons.
- The project can calculate simple maximum position size from entry, stop, and portfolio risk.
- The project can persist news/events and align after-close events to the next trading session.
- The project can aggregate qualitative sentiment by ticker and aligned trading date.
- The project can persist text embeddings and sentiment metadata for news/events.
- The project can fuse TensorFlow technical predictions with qualitative context.
- The project exposes the alpha artifacts through a local FastAPI backend.
- The project can render a local dashboard over the API.
- The project can open simulated paper positions from allowed paper theses and audit them with deterministic risk rules.
- The current improved TensorFlow policy passed walk-forward validation, but current paper trading still blocks all entries because no current signal clears the signal-level entry policy.
- The PyTorch/NLP layer is an MVP sentiment/embedding pipeline, not a production financial language model.
- No Streamlit/React frontend has been implemented yet; the current UI is a FastAPI-served HTML dashboard MVP.
- No background live worker has been implemented yet; the current paper portfolio audit is manual via CLI/API.
- No real broker integration exists, by design.

## Next Copilot Instructions

- Do not skip validation. Run both E2E scripts after changing ingestion, database, feature, model, or backtest code.
- Preserve the alpha scope: no real order execution, no broker integration, no financial promises.
- Next sprint should continue from model-ready data and the new paper/news tables, not from UI-first work.
- Before training TensorFlow, inspect `technical_features` and confirm no target leakage.
- Keep all generated local data out of version control.
- Use `.\.venv\Scripts\python.exe` when running commands in PowerShell.
- Use `.\.venv311\Scripts\python.exe` for TensorFlow, model training, and backtesting.
- Prefer `run-walk-forward` over single optimized backtests when deciding whether paper trading should allow signals.
- Keep the current TensorFlow model behind the evidence gate and promote only runs that satisfy the production validation thresholds.
- The next modeling step should improve probability calibration, signal selectivity, out-of-sample sample size, qualitative model quality, and live paper-trading lifecycle tracking before building a UI.

## Recommended Next Step

Continue with Sprint 12: add data freshness checks, incremental update/recalibration logs, and a safer retraining endpoint that records before/after metrics.