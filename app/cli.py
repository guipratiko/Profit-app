
"""Principais Recursos do argparse:
Automação: Gera automaticamente mensagens de ajuda e uso.
Gestão de Erros: Emite erros quando usuários fornecem argumentos inválidos.
Tipos de Argumento: Suporta argumentos posicionais (obrigatórios) e opcionais (flags).
Conversores de Tipo: Permite converter tipos personalizados de argumentos.
Estrutura: Baseado na instância argparse.ArgumentParser e no método .add_argument() para definir parâmetros"""

import argparse

import pandas as pd

from app.config import DEFAULT_PRICE_INTERVAL, DEFAULT_PRICE_PERIOD
from app.data.database import (
    get_backtest_runs,
    get_feature_counts,
    get_fusion_predictions,
    get_model_runs,
    get_news_events,
    get_operational_predictions,
    get_operational_trade_outcomes,
    get_paper_positions,
    get_paper_trading_signals,
    get_price_counts,
    get_qualitative_features,
    get_risk_alerts,
    get_trade_outcome_runs,
    initialize_database,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profit App alpha CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create the database schema")

    update_parser = subparsers.add_parser("update-prices", help="Download OHLCV prices")
    update_parser.add_argument("--period", default=DEFAULT_PRICE_PERIOD)
    update_parser.add_argument("--interval", default=DEFAULT_PRICE_INTERVAL)
    update_parser.add_argument("--tickers", nargs="*", default=None)

    subparsers.add_parser("price-summary", help="Show stored OHLCV row counts")
    subparsers.add_parser("generate-features", help="Build technical features and future targets")
    subparsers.add_parser("feature-summary", help="Show stored technical feature row counts")

    train_parser = subparsers.add_parser("train-tf-direction", help="Train TensorFlow 7-day direction model (legacy 3-class)")
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=64)

    train_binary_parser = subparsers.add_parser(
        "train-tf-binary",
        help="Train TensorFlow 7-day BINARY enter-long model (replaces 3-class for entry signals)",
    )
    train_binary_parser.add_argument("--epochs", type=int, default=60)
    train_binary_parser.add_argument("--batch-size", type=int, default=128)
    train_binary_parser.add_argument("--seed", type=int, default=42)
    train_binary_parser.add_argument("--l2", type=float, default=0.0, help="L2 weight decay (0 disables)")
    train_binary_parser.add_argument(
        "--dropout-scale",
        type=float,
        default=1.0,
        help="Multiplier on dropout layers (1.0 keeps defaults; 1.3 = +30%% regularization)",
    )

    train_sklearn_binary_parser = subparsers.add_parser(
        "train-sklearn-binary",
        help="Train sklearn gradient-boosting 7-day BINARY enter-long model",
    )
    train_sklearn_binary_parser.add_argument("--max-iter", type=int, default=300)
    train_sklearn_binary_parser.add_argument("--learning-rate", type=float, default=0.04)
    train_sklearn_binary_parser.add_argument("--l2", type=float, default=0.02)
    train_sklearn_binary_parser.add_argument("--max-leaf-nodes", type=int, default=15)
    train_sklearn_binary_parser.add_argument("--min-samples-leaf", type=int, default=40)
    train_sklearn_binary_parser.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("model-summary", help="Show trained model runs")

    inference_parser = subparsers.add_parser(
        "run-current-inference",
        help="Predict the latest available OHLCV rows without future targets",
    )
    inference_parser.add_argument("--run-id", default=None)
    subparsers.add_parser("operational-summary", help="Show current operational model predictions")

    train_to_parser = subparsers.add_parser(
        "train-trade-outcome",
        help="Train the operational trade-outcome model (win/loss/timeout + expected return)",
    )
    train_to_parser.add_argument("--holding-days", type=int, default=7)
    train_to_parser.add_argument("--min-reward-risk", type=float, default=1.5)
    train_to_parser.add_argument("--cost-per-trade", type=float, default=0.002)
    train_to_parser.add_argument("--spread", type=float, default=0.001)
    train_to_parser.add_argument("--slippage", type=float, default=0.001)
    train_to_parser.add_argument("--max-iter", type=int, default=250)
    train_to_parser.add_argument("--learning-rate", type=float, default=0.05)

    to_inference_parser = subparsers.add_parser(
        "run-trade-outcome-inference",
        help="Score the latest OHLCV with the trade-outcome model",
    )
    to_inference_parser.add_argument("--run-id", default=None)

    subparsers.add_parser(
        "trade-outcome-runs", help="Show trained trade-outcome model runs"
    )
    subparsers.add_parser(
        "trade-outcome-summary",
        help="Show current operational trade-outcome predictions",
    )

    backtest_parser = subparsers.add_parser("run-backtest", help="Run probability-threshold backtest")
    backtest_parser.add_argument("--run-id", default=None)
    backtest_parser.add_argument("--threshold", type=float, default=0.45)
    backtest_parser.add_argument("--holding-days", type=int, default=7)
    backtest_parser.add_argument("--cost-per-trade", type=float, default=0.002)
    backtest_parser.add_argument("--use-fixed-costs", action="store_true")

    optimized_backtest_parser = subparsers.add_parser(
        "run-optimized-backtest",
        help="Select threshold on validation split and test it out-of-sample",
    )
    optimized_backtest_parser.add_argument("--run-id", default=None)
    optimized_backtest_parser.add_argument("--holding-days", type=int, default=7)
    optimized_backtest_parser.add_argument("--cost-per-trade", type=float, default=0.002)
    optimized_backtest_parser.add_argument("--use-fixed-costs", action="store_true")
    optimized_backtest_parser.add_argument("--min-threshold", type=float, default=0.35)
    optimized_backtest_parser.add_argument("--max-threshold", type=float, default=0.85)
    optimized_backtest_parser.add_argument("--step", type=float, default=0.025)
    optimized_backtest_parser.add_argument("--min-trades", type=int, default=10)
    optimized_backtest_parser.add_argument("--max-validation-drawdown", type=float, default=0.20)

    walk_forward_parser = subparsers.add_parser(
        "run-walk-forward",
        help="Run policy-level walk-forward validation with per-ticker diagnostics",
    )
    walk_forward_parser.add_argument("--run-id", default=None)
    walk_forward_parser.add_argument("--holding-days", type=int, default=7)
    walk_forward_parser.add_argument("--cost-per-trade", type=float, default=0.002)
    walk_forward_parser.add_argument("--use-fixed-costs", action="store_true")
    walk_forward_parser.add_argument("--window-size", type=int, default=63)
    walk_forward_parser.add_argument("--calibration-lookback-days", type=int, default=504)
    walk_forward_parser.add_argument("--min-threshold", type=float, default=0.35)
    walk_forward_parser.add_argument("--max-threshold", type=float, default=0.85)
    walk_forward_parser.add_argument("--step", type=float, default=0.025)
    walk_forward_parser.add_argument("--min-calibration-trades", type=int, default=10)
    walk_forward_parser.add_argument("--max-calibration-drawdown", type=float, default=0.20)
    walk_forward_parser.add_argument("--max-test-drawdown", type=float, default=0.25)
    walk_forward_parser.add_argument("--min-passing-window-ratio", type=float, default=0.50)
    walk_forward_parser.add_argument("--min-profitable-tickers", type=int, default=2)

    subparsers.add_parser("backtest-summary", help="Show backtest runs")

    paper_parser = subparsers.add_parser("generate-paper-signals", help="Create immutable paper-trading theses")
    paper_parser.add_argument("--run-id", default=None)
    paper_parser.add_argument("--portfolio-value", type=float, default=10000.0)
    paper_parser.add_argument("--max-risk-per-trade", type=float, default=0.01)
    paper_parser.add_argument("--min-confidence", type=float, default=0.48)
    paper_parser.add_argument("--min-reward-risk-ratio", type=float, default=1.5)
    paper_parser.add_argument("--cost-per-trade", type=float, default=0.002)
    paper_parser.add_argument("--spread", type=float, default=0.001)
    paper_parser.add_argument("--slippage", type=float, default=0.001)
    paper_parser.add_argument("--max-volatility-21d", type=float, default=0.045)

    subparsers.add_parser("paper-summary", help="Show paper-trading theses")
    subparsers.add_parser("seed-sample-news", help="Create sample news/events with trading-session alignment")
    subparsers.add_parser("news-summary", help="Show stored news/events")
    subparsers.add_parser("analyze-news-sentiment", help="Generate PyTorch MVP sentiment features from news/events")
    subparsers.add_parser("tag-news-events", help="Generate qualitative features and print event tag distribution")
    subparsers.add_parser("analyze-events-tagged", help="Alias for tag-news-events")
    subparsers.add_parser("sentiment-summary", help="Show qualitative sentiment features")

    fusion_parser = subparsers.add_parser("run-fusion", help="Fuse technical predictions with qualitative context")
    fusion_parser.add_argument("--run-id", default=None)
    fusion_adaptive_parser = subparsers.add_parser(
        "run-fusion-adaptive",
        help="Alias for the v2 regime-adaptive fusion policy",
    )
    fusion_adaptive_parser.add_argument("--run-id", default=None)
    regime_summary_parser = subparsers.add_parser("regime-summary", help="Show latest regime assessment by ticker")
    regime_summary_parser.add_argument("--run-id", default=None)
    subparsers.add_parser("fusion-summary", help="Show fused technical/contextual predictions")
    cost_preview_parser = subparsers.add_parser("cost-preview", help="Preview realistic B3 round-trip costs")
    cost_preview_parser.add_argument("--notional", type=float, default=10000.0)
    cost_preview_parser.add_argument("--adtv", type=float, default=None)
    cost_preview_parser.add_argument("--corretagem-per-leg", type=float, default=0.0)
    cost_preview_parser.add_argument("--half-spread-bps", type=float, default=2.0)
    cost_preview_parser.add_argument("--daytrade", action="store_true")
    subparsers.add_parser("audit-portfolio", help="Open/evaluate simulated paper positions and create risk alerts")
    subparsers.add_parser(
        "audit-portfolio-conselheiro",
        help="EV-aware Conselheiro audit (trailing stop + residual EV exits)",
    )
    subparsers.add_parser("portfolio-summary", help="Show simulated paper positions")
    subparsers.add_parser("risk-alert-summary", help="Show paper portfolio risk alerts")

    subparsers.add_parser("check-staleness", help="Check OHLCV freshness for online learning trigger")

    refresh_parser = subparsers.add_parser(
        "run-refresh",
        help="Online refresh: pull prices, regenerate features, refit head, re-run inference",
    )
    refresh_parser.add_argument("--max-staleness-days", type=int, default=1)
    refresh_parser.add_argument("--refit-window-days", type=int, default=180)
    refresh_parser.add_argument("--skip-price-update", action="store_true")
    refresh_parser.add_argument("--tickers", nargs="*", default=None)

    multi_to_parser = subparsers.add_parser(
        "train-multi-horizon-trade-outcome",
        help="Train operational trade-outcome models for multiple horizons (e.g. 7 21 63)",
    )
    multi_to_parser.add_argument("--horizons", type=int, nargs="+", default=[7, 21, 63])
    multi_to_parser.add_argument("--min-reward-risk", type=float, default=1.5)
    multi_to_parser.add_argument("--cost-per-trade", type=float, default=0.002)
    multi_to_parser.add_argument("--spread", type=float, default=0.001)
    multi_to_parser.add_argument("--slippage", type=float, default=0.001)
    multi_to_parser.add_argument("--max-iter", type=int, default=200)
    multi_to_parser.add_argument("--learning-rate", type=float, default=0.05)

    subparsers.add_parser("alpha-metrics", help="Print consolidated alpha metrics summary")
    subparsers.add_parser("paper-validation-gate", help="Print the 90-day paper-trading validation gate")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-db":
        initialize_database()
        print("Database initialized.")
        return

    if args.command == "update-prices":
        from app.data.market_data import update_all_prices

        updated_rows = update_all_prices(
            tickers=args.tickers,
            period=args.period,
            interval=args.interval,
        )
        for ticker, rows in updated_rows.items():
            print(f"{ticker}: {rows} rows stored")
        return

    if args.command == "price-summary":
        summary = get_price_counts()
        if summary.empty:
            print("No prices stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "generate-features":
        from app.features.technical import generate_technical_features

        rows = generate_technical_features()
        print(f"{rows} technical feature rows stored")
        return

    if args.command == "feature-summary":
        summary = get_feature_counts()
        if summary.empty:
            print("No technical features stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "train-tf-direction":
        from app.models.tensorflow_direction import train_tensorflow_direction_model

        metadata = train_tensorflow_direction_model(epochs=args.epochs, batch_size=args.batch_size)
        print("TensorFlow direction model trained")
        print(f"run_id: {metadata['run_id']}")
        print(f"epochs_ran: {metadata['epochs_ran']}")
        print(f"validation_accuracy: {metadata['validation_accuracy']:.4f}")
        print(f"test_accuracy: {metadata['test_accuracy']:.4f}")
        return

    if args.command == "train-tf-binary":
        from app.models.tensorflow_binary import train_tensorflow_binary_model

        metadata = train_tensorflow_binary_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            l2_strength=args.l2,
            dropout_scale=args.dropout_scale,
        )
        val = metadata["validation_metrics_calibrated"]
        test = metadata["test_metrics_calibrated"]
        print("TensorFlow BINARY enter-long model trained")
        print(f"run_id: {metadata['run_id']}")
        print(f"epochs_ran: {metadata['epochs_ran']}")
        print(f"train_positive_rate: {metadata['train_positive_rate']:.4f}")
        print("-- validation (calibrated) --")
        print(f"  accuracy:        {val['accuracy']:.4f}")
        print(f"  auc:             {val['auc']:.4f}" if val['auc'] is not None else "  auc: n/a")
        print(f"  base_rate(pos):  {val['base_rate_positive']:.4f}")
        print(f"  high_conf_count: {val['high_confidence_count']}")
        if val['precision_at_p60'] is not None:
            print(f"  precision@p60:   {val['precision_at_p60']:.4f}")
        else:
            print("  precision@p60:   n/a (no high-confidence rows)")
        print("-- test (calibrated) --")
        print(f"  accuracy:        {test['accuracy']:.4f}")
        print(f"  auc:             {test['auc']:.4f}" if test['auc'] is not None else "  auc: n/a")
        print(f"  high_conf_count: {test['high_confidence_count']}")
        if test['precision_at_p60'] is not None:
            print(f"  precision@p60:   {test['precision_at_p60']:.4f}")
        return

    if args.command == "train-sklearn-binary":
        from app.models.sklearn_binary import train_sklearn_binary_model

        metadata = train_sklearn_binary_model(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )
        val = metadata["validation_metrics_calibrated"]
        test = metadata["test_metrics_calibrated"]
        raw_val = metadata["validation_metrics_raw"]
        raw_test = metadata["test_metrics_raw"]
        print("Sklearn BINARY enter-long model trained")
        print(f"run_id: {metadata['run_id']}")
        print(f"n_iter: {metadata['n_iter']}")
        print(f"train_positive_rate: {metadata['train_positive_rate']:.4f}")
        print("-- validation --")
        print(f"  accuracy(cal):   {val['accuracy']:.4f}")
        print(f"  auc(cal/raw):    {val['auc']:.4f} / {raw_val['auc']:.4f}")
        print(f"  p60_count(cal):  {val['high_confidence_count']}")
        if val['precision_at_p60'] is not None:
            print(f"  precision@p60:   {val['precision_at_p60']:.4f}")
        else:
            print("  precision@p60:   n/a (no high-confidence rows)")
        print("-- test --")
        print(f"  accuracy(cal):   {test['accuracy']:.4f}")
        print(f"  auc(cal/raw):    {test['auc']:.4f} / {raw_test['auc']:.4f}")
        print(f"  p60_count(cal):  {test['high_confidence_count']}")
        if test['precision_at_p60'] is not None:
            print(f"  precision@p60:   {test['precision_at_p60']:.4f}")
        else:
            print("  precision@p60:   n/a (no high-confidence rows)")
        return

    if args.command == "model-summary":
        summary = get_model_runs()
        if summary.empty:
            print("No model runs stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "run-current-inference":
        from app.models.inference import run_current_inference

        result = run_current_inference(run_id=args.run_id)
        print("Current inference completed")
        print(f"run_id: {result['run_id']}")
        print(f"inference_version: {result['inference_version']}")
        print(f"latest_date: {result['latest_date']}")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"directions: {result['directions']}")
        return

    if args.command == "operational-summary":
        summary = get_operational_predictions()
        if summary.empty:
            print("No operational predictions stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "train-trade-outcome":
        from app.models.trade_outcome import train_trade_outcome_model

        metadata = train_trade_outcome_model(
            holding_days=args.holding_days,
            min_reward_risk=args.min_reward_risk,
            cost_per_trade=args.cost_per_trade,
            spread=args.spread,
            slippage=args.slippage,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
        )
        print("Trade outcome model trained")
        print(f"run_id: {metadata['run_id']}")
        print(f"holding_days: {metadata['holding_days']}")
        print(f"validation_accuracy: {metadata['validation_accuracy']:.4f}")
        print(f"validation_log_loss: {metadata['validation_log_loss']:.4f}")
        print(f"test_accuracy: {metadata['test_accuracy']:.4f}")
        print(f"test_log_loss: {metadata['test_log_loss']:.4f}")
        print(f"simulated_test_trades: {metadata['simulated_test_trades']}")
        print(f"simulated_test_avg_return: {metadata['simulated_test_avg_return']:.4f}")
        print(f"simulated_test_win_rate: {metadata['simulated_test_win_rate']:.4f}")
        print(f"outcome_distribution: {metadata['outcome_distribution']}")
        return

    if args.command == "run-trade-outcome-inference":
        from app.models.trade_outcome import run_trade_outcome_inference

        result = run_trade_outcome_inference(run_id=args.run_id)
        print("Trade outcome inference completed")
        print(f"run_id: {result['run_id']}")
        print(f"inference_version: {result['inference_version']}")
        print(f"latest_date: {result['latest_date']}")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"directions: {result['directions']}")
        return

    if args.command == "trade-outcome-runs":
        summary = get_trade_outcome_runs()
        if summary.empty:
            print("No trade outcome runs stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "trade-outcome-summary":
        summary = get_operational_trade_outcomes()
        if summary.empty:
            print("No operational trade-outcome predictions stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "run-backtest":
        from app.backtesting.strategy import run_probability_backtest

        result = run_probability_backtest(
            run_id=args.run_id,
            threshold=args.threshold,
            holding_days=args.holding_days,
            cost_per_trade=args.cost_per_trade,
            use_b3_costs=not args.use_fixed_costs,
        )
        print("Backtest completed")
        print(f"backtest_id: {result['backtest_id']}")
        print(f"trades: {result['trades']}")
        print(f"win_rate: {result['win_rate']:.4f}")
        print(f"cumulative_return: {result['cumulative_return']:.4f}")
        print(f"max_drawdown: {result['max_drawdown']:.4f}")
        print(f"buy_hold_return_avg: {result['buy_hold_return_avg']:.4f}")
        return

    if args.command == "run-optimized-backtest":
        from app.backtesting.strategy import run_validation_selected_backtest

        result = run_validation_selected_backtest(
            run_id=args.run_id,
            holding_days=args.holding_days,
            cost_per_trade=args.cost_per_trade,
            use_b3_costs=not args.use_fixed_costs,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            step=args.step,
            min_trades=args.min_trades,
            max_validation_drawdown=args.max_validation_drawdown,
        )
        print("Validation-selected backtest completed")
        print(f"backtest_id: {result['backtest_id']}")
        print(f"threshold: {result['threshold']:.4f}")
        print(f"strategy_gate_passed_on_validation: {result['strategy_gate_passed_on_validation']}")
        print(f"strategy_gate_reason: {result['strategy_gate_reason']}")
        print(f"trades: {result['trades']}")
        print(f"win_rate: {result['win_rate']:.4f}")
        print(f"cumulative_return: {result['cumulative_return']:.4f}")
        print(f"max_drawdown: {result['max_drawdown']:.4f}")
        print(f"buy_hold_return_avg: {result['buy_hold_return_avg']:.4f}")
        return

    if args.command == "run-walk-forward":
        from app.backtesting.strategy import run_walk_forward_backtest

        result = run_walk_forward_backtest(
            run_id=args.run_id,
            holding_days=args.holding_days,
            cost_per_trade=args.cost_per_trade,
            use_b3_costs=not args.use_fixed_costs,
            window_size=args.window_size,
            calibration_lookback_days=args.calibration_lookback_days,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            step=args.step,
            min_calibration_trades=args.min_calibration_trades,
            max_calibration_drawdown=args.max_calibration_drawdown,
            max_test_drawdown=args.max_test_drawdown,
            min_passing_window_ratio=args.min_passing_window_ratio,
            min_profitable_tickers=args.min_profitable_tickers,
        )
        print("Walk-forward validation completed")
        print(f"backtest_id: {result['backtest_id']}")
        print(f"strategy_gate_passed: {result['strategy_gate_passed']}")
        print(f"strategy_gate_reason: {result['strategy_gate_reason']}")
        print(f"windows: {result['passing_windows']}/{result['total_windows']}")
        print(f"traded_tickers: {result['traded_tickers']}")
        print(f"profitable_tickers: {result['profitable_tickers']}")
        print(f"trades: {result['trades']}")
        print(f"win_rate: {result['win_rate']:.4f}")
        print(f"cumulative_return: {result['cumulative_return']:.4f}")
        print(f"max_drawdown: {result['max_drawdown']:.4f}")
        print(f"buy_hold_return_avg: {result['buy_hold_return_avg']:.4f}")
        print("Per ticker:")
        print(pd.DataFrame(result["per_ticker"]).to_string(index=False))
        return

    if args.command == "backtest-summary":
        summary = get_backtest_runs()
        if summary.empty:
            print("No backtest runs stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "generate-paper-signals":
        from app.trading.paper import generate_paper_trading_signals

        result = generate_paper_trading_signals(
            run_id=args.run_id,
            portfolio_value=args.portfolio_value,
            max_risk_per_trade=args.max_risk_per_trade,
            min_confidence=args.min_confidence,
            min_reward_risk_ratio=args.min_reward_risk_ratio,
            cost_per_trade=args.cost_per_trade,
            spread=args.spread,
            slippage=args.slippage,
            max_volatility_21d=args.max_volatility_21d,
        )
        print("Paper-trading signals generated")
        print(f"run_id: {result['run_id']}")
        print(f"signal_source: {result['signal_source']}")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"simulate_long: {result['simulate_long']}")
        print(f"no_operate: {result['no_operate']}")
        print(f"operational_actions: {result.get('operational_actions', {})}")
        return

    if args.command == "paper-summary":
        summary = get_paper_trading_signals()
        if summary.empty:
            print("No paper-trading signals stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "seed-sample-news":
        from app.data.news_events import save_sample_news_events

        result = save_sample_news_events()
        print("Sample news/events stored")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"aligned_dates: {', '.join(result['aligned_dates'])}")
        return

    if args.command == "news-summary":
        summary = get_news_events()
        if summary.empty:
            print("No news/events stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "analyze-news-sentiment":
        from app.models.pytorch_sentiment import evaluate_manual_sample, generate_qualitative_features

        result = generate_qualitative_features()
        print("Qualitative sentiment features generated")
        print(f"model_name: {result['model_name']}")
        print(f"torch_available: {result['torch_available']}")
        print(f"events: {result['events']}")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"labels: {result['labels']}")
        print("Manual sample:")
        print(pd.DataFrame(evaluate_manual_sample()).to_string(index=False))
        return

    if args.command in {"tag-news-events", "analyze-events-tagged"}:
        import json as _json

        from app.models.pytorch_sentiment import generate_qualitative_features

        result = generate_qualitative_features()
        features = get_qualitative_features()
        tag_counts: dict[str, int] = {}
        severity_max = 0.0
        if not features.empty:
            for metadata_text in features["metadata_json"].dropna().tolist():
                try:
                    metadata = _json.loads(str(metadata_text))
                except (TypeError, ValueError):
                    continue
                for tag in metadata.get("event_tags", []):
                    tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
                severity_max = max(severity_max, float(metadata.get("event_severity_max", 0.0)))
        print("Tagged qualitative events generated")
        print(f"model_name: {result['model_name']}")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"labels: {result['labels']}")
        print(f"event_tags: {tag_counts}")
        print(f"max_event_severity: {severity_max:.4f}")
        return

    if args.command == "sentiment-summary":
        summary = get_qualitative_features()
        if summary.empty:
            print("No qualitative features stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command in {"run-fusion", "run-fusion-adaptive"}:
        from app.models.fusion import run_fusion_predictions

        result = run_fusion_predictions(run_id=args.run_id)
        print("Fusion predictions generated")
        print(f"run_id: {result['run_id']}")
        print(f"fusion_version: {result['fusion_version']}")
        print(f"generated: {result['generated']}")
        print(f"inserted: {result['inserted']}")
        print(f"directions: {result['directions']}")
        print(f"regimes: {result.get('regimes', {})}")
        print(f"qualitative_overrides: {result.get('qualitative_overrides', 0)}")
        return

    if args.command == "regime-summary":
        import json as _json

        from app.models.fusion import run_fusion_predictions

        result = run_fusion_predictions(run_id=args.run_id)
        summary = get_fusion_predictions()
        if not summary.empty and "fusion_version" in summary.columns:
            summary = summary[summary["fusion_version"].astype(str).str.contains("regime_adaptive", na=False)].copy()
        if not summary.empty:
            summary = summary.sort_values(["ticker", "signal_date", "created_at"]).groupby("ticker", as_index=False).tail(1)
        rows: list[dict] = []
        for record in summary.to_dict(orient="records"):
            try:
                explanation = _json.loads(str(record.get("explanation_json") or "{}"))
            except (TypeError, ValueError):
                explanation = {}
            regime = explanation.get("regime", {})
            contextual = explanation.get("contextual", {})
            if not regime:
                continue
            rows.append(
                {
                    "ticker": record.get("ticker"),
                    "signal_date": record.get("signal_date"),
                    "fused_direction": record.get("fused_direction"),
                    "regime": regime.get("regime"),
                    "override": regime.get("override_qualitative"),
                    "vol_pct": regime.get("volatility_percentile"),
                    "divergence": regime.get("divergence"),
                    "event_tags": ",".join(contextual.get("event_tags", [])),
                }
            )
        print("Regime-adaptive fusion completed")
        print(f"run_id: {result['run_id']}")
        print(f"regimes: {result.get('regimes', {})}")
        if rows:
            print(pd.DataFrame(rows).head(20).to_string(index=False))
        else:
            print("No regime rows available.")
        return

    if args.command == "fusion-summary":
        summary = get_fusion_predictions()
        if summary.empty:
            print("No fusion predictions stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "cost-preview":
        from app.trading.costs import compute_cost_breakdown

        breakdown = compute_cost_breakdown(
            notional_brl=args.notional,
            adtv_brl=args.adtv,
            corretagem_per_leg_brl=args.corretagem_per_leg,
            half_spread_bps=args.half_spread_bps,
            is_daytrade=args.daytrade,
        )
        print("B3 round-trip cost preview")
        print(f"notional_brl: {args.notional:.2f}")
        print(f"emolumentos: {breakdown.emolumentos:.6f}")
        print(f"liquidacao: {breakdown.liquidacao:.6f}")
        print(f"corretagem: {breakdown.corretagem:.6f}")
        print(f"iss: {breakdown.iss:.6f}")
        print(f"spread: {breakdown.spread:.6f}")
        print(f"slippage: {breakdown.slippage:.6f}")
        print(f"total_pre_ir: {breakdown.total_pre_ir:.6f}")
        print(f"ir_on_profit_rate: {breakdown.ir_on_profit_rate:.4f}")
        return

    if args.command == "audit-portfolio":
        from app.trading.risk_advisor import audit_paper_portfolio

        result = audit_paper_portfolio()
        print("Paper portfolio audited")
        print(f"risk_advisor_version: {result['risk_advisor_version']}")
        print(f"opened_positions: {result['opened_positions']}")
        print(f"evaluated_positions: {result['evaluated_positions']}")
        print(f"updated_positions: {result['updated_positions']}")
        print(f"alerts: {result['alerts']}")
        print(f"open_positions: {result['open_positions']}")
        return

    if args.command == "portfolio-summary":
        summary = get_paper_positions()
        if summary.empty:
            print("No simulated paper positions stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "risk-alert-summary":
        summary = get_risk_alerts()
        if summary.empty:
            print("No risk alerts stored yet.")
        else:
            print(summary.to_string(index=False))
        return

    if args.command == "audit-portfolio-conselheiro":
        from app.trading.risk_advisor import audit_paper_portfolio_with_conselheiro

        result = audit_paper_portfolio_with_conselheiro()
        print("Conselheiro audit complete")
        print(f"risk_advisor_version: {result['risk_advisor_version']}")
        print(f"opened_positions: {result['opened_positions']}")
        print(f"evaluated_positions: {result['evaluated_positions']}")
        print(f"updated_positions: {result['updated_positions']}")
        print(f"alerts: {result['alerts']}")
        print(f"open_positions: {result['open_positions']}")
        print(f"actions: {result['actions']}")
        return

    if args.command == "check-staleness":
        from app.pipelines.refresh import detect_staleness

        result = detect_staleness()
        print("Data staleness check")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return

    if args.command == "run-refresh":
        from app.pipelines.refresh import run_refresh_pipeline

        result = run_refresh_pipeline(
            tickers=args.tickers,
            max_staleness_days=args.max_staleness_days,
            refit_window_days=args.refit_window_days,
            skip_price_update=args.skip_price_update,
        )
        print(f"Refresh pipeline status: {result.get('status')}")
        for key, value in result.items():
            if key == "status":
                continue
            print(f"  {key}: {value}")
        return

    if args.command == "train-multi-horizon-trade-outcome":
        from app.models.trade_outcome import train_trade_outcome_model

        for horizon in args.horizons:
            print(f"\n=== Training trade-outcome model for horizon={horizon}d ===")
            result = train_trade_outcome_model(
                holding_days=int(horizon),
                min_reward_risk=args.min_reward_risk,
                cost_per_trade=args.cost_per_trade,
                spread=args.spread,
                slippage=args.slippage,
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
            )
            print(f"  run_id: {result['run_id']}")
            print(f"  validation_accuracy: {result['validation_accuracy']:.4f}")
            print(f"  test_accuracy: {result['test_accuracy']:.4f}")
            print(f"  simulated_test_avg_return: {result['simulated_test_avg_return']:.4f}")
            print(f"  simulated_test_win_rate: {result['simulated_test_win_rate']:.4f}")
        return

    if args.command == "alpha-metrics":
        import json as _json

        from app.pipelines.alpha_metrics import build_alpha_metrics

        metrics = build_alpha_metrics()
        print(_json.dumps(metrics, indent=2, default=str))
        return

    if args.command == "paper-validation-gate":
        import json as _json

        from app.pipelines.paper_validation import build_paper_validation_report

        report = build_paper_validation_report()
        print(_json.dumps(report, indent=2, default=str))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()