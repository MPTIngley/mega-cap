"""
StockPulse Main Entry Point.

Usage:
    stockpulse run          # Run scheduler and scan
    stockpulse dashboard    # Launch Streamlit dashboard
    stockpulse backtest     # Run backtests
    stockpulse ingest       # Run data ingestion
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import date

# Auto-load .env file before anything else
def _load_env():
    """Automatically load .env file from project root."""
    # Find .env file - check current dir and parent dirs
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent.parent / ".env",
        Path.home() / "mega-cap" / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
            break

_load_env()

from stockpulse.utils.config import load_config
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="StockPulse - Automated Stock Scanning & Trading Signal System"
    )

    parser.add_argument(
        "command",
        choices=["run", "dashboard", "backtest", "ingest", "scan", "init", "optimize", "reset"],
        help="Command to execute"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name for backtest"
    )

    parser.add_argument(
        "--keep-market-data",
        action="store_true",
        default=True,
        help="Keep historical price data when resetting (default: True)"
    )

    parser.add_argument(
        "--clear-all",
        action="store_true",
        default=False,
        help="Clear ALL data including historical prices (use with caution)"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script location
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    try:
        load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Execute command
    if args.command == "run":
        run_scheduler()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "backtest":
        run_backtest(args.strategy)
    elif args.command == "ingest":
        run_ingestion()
    elif args.command == "scan":
        run_scan()
    elif args.command == "init":
        run_init()
    elif args.command == "optimize":
        run_optimize()
    elif args.command == "reset":
        run_reset(keep_market_data=not args.clear_all)


def run_scheduler():
    """Run the scheduler for continuous scanning."""
    from stockpulse.scheduler import StockPulseScheduler
    from stockpulse.strategies.signal_generator import SignalGenerator
    from stockpulse.strategies.position_manager import PositionManager
    from stockpulse.scanner.long_term_scanner import LongTermScanner
    from stockpulse.alerts.alert_manager import AlertManager

    logger.info("Starting StockPulse scheduler...")

    scheduler = StockPulseScheduler()
    signal_generator = SignalGenerator()
    position_manager = PositionManager()
    long_term_scanner = LongTermScanner()
    alert_manager = AlertManager()

    def on_intraday_scan(tickers):
        """Callback for intraday scans."""
        signals = signal_generator.generate_signals(tickers)
        alert_manager.process_signals(signals)

        # Update positions
        from stockpulse.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        prices_df = ingestion.get_daily_prices(tickers)
        if not prices_df.empty:
            current_prices = {
                row["ticker"]: row["close"]
                for _, row in prices_df.groupby("ticker").last().reset_index().iterrows()
            }
            updates = position_manager.update_positions(current_prices)
            for update in updates:
                alert_manager.process_position_exit(update)

    def on_daily_scan(tickers):
        """Callback for daily scans."""
        signals = signal_generator.generate_signals(tickers)
        alert_manager.process_signals(signals)
        alert_manager.send_daily_digest()

    def on_long_term_scan(tickers):
        """Callback for long-term scanner."""
        opportunities = long_term_scanner.run_scan(tickers)
        long_term_scanner.send_digest(opportunities)

    scheduler.on_intraday_scan = on_intraday_scan
    scheduler.on_daily_scan = on_daily_scan
    scheduler.on_long_term_scan = on_long_term_scan

    scheduler.start()

    logger.info("Scheduler started. Press Ctrl+C to stop.")

    try:
        import time
        from datetime import datetime
        import pytz

        et = pytz.timezone("US/Eastern")

        while True:
            # Show countdown to next scan
            next_runs = scheduler.get_next_run_times()
            now = datetime.now(et)

            print("\n" + "=" * 60)
            print(f"  StockPulse Scheduler | {now.strftime('%Y-%m-%d %H:%M:%S ET')}")
            print("=" * 60)

            for job_id, next_time in next_runs.items():
                if next_time:
                    delta = next_time - now
                    total_seconds = int(delta.total_seconds())
                    if total_seconds > 0:
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        if hours > 0:
                            countdown = f"{hours}h {minutes}m"
                        else:
                            countdown = f"{minutes}m {seconds}s"
                        print(f"  {job_id}: {next_time.strftime('%H:%M:%S')} (in {countdown})")
                    else:
                        print(f"  {job_id}: running now...")

            print("=" * 60)
            print("  Press Ctrl+C to stop")
            print()

            time.sleep(30)  # Update countdown every 30 seconds
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        scheduler.stop()


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    logger.info("Launching Streamlit dashboard...")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true"
    ])


def run_backtest(strategy_name: str | None = None):
    """Run backtests."""
    from stockpulse.strategies.backtest import Backtester
    from stockpulse.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    from stockpulse.strategies.bollinger_squeeze import BollingerSqueezeStrategy
    from stockpulse.strategies.macd_volume import MACDVolumeStrategy
    from stockpulse.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy
    from stockpulse.strategies.momentum_breakout import MomentumBreakoutStrategy
    from stockpulse.data.universe import UniverseManager
    from stockpulse.utils.config import get_config

    config = get_config()
    backtester = Backtester()
    universe = UniverseManager()

    # Get tickers
    tickers = universe.get_active_tickers()
    if not tickers:
        logger.info("No tickers in universe, using defaults")
        from stockpulse.data.universe import TOP_US_STOCKS
        tickers = TOP_US_STOCKS[:50]

    # Define strategies
    strategy_classes = {
        "rsi_mean_reversion": RSIMeanReversionStrategy,
        "bollinger_squeeze": BollingerSqueezeStrategy,
        "macd_volume": MACDVolumeStrategy,
        "zscore_mean_reversion": ZScoreMeanReversionStrategy,
        "momentum_breakout": MomentumBreakoutStrategy,
    }

    strategies_config = config.get("strategies", {})

    if strategy_name:
        if strategy_name not in strategy_classes:
            logger.error(f"Unknown strategy: {strategy_name}")
            return

        strategies_to_test = {strategy_name: strategy_classes[strategy_name]}
    else:
        strategies_to_test = strategy_classes

    for name, cls in strategies_to_test.items():
        logger.info(f"Running backtest for {name}...")

        strategy_config = strategies_config.get(name, {})
        strategy = cls(strategy_config)

        result = backtester.run_backtest(strategy, tickers)

        logger.info(f"""
        {name} Backtest Results:
        ========================
        Total Return: {result.total_return_pct:.2f}%
        Annualized Return: {result.annualized_return_pct:.2f}%
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Max Drawdown: {result.max_drawdown_pct:.2f}%
        Win Rate: {result.win_rate:.2f}%
        Profit Factor: {result.profit_factor:.2f}
        Total Trades: {result.total_trades}
        """)


def run_ingestion():
    """Run data ingestion."""
    from stockpulse.data.universe import UniverseManager
    from stockpulse.data.ingestion import DataIngestion

    logger.info("Running data ingestion...")

    universe = UniverseManager()
    ingestion = DataIngestion()

    # Refresh universe
    tickers = universe.refresh_universe(force=True)
    logger.info(f"Universe has {len(tickers)} tickers")

    # Run ingestion
    results = ingestion.run_daily_ingestion(tickers)

    logger.info(f"Ingestion complete: {results}")


def run_scan():
    """Run a single scan with market snapshot."""
    from stockpulse.data.universe import UniverseManager
    from stockpulse.strategies.signal_generator import SignalGenerator
    from stockpulse.alerts.alert_manager import AlertManager
    from stockpulse.strategies.position_manager import PositionManager
    from stockpulse.data.ingestion import DataIngestion
    from stockpulse.strategies.base import SignalDirection
    from stockpulse.utils.config import get_config
    import pandas as pd
    import numpy as np

    logger.info("Running scan...")

    universe = UniverseManager()
    signal_generator = SignalGenerator()
    alert_manager = AlertManager()
    position_manager = PositionManager()
    ingestion = DataIngestion()
    config = get_config()

    tickers = universe.get_active_tickers()
    if not tickers:
        logger.warning("No tickers in universe")
        return

    # Get current portfolio positions
    open_positions_df = position_manager.get_open_positions()
    portfolio_tickers = set(open_positions_df["ticker"].tolist()) if not open_positions_df.empty else set()

    # Get allocation weights from config
    allocation_weights = config.get("strategy_allocation", {})
    base_position_pct = config.get("risk_management", {}).get("max_position_size_pct", 5.0)
    initial_capital = config.get("portfolio", {}).get("initial_capital", 100000.0)

    # Generate signals
    all_signals = signal_generator.generate_signals(tickers)

    # Separate BUY and SELL signals
    buy_signals = [s for s in all_signals if s.direction == SignalDirection.BUY]
    sell_signals = [s for s in all_signals if s.direction == SignalDirection.SELL]

    # Filter SELL signals to only stocks we own
    actionable_sells = [s for s in sell_signals if s.ticker in portfolio_tickers]
    informational_sells = [s for s in sell_signals if s.ticker not in portfolio_tickers]

    print("\n" + "=" * 80)
    print("  STOCKPULSE SCAN RESULTS")
    print("=" * 80)

    # === BUY SIGNALS ===
    print(f"\n  📈 BUY SIGNALS: {len(buy_signals)}")
    print("-" * 80)

    if buy_signals:
        # Sort by confidence
        buy_signals.sort(key=lambda s: s.confidence, reverse=True)
        for signal in buy_signals:
            weight = allocation_weights.get(signal.strategy, 1.0)
            position_size = base_position_pct * weight
            dollar_amount = initial_capital * (position_size / 100)
            shares = int(dollar_amount / signal.entry_price)

            print(f"  {signal.ticker:5} | {signal.strategy:25} | Conf: {signal.confidence:.0f}%")
            print(f"         Entry: ${signal.entry_price:.2f} → Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}")
            print(f"         Allocation: {position_size:.1f}% (${dollar_amount:,.0f}) = ~{shares} shares")
            if signal.notes:
                print(f"         Why: {signal.notes}")
            print()
    else:
        print("  No buy signals\n")

    # === ACTIONABLE SELL SIGNALS (stocks we own) ===
    print(f"\n  📉 SELL SIGNALS (Portfolio): {len(actionable_sells)}")
    print("-" * 80)

    if actionable_sells:
        actionable_sells.sort(key=lambda s: s.confidence, reverse=True)
        for signal in actionable_sells:
            print(f"  {signal.ticker:5} | {signal.strategy:25} | Conf: {signal.confidence:.0f}%")
            print(f"         Exit: ${signal.entry_price:.2f} → Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}")
            if signal.notes:
                print(f"         Why: {signal.notes}")
            print()
    else:
        print("  No sell signals for current holdings\n")

    # === INFORMATIONAL SELL SIGNALS (stocks we don't own) ===
    if informational_sells:
        print(f"\n  ℹ️  SELL SIGNALS (Not in Portfolio): {len(informational_sells)}")
        print("-" * 80)
        # Just show tickers, not full details
        sell_tickers = [s.ticker for s in informational_sells[:10]]
        print(f"  {', '.join(sell_tickers)}{'...' if len(informational_sells) > 10 else ''}")
        print("  (These are overbought stocks - avoid buying)\n")

    # === PORTFOLIO STATUS ===
    print("\n  💼 PORTFOLIO STATUS")
    print("-" * 80)
    if not open_positions_df.empty:
        print(f"  Open positions: {len(open_positions_df)}")
        for _, pos in open_positions_df.head(5).iterrows():
            print(f"    {pos['ticker']:5} | Entry: ${pos.get('entry_price', 0):.2f} | P&L: {pos.get('unrealized_pnl_pct', 0):.1f}%")
        if len(open_positions_df) > 5:
            print(f"    ... and {len(open_positions_df) - 5} more")
    else:
        print("  No open positions (paper portfolio empty)")
    print()

    # === MARKET SNAPSHOT ===
    print("-" * 80)
    print("  MARKET SNAPSHOT - Stocks Near Threshold")
    print("-" * 80)

    try:
        from datetime import timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=60)
        prices_df = ingestion.get_daily_prices(tickers[:30], start_date, end_date)

        if not prices_df.empty:
            near_triggers = []

            for ticker in prices_df["ticker"].unique():
                ticker_data = prices_df[prices_df["ticker"] == ticker].copy()
                if len(ticker_data) < 20:
                    continue

                ticker_data = ticker_data.sort_values("date")
                latest = ticker_data.iloc[-1]
                close = latest["close"]

                delta = ticker_data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                mean_20 = ticker_data["close"].rolling(20).mean().iloc[-1]
                std_20 = ticker_data["close"].rolling(20).std().iloc[-1]
                zscore = (close - mean_20) / std_20 if std_20 > 0 else 0

                status = []
                if current_rsi < 35:
                    status.append(f"RSI={current_rsi:.1f}")
                if zscore < -1.5:
                    status.append(f"Z={zscore:.2f}")
                if current_rsi > 65:
                    status.append(f"RSI={current_rsi:.1f}(OB)")

                if status:
                    near_triggers.append((ticker, close, " | ".join(status)))

            if near_triggers:
                for ticker, price, status in near_triggers[:10]:
                    print(f"  {ticker:5} @ ${price:>8.2f} | {status}")
            else:
                print("  No stocks near threshold triggers")

    except Exception as e:
        logger.debug(f"Could not generate market snapshot: {e}")
        print("  Could not generate market snapshot")

    print("\n" + "=" * 80)
    print(f"  Scan complete. {len(buy_signals)} buys, {len(actionable_sells)} actionable sells.")
    print("=" * 80 + "\n")

    # === AUTO-OPEN POSITIONS FOR TOP BUY SIGNALS ===
    positions_opened = 0
    if buy_signals:
        # Open positions for top signals (up to max concurrent)
        max_positions = config.get("risk_management", {}).get("max_concurrent_positions", 20)
        current_positions = len(portfolio_tickers)
        available_slots = max_positions - current_positions

        if available_slots > 0:
            for signal in buy_signals[:available_slots]:
                try:
                    pos_id = position_manager.open_position_from_signal(signal)
                    if pos_id:
                        positions_opened += 1
                        logger.info(f"Opened position for {signal.ticker}")
                except Exception as e:
                    logger.error(f"Failed to open position for {signal.ticker}: {e}")

            if positions_opened > 0:
                print(f"  ✅ Opened {positions_opened} new paper positions")

    # === SEND CONSOLIDATED EMAIL ===
    if buy_signals or actionable_sells:
        sent = alert_manager.send_consolidated_scan_alert(
            buy_signals=buy_signals,
            sell_signals=actionable_sells,
            portfolio_tickers=portfolio_tickers,
            allocation_weights=allocation_weights,
            base_position_pct=base_position_pct,
            initial_capital=initial_capital
        )
        if sent:
            logger.info("Sent consolidated email alert")
        logger.info(f"Sent {sent} alerts")


def run_init():
    """Initialize the database and fetch initial data with full historical preload."""
    from stockpulse.data.database import get_db
    from stockpulse.data.universe import UniverseManager
    from stockpulse.data.ingestion import DataIngestion
    from datetime import timedelta

    logger.info("Initializing StockPulse with full data preload...")

    # Initialize database (creates tables)
    db = get_db()
    logger.info("Database initialized")

    # Refresh universe
    universe = UniverseManager()
    tickers = universe.refresh_universe(force=True)
    logger.info(f"Universe populated with {len(tickers)} tickers")

    # Fetch historical data - full 2 years for backtesting
    ingestion = DataIngestion()

    logger.info("Fetching 2 years of historical daily price data...")
    start_date = date.today() - timedelta(days=730)
    daily_df = ingestion.fetch_daily_prices(tickers, start_date=start_date, progress=True)
    daily_count = ingestion.store_daily_prices(daily_df)
    logger.info(f"Stored {daily_count} daily price records")

    logger.info("Fetching 5 days of intraday data...")
    intraday_df = ingestion.fetch_intraday_prices(tickers, period="5d")
    intraday_count = ingestion.store_intraday_prices(intraday_df)
    logger.info(f"Stored {intraday_count} intraday price records")

    logger.info("Fetching fundamentals...")
    fundamentals_df = ingestion.fetch_fundamentals(tickers[:50])  # Top 50 for fundamentals
    fundamentals_count = ingestion.store_fundamentals(fundamentals_df)
    logger.info(f"Stored {fundamentals_count} fundamental records")

    logger.info(f"""
    ==========================================
    StockPulse Initialization Complete!
    ==========================================
    Universe: {len(tickers)} stocks
    Daily prices: {daily_count} records (2 years)
    Intraday prices: {intraday_count} records (5 days)
    Fundamentals: {fundamentals_count} records

    Ready for trading! Run one of:
      - stockpulse run        # Start scheduler
      - stockpulse dashboard  # Launch dashboard
      - stockpulse scan       # Run single scan
      - stockpulse backtest   # Run backtests
    ==========================================
    """)


def run_optimize():
    """Run hyperparameter optimization for all strategies."""
    from stockpulse.strategies.optimizer import StrategyOptimizer, STRATEGY_CLASSES
    from stockpulse.data.universe import UniverseManager
    from datetime import timedelta
    import time

    logger.info("Starting hyperparameter optimization...")

    print("\n" + "=" * 70)
    print("  STOCKPULSE STRATEGY OPTIMIZER")
    print("  Maximizing returns with max 25% drawdown constraint")
    print("=" * 70)

    # Get tickers
    universe = UniverseManager()
    tickers = universe.get_active_tickers()

    if not tickers:
        print("\n  ⚠ No tickers in universe. Run 'stockpulse init' first.")
        logger.error("No tickers in universe. Run 'stockpulse init' first.")
        return

    # Use subset for faster optimization
    tickers = tickers[:30]
    print(f"\n  Using {len(tickers)} tickers for optimization")
    print(f"  Tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

    # Date range - use 18 months of data
    end_date = date.today()
    start_date = end_date - timedelta(days=540)
    print(f"  Date range: {start_date} to {end_date}")

    # Check if we have price data
    from stockpulse.data.database import get_db
    db = get_db()
    try:
        price_count = db.fetchone("SELECT COUNT(*) FROM prices_daily")
        price_count = price_count[0] if price_count else 0
        print(f"  Price records in database: {price_count:,}")
        if price_count < 1000:
            print("\n  ⚠ WARNING: Not enough price data for reliable optimization.")
            print("  Run 'stockpulse init' to fetch 2 years of historical data.")
            return
    except Exception as e:
        print(f"\n  ⚠ Database error: {e}")
        print("  Run 'stockpulse init' to initialize the database.")
        return

    # Initialize optimizer
    optimizer = StrategyOptimizer(max_drawdown_pct=25.0)

    all_results = {}
    total_strategies = len(STRATEGY_CLASSES)

    for i, strategy_name in enumerate(STRATEGY_CLASSES.keys(), 1):
        print(f"\n  [{i}/{total_strategies}] Optimizing {strategy_name}...")
        print("-" * 70)

        start_time = time.time()

        try:
            def progress_callback(current, total, result):
                bar_width = 40
                filled = int(bar_width * current / total)
                bar = "█" * filled + "░" * (bar_width - filled)
                ret = result.get("total_return_pct", 0)
                dd = result.get("max_drawdown_pct", 0)
                print(f"\r  [{bar}] {current}/{total} | Return: {ret:+.1f}% | DD: {dd:.1f}%", end="", flush=True)

            result = optimizer.optimize(
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                objective="sharpe",
                max_iterations=50,
                progress_callback=progress_callback
            )

            all_results[strategy_name] = result

            elapsed = time.time() - start_time
            print()  # New line after progress bar

            # Show results
            constraint_status = "✓" if result.constraint_satisfied else "✗"
            print(f"\n  {constraint_status} {strategy_name}")
            print(f"    Final Value: ${result.final_value:,.0f} (from $100k)")
            print(f"    Return: {result.best_return:+.2f}% ± {result.return_std_pct:.1f}%")
            print(f"    Sharpe: {result.best_sharpe:.2f}")
            print(f"    Max DD: {result.best_drawdown:.2f}%")
            print(f"    Time: {elapsed:.1f}s")

            # Show key params
            key_params = {k: v for k, v in result.best_params.items()
                         if k not in ["enabled", "_optimized"]}
            print(f"    Params: {key_params}")

        except Exception as e:
            import traceback
            logger.error(f"Failed to optimize {strategy_name}: {e}")
            print(f"\n  ✗ {strategy_name} - FAILED: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 70)

    print("\n  Strategy                     Final Value     Return ± Std    Sharpe    Max DD   Constraint")
    print("  " + "-" * 94)

    for name, result in all_results.items():
        status = "✓ Met" if result.constraint_satisfied else "✗ Exceeded"
        print(f"  {name:28} ${result.final_value:>10,.0f}  {result.best_return:+6.1f}% ± {result.return_std_pct:4.1f}%  {result.best_sharpe:6.2f}  {result.best_drawdown:6.1f}%  {status}")

    # Save to config
    print("\n" + "-" * 70)
    print("  Saving optimized parameters to config/config.yaml...")

    try:
        optimizer.save_optimized_params(all_results)
        print("  ✓ Parameters saved successfully!")

        # Also save detailed results to JSON for future analysis
        json_file = optimizer.save_detailed_results(all_results)
        print(f"  ✓ Detailed results saved to: {json_file}")

        print("\n  To use new parameters:")
        print("    1. Restart stockpulse run/dashboard")
        print("    2. Commit and push: git add config/config.yaml && git commit -m 'Optimized params' && git push")
    except Exception as e:
        logger.error(f"Failed to save params: {e}")
        print(f"  ✗ Failed to save: {e}")

    print("\n" + "=" * 70 + "\n")


def run_reset(keep_market_data: bool = True):
    """Reset trading data while keeping historical prices."""
    from stockpulse.data.database import reset_trading_data

    print("\n" + "=" * 60)
    print("  STOCKPULSE DATA RESET")
    print("=" * 60)

    if keep_market_data:
        print("\n  Mode: Reset trading data ONLY")
        print("  ✓ Will KEEP: prices_daily, prices_intraday, fundamentals, universe")
        print("  ✗ Will DELETE: signals, positions, alerts, backtest_results")
    else:
        print("\n  Mode: FULL RESET (including historical data)")
        print("  ✗ Will DELETE: ALL data including historical prices")
        print("  ⚠ You will need to run 'stockpulse init' to fetch data again")

    print("\n  Resetting...")

    deleted = reset_trading_data(keep_market_data=keep_market_data)

    print("\n  Deleted records:")
    for table, count in deleted.items():
        print(f"    {table}: {count}")

    total = sum(deleted.values())
    print(f"\n  Total: {total} records deleted")

    if keep_market_data:
        print("\n  ✓ Historical market data preserved!")
        print("  Ready to run: stockpulse optimize")
    else:
        print("\n  ✓ All data cleared!")
        print("  Run: stockpulse init")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
