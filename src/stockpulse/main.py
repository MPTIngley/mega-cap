"""
StockPulse Main Entry Point.

Usage:
    stockpulse run          # Run scheduler and scan
    stockpulse dashboard    # Launch Streamlit dashboard
    stockpulse backtest     # Run backtests
    stockpulse ingest       # Run data ingestion
"""

import sys
import argparse
from pathlib import Path
from datetime import date

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
        choices=["run", "dashboard", "backtest", "ingest", "scan", "init"],
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
    from stockpulse.data.ingestion import DataIngestion
    import pandas as pd
    import numpy as np

    logger.info("Running scan...")

    universe = UniverseManager()
    signal_generator = SignalGenerator()
    alert_manager = AlertManager()
    ingestion = DataIngestion()

    tickers = universe.get_active_tickers()
    if not tickers:
        logger.warning("No tickers in universe")
        return

    # Generate signals
    signals = signal_generator.generate_signals(tickers)

    print("\n" + "=" * 70)
    print("  STOCKPULSE SCAN RESULTS")
    print("=" * 70)

    if signals:
        print(f"\n  SIGNALS GENERATED: {len(signals)}")
        print("-" * 70)
        for signal in signals:
            print(f"  {signal.direction.value:4} | {signal.ticker:5} | {signal.strategy:25} | Conf: {signal.confidence:.0f}%")
            print(f"       Entry: ${signal.entry_price:.2f} | Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}")
    else:
        print("\n  NO SIGNALS - Market conditions don't meet strategy thresholds")

    # Show market snapshot - what's close to triggering
    print("\n" + "-" * 70)
    print("  MARKET SNAPSHOT - Stocks Near Threshold")
    print("-" * 70)

    try:
        # Get recent price data
        from datetime import timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=60)
        prices_df = ingestion.get_daily_prices(tickers[:30], start_date, end_date)  # Sample for speed

        if not prices_df.empty:
            near_triggers = []

            for ticker in prices_df["ticker"].unique():
                ticker_data = prices_df[prices_df["ticker"] == ticker].copy()
                if len(ticker_data) < 20:
                    continue

                ticker_data = ticker_data.sort_values("date")
                latest = ticker_data.iloc[-1]
                close = latest["close"]

                # Calculate RSI
                delta = ticker_data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                # Calculate Z-score
                mean_20 = ticker_data["close"].rolling(20).mean().iloc[-1]
                std_20 = ticker_data["close"].rolling(20).std().iloc[-1]
                zscore = (close - mean_20) / std_20 if std_20 > 0 else 0

                # Check for near-threshold conditions
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
                print("  (RSI between 35-65, Z-score between -1.5 and 1.5)")

    except Exception as e:
        logger.debug(f"Could not generate market snapshot: {e}")
        print("  Could not generate market snapshot")

    print("\n" + "=" * 70)
    print(f"  Scan complete. {len(signals)} signals generated.")
    print("=" * 70 + "\n")

    # Send alerts
    if signals:
        sent = alert_manager.process_signals(signals)
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


if __name__ == "__main__":
    main()
