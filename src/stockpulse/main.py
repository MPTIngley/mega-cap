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
        while True:
            time.sleep(60)
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
    """Run a single scan."""
    from stockpulse.data.universe import UniverseManager
    from stockpulse.strategies.signal_generator import SignalGenerator
    from stockpulse.alerts.alert_manager import AlertManager

    logger.info("Running scan...")

    universe = UniverseManager()
    signal_generator = SignalGenerator()
    alert_manager = AlertManager()

    tickers = universe.get_active_tickers()
    if not tickers:
        logger.warning("No tickers in universe")
        return

    signals = signal_generator.generate_signals(tickers)

    logger.info(f"Generated {len(signals)} signals")

    for signal in signals:
        logger.info(f"  {signal.direction.value} {signal.ticker} - {signal.strategy} - Confidence: {signal.confidence:.0f}%")

    # Send alerts
    sent = alert_manager.process_signals(signals)
    logger.info(f"Sent {sent} alerts")


def run_init():
    """Initialize the database and fetch initial data."""
    from stockpulse.data.database import get_db
    from stockpulse.data.universe import UniverseManager
    from stockpulse.data.ingestion import DataIngestion

    logger.info("Initializing StockPulse...")

    # Initialize database (creates tables)
    db = get_db()
    logger.info("Database initialized")

    # Refresh universe
    universe = UniverseManager()
    tickers = universe.refresh_universe(force=True)
    logger.info(f"Universe populated with {len(tickers)} tickers")

    # Fetch historical data
    ingestion = DataIngestion()
    logger.info("Fetching historical price data (this may take a while)...")

    results = ingestion.run_daily_ingestion(tickers[:50])  # Start with top 50

    logger.info(f"Initial data ingestion complete: {results}")
    logger.info("StockPulse is ready! Run 'stockpulse run' to start the scheduler.")


if __name__ == "__main__":
    main()
