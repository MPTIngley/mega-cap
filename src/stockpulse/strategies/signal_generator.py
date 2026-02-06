"""Signal Generator - runs strategies and generates trading signals."""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

from .base import BaseStrategy, Signal, SignalDirection
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy
from .macd_volume import MACDVolumeStrategy
from .zscore_mean_reversion import ZScoreMeanReversionStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .gap_fade import GapFadeStrategy
from .week52_low_bounce import Week52LowBounceStrategy
from .sector_rotation import SectorRotationStrategy

logger = get_logger(__name__)


class SignalGenerator:
    """
    Runs all enabled strategies and generates trading signals.

    Handles:
    - Loading price data
    - Running each strategy
    - Applying ensemble logic
    - Filtering duplicate signals
    - Storing signals in database
    """

    def __init__(self):
        """Initialize signal generator."""
        self.db = get_db()
        self.config = get_config()
        self.data_ingestion = DataIngestion()
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> list[BaseStrategy]:
        """Load and initialize all enabled strategies."""
        strategies = []
        strategy_configs = self.config.get("strategies", {})

        strategy_classes = {
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "bollinger_squeeze": BollingerSqueezeStrategy,
            "macd_volume": MACDVolumeStrategy,
            "zscore_mean_reversion": ZScoreMeanReversionStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
            "gap_fade": GapFadeStrategy,
            "week52_low_bounce": Week52LowBounceStrategy,
            "sector_rotation": SectorRotationStrategy,
        }

        for name, cls in strategy_classes.items():
            config = strategy_configs.get(name, {})
            if config.get("enabled", True):
                try:
                    strategy = cls(config)
                    strategies.append(strategy)
                    logger.info(f"Loaded strategy: {name}")
                except Exception as e:
                    logger.error(f"Failed to load strategy {name}: {e}")

        return strategies

    def generate_signals(self, tickers: list[str]) -> list[Signal]:
        """
        Generate signals for all tickers using all enabled strategies.

        Args:
            tickers: List of ticker symbols to scan

        Returns:
            List of generated signals
        """
        if not tickers:
            return []

        # Clear ALL open signals before generating new ones
        # This ensures dashboard and console show consistent, current data
        # If market conditions changed, old signals are no longer valid
        self.clear_all_open_signals()

        logger.info(f"Generating signals for {len(tickers)} tickers using {len(self.strategies)} strategies")

        all_signals = []

        # Get historical price data from database
        end_date = date.today()
        start_date = end_date - timedelta(days=365)  # 1 year of history

        price_data = self.data_ingestion.get_daily_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )

        if price_data.empty:
            logger.warning("No price data available")
            return []

        # Fetch LIVE prices for all tickers to ensure today's data is accurate
        logger.info(f"Fetching live prices for {len(tickers)} tickers...")
        live_prices = self.data_ingestion.fetch_current_prices(tickers)
        logger.info(f"Got live prices for {len(live_prices)} tickers")

        # Update today's prices in price_data with live data
        if live_prices:
            today_str = str(end_date)
            for ticker, live_price in live_prices.items():
                mask = (price_data["ticker"] == ticker) & (price_data["date"].astype(str) == today_str)
                if mask.any():
                    # Update existing today's row
                    price_data.loc[mask, "close"] = live_price
                    price_data.loc[mask, "high"] = max(price_data.loc[mask, "high"].iloc[0], live_price)
                    price_data.loc[mask, "low"] = min(price_data.loc[mask, "low"].iloc[0], live_price)
                else:
                    # Add today's row if missing
                    last_row = price_data[price_data["ticker"] == ticker].iloc[-1:].copy() if not price_data[price_data["ticker"] == ticker].empty else None
                    if last_row is not None and not last_row.empty:
                        new_row = last_row.copy()
                        new_row["date"] = end_date
                        new_row["open"] = live_price
                        new_row["high"] = live_price
                        new_row["low"] = live_price
                        new_row["close"] = live_price
                        price_data = pd.concat([price_data, new_row], ignore_index=True)

        # Process each ticker
        for ticker in tickers:
            ticker_data = price_data[price_data["ticker"] == ticker].copy()

            if len(ticker_data) < 60:  # Need minimum history
                continue

            ticker_data = ticker_data.sort_values("date")

            # Run each strategy
            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(ticker_data, ticker)

                    for signal in signals:
                        # Check for duplicate/recent signals
                        if not self._is_duplicate_signal(signal):
                            all_signals.append(signal)

                except Exception as e:
                    logger.error(f"Error running {strategy.name} on {ticker}: {e}")

        # Apply ensemble logic
        all_signals = self._apply_ensemble_logic(all_signals)

        # Fetch current/live prices and update entry prices
        if all_signals:
            signal_tickers = list(set(s.ticker for s in all_signals))
            current_prices = self.data_ingestion.fetch_current_prices(signal_tickers)

            for signal in all_signals:
                if signal.ticker in current_prices:
                    old_entry = signal.entry_price
                    current_price = current_prices[signal.ticker]
                    signal.entry_price = current_price

                    # Recalculate target and stop based on current price
                    # Maintain the same percentage distances
                    if old_entry > 0:
                        target_pct = (signal.target_price - old_entry) / old_entry
                        stop_pct = (signal.stop_price - old_entry) / old_entry
                        signal.target_price = current_price * (1 + target_pct)
                        signal.stop_price = current_price * (1 + stop_pct)

        # Store signals
        self._store_signals(all_signals)

        logger.info(f"Generated {len(all_signals)} signals")
        return all_signals

    def _is_duplicate_signal(self, signal: Signal) -> bool:
        """Check if a similar signal was recently generated."""
        risk_config = self.config.get("risk_management", {})
        min_days = risk_config.get("min_days_between_same_ticker", 3)

        # Check database for recent signals (SQLite datetime syntax)
        result = self.db.fetchone("""
            SELECT COUNT(*) FROM signals
            WHERE ticker = ?
            AND strategy = ?
            AND direction = ?
            AND created_at > datetime('now', ?)
            AND status = 'open'
        """, (signal.ticker, signal.strategy, signal.direction.value, f'-{min_days} days'))

        return result[0] > 0 if result else False

    def _apply_ensemble_logic(self, signals: list[Signal]) -> list[Signal]:
        """
        Apply ensemble logic - boost confidence when multiple strategies agree.

        Args:
            signals: List of signals from all strategies

        Returns:
            List of signals with adjusted confidence
        """
        if not signals:
            return []

        # Group signals by ticker and direction
        from collections import defaultdict
        grouped = defaultdict(list)

        for signal in signals:
            key = (signal.ticker, signal.direction)
            grouped[key].append(signal)

        boosted_signals = []

        for (ticker, direction), ticker_signals in grouped.items():
            if len(ticker_signals) >= 2:
                # Multiple strategies agree - boost all
                strategies_agreeing = [s.strategy for s in ticker_signals]
                boost_factor = 1 + (0.05 * (len(ticker_signals) - 1))  # 5% boost per additional strategy

                for signal in ticker_signals:
                    signal.confidence = min(100, signal.confidence * boost_factor)
                    signal.notes += f" | Ensemble: {len(ticker_signals)} strategies agree"

            boosted_signals.extend(ticker_signals)

        return boosted_signals

    def _store_signals(self, signals: list[Signal]) -> None:
        """Store signals in database."""
        stored_count = 0
        for signal in signals:
            try:
                self.db.execute("""
                    INSERT INTO signals (ticker, strategy, direction, confidence, entry_price, target_price, stop_price, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
                """, (
                    signal.ticker,
                    signal.strategy,
                    signal.direction.value,
                    signal.confidence,
                    signal.entry_price,
                    signal.target_price,
                    signal.stop_price,
                    signal.notes
                ))
                stored_count += 1
            except Exception as e:
                logger.error(f"Error storing signal for {signal.ticker}: {e}")

        if stored_count > 0:
            logger.info(f"Stored {stored_count} signals to database")
        elif signals:
            logger.warning(f"Failed to store any of {len(signals)} signals")

    def get_open_signals(self, include_stale: bool = False) -> pd.DataFrame:
        """
        Get open signals.

        Args:
            include_stale: If False (default), only returns signals from today.
                          If True, returns all open signals regardless of age.

        Returns:
            DataFrame of open signals
        """
        if include_stale:
            return self.db.fetchdf("""
                SELECT * FROM signals
                WHERE status = 'open'
                ORDER BY confidence DESC, created_at DESC
            """)
        else:
            # Only return signals from today to ensure consistency with current market conditions
            return self.db.fetchdf("""
                SELECT * FROM signals
                WHERE status = 'open'
                AND date(created_at) = date('now')
                ORDER BY confidence DESC, created_at DESC
            """)

    def get_signals_by_ticker(self, ticker: str) -> pd.DataFrame:
        """Get signals for a specific ticker."""
        return self.db.fetchdf("""
            SELECT * FROM signals
            WHERE ticker = ?
            ORDER BY created_at DESC
        """, (ticker,))

    def close_signal(
        self,
        signal_id: int,
        status: str = "closed",
        notes: str | None = None
    ) -> None:
        """Close a signal."""
        if notes:
            self.db.execute("""
                UPDATE signals
                SET status = ?, notes = COALESCE(notes, '') || ' | ' || ?
                WHERE id = ?
            """, (status, notes, signal_id))
        else:
            self.db.execute("""
                UPDATE signals
                SET status = ?
                WHERE id = ?
            """, (status, signal_id))

    def expire_old_signals(self, max_age_days: int = 1) -> int:
        """
        Expire signals older than max_age_days.

        Signals are time-sensitive - if market conditions have changed,
        old signals may no longer be valid. This ensures we don't act
        on stale signals.

        Args:
            max_age_days: Signals older than this are expired (default 1 day)

        Returns:
            Number of signals expired
        """
        result = self.db.execute("""
            UPDATE signals
            SET status = 'expired'
            WHERE status = 'open'
            AND date(created_at) < date('now', ?)
        """, (f'-{max_age_days} days',))

        expired_count = result.rowcount if hasattr(result, 'rowcount') else 0
        if expired_count > 0:
            logger.info(f"Expired {expired_count} signals older than {max_age_days} day(s)")
        return expired_count

    def clear_all_open_signals(self) -> int:
        """
        Clear all open signals before generating new ones.

        This ensures the database always reflects the CURRENT market state.
        Called at the start of each scan to prevent stale signals from
        showing as actionable when market conditions have changed.

        Returns:
            Number of signals cleared
        """
        result = self.db.execute("""
            UPDATE signals
            SET status = 'expired'
            WHERE status = 'open'
        """)

        cleared_count = result.rowcount if hasattr(result, 'rowcount') else 0
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} open signals before new scan")
        return cleared_count
