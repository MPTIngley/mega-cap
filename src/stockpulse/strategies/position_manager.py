"""Position Manager - manages paper and real trading positions.

Includes smart trading logic:
- Cooldown periods after losses
- Loss tracking per ticker (avoid repeat bad trades)
- Concentration limits (max per sector)
- Churn prevention (no rapid buy/sell cycles)
"""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

from .base import Signal, SignalDirection

logger = get_logger(__name__)

# Smart trading constants
LOSS_COOLDOWN_DAYS = 7  # Days to wait before re-entering a losing ticker
MAX_LOSSES_PER_TICKER = 3  # Max consecutive losses before blocking ticker
CHURN_COOLDOWN_DAYS = 3  # Days to wait before re-entering after any exit
MIN_DAYS_BETWEEN_SAME_TICKER = 2  # Minimum days between trades on same ticker


class PositionManager:
    """
    Manages paper trading positions.

    Handles:
    - Opening positions from signals
    - Tracking open positions
    - Closing positions (target, stop, expiry)
    - P&L calculation with transaction costs
    """

    def __init__(self):
        """Initialize position manager."""
        import os

        self.db = get_db()
        self.config = get_config()
        self.trading_config = self.config.get("trading", {})
        self.risk_config = self.config.get("risk_management", {})
        self.portfolio_config = self.config.get("portfolio", {})
        self.data_ingestion = DataIngestion()

        # Portfolio settings (env vars override config)
        env_capital = os.environ.get("STOCKPULSE_INITIAL_CAPITAL")
        self.initial_capital = float(env_capital) if env_capital else self.portfolio_config.get("initial_capital", 100000.0)

        env_position_pct = os.environ.get("STOCKPULSE_POSITION_SIZE_PCT")
        self.position_size_pct = float(env_position_pct) if env_position_pct else self.portfolio_config.get("base_position_size_pct", 15.0)

        env_max_positions = os.environ.get("STOCKPULSE_MAX_POSITIONS")
        self.max_positions = int(env_max_positions) if env_max_positions else self.portfolio_config.get("max_positions", 25)

        # Concentration limits
        self.max_per_strategy_pct = self.risk_config.get("max_per_strategy_pct", 40.0)
        self.max_position_size_pct = self.risk_config.get("max_position_size_pct", 15.0)
        self.min_position_size_pct = self.risk_config.get("min_position_size_pct", 3.0)

        # Position sizing: base * strategy_weight * confidence_multiplier
        self.confidence_config = self.config.get("confidence_scaling", {})
        self.base_size_pct = self.confidence_config.get("base_size_pct", 5.0)
        self.confidence_75_mult = self.confidence_config.get("confidence_75_multiplier", 2.0)
        self.confidence_85_mult = self.confidence_config.get("confidence_85_multiplier", 3.0)
        self.strategy_allocation = self.config.get("strategy_allocation", {})

        # Transaction costs
        self.commission = self.trading_config.get("commission_per_trade", 0.0)
        self.slippage_pct = self.trading_config.get("slippage_percent", 0.05)
        self.spread_pct = self.trading_config.get("spread_percent", 0.02)

        # Smart trading: track losses and cooldowns (in-memory cache, rebuilt from DB)
        self._loss_count_cache: dict[str, int] = {}
        self._last_exit_cache: dict[str, datetime] = {}
        self._rebuild_trade_history_cache()

        # Position add-on cooldown (days to wait before adding to existing position)
        self.position_add_cooldown_days = self.risk_config.get("position_add_cooldown_days", 7)

    def calculate_transaction_cost(self, price: float, shares: float) -> float:
        """
        Calculate total transaction cost for a trade.

        Includes:
        - Commission (if any)
        - Slippage estimate
        - Spread cost
        """
        trade_value = price * shares

        # Spread cost (half the spread on entry, half on exit)
        spread_cost = trade_value * (self.spread_pct / 100) / 2

        # Slippage
        slippage_cost = trade_value * (self.slippage_pct / 100)

        return self.commission + spread_cost + slippage_cost

    def _rebuild_trade_history_cache(self) -> None:
        """Rebuild loss count and last exit caches from database."""
        try:
            # Get recent closed positions to build loss tracking
            # Use string date format for SQLite comparison
            cutoff_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            recent_closed = self.db.fetchdf("""
                SELECT ticker, pnl, exit_date, exit_reason
                FROM positions_paper
                WHERE status = 'closed'
                AND exit_date >= ?
                ORDER BY ticker, exit_date DESC
            """, (cutoff_date,))

            self._loss_count_cache.clear()
            self._last_exit_cache.clear()

            if recent_closed.empty:
                return

            # Count consecutive losses per ticker and track last exit
            for ticker in recent_closed["ticker"].unique():
                ticker_trades = recent_closed[recent_closed["ticker"] == ticker].sort_values("exit_date", ascending=False)

                # Track last exit time
                last_exit = ticker_trades.iloc[0]["exit_date"]
                if isinstance(last_exit, str):
                    last_exit = datetime.fromisoformat(last_exit)
                self._last_exit_cache[ticker] = last_exit

                # Count consecutive losses (stop at first win)
                loss_count = 0
                for _, trade in ticker_trades.iterrows():
                    if trade["pnl"] < 0:
                        loss_count += 1
                    else:
                        break  # Stop counting at first win

                if loss_count > 0:
                    self._loss_count_cache[ticker] = loss_count

            logger.debug(f"Rebuilt trade cache: {len(self._loss_count_cache)} tickers with losses, {len(self._last_exit_cache)} with recent exits")

        except Exception as e:
            logger.warning(f"Error rebuilding trade history cache: {e}")

    def _check_cooldown(self, ticker: str) -> tuple[bool, str]:
        """
        Check if ticker is in cooldown period.

        Returns:
            (is_allowed, reason) - True if trade is allowed
        """
        last_exit = self._last_exit_cache.get(ticker)
        if last_exit is None:
            return True, ""

        days_since_exit = (datetime.now() - last_exit).days

        # Check churn cooldown (any exit)
        if days_since_exit < CHURN_COOLDOWN_DAYS:
            return False, f"Churn cooldown: {CHURN_COOLDOWN_DAYS - days_since_exit} days remaining"

        # Check loss cooldown (if last trade was a loss)
        loss_count = self._loss_count_cache.get(ticker, 0)
        if loss_count > 0 and days_since_exit < LOSS_COOLDOWN_DAYS:
            return False, f"Loss cooldown: {LOSS_COOLDOWN_DAYS - days_since_exit} days remaining"

        return True, ""

    def _check_loss_limit(self, ticker: str) -> tuple[bool, str]:
        """
        Check if ticker has exceeded max consecutive losses.

        Returns:
            (is_allowed, reason) - True if trade is allowed
        """
        loss_count = self._loss_count_cache.get(ticker, 0)
        if loss_count >= MAX_LOSSES_PER_TICKER:
            return False, f"Max losses ({MAX_LOSSES_PER_TICKER}) reached for {ticker}"
        return True, ""

    def _check_sector_concentration(self, ticker: str) -> tuple[bool, str]:
        """
        Check if adding this ticker would exceed sector concentration limits.

        Returns:
            (is_allowed, reason) - True if trade is allowed
        """
        try:
            # Get sector for this ticker
            ticker_sector = self.db.fetchone(
                "SELECT sector FROM universe WHERE ticker = ?", (ticker,)
            )
            if not ticker_sector or not ticker_sector[0]:
                return True, ""  # Unknown sector, allow

            sector = ticker_sector[0]

            # Get current open positions and their sectors
            open_positions = self.db.fetchdf("""
                SELECT pp.ticker, pp.entry_price, pp.shares, u.sector
                FROM positions_paper pp
                LEFT JOIN universe u ON pp.ticker = u.ticker
                WHERE pp.status = 'open'
            """)

            # Calculate current sector exposure
            sector_value = 0.0
            if not open_positions.empty:
                sector_positions = open_positions[open_positions["sector"] == sector]
                if not sector_positions.empty:
                    sector_value = (sector_positions["entry_price"] * sector_positions["shares"]).sum()

            # Calculate new position value (use base size for estimation)
            new_position_value = self.initial_capital * (self.base_size_pct / 100)

            # Check concentration relative to TOTAL CAPITAL (not just invested amount)
            new_sector_value = sector_value + new_position_value
            new_concentration_pct = (new_sector_value / self.initial_capital) * 100

            # Use configurable limit (default 40%)
            max_sector_pct = self.risk_config.get("max_sector_concentration_pct", 40.0)

            if new_concentration_pct > max_sector_pct:
                return False, f"Sector {sector} would be {new_concentration_pct:.0f}% of capital (max {max_sector_pct:.0f}%)"

            return True, ""

        except Exception as e:
            logger.warning(f"Error checking sector concentration: {e}")
            return True, ""  # Allow on error

    def _update_loss_cache_on_close(self, ticker: str, pnl: float) -> None:
        """Update loss tracking cache when a position closes."""
        self._last_exit_cache[ticker] = datetime.now()

        if pnl < 0:
            self._loss_count_cache[ticker] = self._loss_count_cache.get(ticker, 0) + 1
        else:
            # Win resets the consecutive loss count
            self._loss_count_cache[ticker] = 0

    def calculate_position_size_pct(self, signal: Signal, return_details: bool = False) -> float | tuple[float, dict]:
        """
        Calculate position size percentage based on confidence and strategy.

        Uses CONTINUOUS confidence scaling:
        - multiplier = 1.0 + (confidence - min_conf) / (100 - min_conf) * (max_mult - 1.0)
        - This gives smooth scaling from 1.0x at min_confidence to max_mult at 100%

        Formula: base_size * strategy_weight * confidence_multiplier
        Capped at max_position_size_pct.

        Args:
            signal: The signal to size
            return_details: If True, also return calculation details dict

        Returns:
            Position size as percentage of capital (or tuple with details if return_details=True)
        """
        # Get strategy allocation weight (default 1.0)
        strategy_weight = self.strategy_allocation.get(signal.strategy, 1.0)

        # Continuous confidence multiplier
        # Scale from 1.0x at min_confidence to max_mult at 100%
        confidence = signal.confidence
        min_conf = self.confidence_config.get("min_confidence", 60)  # Below this, use 1.0x
        max_mult = self.confidence_config.get("max_multiplier", 2.5)  # At 100% confidence

        if confidence <= min_conf:
            confidence_mult = 1.0
        elif min_conf >= 100:
            # Edge case: prevent division by zero if min_conf >= 100
            confidence_mult = max_mult
        else:
            # Linear interpolation: 1.0 at min_conf, max_mult at 100
            confidence_mult = 1.0 + (confidence - min_conf) / (100 - min_conf) * (max_mult - 1.0)

        # Calculate raw size before caps
        raw_size = self.base_size_pct * strategy_weight * confidence_mult

        # Apply caps
        final_size = min(raw_size, self.max_position_size_pct)
        final_size = max(final_size, self.min_position_size_pct)

        was_capped = raw_size > self.max_position_size_pct

        logger.debug(
            f"Position sizing for {signal.ticker}: "
            f"base={self.base_size_pct}% × strategy={strategy_weight:.1f} × conf_mult={confidence_mult:.2f} "
            f"= {raw_size:.1f}% → {final_size:.1f}% (conf={confidence}%{', CAPPED' if was_capped else ''})"
        )

        if return_details:
            details = {
                "base_size_pct": self.base_size_pct,
                "strategy_weight": strategy_weight,
                "confidence": confidence,
                "confidence_mult": round(confidence_mult, 2),
                "raw_size_pct": round(raw_size, 1),
                "final_size_pct": round(final_size, 1),
                "was_capped": was_capped,
                "max_position_pct": self.max_position_size_pct,
            }
            return final_size, details

        return final_size

    def open_position_from_signal(
        self,
        signal: Signal,
        capital: float | None = None,
        override_size_pct: float | None = None
    ) -> int | None:
        """
        Open a paper position from a signal.

        Args:
            signal: Signal to open position from
            capital: Capital to allocate (defaults to initial_capital from config)
            override_size_pct: If provided, use this position size instead of calculating.
                               Useful when caller has already reduced size to fit limits.

        Returns:
            Position ID or None if failed
        """
        if capital is None:
            capital = self.initial_capital

        # Check risk limits (pass override size for accurate concentration check)
        if not self._check_risk_limits(signal, override_size_pct):
            logger.info(f"Risk limits prevent opening position for {signal.ticker}")
            return None

        # Calculate position size using confidence-based formula, or use override
        if override_size_pct is not None:
            position_size_pct = override_size_pct
        else:
            position_size_pct = self.calculate_position_size_pct(signal)
        max_position_value = capital * (position_size_pct / 100)

        # Apply slippage to entry price
        if signal.direction == SignalDirection.BUY:
            entry_price = signal.entry_price * (1 + self.slippage_pct / 100)
        else:
            entry_price = signal.entry_price * (1 - self.slippage_pct / 100)

        shares = max_position_value / entry_price

        # Calculate entry transaction cost
        entry_cost = self.calculate_transaction_cost(entry_price, shares)

        try:
            # Insert position (id auto-generated via sequence default)
            self.db.execute("""
                INSERT INTO positions_paper (signal_id, ticker, direction, entry_price, entry_date, shares, status, strategy)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
            """, (
                None,  # signal_id can be linked if needed
                signal.ticker,
                signal.direction.value,
                entry_price,
                datetime.now(),
                shares,
                signal.strategy
            ))

            # Get the inserted ID (most recent for this ticker/strategy combo)
            result = self.db.fetchone("""
                SELECT id FROM positions_paper
                WHERE ticker = ? AND strategy = ? AND status = 'open'
                ORDER BY entry_date DESC LIMIT 1
            """, (signal.ticker, signal.strategy))
            position_id = result[0] if result else None

            logger.info(f"Opened position {position_id}: {signal.direction.value} {shares:.2f} shares of {signal.ticker} @ ${entry_price:.2f} ({position_size_pct:.1f}% of capital, conf={signal.confidence}%)")

            return position_id

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None

    def get_strategy_current_exposure_pct(self, strategy: str) -> float:
        """
        Get current exposure percentage for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Current exposure as percentage of initial capital
        """
        try:
            strategy_positions = self.db.fetchdf("""
                SELECT ticker, entry_price, shares
                FROM positions_paper
                WHERE status = 'open' AND strategy = ?
            """, (strategy,))

            if strategy_positions.empty:
                return 0.0

            current_strategy_value = (strategy_positions["entry_price"] * strategy_positions["shares"]).sum()
            return (current_strategy_value / self.initial_capital) * 100

        except Exception as e:
            logger.warning(f"Error getting strategy exposure: {e}")
            return 0.0

    def get_strategy_remaining_capacity_pct(self, strategy: str) -> float:
        """
        Get remaining capacity percentage for a strategy before hitting the limit.

        Args:
            strategy: Strategy name

        Returns:
            Remaining capacity as percentage of initial capital
        """
        current_pct = self.get_strategy_current_exposure_pct(strategy)
        remaining = self.max_per_strategy_pct - current_pct
        return max(0.0, remaining)

    def _check_strategy_concentration(self, signal: Signal, override_size_pct: float | None = None) -> tuple[bool, str]:
        """
        Check if adding this position would exceed per-strategy concentration limit.

        Args:
            signal: The signal being considered
            override_size_pct: If provided, use this size instead of calculating

        Returns:
            (is_allowed, reason) - True if trade is allowed
        """
        try:
            strategy = signal.strategy
            current_strategy_pct = self.get_strategy_current_exposure_pct(strategy)

            # Calculate the size this position would be
            if override_size_pct is not None:
                new_position_pct = override_size_pct
            else:
                new_position_pct = self.calculate_position_size_pct(signal)

            new_strategy_pct = current_strategy_pct + new_position_pct

            if new_strategy_pct > self.max_per_strategy_pct:
                return False, f"Strategy {strategy} would be {new_strategy_pct:.1f}% of capital (max {self.max_per_strategy_pct:.0f}%)"

            return True, ""

        except Exception as e:
            logger.warning(f"Error checking strategy concentration: {e}")
            return True, ""  # Allow on error

    def _can_add_to_position(self, ticker: str, new_size_pct: float) -> tuple[bool, str]:
        """
        Check if we can add to an existing position for this ticker.

        Allows adding to positions if:
        1. Current position + new size <= max_position_size_pct
        2. At least position_add_cooldown_days (default 7) since last purchase

        Args:
            ticker: The ticker symbol
            new_size_pct: The position size we want to add

        Returns:
            (can_add, reason) - True if we can add, reason if not
        """
        # Get current position info for this ticker
        result = self.db.fetchone("""
            SELECT SUM(entry_price * shares) as total_value,
                   MAX(entry_date) as last_buy_date
            FROM positions_paper
            WHERE ticker = ? AND status = 'open'
        """, (ticker,))

        if not result or result[0] is None:
            # No existing position - can add (this is a new position)
            return True, ""

        current_value = result[0]
        last_buy_date_str = result[1]

        # Check position size limit
        current_pct = (current_value / self.initial_capital) * 100
        new_total_pct = current_pct + new_size_pct

        if new_total_pct > self.max_position_size_pct:
            return False, f"Position would be {new_total_pct:.1f}% (max {self.max_position_size_pct:.0f}%)"

        # Check cooldown since last purchase
        try:
            if last_buy_date_str:
                if isinstance(last_buy_date_str, str):
                    last_buy_date = datetime.fromisoformat(last_buy_date_str.replace('Z', '+00:00'))
                else:
                    last_buy_date = last_buy_date_str

                days_since_buy = (datetime.now() - last_buy_date).days
                if days_since_buy < self.position_add_cooldown_days:
                    remaining = self.position_add_cooldown_days - days_since_buy
                    return False, f"Add-on cooldown: {remaining} days remaining"
        except Exception as e:
            logger.warning(f"Error parsing last buy date for {ticker}: {e}")

        return True, ""

    def _check_risk_limits(self, signal: Signal, override_size_pct: float | None = None) -> bool:
        """
        Check if opening a position would violate risk limits.

        Args:
            signal: The signal to check
            override_size_pct: If provided, use this size for concentration checks

        Checks:
        1. Max concurrent positions (for NEW positions only)
        2. Per-stock position limit and add-on cooldown
        3. Cooldown periods (churn prevention)
        4. Loss limits per ticker
        5. Sector concentration limits
        6. Per-strategy concentration limits
        """
        ticker = signal.ticker

        # Calculate position size for this signal
        if override_size_pct is not None:
            new_size_pct = override_size_pct
        else:
            new_size_pct = self.calculate_position_size_pct(signal)

        # Check if we already have a position in this ticker
        existing = self.db.fetchone(
            "SELECT COUNT(*) FROM positions_paper WHERE ticker = ? AND status = 'open'",
            (ticker,)
        )
        has_existing_position = existing and existing[0] > 0

        if has_existing_position:
            # Check if we can add to the existing position
            can_add, add_reason = self._can_add_to_position(ticker, new_size_pct)
            if not can_add:
                logger.info(f"Cannot add to {ticker}: {add_reason}")
                return False
        else:
            # New position - check max concurrent positions
            open_count = self.db.fetchone(
                "SELECT COUNT(*) FROM positions_paper WHERE status = 'open'"
            )
            if open_count and open_count[0] >= self.max_positions:
                logger.warning(f"Max positions ({self.max_positions}) reached")
                return False

        # Check cooldown periods (churn and loss cooldowns)
        cooldown_ok, cooldown_reason = self._check_cooldown(ticker)
        if not cooldown_ok:
            logger.info(f"Cooldown prevents {ticker}: {cooldown_reason}")
            return False

        # Check consecutive loss limit
        loss_ok, loss_reason = self._check_loss_limit(ticker)
        if not loss_ok:
            logger.info(f"Loss limit prevents {ticker}: {loss_reason}")
            return False

        # Check sector concentration
        sector_ok, sector_reason = self._check_sector_concentration(ticker)
        if not sector_ok:
            logger.info(f"Concentration prevents {ticker}: {sector_reason}")
            return False

        # Check per-strategy concentration (pass override size if provided)
        strategy_ok, strategy_reason = self._check_strategy_concentration(signal, override_size_pct)
        if not strategy_ok:
            logger.info(f"Strategy concentration prevents {ticker}: {strategy_reason}")
            return False

        return True

    def get_signal_blocking_reasons(self, ticker: str, strategy: str, confidence: float = 70) -> dict:
        """
        Get comprehensive blocking reasons for a signal.

        Args:
            ticker: The ticker symbol
            strategy: The strategy name
            confidence: Signal confidence (0-100) for accurate position size calculation

        Returns:
            Dict with:
                - can_trade: bool
                - status: str (emoji + status)
                - reason: str (human readable)
                - details: list of all checks and their results
        """
        details = []
        blocking_reasons = []

        # Calculate position size for accurate checks
        dummy_signal = Signal(
            ticker=ticker,
            strategy=strategy,
            direction=SignalDirection.BUY,
            confidence=confidence,
            entry_price=100,
            target_price=110,
            stop_price=95
        )
        new_size_pct = self.calculate_position_size_pct(dummy_signal)

        # Check if already have position
        existing = self.db.fetchone(
            "SELECT COUNT(*) FROM positions_paper WHERE ticker = ? AND status = 'open'",
            (ticker,)
        )[0] or 0
        already_held = existing > 0

        # 1. Check max positions (only for NEW positions)
        open_count = self.db.fetchone(
            "SELECT COUNT(*) FROM positions_paper WHERE status = 'open'"
        )[0] or 0

        if not already_held:
            at_max_positions = open_count >= self.max_positions
            details.append({
                "check": "Max positions",
                "passed": not at_max_positions,
                "info": f"{open_count}/{self.max_positions} open"
            })
            if at_max_positions:
                blocking_reasons.append(f"Max positions ({self.max_positions}) reached")
        else:
            details.append({
                "check": "Max positions",
                "passed": True,
                "info": f"{open_count}/{self.max_positions} (adding to existing)"
            })

        # 2. Check if can add to existing position or open new
        if already_held:
            can_add, add_reason = self._can_add_to_position(ticker, new_size_pct)
            details.append({
                "check": "Add to position",
                "passed": can_add,
                "info": add_reason if not can_add else "Can add more"
            })
            if not can_add:
                blocking_reasons.append(add_reason)
        else:
            details.append({
                "check": "Add to position",
                "passed": True,
                "info": "New position"
            })

        # 3. Check cooldown
        cooldown_ok, cooldown_reason = self._check_cooldown(ticker)
        details.append({
            "check": "Cooldown",
            "passed": cooldown_ok,
            "info": cooldown_reason if not cooldown_ok else "Clear"
        })
        if not cooldown_ok:
            blocking_reasons.append(cooldown_reason)

        # 4. Check loss limit
        loss_ok, loss_reason = self._check_loss_limit(ticker)
        details.append({
            "check": "Loss limit",
            "passed": loss_ok,
            "info": loss_reason if not loss_ok else "Clear"
        })
        if not loss_ok:
            blocking_reasons.append(loss_reason)

        # 5. Check sector concentration
        sector_ok, sector_reason = self._check_sector_concentration(ticker)
        details.append({
            "check": "Sector limit",
            "passed": sector_ok,
            "info": sector_reason if not sector_ok else "Clear"
        })
        if not sector_ok:
            blocking_reasons.append(sector_reason)

        # 6. Check portfolio exposure limit
        max_exposure_pct = self.risk_config.get("max_portfolio_exposure_pct", 80.0)
        total_invested = self.db.fetchone(
            "SELECT COALESCE(SUM(entry_price * shares), 0) FROM positions_paper WHERE status = 'open'"
        )[0] or 0
        current_exposure_pct = (total_invested / self.initial_capital) * 100 if self.initial_capital > 0 else 0

        # Calculate what this position would add
        dummy_signal = Signal(
            ticker=ticker,
            strategy=strategy,
            direction=SignalDirection.BUY,
            confidence=confidence,
            entry_price=100,
            target_price=110,
            stop_price=95
        )
        new_position_pct = self.calculate_position_size_pct(dummy_signal)
        new_exposure_pct = current_exposure_pct + new_position_pct

        # Check if remaining capacity is below minimum position size
        remaining_exposure_pct = max_exposure_pct - current_exposure_pct
        exposure_ok = remaining_exposure_pct >= self.min_position_size_pct
        details.append({
            "check": "Portfolio exposure",
            "passed": exposure_ok,
            "info": f"{current_exposure_pct:.0f}%/{max_exposure_pct:.0f}% used" if exposure_ok else f"At {current_exposure_pct:.0f}% (max {max_exposure_pct:.0f}%)"
        })
        if not exposure_ok:
            blocking_reasons.append(f"Portfolio exposure at {current_exposure_pct:.0f}% (max {max_exposure_pct:.0f}%)")

        # 7. Check strategy concentration using actual confidence
        current_strategy_pct = self.get_strategy_current_exposure_pct(strategy)
        strategy_remaining = self.max_per_strategy_pct - current_strategy_pct

        # Check if remaining capacity is below minimum position size
        strategy_ok = strategy_remaining >= self.min_position_size_pct
        strategy_info = f"{current_strategy_pct:.0f}%/{self.max_per_strategy_pct:.0f}% used" if strategy_ok else f"At {current_strategy_pct:.0f}% (max {self.max_per_strategy_pct:.0f}%)"
        details.append({
            "check": "Strategy limit",
            "passed": strategy_ok,
            "info": strategy_info
        })
        if not strategy_ok:
            blocking_reasons.append(f"Strategy {strategy} at {current_strategy_pct:.0f}% (max {self.max_per_strategy_pct:.0f}%)")

        # Determine overall status
        can_trade = len(blocking_reasons) == 0

        # Check if we can add to existing position
        if already_held:
            can_add, add_reason = self._can_add_to_position(ticker, new_size_pct)

        if can_trade:
            if already_held:
                status = "✅ ADD MORE"
                reason = "Can add to existing position"
            else:
                status = "✅ ACTIONABLE"
                reason = "Ready to trade"
        elif already_held and not can_add:
            status = "📌 HELD"
            reason = add_reason
        elif not cooldown_ok:
            status = "⏱️ COOLDOWN"
            reason = cooldown_reason
        else:
            status = "🚫 BLOCKED"
            reason = blocking_reasons[0] if blocking_reasons else "Unknown"

        return {
            "can_trade": can_trade,
            "status": status,
            "reason": reason,
            "details": details,
            "all_reasons": blocking_reasons
        }

    def update_positions(self, current_prices: dict[str, float]) -> list[dict]:
        """
        Update all open positions and check for exit conditions.

        Args:
            current_prices: Dictionary of ticker -> current price

        Returns:
            List of position updates (closures, status changes)
        """
        updates = []

        # Get open positions
        open_positions = self.db.fetchdf("""
            SELECT pp.*, s.target_price, s.stop_price
            FROM positions_paper pp
            LEFT JOIN signals s ON pp.signal_id = s.id
            WHERE pp.status = 'open'
        """)

        if open_positions.empty:
            return updates

        for _, position in open_positions.iterrows():
            ticker = position["ticker"]
            current_price = current_prices.get(ticker)

            if current_price is None:
                continue

            # Get signal data for targets/stops
            signal_data = self.db.fetchone("""
                SELECT target_price, stop_price FROM signals
                WHERE ticker = ? AND strategy = ?
                ORDER BY created_at DESC LIMIT 1
            """, (ticker, position.get("strategy", "")))

            if signal_data:
                target_price, stop_price = signal_data
            else:
                # Use default percentages
                entry = position["entry_price"]
                if position["direction"] == "BUY":
                    target_price = entry * 1.10
                    stop_price = entry * 0.95
                else:
                    target_price = entry * 0.90
                    stop_price = entry * 1.05

            # Check exit conditions
            exit_reason = None
            exit_price = None

            if position["direction"] == "BUY":
                if current_price >= target_price:
                    exit_reason = "target"
                    exit_price = current_price * (1 - self.slippage_pct / 100)
                elif current_price <= stop_price:
                    exit_reason = "stop"
                    exit_price = current_price * (1 - self.slippage_pct / 100)
            else:  # SELL/SHORT
                if current_price <= target_price:
                    exit_reason = "target"
                    exit_price = current_price * (1 + self.slippage_pct / 100)
                elif current_price >= stop_price:
                    exit_reason = "stop"
                    exit_price = current_price * (1 + self.slippage_pct / 100)

            # Check for expiry (max hold days)
            entry_date = position["entry_date"]
            if isinstance(entry_date, str):
                entry_date = datetime.fromisoformat(entry_date)
            hold_days = (datetime.now() - entry_date).days

            strategy_config = self.config.get("strategies", {}).get(position.get("strategy", ""), {})
            max_hold = strategy_config.get("hold_days_max", 30)

            if hold_days >= max_hold and exit_reason is None:
                exit_reason = "expired"
                exit_price = current_price

            # Close position if exit condition met
            if exit_reason:
                update = self.close_position(
                    position["id"],
                    exit_price,
                    exit_reason
                )
                if update:
                    updates.append(update)

        return updates

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str
    ) -> dict | None:
        """
        Close a position and calculate P&L.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_reason: Reason for exit (target, stop, expired, manual)

        Returns:
            Dictionary with position closure details
        """
        # Get position details with explicit columns
        position = self.db.fetchone("""
            SELECT id, signal_id, ticker, direction, entry_price, entry_date, shares, status, strategy
            FROM positions_paper WHERE id = ?
        """, (position_id,))

        if not position:
            return None

        # Unpack position
        pos_id, signal_id, ticker, direction, entry_price, entry_date, shares, status, strategy = position

        if status != "open":
            return None

        # Calculate P&L
        if direction == "BUY":
            gross_pnl = (exit_price - entry_price) * shares
        else:
            gross_pnl = (entry_price - exit_price) * shares

        # Subtract transaction costs (entry + exit)
        entry_cost = self.calculate_transaction_cost(entry_price, shares)
        exit_cost = self.calculate_transaction_cost(exit_price, shares)
        total_costs = entry_cost + exit_cost

        net_pnl = gross_pnl - total_costs
        cost_basis = entry_price * shares if entry_price and shares else 0
        pnl_pct = (net_pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Update position
        self.db.execute("""
            UPDATE positions_paper
            SET exit_price = ?,
                exit_date = ?,
                pnl = ?,
                pnl_pct = ?,
                status = 'closed',
                exit_reason = ?
            WHERE id = ?
        """, (exit_price, datetime.now(), net_pnl, pnl_pct, exit_reason, position_id))

        result = {
            "position_id": position_id,
            "ticker": ticker,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "gross_pnl": gross_pnl,
            "transaction_costs": total_costs,
            "net_pnl": net_pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "strategy": strategy
        }

        logger.info(f"Closed position {position_id}: {ticker} {direction} PnL=${net_pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")

        # Update loss tracking cache
        self._update_loss_cache_on_close(ticker, net_pnl)

        return result

    def get_open_positions(self) -> pd.DataFrame:
        """Get all open positions with current P&L."""
        return self.db.fetchdf("""
            SELECT * FROM positions_paper
            WHERE status = 'open'
            ORDER BY entry_date DESC
        """)

    def get_closed_positions(
        self,
        start_date: date | None = None,
        end_date: date | None = None
    ) -> pd.DataFrame:
        """Get closed positions within date range."""
        query = "SELECT * FROM positions_paper WHERE status = 'closed'"
        params = []

        if start_date:
            query += " AND exit_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND exit_date <= ?"
            params.append(end_date)

        query += " ORDER BY exit_date DESC"

        return self.db.fetchdf(query, tuple(params) if params else None)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance summary."""
        # Get all closed positions
        closed = self.get_closed_positions()

        if closed.empty:
            return {
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
            }

        total_trades = len(closed)
        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] <= 0]

        total_pnl = closed["pnl"].sum()
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

        avg_pnl = closed["pnl"].mean()
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_wins": len(wins),
            "total_losses": len(losses),
        }

    def get_strategy_performance(self) -> pd.DataFrame:
        """Get performance breakdown by strategy."""
        return self.db.fetchdf("""
            SELECT
                strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl <= 0 THEN pnl END) as avg_loss,
                AVG(pnl_pct) as avg_pnl_pct
            FROM positions_paper
            WHERE status = 'closed'
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """)

    def get_equity_curve(self, is_paper: bool = True) -> pd.DataFrame:
        """
        Generate equity curve time series from closed positions.

        Args:
            is_paper: True for paper positions, False for real trades

        Returns:
            DataFrame with date, equity, and drawdown columns
        """
        table = "positions_paper" if is_paper else "positions_real"

        # Get all closed positions ordered by exit date
        closed = self.db.fetchdf(f"""
            SELECT exit_date, pnl
            FROM {table}
            WHERE status = 'closed' AND exit_date IS NOT NULL
            ORDER BY exit_date ASC
        """)

        if closed.empty:
            # Return initial capital as starting point
            return pd.DataFrame({
                "date": [datetime.now().date()],
                "equity": [self.initial_capital],
                "drawdown": [0.0]
            })

        # Convert dates and ensure proper format
        closed["exit_date"] = pd.to_datetime(closed["exit_date"]).dt.date

        # Calculate cumulative P&L by date
        daily_pnl = closed.groupby("exit_date")["pnl"].sum().reset_index()
        daily_pnl.columns = ["date", "daily_pnl"]

        # Calculate cumulative equity
        daily_pnl["cumulative_pnl"] = daily_pnl["daily_pnl"].cumsum()
        daily_pnl["equity"] = self.initial_capital + daily_pnl["cumulative_pnl"]

        # Calculate drawdown (with division by zero protection)
        daily_pnl["peak"] = daily_pnl["equity"].cummax()
        daily_pnl["drawdown"] = daily_pnl.apply(
            lambda row: (row["equity"] - row["peak"]) / row["peak"] if row["peak"] > 0 else 0,
            axis=1
        )

        # Add starting point
        start_row = pd.DataFrame({
            "date": [daily_pnl["date"].min() - timedelta(days=1)],
            "equity": [self.initial_capital],
            "drawdown": [0.0]
        })
        result = pd.concat([start_row, daily_pnl[["date", "equity", "drawdown"]]], ignore_index=True)

        return result

    def get_equity_curve_with_open_positions(
        self,
        current_prices: dict[str, float],
        is_paper: bool = True
    ) -> pd.DataFrame:
        """
        Generate equity curve including mark-to-market of open positions.

        Args:
            current_prices: Current prices for open positions
            is_paper: True for paper positions, False for real trades

        Returns:
            DataFrame with date, equity, and drawdown columns
        """
        # Get base equity curve from closed positions
        equity_curve = self.get_equity_curve(is_paper)

        if equity_curve.empty:
            return equity_curve

        # Calculate unrealized P&L from open positions
        table = "positions_paper" if is_paper else "positions_real"
        open_positions = self.db.fetchdf(f"""
            SELECT ticker, direction, entry_price, shares
            FROM {table}
            WHERE status = 'open'
        """)

        unrealized_pnl = 0.0
        if not open_positions.empty:
            for _, pos in open_positions.iterrows():
                ticker = pos["ticker"]
                current_price = current_prices.get(ticker)
                if current_price:
                    if pos["direction"] == "BUY":
                        unrealized_pnl += (current_price - pos["entry_price"]) * pos["shares"]
                    else:
                        unrealized_pnl += (pos["entry_price"] - current_price) * pos["shares"]

        # Add current point with unrealized P&L
        last_equity = equity_curve.iloc[-1]["equity"]
        current_equity = last_equity + unrealized_pnl
        peak = max(equity_curve["equity"].max(), current_equity)
        current_drawdown = (current_equity - peak) / peak if peak > 0 else 0

        current_row = pd.DataFrame({
            "date": [datetime.now().date()],
            "equity": [current_equity],
            "drawdown": [current_drawdown]
        })

        return pd.concat([equity_curve, current_row], ignore_index=True)

    def get_blocked_tickers(self) -> list[dict]:
        """
        Get list of tickers currently blocked from trading.

        Returns:
            List of dicts with ticker, reason, and days_remaining
        """
        # Always refresh cache from DB to get latest closed positions
        self._rebuild_trade_history_cache()

        blocked = []

        # Iterate over ALL recent exits, not just losers
        for ticker, last_exit in self._last_exit_cache.items():
            loss_count = self._loss_count_cache.get(ticker, 0)
            days_since = (datetime.now() - last_exit).days

            # Check if blocked due to max losses
            if loss_count >= MAX_LOSSES_PER_TICKER:
                blocked.append({
                    "ticker": ticker,
                    "reason": f"Max losses ({MAX_LOSSES_PER_TICKER} consecutive)",
                    "loss_count": loss_count,
                    "days_since_exit": days_since
                })
            # Check if blocked due to loss cooldown
            elif loss_count > 0 and days_since < LOSS_COOLDOWN_DAYS:
                blocked.append({
                    "ticker": ticker,
                    "reason": f"Loss cooldown ({LOSS_COOLDOWN_DAYS - days_since} days remaining)",
                    "loss_count": loss_count,
                    "days_since_exit": days_since
                })
            # Check if blocked due to churn cooldown (applies to ALL exits including wins)
            elif days_since < CHURN_COOLDOWN_DAYS:
                blocked.append({
                    "ticker": ticker,
                    "reason": f"Churn cooldown ({CHURN_COOLDOWN_DAYS - days_since} days remaining)",
                    "loss_count": loss_count,
                    "days_since_exit": days_since
                })

        return blocked
