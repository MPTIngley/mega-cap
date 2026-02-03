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
MAX_SECTOR_CONCENTRATION = 0.30  # Max 30% of portfolio in one sector
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
            recent_closed = self.db.fetchdf("""
                SELECT ticker, pnl, exit_date, exit_reason
                FROM positions_paper
                WHERE status = 'closed'
                AND exit_date >= ?
                ORDER BY ticker, exit_date DESC
            """, (datetime.now() - timedelta(days=90),))

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

            if open_positions.empty:
                return True, ""

            # Calculate total portfolio value and sector exposure
            total_value = (open_positions["entry_price"] * open_positions["shares"]).sum()
            sector_positions = open_positions[open_positions["sector"] == sector]
            sector_value = (sector_positions["entry_price"] * sector_positions["shares"]).sum() if not sector_positions.empty else 0

            # Calculate new position value
            new_position_value = self.initial_capital * (self.position_size_pct / 100)

            # Check if new position would exceed concentration limit
            new_sector_value = sector_value + new_position_value
            new_total_value = total_value + new_position_value
            new_concentration = new_sector_value / new_total_value if new_total_value > 0 else 0

            if new_concentration > MAX_SECTOR_CONCENTRATION:
                return False, f"Sector {sector} would be {new_concentration:.0%} of portfolio (max {MAX_SECTOR_CONCENTRATION:.0%})"

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

    def calculate_position_size_pct(self, signal: Signal) -> float:
        """
        Calculate position size percentage based on confidence and strategy.

        Formula: base_size * strategy_weight * confidence_multiplier
        Capped at max_position_size_pct.

        Args:
            signal: The signal to size

        Returns:
            Position size as percentage of capital
        """
        # Get strategy allocation weight (default 1.0)
        strategy_weight = self.strategy_allocation.get(signal.strategy, 1.0)

        # Get confidence multiplier
        confidence = signal.confidence
        if confidence >= 85:
            confidence_mult = self.confidence_85_mult  # 3.0x for super confident
        elif confidence >= 75:
            confidence_mult = self.confidence_75_mult  # 2.0x for confident
        else:
            confidence_mult = 1.0  # Base size for lower confidence

        # Calculate final size
        final_size = self.base_size_pct * strategy_weight * confidence_mult

        # Apply caps
        final_size = min(final_size, self.max_position_size_pct)
        final_size = max(final_size, self.min_position_size_pct)

        logger.debug(
            f"Position sizing for {signal.ticker}: "
            f"base={self.base_size_pct}% × strategy={strategy_weight} × conf_mult={confidence_mult} "
            f"= {final_size:.1f}% (conf={confidence}%)"
        )

        return final_size

    def open_position_from_signal(
        self,
        signal: Signal,
        capital: float | None = None
    ) -> int | None:
        """
        Open a paper position from a signal.

        Args:
            signal: Signal to open position from
            capital: Capital to allocate (defaults to initial_capital from config)

        Returns:
            Position ID or None if failed
        """
        if capital is None:
            capital = self.initial_capital

        # Check risk limits
        if not self._check_risk_limits(signal):
            logger.info(f"Risk limits prevent opening position for {signal.ticker}")
            return None

        # Calculate position size using confidence-based formula
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

    def _check_strategy_concentration(self, strategy: str) -> tuple[bool, str]:
        """
        Check if adding this position would exceed per-strategy concentration limit.

        Returns:
            (is_allowed, reason) - True if trade is allowed
        """
        try:
            # Get current open positions for this strategy
            strategy_positions = self.db.fetchdf("""
                SELECT ticker, entry_price, shares
                FROM positions_paper
                WHERE status = 'open' AND strategy = ?
            """, (strategy,))

            if strategy_positions.empty:
                return True, ""

            # Calculate current strategy exposure
            strategy_value = (strategy_positions["entry_price"] * strategy_positions["shares"]).sum()
            new_position_value = self.initial_capital * (self.position_size_pct / 100)
            new_strategy_value = strategy_value + new_position_value

            strategy_pct = (new_strategy_value / self.initial_capital) * 100

            if strategy_pct > self.max_per_strategy_pct:
                return False, f"Strategy {strategy} would be {strategy_pct:.1f}% of capital (max {self.max_per_strategy_pct:.0f}%)"

            return True, ""

        except Exception as e:
            logger.warning(f"Error checking strategy concentration: {e}")
            return True, ""  # Allow on error

    def _check_risk_limits(self, signal: Signal) -> bool:
        """
        Check if opening a position would violate risk limits.

        Checks:
        1. Max concurrent positions
        2. No duplicate open positions
        3. Cooldown periods (churn prevention)
        4. Loss limits per ticker
        5. Sector concentration limits
        6. Per-strategy concentration limits
        """
        ticker = signal.ticker

        # Check max concurrent positions
        open_count = self.db.fetchone(
            "SELECT COUNT(*) FROM positions_paper WHERE status = 'open'"
        )
        if open_count and open_count[0] >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached")
            return False

        # Check if already have position in this ticker
        existing = self.db.fetchone(
            "SELECT COUNT(*) FROM positions_paper WHERE ticker = ? AND status = 'open'",
            (ticker,)
        )
        if existing and existing[0] > 0:
            logger.info(f"Already have open position in {ticker}")
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

        # Check per-strategy concentration
        strategy_ok, strategy_reason = self._check_strategy_concentration(signal.strategy)
        if not strategy_ok:
            logger.info(f"Strategy concentration prevents {ticker}: {strategy_reason}")
            return False

        return True

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
        pnl_pct = (net_pnl / (entry_price * shares)) * 100

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

        # Calculate drawdown
        daily_pnl["peak"] = daily_pnl["equity"].cummax()
        daily_pnl["drawdown"] = (daily_pnl["equity"] - daily_pnl["peak"]) / daily_pnl["peak"]

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
        blocked = []

        for ticker, loss_count in self._loss_count_cache.items():
            last_exit = self._last_exit_cache.get(ticker)
            if last_exit is None:
                continue

            days_since = (datetime.now() - last_exit).days

            # Check if blocked due to losses
            if loss_count >= MAX_LOSSES_PER_TICKER:
                blocked.append({
                    "ticker": ticker,
                    "reason": f"Max losses ({MAX_LOSSES_PER_TICKER} consecutive)",
                    "loss_count": loss_count,
                    "days_since_exit": days_since
                })
            elif loss_count > 0 and days_since < LOSS_COOLDOWN_DAYS:
                blocked.append({
                    "ticker": ticker,
                    "reason": f"Loss cooldown ({LOSS_COOLDOWN_DAYS - days_since} days remaining)",
                    "loss_count": loss_count,
                    "days_since_exit": days_since
                })
            elif days_since < CHURN_COOLDOWN_DAYS:
                blocked.append({
                    "ticker": ticker,
                    "reason": f"Churn cooldown ({CHURN_COOLDOWN_DAYS - days_since} days remaining)",
                    "loss_count": loss_count,
                    "days_since_exit": days_since
                })

        return blocked
