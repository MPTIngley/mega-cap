"""Position Manager - manages paper and real trading positions."""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

from .base import Signal, SignalDirection

logger = get_logger(__name__)


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
        self.position_size_pct = float(env_position_pct) if env_position_pct else self.risk_config.get("max_position_size_pct", 5.0)

        env_max_positions = os.environ.get("STOCKPULSE_MAX_POSITIONS")
        self.max_positions = int(env_max_positions) if env_max_positions else self.risk_config.get("max_concurrent_positions", 20)

        # Transaction costs
        self.commission = self.trading_config.get("commission_per_trade", 0.0)
        self.slippage_pct = self.trading_config.get("slippage_percent", 0.05)
        self.spread_pct = self.trading_config.get("spread_percent", 0.02)

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

        # Calculate position size using configured percentage
        max_position_value = capital * (self.position_size_pct / 100)

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

            logger.info(f"Opened position {position_id}: {signal.direction.value} {shares:.2f} shares of {signal.ticker} @ ${entry_price:.2f}")

            return position_id

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None

    def _check_risk_limits(self, signal: Signal) -> bool:
        """Check if opening a position would violate risk limits."""
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
            (signal.ticker,)
        )
        if existing and existing[0] > 0:
            logger.info(f"Already have open position in {signal.ticker}")
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
