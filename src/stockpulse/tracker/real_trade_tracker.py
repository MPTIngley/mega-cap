"""Real Trade Tracker.

Allows manual entry and tracking of real trades,
with comparison to paper trading performance.
"""

from datetime import datetime, date, timedelta
from typing import Any
import csv
from pathlib import Path

import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

logger = get_logger(__name__)


class RealTradeTracker:
    """
    Tracks real (manual) trades for comparison with paper performance.

    Features:
    - Manual trade entry (buy/sell)
    - Live P&L tracking using market data
    - Comparison with paper trading
    - CSV export
    """

    def __init__(self):
        """Initialize trade tracker."""
        self.db = get_db()
        self.config = get_config()
        self.data_ingestion = DataIngestion()

    def add_trade(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        shares: float,
        entry_date: datetime | date | None = None,
        strategy: str | None = None,
        commission: float = 0.0,
        notes: str | None = None
    ) -> int | None:
        """
        Add a new real trade.

        Args:
            ticker: Stock ticker symbol
            direction: "BUY" or "SELL"
            entry_price: Entry price per share
            shares: Number of shares
            entry_date: Date of trade (default: now)
            strategy: Strategy that generated signal (optional)
            commission: Commission paid
            notes: Additional notes

        Returns:
            Trade ID or None if failed
        """
        if entry_date is None:
            entry_date = datetime.now()
        elif isinstance(entry_date, date):
            entry_date = datetime.combine(entry_date, datetime.min.time())

        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            logger.error(f"Invalid direction: {direction}")
            return None

        try:
            self.db.execute("""
                INSERT INTO positions_real (
                    ticker, direction, entry_price, entry_date,
                    shares, status, strategy, commission, notes
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
            """, (
                ticker.upper(),
                direction,
                entry_price,
                entry_date,
                shares,
                strategy,
                commission,
                notes
            ))

            # Get the inserted trade ID
            result = self.db.fetchone("""
                SELECT id FROM positions_real
                WHERE ticker = ? AND status = 'open'
                ORDER BY entry_date DESC LIMIT 1
            """, (ticker.upper(),))
            trade_id = result[0] if result else None

            logger.info(f"Added real trade {trade_id}: {direction} {shares} {ticker} @ ${entry_price:.2f}")

            return trade_id

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return None

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_date: datetime | date | None = None,
        exit_reason: str = "manual",
        commission: float = 0.0
    ) -> dict | None:
        """
        Close an open trade.

        Args:
            trade_id: ID of trade to close
            exit_price: Exit price per share
            exit_date: Date of exit (default: now)
            exit_reason: Reason for closing (manual, target, stop)
            commission: Exit commission

        Returns:
            Trade closure details or None if failed
        """
        if exit_date is None:
            exit_date = datetime.now()
        elif isinstance(exit_date, date):
            exit_date = datetime.combine(exit_date, datetime.min.time())

        # Get trade details
        trade = self.db.fetchone("""
            SELECT * FROM positions_real WHERE id = ? AND status = 'open'
        """, (trade_id,))

        if not trade:
            logger.error(f"Trade {trade_id} not found or already closed")
            return None

        # Unpack trade (schema: id, ticker, direction, entry_price, entry_date, shares, exit_price, exit_date, pnl, pnl_pct, status, exit_reason, strategy, commission, notes)
        _, ticker, direction, entry_price, entry_date_str, shares, _, _, _, _, _, _, strategy, entry_commission, notes = trade

        total_commission = (entry_commission or 0) + commission

        # Calculate P&L
        if direction == "BUY":
            gross_pnl = (exit_price - entry_price) * shares
        else:
            gross_pnl = (entry_price - exit_price) * shares

        net_pnl = gross_pnl - total_commission
        pnl_pct = (net_pnl / (entry_price * shares)) * 100

        try:
            self.db.execute("""
                UPDATE positions_real
                SET exit_price = ?,
                    exit_date = ?,
                    pnl = ?,
                    pnl_pct = ?,
                    status = 'closed',
                    exit_reason = ?,
                    commission = ?
                WHERE id = ?
            """, (exit_price, exit_date, net_pnl, pnl_pct, exit_reason, total_commission, trade_id))

            result = {
                "trade_id": trade_id,
                "ticker": ticker,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "gross_pnl": gross_pnl,
                "commission": total_commission,
                "net_pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason
            }

            logger.info(f"Closed trade {trade_id}: {ticker} PnL=${net_pnl:+.2f} ({pnl_pct:+.2f}%)")

            return result

        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return None

    def get_open_trades(self) -> pd.DataFrame:
        """Get all open real trades."""
        return self.db.fetchdf("""
            SELECT * FROM positions_real
            WHERE status = 'open'
            ORDER BY entry_date DESC
        """)

    def get_closed_trades(
        self,
        start_date: date | None = None,
        end_date: date | None = None
    ) -> pd.DataFrame:
        """Get closed trades within date range."""
        query = "SELECT * FROM positions_real WHERE status = 'closed'"
        params = []

        if start_date:
            query += " AND exit_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND exit_date <= ?"
            params.append(end_date)

        query += " ORDER BY exit_date DESC"

        return self.db.fetchdf(query, tuple(params) if params else None)

    def get_all_trades(self) -> pd.DataFrame:
        """Get all trades (open and closed)."""
        return self.db.fetchdf("""
            SELECT * FROM positions_real
            ORDER BY entry_date DESC
        """)

    def update_live_pnl(self) -> pd.DataFrame:
        """
        Update unrealized P&L for open trades using live prices.

        Returns:
            DataFrame with open trades and current unrealized P&L
        """
        open_trades = self.get_open_trades()

        if open_trades.empty:
            return open_trades

        # Get current prices
        tickers = open_trades["ticker"].unique().tolist()

        # Get latest prices from database
        prices_df = self.data_ingestion.get_daily_prices(tickers)

        if prices_df.empty:
            return open_trades

        # Get most recent price for each ticker
        current_prices = {}
        for ticker in tickers:
            ticker_prices = prices_df[prices_df["ticker"] == ticker]
            if not ticker_prices.empty:
                current_prices[ticker] = ticker_prices["close"].iloc[-1]

        # Calculate unrealized P&L
        unrealized_pnl = []
        unrealized_pnl_pct = []

        for _, trade in open_trades.iterrows():
            ticker = trade["ticker"]
            current_price = current_prices.get(ticker)

            if current_price:
                if trade["direction"] == "BUY":
                    pnl = (current_price - trade["entry_price"]) * trade["shares"]
                else:
                    pnl = (trade["entry_price"] - current_price) * trade["shares"]

                pnl_pct = (pnl / (trade["entry_price"] * trade["shares"])) * 100
            else:
                pnl = 0
                pnl_pct = 0

            unrealized_pnl.append(pnl)
            unrealized_pnl_pct.append(pnl_pct)

        open_trades["current_price"] = open_trades["ticker"].map(current_prices)
        open_trades["unrealized_pnl"] = unrealized_pnl
        open_trades["unrealized_pnl_pct"] = unrealized_pnl_pct

        return open_trades

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall real trading performance."""
        closed = self.get_closed_trades()

        if closed.empty:
            return {
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "total_commission": 0
            }

        total_trades = len(closed)
        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] <= 0]

        total_pnl = closed["pnl"].sum()
        total_commission = closed["commission"].sum()
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
            "total_commission": total_commission,
            "total_wins": len(wins),
            "total_losses": len(losses)
        }

    def compare_to_paper(self) -> dict[str, Any]:
        """
        Compare real trading performance to paper trading.

        Returns:
            Dictionary with comparison metrics
        """
        # Real performance
        real_closed = self.get_closed_trades()
        real_perf = self.get_performance_summary()

        # Paper performance
        paper_closed = self.db.fetchdf("""
            SELECT * FROM positions_paper WHERE status = 'closed'
        """)

        if paper_closed.empty:
            paper_perf = {
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0
            }
        else:
            paper_wins = paper_closed[paper_closed["pnl"] > 0]
            paper_perf = {
                "total_trades": len(paper_closed),
                "total_pnl": paper_closed["pnl"].sum(),
                "win_rate": len(paper_wins) / len(paper_closed) * 100 if len(paper_closed) > 0 else 0
            }

        return {
            "real": real_perf,
            "paper": paper_perf,
            "pnl_difference": real_perf["total_pnl"] - paper_perf["total_pnl"],
            "win_rate_difference": real_perf["win_rate"] - paper_perf["win_rate"],
            "real_outperforming": real_perf["total_pnl"] > paper_perf["total_pnl"]
        }

    def export_to_csv(self, filepath: str | Path | None = None) -> Path:
        """
        Export all trades to CSV.

        Args:
            filepath: Output file path (default: data/real_trades_export.csv)

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = Path("data") / f"real_trades_export_{date.today()}.csv"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_trades = self.get_all_trades()

        if not all_trades.empty:
            all_trades.to_csv(filepath, index=False)
            logger.info(f"Exported {len(all_trades)} trades to {filepath}")
        else:
            # Create empty file with headers
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "ticker", "direction", "entry_price", "entry_date",
                    "shares", "exit_price", "exit_date", "pnl", "pnl_pct",
                    "status", "exit_reason", "strategy", "commission", "notes"
                ])
            logger.info(f"Created empty export file at {filepath}")

        return filepath

    def import_from_csv(self, filepath: str | Path) -> int:
        """
        Import trades from CSV.

        Args:
            filepath: Path to CSV file

        Returns:
            Number of trades imported
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return 0

        try:
            df = pd.read_csv(filepath)
            count = 0

            for _, row in df.iterrows():
                trade_id = self.add_trade(
                    ticker=row.get("ticker", ""),
                    direction=row.get("direction", "BUY"),
                    entry_price=float(row.get("entry_price", 0)),
                    shares=float(row.get("shares", 0)),
                    entry_date=pd.to_datetime(row.get("entry_date")).to_pydatetime() if row.get("entry_date") else None,
                    strategy=row.get("strategy"),
                    commission=float(row.get("commission", 0)),
                    notes=row.get("notes")
                )

                if trade_id:
                    count += 1

                    # Close if already closed in import
                    if row.get("status") == "closed" and row.get("exit_price"):
                        self.close_trade(
                            trade_id=trade_id,
                            exit_price=float(row.get("exit_price")),
                            exit_date=pd.to_datetime(row.get("exit_date")).to_pydatetime() if row.get("exit_date") else None,
                            exit_reason=row.get("exit_reason", "imported")
                        )

            logger.info(f"Imported {count} trades from {filepath}")
            return count

        except Exception as e:
            logger.error(f"Error importing trades: {e}")
            return 0

    def delete_trade(self, trade_id: int) -> bool:
        """
        Delete a trade.

        Args:
            trade_id: ID of trade to delete

        Returns:
            True if deleted successfully
        """
        try:
            self.db.execute("DELETE FROM positions_real WHERE id = ?", (trade_id,))
            logger.info(f"Deleted trade {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting trade: {e}")
            return False
