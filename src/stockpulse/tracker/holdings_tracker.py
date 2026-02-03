"""Holdings Tracker for recording actual purchases.

Tracks both:
- Active (short-term) holdings: positions from the active trading strategies
- Long-term holdings: positions from the long-term investment strategy

This allows comparison of signal performance vs actual execution.

Usage:
    from stockpulse.tracker.holdings_tracker import HoldingsTracker

    tracker = HoldingsTracker()

    # Record a purchase
    tracker.add_holding(
        ticker="AAPL",
        buy_date="2024-01-15",
        buy_price=185.50,
        shares=10,
        strategy_type="long_term",  # or "active"
        notes="Strong value score"
    )

    # Record a sale
    tracker.close_holding(
        holding_id=1,
        sell_date="2024-06-15",
        sell_price=195.00
    )

    # Get current holdings
    holdings = tracker.get_open_holdings(strategy_type="long_term")
"""

from datetime import date, datetime
from typing import Any
import json

import pandas as pd
import yfinance as yf

from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db

logger = get_logger(__name__)


class HoldingsTracker:
    """Tracks actual holdings for both active and long-term strategies."""

    def __init__(self):
        """Initialize holdings tracker."""
        self.db = get_db()
        self._ensure_tables()

    def _ensure_tables(self):
        """Create holdings tables if they don't exist."""
        # Main holdings table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS actual_holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                strategy_type TEXT NOT NULL,  -- 'active' or 'long_term'
                buy_date DATE NOT NULL,
                buy_price REAL NOT NULL,
                shares REAL NOT NULL,
                cost_basis REAL NOT NULL,
                sell_date DATE,
                sell_price REAL,
                realized_pnl REAL,
                realized_pnl_pct REAL,
                status TEXT DEFAULT 'open',  -- 'open' or 'closed'
                sector TEXT,
                signal_score REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Holdings snapshots for tracking value over time
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS holdings_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date DATE NOT NULL,
                strategy_type TEXT NOT NULL,
                total_value REAL,
                total_cost_basis REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                num_positions INTEGER,
                holdings_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for performance
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_holdings_status
            ON actual_holdings(status, strategy_type)
        """)

        logger.info("Holdings tracker tables initialized")

    def add_holding(
        self,
        ticker: str,
        buy_date: str | date,
        buy_price: float,
        shares: float,
        strategy_type: str = "long_term",
        signal_score: float | None = None,
        notes: str | None = None
    ) -> int:
        """
        Record a new holding (purchase).

        Args:
            ticker: Stock ticker symbol
            buy_date: Purchase date (YYYY-MM-DD string or date object)
            buy_price: Price per share
            shares: Number of shares purchased
            strategy_type: 'active' or 'long_term'
            signal_score: Optional score from scanner at time of purchase
            notes: Optional notes about the purchase

        Returns:
            Holding ID
        """
        if isinstance(buy_date, str):
            buy_date = datetime.strptime(buy_date, "%Y-%m-%d").date()

        cost_basis = buy_price * shares

        # Get sector from universe table
        sector = self._get_sector(ticker)

        cursor = self.db.execute("""
            INSERT INTO actual_holdings
            (ticker, strategy_type, buy_date, buy_price, shares, cost_basis,
             sector, signal_score, notes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (
            ticker.upper(),
            strategy_type,
            buy_date,
            buy_price,
            shares,
            cost_basis,
            sector,
            signal_score,
            notes
        ))

        holding_id = cursor.lastrowid
        logger.info(f"Added holding: {ticker} {shares} shares @ ${buy_price:.2f} ({strategy_type})")

        return holding_id

    def close_holding(
        self,
        holding_id: int,
        sell_date: str | date,
        sell_price: float
    ) -> dict:
        """
        Close a holding (record a sale).

        Args:
            holding_id: ID of the holding to close
            sell_date: Sale date
            sell_price: Price per share

        Returns:
            Dictionary with P&L details
        """
        if isinstance(sell_date, str):
            sell_date = datetime.strptime(sell_date, "%Y-%m-%d").date()

        # Get holding details
        row = self.db.fetchone("""
            SELECT ticker, buy_price, shares, cost_basis
            FROM actual_holdings
            WHERE id = ? AND status = 'open'
        """, (holding_id,))

        if not row:
            raise ValueError(f"Holding {holding_id} not found or already closed")

        ticker, buy_price, shares, cost_basis = row

        # Calculate P&L
        proceeds = sell_price * shares
        realized_pnl = proceeds - cost_basis
        realized_pnl_pct = (realized_pnl / cost_basis) * 100

        # Update holding
        self.db.execute("""
            UPDATE actual_holdings
            SET sell_date = ?, sell_price = ?, realized_pnl = ?,
                realized_pnl_pct = ?, status = 'closed'
            WHERE id = ?
        """, (sell_date, sell_price, realized_pnl, realized_pnl_pct, holding_id))

        logger.info(f"Closed holding {holding_id}: {ticker} @ ${sell_price:.2f}, P&L: ${realized_pnl:+,.2f} ({realized_pnl_pct:+.1f}%)")

        return {
            "ticker": ticker,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "shares": shares,
            "realized_pnl": realized_pnl,
            "realized_pnl_pct": realized_pnl_pct
        }

    def get_open_holdings(self, strategy_type: str | None = None) -> pd.DataFrame:
        """
        Get all open holdings.

        Args:
            strategy_type: Filter by 'active' or 'long_term', or None for all

        Returns:
            DataFrame with open holdings
        """
        if strategy_type:
            df = self.db.fetchdf("""
                SELECT id, ticker, strategy_type, buy_date, buy_price, shares,
                       cost_basis, sector, signal_score, notes
                FROM actual_holdings
                WHERE status = 'open' AND strategy_type = ?
                ORDER BY buy_date DESC
            """, (strategy_type,))
        else:
            df = self.db.fetchdf("""
                SELECT id, ticker, strategy_type, buy_date, buy_price, shares,
                       cost_basis, sector, signal_score, notes
                FROM actual_holdings
                WHERE status = 'open'
                ORDER BY buy_date DESC
            """)

        return df

    def get_closed_holdings(
        self,
        strategy_type: str | None = None,
        start_date: date | None = None
    ) -> pd.DataFrame:
        """Get closed holdings with P&L."""
        query = """
            SELECT id, ticker, strategy_type, buy_date, buy_price, shares,
                   sell_date, sell_price, realized_pnl, realized_pnl_pct,
                   sector, signal_score, notes
            FROM actual_holdings
            WHERE status = 'closed'
        """
        params = []

        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)

        if start_date:
            query += " AND sell_date >= ?"
            params.append(start_date)

        query += " ORDER BY sell_date DESC"

        return self.db.fetchdf(query, tuple(params) if params else None)

    def get_holding_by_id(self, holding_id: int) -> dict | None:
        """Get a specific holding by ID."""
        row = self.db.fetchone("""
            SELECT * FROM actual_holdings WHERE id = ?
        """, (holding_id,))

        if not row:
            return None

        columns = ["id", "ticker", "strategy_type", "buy_date", "buy_price",
                   "shares", "cost_basis", "sell_date", "sell_price",
                   "realized_pnl", "realized_pnl_pct", "status", "sector",
                   "signal_score", "notes", "created_at"]

        return dict(zip(columns, row))

    def get_holdings_with_current_value(
        self,
        strategy_type: str | None = None
    ) -> pd.DataFrame:
        """Get open holdings with current market value and unrealized P&L."""
        holdings_df = self.get_open_holdings(strategy_type)

        if holdings_df.empty:
            return holdings_df

        # Fetch current prices
        tickers = holdings_df["ticker"].tolist()
        current_prices = self._fetch_current_prices(tickers)

        # Calculate current values
        holdings_df["current_price"] = holdings_df["ticker"].map(
            lambda t: current_prices.get(t, 0)
        )
        holdings_df["current_value"] = holdings_df["current_price"] * holdings_df["shares"]
        holdings_df["unrealized_pnl"] = holdings_df["current_value"] - holdings_df["cost_basis"]
        holdings_df["unrealized_pnl_pct"] = (
            holdings_df["unrealized_pnl"] / holdings_df["cost_basis"] * 100
        ).round(2)

        return holdings_df

    def get_portfolio_summary(self, strategy_type: str | None = None) -> dict:
        """Get portfolio summary with totals."""
        holdings_df = self.get_holdings_with_current_value(strategy_type)

        if holdings_df.empty:
            return {
                "num_positions": 0,
                "total_cost_basis": 0,
                "total_current_value": 0,
                "total_unrealized_pnl": 0,
                "total_unrealized_pnl_pct": 0,
                "best_performer": None,
                "worst_performer": None
            }

        total_cost = holdings_df["cost_basis"].sum()
        total_value = holdings_df["current_value"].sum()
        total_pnl = holdings_df["unrealized_pnl"].sum()

        # Best and worst
        best_idx = holdings_df["unrealized_pnl_pct"].idxmax()
        worst_idx = holdings_df["unrealized_pnl_pct"].idxmin()

        return {
            "num_positions": len(holdings_df),
            "total_cost_basis": round(total_cost, 2),
            "total_current_value": round(total_value, 2),
            "total_unrealized_pnl": round(total_pnl, 2),
            "total_unrealized_pnl_pct": round((total_pnl / total_cost * 100), 2) if total_cost > 0 else 0,
            "best_performer": {
                "ticker": holdings_df.loc[best_idx, "ticker"],
                "pnl_pct": holdings_df.loc[best_idx, "unrealized_pnl_pct"]
            },
            "worst_performer": {
                "ticker": holdings_df.loc[worst_idx, "ticker"],
                "pnl_pct": holdings_df.loc[worst_idx, "unrealized_pnl_pct"]
            }
        }

    def take_snapshot(self, strategy_type: str = "long_term"):
        """Take a snapshot of current holdings for historical tracking."""
        holdings_df = self.get_holdings_with_current_value(strategy_type)

        if holdings_df.empty:
            return

        total_value = holdings_df["current_value"].sum()
        total_cost = holdings_df["cost_basis"].sum()
        unrealized_pnl = total_value - total_cost
        unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0

        # Create holdings summary
        holdings_json = holdings_df[[
            "ticker", "shares", "buy_price", "current_price",
            "cost_basis", "current_value", "unrealized_pnl_pct"
        ]].to_dict("records")

        self.db.execute("""
            INSERT INTO holdings_snapshots
            (snapshot_date, strategy_type, total_value, total_cost_basis,
             unrealized_pnl, unrealized_pnl_pct, num_positions, holdings_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date.today(),
            strategy_type,
            total_value,
            total_cost,
            unrealized_pnl,
            unrealized_pnl_pct,
            len(holdings_df),
            json.dumps(holdings_json)
        ))

        logger.info(f"Snapshot taken: {strategy_type}, {len(holdings_df)} positions, ${total_value:,.2f}")

    def get_snapshots(
        self,
        strategy_type: str = "long_term",
        days: int = 365
    ) -> pd.DataFrame:
        """Get historical snapshots for equity curve."""
        start_date = date.today() - pd.Timedelta(days=days)

        return self.db.fetchdf("""
            SELECT snapshot_date, total_value, total_cost_basis,
                   unrealized_pnl, unrealized_pnl_pct, num_positions
            FROM holdings_snapshots
            WHERE strategy_type = ? AND snapshot_date >= ?
            ORDER BY snapshot_date
        """, (strategy_type, start_date))

    def _get_sector(self, ticker: str) -> str | None:
        """Get sector for a ticker from universe table."""
        try:
            row = self.db.fetchone("""
                SELECT sector FROM universe WHERE ticker = ?
            """, (ticker.upper(),))
            return row[0] if row else None
        except Exception:
            return None

    def _fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch current prices for tickers."""
        prices = {}

        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(period="1d")
                if not hist.empty:
                    prices[ticker] = hist["Close"].iloc[-1]
            except Exception as e:
                logger.debug(f"Error fetching price for {ticker}: {e}")

        return prices


def print_holdings_summary(strategy_type: str | None = None):
    """Print a formatted holdings summary."""
    tracker = HoldingsTracker()

    print("\n" + "=" * 70)
    strategy_label = strategy_type.upper() if strategy_type else "ALL"
    print(f"  {strategy_label} HOLDINGS SUMMARY")
    print("=" * 70)

    summary = tracker.get_portfolio_summary(strategy_type)

    print(f"\n  Positions:           {summary['num_positions']}")
    print(f"  Total Cost Basis:    ${summary['total_cost_basis']:,.2f}")
    print(f"  Current Value:       ${summary['total_current_value']:,.2f}")
    print(f"  Unrealized P&L:      ${summary['total_unrealized_pnl']:+,.2f} ({summary['total_unrealized_pnl_pct']:+.1f}%)")

    if summary["best_performer"]:
        print(f"\n  Best Performer:      {summary['best_performer']['ticker']} ({summary['best_performer']['pnl_pct']:+.1f}%)")
        print(f"  Worst Performer:     {summary['worst_performer']['ticker']} ({summary['worst_performer']['pnl_pct']:+.1f}%)")

    # Show individual holdings
    holdings_df = tracker.get_holdings_with_current_value(strategy_type)

    if not holdings_df.empty:
        print("\n" + "-" * 70)
        print("  INDIVIDUAL HOLDINGS")
        print("-" * 70)
        print("  Ticker   Shares     Buy      Current     Cost      Value      P&L")
        print("  " + "-" * 66)

        for _, row in holdings_df.iterrows():
            pnl_str = f"{row['unrealized_pnl']:+,.0f}"
            pct_str = f"({row['unrealized_pnl_pct']:+.1f}%)"
            print(f"  {row['ticker']:6} {row['shares']:8.2f}  ${row['buy_price']:7.2f}  "
                  f"${row['current_price']:7.2f}  ${row['cost_basis']:8,.0f}  "
                  f"${row['current_value']:8,.0f}  {pnl_str:>8} {pct_str}")

    print("\n" + "=" * 70 + "\n")
