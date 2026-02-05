"""Database management for StockPulse using SQLite with WAL mode.

SQLite with WAL (Write-Ahead Logging) mode supports:
- One writer at a time
- Multiple concurrent readers (even while writing)
- No locking conflicts between dashboard and scheduler
"""

from datetime import datetime, date
from pathlib import Path
from typing import Any
import sqlite3

import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)

_db_instance: "Database | None" = None


class Database:
    """SQLite database manager for StockPulse with WAL mode for concurrency."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path | None = None):
        """Initialize database connection with WAL mode."""
        if db_path is None:
            config = get_config()
            db_path = config["database"]["path"]
            # Change extension from .duckdb to .sqlite if needed
            if str(db_path).endswith(".duckdb"):
                db_path = str(db_path).replace(".duckdb", ".sqlite")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with optimized settings for concurrent access
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow multi-threaded access
            timeout=30.0  # Wait up to 30 seconds for locks
        )

        # Enable WAL mode for concurrent read/write access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety/speed
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout

        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()
        logger.info(f"Database initialized at {self.db_path} (SQLite WAL mode)")

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Universe table - stocks we track
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                is_active INTEGER DEFAULT 1,
                added_date TEXT,
                last_refreshed TEXT
            )
        """)

        # Daily price data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)

        # Intraday price data (15-min bars)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices_intraday (
                ticker TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, timestamp)
            )
        """)

        # Fundamental data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker TEXT,
                date TEXT,
                pe_ratio REAL,
                forward_pe REAL,
                pb_ratio REAL,
                peg_ratio REAL,
                dividend_yield REAL,
                eps REAL,
                revenue REAL,
                profit_margin REAL,
                roe REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                fifty_two_week_high REAL,
                fifty_two_week_low REAL,
                avg_volume_10d INTEGER,
                beta REAL,
                PRIMARY KEY (ticker, date)
            )
        """)

        # Trading signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                ticker TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_price REAL,
                status TEXT DEFAULT 'open',
                notes TEXT
            )
        """)

        # Paper trading positions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions_paper (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_date TEXT NOT NULL,
                shares REAL NOT NULL,
                exit_price REAL,
                exit_date TEXT,
                pnl REAL,
                pnl_pct REAL,
                status TEXT DEFAULT 'open',
                exit_reason TEXT,
                strategy TEXT
            )
        """)

        # Real trading positions (Phase 5)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions_real (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_date TEXT NOT NULL,
                shares REAL NOT NULL,
                exit_price REAL,
                exit_date TEXT,
                pnl REAL,
                pnl_pct REAL,
                status TEXT DEFAULT 'open',
                exit_reason TEXT,
                strategy TEXT,
                commission REAL DEFAULT 0,
                notes TEXT
            )
        """)

        # Strategy configuration (runtime state)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_state (
                strategy_name TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                params TEXT,
                last_run TEXT,
                total_signals INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                disabled_reason TEXT
            )
        """)

        # Alert log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                signal_id INTEGER,
                alert_type TEXT NOT NULL,
                recipient TEXT,
                subject TEXT,
                body TEXT,
                sent_successfully INTEGER,
                error_message TEXT
            )
        """)

        # Long-term watchlist with full score breakdown
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                composite_score REAL,
                valuation_score REAL,
                technical_score REAL,
                dividend_score REAL,
                quality_score REAL,
                insider_score REAL,
                fcf_score REAL,
                earnings_score REAL,
                peer_score REAL,
                pe_percentile REAL,
                price_vs_52w_low_pct REAL,
                reasoning TEXT,
                UNIQUE (ticker, scan_date)
            )
        """)

        # Add missing columns if table already exists (migration)
        migration_columns = [
            ("insider_score", "REAL"),
            ("fcf_score", "REAL"),
            ("earnings_score", "REAL"),
            ("peer_score", "REAL"),
            ("company_name", "TEXT"),
            ("sector", "TEXT"),
            ("week52_high", "REAL"),
            ("week52_low", "REAL"),
            ("current_price", "REAL"),
        ]
        for col_name, col_type in migration_columns:
            try:
                cursor.execute(f"ALTER TABLE long_term_watchlist ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass  # Column already exists

        # Backtest results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                run_date TEXT DEFAULT (datetime('now')),
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                final_value REAL,
                total_return_pct REAL,
                annualized_return_pct REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown_pct REAL,
                win_rate REAL,
                profit_factor REAL,
                total_trades INTEGER,
                avg_trade_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                avg_hold_days REAL,
                params TEXT
            )
        """)

        # Optimization runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                run_date TEXT DEFAULT (datetime('now')),
                best_params TEXT,
                best_return_pct REAL,
                best_sharpe REAL,
                best_drawdown_pct REAL,
                constraint_satisfied INTEGER,
                optimization_time_seconds REAL
            )
        """)

        # System state / metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_daily_date ON prices_daily(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_intraday_timestamp ON prices_intraday(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_paper_status ON positions_paper(status)")

        self.conn.commit()
        logger.info("Database schema initialized")

    def execute(self, query: str, params: tuple | None = None) -> sqlite3.Cursor:
        """Execute a query."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.conn.commit()
        return cursor

    def fetchdf(self, query: str, params: tuple | None = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        if params:
            return pd.read_sql_query(query, self.conn, params=params)
        return pd.read_sql_query(query, self.conn)

    def fetchone(self, query: str, params: tuple | None = None) -> tuple | None:
        """Execute query and return single row."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchone()

    def fetchall(self, query: str, params: tuple | None = None) -> list[tuple]:
        """Execute query and return all rows."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()

    def insert_df(self, table: str, df: pd.DataFrame, on_conflict: str = "ignore") -> int:
        """Insert DataFrame into table."""
        if df.empty:
            return 0

        # Convert DataFrame to records
        cols = list(df.columns)
        placeholders = ", ".join(["?" for _ in cols])
        cols_str = ", ".join(cols)

        if on_conflict == "replace":
            query = f"INSERT OR REPLACE INTO {table} ({cols_str}) VALUES ({placeholders})"
        else:
            query = f"INSERT OR IGNORE INTO {table} ({cols_str}) VALUES ({placeholders})"

        cursor = self.conn.cursor()

        # Convert DataFrame rows to tuples, handling NaN values
        records = []
        for _, row in df.iterrows():
            record = tuple(None if pd.isna(v) else v for v in row)
            records.append(record)

        cursor.executemany(query, records)
        self.conn.commit()

        return len(df)

    def get_latest_price_date(self, ticker: str) -> date | None:
        """Get the latest price date for a ticker."""
        result = self.fetchone(
            "SELECT MAX(date) FROM prices_daily WHERE ticker = ?",
            (ticker,)
        )
        return result[0] if result and result[0] else None

    def get_latest_intraday_timestamp(self, ticker: str) -> datetime | None:
        """Get the latest intraday timestamp for a ticker."""
        result = self.fetchone(
            "SELECT MAX(timestamp) FROM prices_intraday WHERE ticker = ?",
            (ticker,)
        )
        return result[0] if result and result[0] else None

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")


def get_db() -> Database:
    """Get or create database instance (singleton)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def release_db_locks() -> None:
    """Close database connection to release any locks.

    With SQLite WAL mode, this is rarely needed but kept for API compatibility.
    """
    global _db_instance

    if _db_instance is not None:
        try:
            _db_instance.close()
        except Exception:
            pass
        _db_instance = None

    logger.debug("Released database connection")


# Keep for API compatibility but not needed with SQLite
def set_read_only_mode(enabled: bool = True) -> None:
    """No-op for SQLite. Kept for API compatibility."""
    pass


def reset_trading_data(keep_market_data: bool = True) -> dict[str, int]:
    """Reset all personal trading data while optionally keeping market history.

    Args:
        keep_market_data: If True (default), keeps prices, fundamentals, universe.
                         If False, clears everything.

    Returns:
        Dictionary with counts of deleted records per table.

    Usage:
        from stockpulse.data.database import reset_trading_data
        reset_trading_data()  # Clear trades, keep market data
        reset_trading_data(keep_market_data=False)  # Clear everything
    """
    db = get_db()
    deleted = {}

    # Tables with personal trading data (always cleared)
    trading_tables = [
        "signals",
        "positions_paper",
        "positions_real",
        "alerts_log",
        "backtest_results",
        "long_term_watchlist",
        "strategy_state",
    ]

    # Tables with market data (only cleared if keep_market_data=False)
    market_tables = [
        "prices_daily",
        "prices_intraday",
        "fundamentals",
        "universe",
        "system_state",
    ]

    tables_to_clear = trading_tables if keep_market_data else trading_tables + market_tables

    for table in tables_to_clear:
        try:
            count_result = db.fetchone(f"SELECT COUNT(*) FROM {table}")
            count = count_result[0] if count_result else 0
            db.execute(f"DELETE FROM {table}")
            deleted[table] = count
            logger.info(f"Cleared {count} records from {table}")
        except Exception as e:
            logger.warning(f"Error clearing {table}: {e}")
            deleted[table] = 0

    logger.info(f"Trading data reset complete. Deleted: {deleted}")
    return deleted


def get_data_summary() -> dict[str, Any]:
    """Get summary of data in the database.

    Returns:
        Dictionary with record counts and date ranges.
    """
    db = get_db()
    summary = {}

    # Table counts
    tables = [
        "universe", "prices_daily", "prices_intraday", "fundamentals",
        "signals", "positions_paper", "positions_real", "alerts_log",
        "backtest_results", "long_term_watchlist"
    ]

    for table in tables:
        try:
            result = db.fetchone(f"SELECT COUNT(*) FROM {table}")
            summary[f"{table}_count"] = result[0] if result else 0
        except Exception:
            summary[f"{table}_count"] = 0

    # Date ranges for market data
    try:
        result = db.fetchone("SELECT MIN(date), MAX(date) FROM prices_daily")
        if result:
            summary["prices_daily_start"] = result[0]
            summary["prices_daily_end"] = result[1]
    except Exception:
        pass

    # Active tickers
    try:
        result = db.fetchone("SELECT COUNT(*) FROM universe WHERE is_active = 1")
        summary["active_tickers"] = result[0] if result else 0
    except Exception:
        summary["active_tickers"] = 0

    return summary
