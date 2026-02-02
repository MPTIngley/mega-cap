"""Database management for StockPulse using DuckDB."""

from datetime import datetime, date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)

_db_instance: "Database | None" = None
_db_readonly_instance: "Database | None" = None
_force_read_only: bool = False  # Set True in dashboard to prevent write conflicts


def set_read_only_mode(enabled: bool = True) -> None:
    """Force all subsequent get_db() calls to return read-only connections.

    Use this in the dashboard to avoid conflicts with the scheduler.
    """
    global _force_read_only
    _force_read_only = enabled
    logger.info(f"Database read-only mode: {enabled}")


class Database:
    """DuckDB database manager for StockPulse."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path | None = None, read_only: bool = False):
        """Initialize database connection.

        Args:
            db_path: Path to database file
            read_only: If True, open in read-only mode (allows concurrent access)
        """
        import time

        if db_path is None:
            config = get_config()
            db_path = config["database"]["path"]

        self.db_path = Path(db_path)
        self.read_only = read_only

        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Retry logic for database connections that may be locked
        max_retries = 10 if read_only else 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                self.conn = duckdb.connect(str(self.db_path), read_only=read_only)
                break
            except duckdb.IOException as e:
                if "lock" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 5.0)  # Cap at 5 seconds
                else:
                    raise

        if not read_only:
            self._init_schema()
        logger.info(f"Database initialized at {self.db_path} (read_only={read_only})")

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Universe table - stocks we track
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                ticker VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap DOUBLE,
                is_active BOOLEAN DEFAULT true,
                added_date DATE,
                last_refreshed TIMESTAMP
            )
        """)

        # Daily price data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices_daily (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (ticker, date)
            )
        """)

        # Intraday price data (15-min bars)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices_intraday (
                ticker VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (ticker, timestamp)
            )
        """)

        # Fundamental data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker VARCHAR,
                date DATE,
                pe_ratio DOUBLE,
                forward_pe DOUBLE,
                pb_ratio DOUBLE,
                peg_ratio DOUBLE,
                dividend_yield DOUBLE,
                eps DOUBLE,
                revenue DOUBLE,
                profit_margin DOUBLE,
                roe DOUBLE,
                debt_to_equity DOUBLE,
                current_ratio DOUBLE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low DOUBLE,
                avg_volume_10d BIGINT,
                beta DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)

        # Trading signals (using autoincrement via sequence default)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS signals_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER DEFAULT nextval('signals_id_seq') PRIMARY KEY,
                created_at TIMESTAMP DEFAULT current_timestamp,
                ticker VARCHAR NOT NULL,
                strategy VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                entry_price DOUBLE,
                target_price DOUBLE,
                stop_price DOUBLE,
                status VARCHAR DEFAULT 'open',
                notes TEXT
            )
        """)

        # Paper trading positions
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS positions_paper_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions_paper (
                id INTEGER DEFAULT nextval('positions_paper_id_seq') PRIMARY KEY,
                signal_id INTEGER,
                ticker VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                entry_price DOUBLE NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                shares DOUBLE NOT NULL,
                exit_price DOUBLE,
                exit_date TIMESTAMP,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                status VARCHAR DEFAULT 'open',
                exit_reason VARCHAR,
                strategy VARCHAR
            )
        """)

        # Real trading positions (Phase 5)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS positions_real_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions_real (
                id INTEGER DEFAULT nextval('positions_real_id_seq') PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                entry_price DOUBLE NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                shares DOUBLE NOT NULL,
                exit_price DOUBLE,
                exit_date TIMESTAMP,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                status VARCHAR DEFAULT 'open',
                exit_reason VARCHAR,
                strategy VARCHAR,
                commission DOUBLE DEFAULT 0,
                notes TEXT
            )
        """)

        # Strategy configuration (runtime state)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_state (
                strategy_name VARCHAR PRIMARY KEY,
                enabled BOOLEAN DEFAULT true,
                params TEXT,  -- JSON
                last_run TIMESTAMP,
                total_signals INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                total_pnl DOUBLE DEFAULT 0,
                max_drawdown DOUBLE DEFAULT 0,
                disabled_reason VARCHAR
            )
        """)

        # Alert log
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS alerts_log_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER DEFAULT nextval('alerts_log_id_seq') PRIMARY KEY,
                created_at TIMESTAMP DEFAULT current_timestamp,
                signal_id INTEGER,
                alert_type VARCHAR NOT NULL,
                recipient VARCHAR,
                subject VARCHAR,
                body TEXT,
                sent_successfully BOOLEAN,
                error_message VARCHAR
            )
        """)

        # Long-term watchlist
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS long_term_watchlist_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_watchlist (
                id INTEGER DEFAULT nextval('long_term_watchlist_id_seq') PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                scan_date DATE NOT NULL,
                composite_score DOUBLE,
                valuation_score DOUBLE,
                technical_score DOUBLE,
                dividend_score DOUBLE,
                quality_score DOUBLE,
                pe_percentile DOUBLE,
                price_vs_52w_low_pct DOUBLE,
                reasoning TEXT,
                UNIQUE (ticker, scan_date)
            )
        """)

        # Backtest results
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS backtest_results_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER DEFAULT nextval('backtest_results_id_seq') PRIMARY KEY,
                strategy VARCHAR NOT NULL,
                run_date TIMESTAMP DEFAULT current_timestamp,
                start_date DATE,
                end_date DATE,
                initial_capital DOUBLE,
                final_value DOUBLE,
                total_return_pct DOUBLE,
                annualized_return_pct DOUBLE,
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                max_drawdown_pct DOUBLE,
                win_rate DOUBLE,
                profit_factor DOUBLE,
                total_trades INTEGER,
                avg_trade_pnl DOUBLE,
                avg_win DOUBLE,
                avg_loss DOUBLE,
                avg_hold_days DOUBLE,
                params TEXT
            )
        """)

        # System state / metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                updated_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_daily_date ON prices_daily(date)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_intraday_timestamp ON prices_intraday(timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_positions_paper_status ON positions_paper(status)
        """)

        logger.info("Database schema initialized")

    def execute(self, query: str, params: tuple | None = None) -> duckdb.DuckDBPyRelation:
        """Execute a query."""
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def fetchdf(self, query: str, params: tuple | None = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        result = self.execute(query, params)
        return result.fetchdf()

    def fetchone(self, query: str, params: tuple | None = None) -> tuple | None:
        """Execute query and return single row."""
        result = self.execute(query, params)
        return result.fetchone()

    def fetchall(self, query: str, params: tuple | None = None) -> list[tuple]:
        """Execute query and return all rows."""
        result = self.execute(query, params)
        return result.fetchall()

    def insert_df(self, table: str, df: pd.DataFrame, on_conflict: str = "ignore") -> int:
        """Insert DataFrame into table using DuckDB syntax."""
        if df.empty:
            return 0

        # Register DataFrame as a view
        self.conn.register("_temp_insert_df", df)

        cols = ", ".join(df.columns)

        try:
            if on_conflict == "replace":
                # For tables with primary key, delete existing then insert
                # First, try to identify the primary key column
                if "ticker" in df.columns and "date" in df.columns:
                    # For price tables with composite key
                    self.conn.execute(f"""
                        DELETE FROM {table} WHERE (ticker, date) IN (
                            SELECT ticker, date FROM _temp_insert_df
                        )
                    """)
                elif "ticker" in df.columns and "timestamp" in df.columns:
                    self.conn.execute(f"""
                        DELETE FROM {table} WHERE (ticker, timestamp) IN (
                            SELECT ticker, timestamp FROM _temp_insert_df
                        )
                    """)
                self.conn.execute(f"""
                    INSERT INTO {table} ({cols})
                    SELECT {cols} FROM _temp_insert_df
                """)
            else:
                # Insert and ignore conflicts using ON CONFLICT DO NOTHING
                # DuckDB requires specifying the conflict target for composite keys
                try:
                    self.conn.execute(f"""
                        INSERT INTO {table} ({cols})
                        SELECT {cols} FROM _temp_insert_df
                        ON CONFLICT DO NOTHING
                    """)
                except Exception:
                    # Fallback: just insert (may fail on duplicates)
                    self.conn.execute(f"""
                        INSERT INTO {table} ({cols})
                        SELECT {cols} FROM _temp_insert_df
                    """)
        finally:
            self.conn.unregister("_temp_insert_df")

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


def get_db(read_only: bool = False, fresh: bool = False) -> Database:
    """Get or create database instance.

    Args:
        read_only: If True, return a read-only connection that can run
                   concurrently with a write connection. Use this for
                   dashboard/read-only operations.
        fresh: If True, always create a new connection instead of reusing.
               Use this when you need to release locks between operations.

    Note:
        If set_read_only_mode(True) was called, this always returns a
        read-only connection regardless of the read_only parameter.
    """
    global _db_instance, _db_readonly_instance, _force_read_only

    # Force read-only mode if set (used by dashboard)
    if _force_read_only:
        read_only = True

    # Fresh connection requested - don't use singleton
    if fresh:
        return Database(read_only=read_only)

    if read_only:
        if _db_readonly_instance is None:
            _db_readonly_instance = Database(read_only=True)
        return _db_readonly_instance
    else:
        if _db_instance is None:
            _db_instance = Database()
        return _db_instance


def release_db_locks() -> None:
    """Close all database connections to release locks.

    Call this periodically in long-running processes to allow
    other processes to access the database.
    """
    global _db_instance, _db_readonly_instance

    if _db_instance is not None:
        try:
            _db_instance.close()
        except Exception:
            pass
        _db_instance = None

    if _db_readonly_instance is not None:
        try:
            _db_readonly_instance.close()
        except Exception:
            pass
        _db_readonly_instance = None

    logger.debug("Released database locks")
