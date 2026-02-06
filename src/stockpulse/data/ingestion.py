"""Data ingestion module - fetching price and fundamental data."""

import time
import signal
from datetime import datetime, date, timedelta
from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import pandas as pd
import yfinance as yf
import pytz

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db

logger = get_logger(__name__)

# Rate limiting settings
MIN_REQUEST_INTERVAL = 0.2  # Minimum seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds to wait before retry
YF_DOWNLOAD_TIMEOUT = 60  # Max seconds for a single yfinance download call


class DataIngestion:
    """Handles fetching and storing price and fundamental data."""

    def __init__(self):
        """Initialize data ingestion."""
        self.db = get_db()
        self.config = get_config()
        self.last_request_time = 0
        self.eastern = pytz.timezone("US/Eastern")

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _download_with_timeout(self, tickers_str: str, timeout: int = YF_DOWNLOAD_TIMEOUT, **kwargs) -> pd.DataFrame:
        """
        Download data from yfinance with a timeout.

        Args:
            tickers_str: Space-separated ticker symbols
            timeout: Max seconds to wait
            **kwargs: Arguments to pass to yf.download

        Returns:
            DataFrame from yfinance or empty DataFrame on timeout
        """
        def _download():
            return yf.download(tickers_str, progress=False, **kwargs)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_download)
                return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(f"yfinance download timed out after {timeout}s for: {tickers_str[:50]}...")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"yfinance download error: {e}")
            return pd.DataFrame()

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        progress: bool = False
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (default: 2 years ago)
            end_date: End date (default: today)
            progress: Show download progress

        Returns:
            DataFrame with daily prices
        """
        if not tickers:
            return pd.DataFrame()

        if start_date is None:
            start_date = date.today() - timedelta(days=730)  # 2 years
        if end_date is None:
            end_date = date.today()

        logger.info(f"Fetching daily prices for {len(tickers)} tickers from {start_date} to {end_date}")

        all_data = []
        batch_size = 50  # yfinance works well with batches

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            for attempt in range(MAX_RETRIES):
                try:
                    self._rate_limit()

                    tickers_str = " ".join(batch)
                    data = self._download_with_timeout(
                        tickers_str,
                        timeout=YF_DOWNLOAD_TIMEOUT,
                        start=start_date.isoformat(),
                        end=(end_date + timedelta(days=1)).isoformat(),  # Include end date
                        group_by="ticker",
                        auto_adjust=False,
                        threads=True
                    )

                    if data.empty:
                        logger.warning(f"No data returned for batch {i // batch_size + 1}")
                        break

                    # Process based on single vs multiple tickers
                    if len(batch) == 1:
                        ticker = batch[0]
                        df = data.copy()
                        df["ticker"] = ticker
                        df = df.reset_index()
                        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                        if "date" not in df.columns and "index" in df.columns:
                            df = df.rename(columns={"index": "date"})
                        all_data.append(df)
                    else:
                        for ticker in batch:
                            try:
                                if ticker in data.columns.get_level_values(0):
                                    df = data[ticker].copy()
                                    df["ticker"] = ticker
                                    df = df.reset_index()
                                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                                    if "date" not in df.columns and "index" in df.columns:
                                        df = df.rename(columns={"index": "date"})
                                    all_data.append(df)
                            except Exception as e:
                                logger.debug(f"Error extracting {ticker} from batch: {e}")

                    break  # Success, exit retry loop

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for batch: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))

            # Progress logging
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(tickers):
                logger.info(f"Processed {min(i + batch_size, len(tickers))}/{len(tickers)} tickers")

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)

        # Ensure proper column names
        column_mapping = {
            "adj_close": "adj_close",
            "adj close": "adj_close",
            "adjclose": "adj_close",
        }
        combined = combined.rename(columns=column_mapping)

        # Select and order columns
        expected_cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        available_cols = [c for c in expected_cols if c in combined.columns]
        combined = combined[available_cols]

        # Clean data
        combined = combined.dropna(subset=["close"])
        combined["date"] = pd.to_datetime(combined["date"]).dt.date

        logger.info(f"Fetched {len(combined)} daily price records")
        return combined

    def fetch_intraday_prices(
        self,
        tickers: list[str],
        period: str = "5d"
    ) -> pd.DataFrame:
        """
        Fetch 15-minute intraday OHLCV data.

        Args:
            tickers: List of ticker symbols
            period: Data period (default: 5 days)

        Returns:
            DataFrame with intraday prices
        """
        if not tickers:
            return pd.DataFrame()

        logger.info(f"Fetching intraday prices for {len(tickers)} tickers")

        all_data = []
        batch_size = 20  # Smaller batches for intraday

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            for attempt in range(MAX_RETRIES):
                try:
                    self._rate_limit()

                    tickers_str = " ".join(batch)
                    data = self._download_with_timeout(
                        tickers_str,
                        timeout=YF_DOWNLOAD_TIMEOUT,
                        period=period,
                        interval="15m",
                        group_by="ticker",
                        auto_adjust=True,
                        threads=True
                    )

                    if data.empty:
                        break

                    # Process based on single vs multiple tickers
                    if len(batch) == 1:
                        ticker = batch[0]
                        df = data.copy()
                        df["ticker"] = ticker
                        df = df.reset_index()
                        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                        if "datetime" in df.columns:
                            df = df.rename(columns={"datetime": "timestamp"})
                        elif "index" in df.columns:
                            df = df.rename(columns={"index": "timestamp"})
                        all_data.append(df)
                    else:
                        for ticker in batch:
                            try:
                                if ticker in data.columns.get_level_values(0):
                                    df = data[ticker].copy()
                                    df["ticker"] = ticker
                                    df = df.reset_index()
                                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                                    if "datetime" in df.columns:
                                        df = df.rename(columns={"datetime": "timestamp"})
                                    elif "index" in df.columns:
                                        df = df.rename(columns={"index": "timestamp"})
                                    all_data.append(df)
                            except Exception as e:
                                logger.debug(f"Error extracting {ticker}: {e}")

                    break

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Select columns
        expected_cols = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in expected_cols if c in combined.columns]
        combined = combined[available_cols]

        # Clean data
        combined = combined.dropna(subset=["close"])

        logger.info(f"Fetched {len(combined)} intraday price records")
        return combined

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """
        Fetch current/live prices for tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to current price
        """
        if not tickers:
            return {}

        logger.info(f"Fetching current prices for {len(tickers)} tickers")
        prices = {}

        # Batch fetch using yfinance download with period="1d"
        # This gets the most recent trading data
        try:
            self._rate_limit()
            # Use 1d period with 1m interval to get latest price
            tickers_str = " ".join(tickers) if isinstance(tickers, list) else tickers
            data = self._download_with_timeout(
                tickers_str,
                timeout=YF_DOWNLOAD_TIMEOUT,
                period="1d",
                interval="1m",
                threads=True
            )

            if not data.empty:
                # Get the last available price for each ticker
                if len(tickers) == 1:
                    # Single ticker - data is flat
                    if "Close" in data.columns:
                        last_price = data["Close"].dropna().iloc[-1] if not data["Close"].dropna().empty else None
                        if last_price is not None:
                            prices[tickers[0]] = float(last_price)
                else:
                    # Multiple tickers - data has multi-level columns
                    for ticker in tickers:
                        try:
                            if ticker in data["Close"].columns:
                                ticker_close = data["Close"][ticker].dropna()
                                if not ticker_close.empty:
                                    prices[ticker] = float(ticker_close.iloc[-1])
                        except Exception as e:
                            logger.debug(f"Could not get price for {ticker}: {e}")

        except Exception as e:
            logger.warning(f"Error fetching current prices: {e}")

        logger.info(f"Got current prices for {len(prices)} tickers")
        return prices

    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with fundamental data
        """
        if not tickers:
            return pd.DataFrame()

        logger.info(f"Fetching fundamentals for {len(tickers)} tickers")

        data = []
        today = date.today()

        for ticker in tickers:
            try:
                self._rate_limit()

                stock = yf.Ticker(ticker)
                info = stock.info

                record = {
                    "ticker": ticker,
                    "date": today,
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "peg_ratio": info.get("pegRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "eps": info.get("trailingEps"),
                    "revenue": info.get("totalRevenue"),
                    "profit_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    "beta": info.get("beta"),
                }
                data.append(record)

            except Exception as e:
                logger.warning(f"Error fetching fundamentals for {ticker}: {e}")

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info(f"Fetched fundamentals for {len(df)} tickers")
        return df

    def store_daily_prices(self, df: pd.DataFrame) -> int:
        """Store daily prices in database."""
        if df.empty:
            return 0

        # Convert date to string for SQLite compatibility
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        count = self.db.insert_df("prices_daily", df, on_conflict="replace")
        logger.info(f"Stored {count} daily price records")
        return count

    def store_intraday_prices(self, df: pd.DataFrame) -> int:
        """Store intraday prices in database."""
        if df.empty:
            return 0

        # Convert timestamp to string for SQLite compatibility
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = df["timestamp"].astype(str)

        count = self.db.insert_df("prices_intraday", df, on_conflict="replace")
        logger.info(f"Stored {count} intraday price records")
        return count

    def store_fundamentals(self, df: pd.DataFrame) -> int:
        """Store fundamentals in database."""
        if df.empty:
            return 0

        count = self.db.insert_df("fundamentals", df, on_conflict="replace")
        logger.info(f"Stored {count} fundamental records")
        return count

    def run_daily_ingestion(self, tickers: list[str]) -> dict[str, int]:
        """
        Run full daily data ingestion.

        Returns:
            Dictionary with counts of records stored
        """
        logger.info("Starting daily data ingestion")

        results = {
            "daily_prices": 0,
            "intraday_prices": 0,
            "fundamentals": 0
        }

        # Fetch and store daily prices
        daily_df = self.fetch_daily_prices(tickers)
        results["daily_prices"] = self.store_daily_prices(daily_df)

        # Fetch and store intraday prices
        intraday_df = self.fetch_intraday_prices(tickers)
        results["intraday_prices"] = self.store_intraday_prices(intraday_df)

        # Fetch and store fundamentals
        fundamentals_df = self.fetch_fundamentals(tickers)
        results["fundamentals"] = self.store_fundamentals(fundamentals_df)

        # Update last ingestion timestamp
        self.db.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('last_daily_ingestion', ?, current_timestamp)
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = current_timestamp
        """, (datetime.now().isoformat(),))

        logger.info(f"Daily ingestion complete: {results}")
        return results

    def run_intraday_ingestion(self, tickers: list[str]) -> int:
        """
        Run intraday data ingestion (15-min bars).

        Returns:
            Count of records stored
        """
        logger.info("Starting intraday data ingestion")

        intraday_df = self.fetch_intraday_prices(tickers, period="1d")
        count = self.store_intraday_prices(intraday_df)

        self.db.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('last_intraday_ingestion', ?, current_timestamp)
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = current_timestamp
        """, (datetime.now().isoformat(),))

        logger.info(f"Intraday ingestion complete: {count} records")
        return count

    def get_daily_prices(
        self,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None
    ) -> pd.DataFrame:
        """Get daily prices from database."""
        query = "SELECT * FROM prices_daily WHERE 1=1"
        params = []

        if tickers:
            placeholders = ", ".join(["?" for _ in tickers])
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        if start_date:
            query += " AND date >= ?"
            # Convert to string for SQLite comparison
            params.append(start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date))

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date))

        query += " ORDER BY ticker, date"

        return self.db.fetchdf(query, tuple(params) if params else None)

    def get_intraday_prices(
        self,
        tickers: list[str] | None = None,
        hours_back: int = 24
    ) -> pd.DataFrame:
        """Get recent intraday prices from database."""
        query = "SELECT * FROM prices_intraday WHERE timestamp >= ?"
        params = [datetime.now() - timedelta(hours=hours_back)]

        if tickers:
            placeholders = ", ".join(["?" for _ in tickers])
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        query += " ORDER BY ticker, timestamp"

        return self.db.fetchdf(query, tuple(params))

    def check_data_staleness(self, max_hours: int = 24) -> dict[str, Any]:
        """
        Check if data is stale.

        Returns:
            Dictionary with staleness status
        """
        result = {
            "is_stale": False,
            "last_daily": None,
            "last_intraday": None,
            "hours_since_daily": None,
            "hours_since_intraday": None
        }

        # Check daily ingestion
        row = self.db.fetchone(
            "SELECT value FROM system_state WHERE key = 'last_daily_ingestion'"
        )
        if row and row[0]:
            last_daily = datetime.fromisoformat(row[0])
            result["last_daily"] = last_daily
            hours = (datetime.now() - last_daily).total_seconds() / 3600
            result["hours_since_daily"] = hours
            if hours > max_hours:
                result["is_stale"] = True

        # Check intraday ingestion
        row = self.db.fetchone(
            "SELECT value FROM system_state WHERE key = 'last_intraday_ingestion'"
        )
        if row and row[0]:
            last_intraday = datetime.fromisoformat(row[0])
            result["last_intraday"] = last_intraday
            hours = (datetime.now() - last_intraday).total_seconds() / 3600
            result["hours_since_intraday"] = hours

        return result
