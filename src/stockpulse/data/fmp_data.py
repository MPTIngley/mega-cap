"""
Financial Modeling Prep (FMP) data source for historical fundamentals.

Provides historical P/E ratios and other fundamentals that yfinance doesn't offer.
Free tier: 250 requests/day.
"""

import os
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from stockpulse.data.database import get_db

logger = logging.getLogger(__name__)

# Rate limiting
MIN_REQUEST_INTERVAL = 0.25  # 4 requests per second max


class FMPDataSource:
    """Financial Modeling Prep API data source."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self):
        """Initialize FMP data source."""
        self.api_key = os.environ.get("FMP_API_KEY", "")
        self.db = get_db()
        self._last_request_time = 0

        if not self.api_key:
            logger.warning("FMP_API_KEY not set. Historical P/E data will be limited.")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict | list]:
        """Make API request with rate limiting and error handling."""
        if not self.api_key:
            return None

        self._rate_limit()

        params = params or {}
        params["apikey"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FMP API error for {endpoint}: {e}")
            return None

    def fetch_historical_ratios(self, ticker: str, limit: int = 40) -> pd.DataFrame:
        """
        Fetch historical financial ratios for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Number of periods to fetch (default 40 = ~10 years quarterly)

        Returns:
            DataFrame with historical ratios including P/E
        """
        data = self._make_request(f"ratios/{ticker}", {"limit": limit})

        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append({
                "ticker": ticker,
                "date": item.get("date"),
                "pe_ratio": item.get("priceEarningsRatio"),
                "pb_ratio": item.get("priceToBookRatio"),
                "ps_ratio": item.get("priceToSalesRatio"),
                "peg_ratio": item.get("priceEarningsToGrowthRatio"),
                "dividend_yield": item.get("dividendYield"),
                "roe": item.get("returnOnEquity"),
                "roa": item.get("returnOnAssets"),
                "debt_to_equity": item.get("debtEquityRatio"),
                "current_ratio": item.get("currentRatio"),
                "gross_margin": item.get("grossProfitMargin"),
                "operating_margin": item.get("operatingProfitMargin"),
                "net_margin": item.get("netProfitMargin"),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_key_metrics(self, ticker: str, limit: int = 40) -> pd.DataFrame:
        """
        Fetch historical key metrics for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Number of periods to fetch

        Returns:
            DataFrame with historical key metrics
        """
        data = self._make_request(f"key-metrics/{ticker}", {"limit": limit})

        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append({
                "ticker": ticker,
                "date": item.get("date"),
                "pe_ratio": item.get("peRatio"),
                "market_cap": item.get("marketCap"),
                "enterprise_value": item.get("enterpriseValue"),
                "ev_to_ebitda": item.get("enterpriseValueOverEBITDA"),
                "eps": item.get("netIncomePerShare"),
                "revenue_per_share": item.get("revenuePerShare"),
                "fcf_per_share": item.get("freeCashFlowPerShare"),
                "book_value_per_share": item.get("bookValuePerShare"),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def backfill_historical_pe(self, tickers: list[str], years: int = 5) -> int:
        """
        Backfill historical P/E ratios for multiple tickers.

        Args:
            tickers: List of ticker symbols
            years: Years of history to fetch (default 5)

        Returns:
            Number of records stored
        """
        if not self.api_key:
            logger.error("FMP_API_KEY not set. Cannot backfill historical P/E data.")
            return 0

        total_records = 0
        limit = years * 4 + 4  # Quarterly data + buffer

        logger.info(f"Backfilling historical P/E for {len(tickers)} tickers from FMP...")

        for i, ticker in enumerate(tickers):
            try:
                # Fetch ratios (has P/E)
                ratios_df = self.fetch_historical_ratios(ticker, limit=limit)

                if not ratios_df.empty:
                    # Store in fundamentals table
                    for _, row in ratios_df.iterrows():
                        if pd.notna(row["pe_ratio"]) and row["pe_ratio"] > 0:
                            self.db.execute("""
                                INSERT OR REPLACE INTO fundamentals (
                                    ticker, date, pe_ratio, pb_ratio, peg_ratio,
                                    dividend_yield, roe, debt_to_equity, current_ratio,
                                    profit_margin
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                ticker,
                                str(row["date"]),
                                row["pe_ratio"],
                                row.get("pb_ratio"),
                                row.get("peg_ratio"),
                                row.get("dividend_yield"),
                                row.get("roe"),
                                row.get("debt_to_equity"),
                                row.get("current_ratio"),
                                row.get("net_margin"),
                            ))
                            total_records += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(tickers)} tickers...")

            except Exception as e:
                logger.error(f"Error fetching FMP data for {ticker}: {e}")

        logger.info(f"Backfilled {total_records} historical P/E records from FMP")
        return total_records

    def is_available(self) -> bool:
        """Check if FMP API is available and configured."""
        return bool(self.api_key)


def calculate_pe_from_price_eps(ticker: str, as_of_date: date = None) -> Optional[float]:
    """
    Calculate P/E ratio from historical price and trailing EPS.

    This is a fallback when FMP data isn't available.
    Uses price on the given date divided by trailing 12-month EPS.

    Args:
        ticker: Stock ticker symbol
        as_of_date: Date to calculate P/E for (default: today)

    Returns:
        P/E ratio or None if data unavailable
    """
    import yfinance as yf

    as_of_date = as_of_date or date.today()
    db = get_db()

    try:
        # Get price on or before the target date
        price_row = db.fetchone("""
            SELECT close FROM prices_daily
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (ticker, str(as_of_date)))

        if not price_row:
            return None

        price = price_row[0]

        # Get EPS from yfinance (trailing 12 months)
        stock = yf.Ticker(ticker)
        eps = stock.info.get("trailingEps")

        if not eps or eps <= 0:
            return None

        pe_ratio = price / eps
        return pe_ratio if pe_ratio > 0 else None

    except Exception as e:
        logger.debug(f"Could not calculate P/E for {ticker}: {e}")
        return None


def backfill_calculated_pe(tickers: list[str], days_back: int = 365, verbose: bool = True) -> int:
    """
    Backfill P/E ratios by calculating from historical prices and current EPS.

    Note: This uses current EPS which isn't perfect for historical periods,
    but provides reasonable estimates for P/E percentile calculations.

    Args:
        tickers: List of ticker symbols
        days_back: Number of days to backfill
        verbose: Print progress to console

    Returns:
        Number of records stored
    """
    import yfinance as yf

    db = get_db()
    total_records = 0
    skipped_no_eps = 0
    skipped_no_prices = 0
    errors = 0
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    if verbose:
        print(f"  Processing {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            # Get current EPS (with timeout handling)
            stock = yf.Ticker(ticker)
            info = stock.info
            eps = info.get("trailingEps") if info else None

            if not eps or eps <= 0:
                skipped_no_eps += 1
                continue

            # Get historical prices from local DB
            prices = db.fetchdf("""
                SELECT date, close FROM prices_daily
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (ticker, str(start_date), str(end_date)))

            if prices.empty:
                skipped_no_prices += 1
                continue

            ticker_records = 0
            # Calculate P/E for each date and store
            for _, row in prices.iterrows():
                price = row["close"]
                pe_ratio = price / eps

                if pe_ratio > 0 and pe_ratio < 1000:  # Sanity check
                    db.execute("""
                        INSERT OR IGNORE INTO fundamentals (ticker, date, pe_ratio)
                        VALUES (?, ?, ?)
                    """, (ticker, str(row["date"]), pe_ratio))
                    ticker_records += 1

            total_records += ticker_records

            if verbose and (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(tickers)} tickers ({total_records} records)...")

        except Exception as e:
            errors += 1
            logger.debug(f"Error calculating P/E for {ticker}: {e}")

    if verbose:
        print(f"  Summary: {total_records} records stored")
        print(f"    Skipped (no EPS): {skipped_no_eps}")
        print(f"    Skipped (no prices): {skipped_no_prices}")
        print(f"    Errors: {errors}")

    return total_records
