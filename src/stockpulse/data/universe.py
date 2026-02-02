"""Universe management - tracking the top stocks by market cap."""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db

logger = get_logger(__name__)

# Pre-defined list of top 100+ US stocks by market cap (as of 2024)
# This serves as a fallback and starting point
TOP_US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "LLY",
    "JPM", "XOM", "JNJ", "V", "PG", "MA", "AVGO", "HD", "CVX", "MRK",
    "ABBV", "COST", "PEP", "KO", "ADBE", "WMT", "BAC", "CRM", "MCD", "CSCO",
    "TMO", "ACN", "ABT", "NFLX", "LIN", "AMD", "DHR", "ORCL", "CMCSA", "NKE",
    "TXN", "PFE", "PM", "WFC", "INTC", "UPS", "DIS", "VZ", "NEE", "RTX",
    "COP", "BMY", "QCOM", "HON", "SPGI", "UNP", "INTU", "LOW", "ELV", "BA",
    "CAT", "GE", "AMGN", "IBM", "DE", "SBUX", "ISRG", "AMAT", "GS", "PLD",
    "NOW", "MS", "BLK", "AXP", "MDLZ", "GILD", "LMT", "ADI", "BKNG", "SYK",
    "TJX", "VRTX", "MMC", "CB", "REGN", "ADP", "SCHW", "CVS", "CI", "MO",
    "TMUS", "SO", "LRCX", "ZTS", "DUK", "PGR", "BSX", "BDX", "EOG", "CME",
    "PANW", "ETN", "ITW", "CL", "SNPS", "CDNS", "NOC", "FI", "SLB", "MU",
    "ICE", "EQIX", "APD", "SHW", "FCX", "HUM", "MCK", "AON", "WM", "NSC",
]


class UniverseManager:
    """Manages the stock universe - which stocks to track."""

    def __init__(self):
        """Initialize universe manager."""
        self.db = get_db()
        self.config = get_config()["universe"]

    def get_active_tickers(self) -> list[str]:
        """Get list of active tickers in the universe."""
        df = self.db.fetchdf(
            "SELECT ticker FROM universe WHERE is_active = true ORDER BY market_cap DESC"
        )
        return df["ticker"].tolist() if not df.empty else []

    def refresh_universe(self, force: bool = False) -> list[str]:
        """
        Refresh the universe of stocks to track.

        Args:
            force: If True, refresh even if recently updated

        Returns:
            List of active tickers
        """
        # Check if refresh is needed
        if not force:
            last_refresh = self._get_last_refresh()
            if last_refresh:
                days_since = (datetime.now() - last_refresh).days
                if days_since < self.config.get("refresh_interval_days", 7):
                    logger.info(f"Universe refreshed {days_since} days ago, skipping")
                    return self.get_active_tickers()

        logger.info("Refreshing stock universe...")

        if self.config.get("source") == "manual":
            tickers = self.config.get("manual_tickers", TOP_US_STOCKS[:100])
        else:
            tickers = self._fetch_top_stocks_by_market_cap()

        # Update database
        self._update_universe_db(tickers)

        logger.info(f"Universe refreshed with {len(tickers)} stocks")
        return tickers

    def _fetch_top_stocks_by_market_cap(self) -> list[str]:
        """Fetch top stocks by market cap using yfinance."""
        target_count = self.config.get("count", 100)
        candidates = TOP_US_STOCKS[:150]  # Start with more to filter down

        logger.info(f"Fetching market cap data for {len(candidates)} candidates...")

        stock_data = []
        batch_size = 50  # Process in batches to avoid rate limits

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            try:
                tickers_str = " ".join(batch)
                data = yf.download(
                    tickers_str,
                    period="1d",
                    progress=False,
                    group_by="ticker"
                )

                for ticker in batch:
                    try:
                        info = yf.Ticker(ticker).fast_info
                        market_cap = getattr(info, "market_cap", None)
                        if market_cap and market_cap > 0:
                            stock_data.append({
                                "ticker": ticker,
                                "market_cap": market_cap
                            })
                    except Exception as e:
                        logger.debug(f"Error getting info for {ticker}: {e}")
                        # Use fallback - include it anyway
                        stock_data.append({
                            "ticker": ticker,
                            "market_cap": 0
                        })

            except Exception as e:
                logger.warning(f"Error fetching batch {i}: {e}")
                # Add all tickers from failed batch with 0 market cap
                for ticker in batch:
                    stock_data.append({"ticker": ticker, "market_cap": 0})

        # Sort by market cap and take top N
        stock_data.sort(key=lambda x: x["market_cap"], reverse=True)

        # If we couldn't get market cap data, just use the predefined order
        if all(s["market_cap"] == 0 for s in stock_data):
            logger.warning("Could not fetch market cap data, using predefined list")
            return TOP_US_STOCKS[:target_count]

        return [s["ticker"] for s in stock_data[:target_count]]

    def _update_universe_db(self, tickers: list[str]) -> None:
        """Update universe in database."""
        now = datetime.now()
        today = date.today()

        # Mark all existing as inactive first
        self.db.execute("UPDATE universe SET is_active = false")

        # Fetch company info for new/updated tickers
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                company_name = info.get("shortName") or info.get("longName") or ticker
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")
                market_cap = info.get("marketCap", 0)

                # Upsert into database
                self.db.execute("""
                    INSERT INTO universe (ticker, company_name, sector, industry, market_cap, is_active, added_date, last_refreshed)
                    VALUES (?, ?, ?, ?, ?, true, ?, ?)
                    ON CONFLICT (ticker) DO UPDATE SET
                        company_name = excluded.company_name,
                        sector = excluded.sector,
                        industry = excluded.industry,
                        market_cap = excluded.market_cap,
                        is_active = true,
                        last_refreshed = excluded.last_refreshed
                """, (ticker, company_name, sector, industry, market_cap, today, now))

            except Exception as e:
                logger.warning(f"Error updating {ticker} info: {e}")
                # Insert with minimal info
                self.db.execute("""
                    INSERT INTO universe (ticker, company_name, is_active, added_date, last_refreshed)
                    VALUES (?, ?, true, ?, ?)
                    ON CONFLICT (ticker) DO UPDATE SET
                        is_active = true,
                        last_refreshed = excluded.last_refreshed
                """, (ticker, ticker, today, now))

        # Update system state
        self.db.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('universe_last_refresh', ?, ?)
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (now.isoformat(), now))

    def _get_last_refresh(self) -> datetime | None:
        """Get timestamp of last universe refresh."""
        result = self.db.fetchone(
            "SELECT value FROM system_state WHERE key = 'universe_last_refresh'"
        )
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        return None

    def add_ticker(self, ticker: str) -> bool:
        """Manually add a ticker to the universe."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            self.db.execute("""
                INSERT INTO universe (ticker, company_name, sector, industry, market_cap, is_active, added_date, last_refreshed)
                VALUES (?, ?, ?, ?, ?, true, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET is_active = true
            """, (
                ticker,
                info.get("shortName", ticker),
                info.get("sector", "Unknown"),
                info.get("industry", "Unknown"),
                info.get("marketCap", 0),
                date.today(),
                datetime.now()
            ))
            logger.info(f"Added {ticker} to universe")
            return True

        except Exception as e:
            logger.error(f"Error adding {ticker}: {e}")
            return False

    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from the active universe."""
        try:
            self.db.execute(
                "UPDATE universe SET is_active = false WHERE ticker = ?",
                (ticker,)
            )
            logger.info(f"Removed {ticker} from universe")
            return True
        except Exception as e:
            logger.error(f"Error removing {ticker}: {e}")
            return False

    def get_universe_df(self) -> pd.DataFrame:
        """Get full universe as DataFrame."""
        return self.db.fetchdf("""
            SELECT ticker, company_name, sector, industry, market_cap, is_active, added_date, last_refreshed
            FROM universe
            ORDER BY market_cap DESC
        """)
