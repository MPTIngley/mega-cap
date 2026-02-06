"""Long-term Investment Scanner.

Identifies stocks that may be attractive for long-term investment based on:
- Valuation metrics (P/E, P/B relative to history)
- Technical accumulation signals
- Dividend yield
- Earnings quality
- Insider buying activity
- Free cash flow yield
- Earnings momentum (beat streak)
- Peer relative valuation
"""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd
import numpy as np
import yfinance as yf

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion
from stockpulse.alerts.alert_manager import AlertManager

logger = get_logger(__name__)


class LongTermScanner:
    """
    Scans for long-term investment opportunities.

    Scoring components (8 total):
    - Valuation Score: P/E percentile vs history, P/B, PEG
    - Technical Score: Price near 52-week low, accumulation signals
    - Dividend Score: Yield vs history, payout sustainability
    - Quality Score: Earnings consistency, profitability metrics
    - Insider Score: Recent insider buying activity (NEW)
    - FCF Yield Score: Free cash flow yield vs market (NEW)
    - Earnings Momentum Score: EPS beat streak (NEW)
    - Peer Valuation Score: Valuation vs sector peers (NEW)
    """

    # Default scoring weights (can be optimized)
    DEFAULT_WEIGHTS = {
        "valuation": 0.15,
        "technical": 0.15,
        "dividend": 0.10,
        "quality": 0.15,
        "insider": 0.15,      # High signal value
        "fcf_yield": 0.12,    # Better than P/E
        "earnings_momentum": 0.10,
        "peer_valuation": 0.08,
    }

    # Sector ETF mapping for peer comparison
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financial Services": "XLF",
        "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Basic Materials": "XLB",
        "Communication Services": "XLC",
    }

    def __init__(self, weights: dict | None = None):
        """Initialize long-term scanner."""
        self.db = get_db()
        self.config = get_config()
        self.scanner_config = self.config.get("long_term_scanner", {})
        self.data_ingestion = DataIngestion()
        self.alert_manager = AlertManager()

        # Scoring thresholds
        self.pe_percentile_threshold = self.scanner_config.get("pe_percentile_threshold", 30)
        self.price_near_52w_low_pct = self.scanner_config.get("price_near_52w_low_pct", 15)
        self.min_score = self.scanner_config.get("min_score", 60)

        # Scoring weights (can be customized or optimized)
        self.weights = weights or self.scanner_config.get("weights", self.DEFAULT_WEIGHTS)

        # Cache for yfinance data to avoid repeated API calls
        self._yf_cache = {}

    def _ensure_prices_current(self, tickers: list[str]) -> None:
        """
        Check if prices_daily is up to date and update if stale.

        Considers data stale if the latest date is more than 1 trading day old.
        """
        # Check latest date in prices_daily
        result = self.db.fetchone("""
            SELECT MAX(date) FROM prices_daily
        """)

        if not result or not result[0]:
            logger.warning("No price data in database - running full ingestion")
            self._run_price_update(tickers)
            return

        latest_date_str = result[0]
        try:
            latest_date = datetime.strptime(str(latest_date_str)[:10], "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Could not parse latest date: {latest_date_str}")
            self._run_price_update(tickers)
            return

        today = date.today()
        days_behind = (today - latest_date).days

        # Allow 1 day grace for weekends/holidays, but if 2+ days behind, update
        # Also check if it's a weekday and we're behind
        is_weekday = today.weekday() < 5

        if days_behind >= 2 or (days_behind >= 1 and is_weekday and today.weekday() != 0):
            # More than 1 day behind (and not Monday checking Sunday)
            logger.info(f"Price data is {days_behind} days old (latest: {latest_date}). Updating...")
            self._run_price_update(tickers)
        else:
            logger.info(f"Price data is current (latest: {latest_date})")

    def _run_price_update(self, tickers: list[str]) -> None:
        """Run price ingestion to update database."""
        try:
            logger.info(f"Updating daily prices for {len(tickers)} tickers...")
            results = self.data_ingestion.run_daily_ingestion(tickers)
            logger.info(f"Price update complete: {results}")
        except Exception as e:
            logger.error(f"Failed to update prices: {e}")
            # Continue with scan anyway - will use live prices from yfinance

    def _ensure_watchlist_history(self, min_days: int = 7) -> None:
        """
        Check if long_term_watchlist has enough historical data for trend analysis.

        If not enough history exists OR there are gaps in recent data, runs backfill.

        Args:
            min_days: Minimum number of unique scan dates required (default 7)
        """
        # Check how many unique dates we have in the watchlist (last 14 days)
        result = self.db.fetchone("""
            SELECT COUNT(DISTINCT scan_date) FROM long_term_watchlist
            WHERE scan_date >= date('now', '-14 days')
        """)
        watchlist_dates_14d = result[0] if result and result[0] else 0

        # Check how many trading days exist in prices_daily (last 14 days)
        result = self.db.fetchone("""
            SELECT COUNT(DISTINCT date) FROM prices_daily
            WHERE date >= date('now', '-14 days')
        """)
        price_dates_14d = result[0] if result and result[0] else 0

        # If watchlist has fewer dates than prices, we have gaps to fill
        needs_backfill = False
        missing_days = price_dates_14d - watchlist_dates_14d

        if missing_days >= 2:
            needs_backfill = True
            logger.info(
                f"Watchlist missing {missing_days} days in last 14 days "
                f"(watchlist: {watchlist_dates_14d}, prices: {price_dates_14d}). Running backfill..."
            )

        if watchlist_dates_14d < min_days:
            needs_backfill = True
            logger.info(
                f"Watchlist history insufficient ({watchlist_dates_14d} days in last 14d, need {min_days}). "
                f"Running backfill..."
            )

        if needs_backfill:
            try:
                # Backfill 3 weeks to cover gaps
                records = self.backfill_history(days=21)
                logger.info(f"Backfill complete: {records} records created")
            except Exception as e:
                logger.error(f"Failed to backfill watchlist history: {e}")
        else:
            logger.info(f"Watchlist history sufficient ({watchlist_dates_14d} days in last 14d, no gaps)")

    def run_scan(self, tickers: list[str]) -> list[dict]:
        """
        Run long-term scan on given tickers.

        Args:
            tickers: List of tickers to scan

        Returns:
            List of opportunity dictionaries, sorted by score
        """
        # Ensure price database is up to date before scanning
        self._ensure_prices_current(tickers)

        # Ensure watchlist has enough history for trend analysis
        self._ensure_watchlist_history()

        logger.info(f"Running long-term scan on {len(tickers)} tickers")

        opportunities = []
        today = date.today()

        for ticker in tickers:
            try:
                score_data = self._score_ticker(ticker)

                if score_data and score_data["composite_score"] >= self.min_score:
                    opportunities.append(score_data)

            except Exception as e:
                logger.debug(f"Error scoring {ticker}: {e}")

        # Sort by composite score
        opportunities.sort(key=lambda x: x["composite_score"], reverse=True)

        # Store results
        self._store_opportunities(opportunities)

        # Enrich with trend data
        opportunities = self.enrich_with_trends(opportunities)

        logger.info(f"Found {len(opportunities)} long-term opportunities")

        return opportunities

    def _get_yf_ticker(self, ticker: str) -> yf.Ticker:
        """Get cached yfinance Ticker object."""
        if ticker not in self._yf_cache:
            self._yf_cache[ticker] = yf.Ticker(ticker)
        return self._yf_cache[ticker]

    def _score_ticker(self, ticker: str) -> dict | None:
        """
        Calculate composite score for a ticker.

        Returns None if insufficient data.
        """
        # Get price data from database (historical)
        price_data = self.data_ingestion.get_daily_prices(
            [ticker],
            start_date=date.today() - timedelta(days=365)
        )

        if price_data.empty or len(price_data) < 200:
            return None

        # Get fundamentals
        fundamentals = self.db.fetchone("""
            SELECT * FROM fundamentals
            WHERE ticker = ?
            ORDER BY date DESC LIMIT 1
        """, (ticker,))

        if not fundamentals:
            return None

        # Get yfinance ticker for additional data AND live price
        yf_ticker = self._get_yf_ticker(ticker)

        # Fetch live/current price from yfinance (not stale DB price)
        # Initialize variables that may be set in try block
        yf_info = None
        live_price = None
        try:
            yf_info = yf_ticker.info
            live_price = yf_info.get('regularMarketPrice') or yf_info.get('currentPrice')
            if live_price and live_price > 0:
                # Update the last row of price_data with live price for technical calcs
                price_data = price_data.sort_values('date').reset_index(drop=True)
                price_data.loc[price_data.index[-1], 'close'] = live_price
            else:
                live_price = None  # Ensure it's None if invalid
        except Exception as e:
            logger.debug(f"Could not fetch live price for {ticker}: {e}")

        # Calculate all individual scores
        valuation_score = self._calculate_valuation_score(ticker, fundamentals)
        technical_score = self._calculate_technical_score(price_data)
        dividend_score = self._calculate_dividend_score(fundamentals)
        quality_score = self._calculate_quality_score(fundamentals)

        # NEW: Additional scores
        insider_score = self._calculate_insider_score(yf_ticker)
        fcf_yield_score = self._calculate_fcf_yield_score(yf_ticker)
        earnings_momentum_score = self._calculate_earnings_momentum_score(yf_ticker)
        peer_valuation_score = self._calculate_peer_valuation_score(ticker, yf_ticker)

        # Weighted composite using configurable weights
        w = self.weights
        composite_score = (
            valuation_score * w.get("valuation", 0.15) +
            technical_score * w.get("technical", 0.15) +
            dividend_score * w.get("dividend", 0.10) +
            quality_score * w.get("quality", 0.15) +
            insider_score * w.get("insider", 0.15) +
            fcf_yield_score * w.get("fcf_yield", 0.12) +
            earnings_momentum_score * w.get("earnings_momentum", 0.10) +
            peer_valuation_score * w.get("peer_valuation", 0.08)
        )

        # Calculate supporting metrics - use live price if available
        # (live_price was set earlier when we fetched yf_ticker.info)
        current_price = live_price if live_price else price_data["close"].iloc[-1]

        # Get company info (reuse yf_info from live price fetch if available)
        try:
            if yf_info is None:
                yf_info = yf_ticker.info
            company_name = yf_info.get("shortName", yf_info.get("longName", ticker))
            sector = yf_info.get("sector", "Unknown")
        except Exception:
            company_name = ticker
            sector = "Unknown"
        fifty_two_week_low = price_data["low"].min()
        fifty_two_week_high = price_data["high"].max()

        price_vs_52w_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100

        # P/E percentile calculation
        pe_ratio = fundamentals[2] if len(fundamentals) > 2 else None  # pe_ratio column
        pe_percentile = self._calculate_pe_percentile(ticker, pe_ratio) if pe_ratio else 50

        # Generate reasoning with new scores
        reasoning = self._generate_reasoning_enhanced(
            ticker=ticker,
            scores={
                "valuation": valuation_score,
                "technical": technical_score,
                "dividend": dividend_score,
                "quality": quality_score,
                "insider": insider_score,
                "fcf_yield": fcf_yield_score,
                "earnings_momentum": earnings_momentum_score,
                "peer_valuation": peer_valuation_score,
            },
            pe_percentile=pe_percentile,
            price_vs_52w_low=price_vs_52w_low
        )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "scan_date": date.today(),
            "composite_score": composite_score,
            "valuation_score": valuation_score,
            "technical_score": technical_score,
            "dividend_score": dividend_score,
            "quality_score": quality_score,
            "insider_score": insider_score,
            "fcf_score": fcf_yield_score,  # Alias for email compatibility
            "earnings_score": earnings_momentum_score,  # Alias for email compatibility
            "peer_score": peer_valuation_score,  # Alias for email compatibility
            "fcf_yield_score": fcf_yield_score,
            "earnings_momentum_score": earnings_momentum_score,
            "peer_valuation_score": peer_valuation_score,
            "pe_percentile": pe_percentile,
            "price_vs_52w_low_pct": price_vs_52w_low,
            "week52_high": fifty_two_week_high,
            "week52_low": fifty_two_week_low,
            "current_price": current_price,
            "reasoning": reasoning
        }

    def _calculate_valuation_score(self, ticker: str, fundamentals: tuple) -> float:
        """Calculate valuation score based on P/E, P/B, PEG."""
        score = 50  # Base score

        # Unpack fundamentals (based on schema order)
        # ticker, date, pe_ratio, forward_pe, pb_ratio, peg_ratio, dividend_yield, ...
        pe_ratio = fundamentals[2] if len(fundamentals) > 2 else None
        forward_pe = fundamentals[3] if len(fundamentals) > 3 else None
        pb_ratio = fundamentals[4] if len(fundamentals) > 4 else None
        peg_ratio = fundamentals[5] if len(fundamentals) > 5 else None

        # P/E score (lower is better for value)
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 10:
                score += 25
            elif pe_ratio < 15:
                score += 15
            elif pe_ratio < 20:
                score += 5
            elif pe_ratio > 40:
                score -= 15

        # Forward P/E (growth expectation)
        if forward_pe and pe_ratio and forward_pe < pe_ratio:
            score += 10  # Earnings expected to grow

        # P/B ratio (lower = more value)
        if pb_ratio and pb_ratio > 0:
            if pb_ratio < 1.0:
                score += 15
            elif pb_ratio < 2.0:
                score += 5
            elif pb_ratio > 5.0:
                score -= 10

        # PEG ratio (growth at reasonable price)
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 1.0:
                score += 15  # Undervalued relative to growth
            elif peg_ratio < 1.5:
                score += 5
            elif peg_ratio > 2.5:
                score -= 10

        return max(0, min(100, score))

    def _calculate_technical_score(self, price_data: pd.DataFrame) -> float:
        """Calculate technical score based on price position and accumulation."""
        score = 50  # Base score

        if price_data.empty:
            return score

        current_price = price_data["close"].iloc[-1]

        # 52-week range position
        low_52w = price_data["low"].min()
        high_52w = price_data["high"].max()
        range_52w = high_52w - low_52w

        if range_52w > 0:
            position_in_range = (current_price - low_52w) / range_52w

            # Near 52-week low is attractive
            if position_in_range < 0.15:
                score += 25
            elif position_in_range < 0.25:
                score += 15
            elif position_in_range < 0.35:
                score += 5
            elif position_in_range > 0.90:
                score -= 10  # Near highs = less attractive

        # Price vs moving averages
        sma_50 = price_data["close"].rolling(50).mean().iloc[-1]
        sma_200 = price_data["close"].rolling(200).mean().iloc[-1]

        if pd.notna(sma_200):
            # Price below 200-day MA can indicate value
            if current_price < sma_200 * 0.95:
                score += 10
            elif current_price < sma_200:
                score += 5

        # Accumulation signal (price rising on increasing volume)
        if "volume" in price_data.columns:
            recent_volume = price_data["volume"].iloc[-20:].mean()
            older_volume = price_data["volume"].iloc[-40:-20].mean()

            recent_price_change = (
                price_data["close"].iloc[-1] / price_data["close"].iloc[-20] - 1
            )

            if recent_volume > older_volume and recent_price_change > 0:
                score += 10  # Accumulation detected

        # RSI for oversold
        delta = price_data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if pd.notna(current_rsi):
            if current_rsi < 30:
                score += 15  # Oversold
            elif current_rsi < 40:
                score += 5

        return max(0, min(100, score))

    def _calculate_dividend_score(self, fundamentals: tuple) -> float:
        """Calculate dividend score."""
        score = 50  # Base score

        dividend_yield = fundamentals[6] if len(fundamentals) > 6 else None

        if dividend_yield and dividend_yield > 0:
            # Higher yield is better (to a point)
            if dividend_yield > 0.05:  # > 5%
                score += 20  # But watch for dividend traps
            elif dividend_yield > 0.03:  # > 3%
                score += 15
            elif dividend_yield > 0.02:  # > 2%
                score += 10
            elif dividend_yield > 0.01:  # > 1%
                score += 5

            # Very high yields might indicate distress
            if dividend_yield > 0.08:  # > 8%
                score -= 10  # Potential dividend cut risk
        else:
            # No dividend - neutral for growth stocks
            pass

        return max(0, min(100, score))

    def _calculate_quality_score(self, fundamentals: tuple) -> float:
        """Calculate quality score based on profitability metrics."""
        score = 50  # Base score

        # Unpack (schema: ticker, date, pe, fwd_pe, pb, peg, div_yield, eps, revenue, profit_margin, roe, ...)
        profit_margin = fundamentals[9] if len(fundamentals) > 9 else None
        roe = fundamentals[10] if len(fundamentals) > 10 else None
        debt_to_equity = fundamentals[11] if len(fundamentals) > 11 else None
        current_ratio = fundamentals[12] if len(fundamentals) > 12 else None

        # Profit margin
        if profit_margin and profit_margin > 0:
            if profit_margin > 0.20:  # > 20%
                score += 20
            elif profit_margin > 0.10:  # > 10%
                score += 10
            elif profit_margin > 0.05:  # > 5%
                score += 5

        # ROE
        if roe and roe > 0:
            if roe > 0.20:  # > 20%
                score += 15
            elif roe > 0.15:  # > 15%
                score += 10
            elif roe > 0.10:  # > 10%
                score += 5

        # Debt to equity (lower is better)
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                score += 10
            elif debt_to_equity < 1.0:
                score += 5
            elif debt_to_equity > 2.0:
                score -= 10

        # Current ratio (liquidity)
        if current_ratio and current_ratio > 0:
            if current_ratio > 2.0:
                score += 5
            elif current_ratio > 1.5:
                score += 3
            elif current_ratio < 1.0:
                score -= 10  # Liquidity concern

        return max(0, min(100, score))

    def _calculate_insider_score(self, yf_ticker: yf.Ticker) -> float:
        """
        Calculate insider buying score.

        Strong insider buying (especially by multiple insiders or large purchases)
        is one of the most predictive signals for long-term outperformance.
        """
        score = 50  # Base score

        try:
            # Get insider transactions
            insider_txns = yf_ticker.insider_transactions
            if insider_txns is None or insider_txns.empty:
                return score

            # Look at last 90 days of transactions
            ninety_days_ago = datetime.now() - timedelta(days=90)

            # Filter recent transactions
            if "Start Date" in insider_txns.columns:
                insider_txns["Start Date"] = pd.to_datetime(insider_txns["Start Date"], errors="coerce")
                recent = insider_txns[insider_txns["Start Date"] >= ninety_days_ago]
            else:
                recent = insider_txns.head(10)  # Fallback to recent rows

            if recent.empty:
                return score

            # Count buys vs sells
            buys = 0
            sells = 0
            buy_value = 0
            sell_value = 0

            for _, txn in recent.iterrows():
                txn_type = str(txn.get("Transaction", "")).lower()
                shares = abs(txn.get("Shares", 0) or 0)
                value = abs(txn.get("Value", 0) or 0)

                if "purchase" in txn_type or "buy" in txn_type or "acquisition" in txn_type:
                    buys += 1
                    buy_value += value
                elif "sale" in txn_type or "sell" in txn_type:
                    sells += 1
                    sell_value += value

            # Scoring based on net insider activity
            net_buys = buys - sells

            if net_buys >= 3:
                score += 30  # Strong cluster of insider buying
            elif net_buys >= 2:
                score += 20
            elif net_buys >= 1:
                score += 10
            elif net_buys <= -3:
                score -= 20  # Heavy insider selling
            elif net_buys <= -1:
                score -= 10

            # Bonus for large purchases
            if buy_value > 1_000_000:  # > $1M in purchases
                score += 15
            elif buy_value > 500_000:
                score += 10
            elif buy_value > 100_000:
                score += 5

        except Exception as e:
            logger.debug(f"Error getting insider data: {e}")

        return max(0, min(100, score))

    def _calculate_fcf_yield_score(self, yf_ticker: yf.Ticker) -> float:
        """
        Calculate Free Cash Flow Yield score.

        FCF Yield = Free Cash Flow / Market Cap
        Higher is better (indicates undervaluation relative to cash generation).
        FCF Yield > 8% is very attractive for value investors.
        """
        score = 50  # Base score

        try:
            info = yf_ticker.info

            # Get market cap
            market_cap = info.get("marketCap", 0)
            if not market_cap or market_cap <= 0:
                return score

            # Try to get FCF from cash flow statement
            cashflow = yf_ticker.cashflow
            if cashflow is None or cashflow.empty:
                return score

            # FCF = Operating Cash Flow - Capital Expenditures
            ocf = None
            capex = None

            # Look for operating cash flow
            ocf_keys = ["Operating Cash Flow", "Total Cash From Operating Activities"]
            for key in ocf_keys:
                if key in cashflow.index:
                    ocf = cashflow.loc[key].iloc[0]  # Most recent year
                    break

            # Look for capex (usually negative)
            capex_keys = ["Capital Expenditure", "Capital Expenditures"]
            for key in capex_keys:
                if key in cashflow.index:
                    capex = cashflow.loc[key].iloc[0]
                    break

            if ocf is None:
                # Fallback: try using freeCashflow from info
                fcf = info.get("freeCashflow", 0)
            else:
                # capex is typically negative, so we add it
                capex = capex or 0
                fcf = ocf + capex if capex < 0 else ocf - abs(capex)

            if not fcf or fcf <= 0:
                return score  # No positive FCF

            # Calculate FCF Yield
            fcf_yield = (fcf / market_cap) * 100  # As percentage

            # Score based on FCF yield
            if fcf_yield >= 10:
                score += 30  # Excellent - very undervalued
            elif fcf_yield >= 8:
                score += 25
            elif fcf_yield >= 6:
                score += 20
            elif fcf_yield >= 5:
                score += 15
            elif fcf_yield >= 4:
                score += 10
            elif fcf_yield >= 3:
                score += 5
            elif fcf_yield < 1:
                score -= 10  # Very low FCF yield

        except Exception as e:
            logger.debug(f"Error calculating FCF yield: {e}")

        return max(0, min(100, score))

    def _calculate_earnings_momentum_score(self, yf_ticker: yf.Ticker) -> float:
        """
        Calculate earnings momentum score based on EPS beat streak.

        Consistent earnings beats indicate management execution and
        conservative guidance - both positive for long-term performance.
        """
        score = 50  # Base score

        try:
            # Get earnings history
            earnings = yf_ticker.earnings_history
            if earnings is None or earnings.empty:
                return score

            # Count beats vs misses in recent quarters
            beats = 0
            misses = 0
            beat_streak = 0
            current_streak = 0

            for _, row in earnings.iterrows():
                surprise_pct = row.get("surprisePercent", row.get("Surprise(%)", None))

                if surprise_pct is not None:
                    if surprise_pct > 0:
                        beats += 1
                        if current_streak >= 0:
                            current_streak += 1
                        else:
                            current_streak = 1
                        beat_streak = max(beat_streak, current_streak)
                    elif surprise_pct < 0:
                        misses += 1
                        if current_streak <= 0:
                            current_streak -= 1
                        else:
                            current_streak = -1

            total_quarters = beats + misses

            if total_quarters == 0:
                return score

            # Score based on beat rate
            beat_rate = beats / total_quarters

            if beat_rate >= 0.9:  # 90%+ beats
                score += 25
            elif beat_rate >= 0.75:
                score += 20
            elif beat_rate >= 0.6:
                score += 10
            elif beat_rate < 0.4:
                score -= 15  # Frequent misses

            # Bonus for beat streak
            if beat_streak >= 4:
                score += 15  # 4+ consecutive beats
            elif beat_streak >= 3:
                score += 10
            elif beat_streak >= 2:
                score += 5

        except Exception as e:
            logger.debug(f"Error calculating earnings momentum: {e}")

        return max(0, min(100, score))

    def _calculate_peer_valuation_score(self, ticker: str, yf_ticker: yf.Ticker) -> float:
        """
        Calculate peer relative valuation score.

        Compares P/E and P/B to sector median.
        Being cheaper than peers (while maintaining quality) is bullish.
        """
        score = 50  # Base score

        try:
            info = yf_ticker.info

            # Get company's metrics
            pe_ratio = info.get("trailingPE", info.get("forwardPE"))
            pb_ratio = info.get("priceToBook")
            sector = info.get("sector")

            if not sector:
                return score

            # Get peer companies in same sector
            # Note: For a full implementation, we'd query all stocks in the sector
            # Here we'll use sector averages from yfinance info
            sector_pe = info.get("sectorPE")
            sector_pb = info.get("sectorPriceToBook")

            # If sector averages not available, use reasonable defaults
            if not sector_pe:
                sector_pe = 20  # Market average
            if not sector_pb:
                sector_pb = 3  # Market average

            # Compare P/E to sector
            if pe_ratio and pe_ratio > 0:
                pe_discount = (sector_pe - pe_ratio) / sector_pe * 100

                if pe_discount >= 30:  # 30%+ cheaper than sector
                    score += 20
                elif pe_discount >= 20:
                    score += 15
                elif pe_discount >= 10:
                    score += 10
                elif pe_discount >= 0:
                    score += 5
                elif pe_discount < -30:  # 30%+ more expensive
                    score -= 15
                elif pe_discount < -20:
                    score -= 10

            # Compare P/B to sector
            if pb_ratio and pb_ratio > 0:
                pb_discount = (sector_pb - pb_ratio) / sector_pb * 100

                if pb_discount >= 30:
                    score += 10
                elif pb_discount >= 20:
                    score += 7
                elif pb_discount >= 10:
                    score += 5
                elif pb_discount < -30:
                    score -= 10
                elif pb_discount < -20:
                    score -= 5

        except Exception as e:
            logger.debug(f"Error calculating peer valuation: {e}")

        return max(0, min(100, score))

    def _generate_reasoning_enhanced(
        self,
        ticker: str,
        scores: dict,
        pe_percentile: float,
        price_vs_52w_low: float
    ) -> str:
        """Generate enhanced human-readable reasoning with all score components."""
        reasons = []

        # Valuation
        if scores["valuation"] >= 70:
            reasons.append("attractive valuation metrics")
        elif scores["valuation"] >= 60:
            reasons.append("reasonable valuation")

        # Technical
        if scores["technical"] >= 70:
            reasons.append("strong technical setup near support")
        elif scores["technical"] >= 60:
            reasons.append("favorable technical position")

        # P/E percentile
        if pe_percentile < 30:
            reasons.append(f"P/E in bottom {pe_percentile:.0f}% of historical range")

        # Price vs 52-week low
        if price_vs_52w_low < 15:
            reasons.append(f"trading {price_vs_52w_low:.1f}% above 52-week low")

        # Dividend
        if scores["dividend"] >= 70:
            reasons.append("attractive dividend yield")

        # Quality
        if scores["quality"] >= 70:
            reasons.append("strong profitability metrics")

        # NEW: Insider activity
        if scores["insider"] >= 75:
            reasons.append("significant insider buying")
        elif scores["insider"] >= 65:
            reasons.append("positive insider activity")

        # NEW: FCF Yield
        if scores["fcf_yield"] >= 75:
            reasons.append("excellent free cash flow yield")
        elif scores["fcf_yield"] >= 65:
            reasons.append("strong cash generation")

        # NEW: Earnings momentum
        if scores["earnings_momentum"] >= 75:
            reasons.append("consistent earnings beats")
        elif scores["earnings_momentum"] >= 65:
            reasons.append("positive earnings momentum")

        # NEW: Peer valuation
        if scores["peer_valuation"] >= 70:
            reasons.append("cheapest in sector")
        elif scores["peer_valuation"] >= 60:
            reasons.append("below-average sector valuation")

        if not reasons:
            reasons.append("meets minimum screening criteria")

        return f"{ticker}: " + ", ".join(reasons) + "."

    def _calculate_pe_percentile(self, ticker: str, current_pe: float) -> float:
        """Calculate where current P/E falls in historical distribution."""
        # Get historical P/E data from fundamentals table
        historical = self.db.fetchdf("""
            SELECT pe_ratio FROM fundamentals
            WHERE ticker = ? AND pe_ratio IS NOT NULL AND pe_ratio > 0
            ORDER BY date
        """, (ticker,))

        if historical.empty or len(historical) < 5:
            # Try to calculate P/E from price history if we don't have enough data
            try:
                from stockpulse.data.fmp_data import calculate_pe_from_price_eps
                # If current_pe is valid, compare to a simple market average
                if current_pe and current_pe > 0:
                    # Use S&P 500 historical average P/E range (15-25) as reference
                    # Percentile based on where current P/E falls in typical range
                    if current_pe < 12:
                        return 10  # Very cheap
                    elif current_pe < 15:
                        return 25  # Cheap
                    elif current_pe < 20:
                        return 50  # Average
                    elif current_pe < 25:
                        return 70  # Expensive
                    elif current_pe < 35:
                        return 85  # Very expensive
                    else:
                        return 95  # Extremely expensive
            except Exception:
                pass
            return 50  # Default to middle if nothing works

        pe_values = historical["pe_ratio"].values
        percentile = (pe_values < current_pe).sum() / len(pe_values) * 100

        return percentile

    def _generate_reasoning(
        self,
        ticker: str,
        valuation_score: float,
        technical_score: float,
        dividend_score: float,
        quality_score: float,
        pe_percentile: float,
        price_vs_52w_low: float
    ) -> str:
        """Generate human-readable reasoning for the opportunity."""
        reasons = []

        if valuation_score >= 70:
            reasons.append("attractive valuation metrics")
        elif valuation_score >= 60:
            reasons.append("reasonable valuation")

        if technical_score >= 70:
            reasons.append("strong technical setup near support")
        elif technical_score >= 60:
            reasons.append("favorable technical position")

        if pe_percentile < 30:
            reasons.append(f"P/E in bottom {pe_percentile:.0f}% of historical range")

        if price_vs_52w_low < 15:
            reasons.append(f"trading {price_vs_52w_low:.1f}% above 52-week low")

        if dividend_score >= 70:
            reasons.append("attractive dividend yield")

        if quality_score >= 70:
            reasons.append("strong profitability metrics")

        if not reasons:
            reasons.append("meets minimum screening criteria")

        return f"{ticker}: " + ", ".join(reasons) + "."

    def _store_opportunities(self, opportunities: list[dict]) -> None:
        """Store opportunities in database with full score breakdown."""
        for opp in opportunities:
            try:
                self.db.execute("""
                    INSERT INTO long_term_watchlist (
                        ticker, scan_date, composite_score, valuation_score,
                        technical_score, dividend_score, quality_score,
                        insider_score, fcf_score, earnings_score, peer_score,
                        pe_percentile, price_vs_52w_low_pct, reasoning,
                        company_name, sector, week52_high, week52_low, current_price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, scan_date) DO UPDATE SET
                        composite_score = excluded.composite_score,
                        valuation_score = excluded.valuation_score,
                        technical_score = excluded.technical_score,
                        dividend_score = excluded.dividend_score,
                        quality_score = excluded.quality_score,
                        insider_score = excluded.insider_score,
                        fcf_score = excluded.fcf_score,
                        earnings_score = excluded.earnings_score,
                        peer_score = excluded.peer_score,
                        pe_percentile = excluded.pe_percentile,
                        price_vs_52w_low_pct = excluded.price_vs_52w_low_pct,
                        reasoning = excluded.reasoning,
                        company_name = excluded.company_name,
                        sector = excluded.sector,
                        week52_high = excluded.week52_high,
                        week52_low = excluded.week52_low,
                        current_price = excluded.current_price
                """, (
                    opp["ticker"],
                    opp["scan_date"],
                    opp["composite_score"],
                    opp["valuation_score"],
                    opp["technical_score"],
                    opp.get("dividend_score", 50),
                    opp["quality_score"],
                    opp.get("insider_score", 50),
                    opp.get("fcf_score", 50),
                    opp.get("earnings_score", 50),
                    opp.get("peer_score", 50),
                    opp["pe_percentile"],
                    opp["price_vs_52w_low_pct"],
                    opp["reasoning"],
                    opp.get("company_name", opp["ticker"]),
                    opp.get("sector", "Unknown"),
                    opp.get("week52_high"),
                    opp.get("week52_low"),
                    opp.get("current_price")
                ))
            except Exception as e:
                logger.error(f"Error storing opportunity for {opp['ticker']}: {e}")

    def get_watchlist(self, days_back: int = 7) -> pd.DataFrame:
        """Get recent watchlist entries."""
        return self.db.fetchdf("""
            SELECT * FROM long_term_watchlist
            WHERE scan_date >= date('now', ?)
            ORDER BY composite_score DESC
        """, (f'-{days_back} days',))

    def get_trend_data(self, ticker: str, days_back: int = 30) -> dict:
        """
        Get trend data for a ticker including consecutive days and score history.

        Returns:
            dict with keys: consecutive_days, score_history, avg_5d, avg_20d,
                           change_vs_yesterday, change_vs_5d_avg, trend_symbol
        """
        history = self.db.fetchdf("""
            SELECT scan_date, composite_score
            FROM long_term_watchlist
            WHERE ticker = ? AND scan_date >= date('now', ?)
            ORDER BY scan_date DESC
        """, (ticker, f'-{days_back} days'))

        if history.empty:
            return {
                "consecutive_days": 0,
                "score_history": [],
                "avg_5d": 0,
                "avg_20d": 0,
                "change_vs_yesterday": 0,
                "change_vs_5d_avg": 0,
                "trend_symbol": "🆕",
                "is_new": True
            }

        scores = history["composite_score"].tolist()
        dates = history["scan_date"].tolist()

        # Calculate consecutive days (how many recent consecutive days on list)
        consecutive_days = 1
        for i in range(1, len(dates)):
            # Check if dates are consecutive trading days (within 3 calendar days to account for weekends)
            from datetime import datetime, timedelta
            # Handle various date formats from pandas/SQLite
            try:
                d1_raw = dates[i-1]
                d2_raw = dates[i]
                # Convert to datetime if needed
                if hasattr(d1_raw, 'strftime'):
                    d1 = d1_raw if isinstance(d1_raw, datetime) else datetime.combine(d1_raw, datetime.min.time())
                else:
                    d1 = datetime.strptime(str(d1_raw)[:10], "%Y-%m-%d")
                if hasattr(d2_raw, 'strftime'):
                    d2 = d2_raw if isinstance(d2_raw, datetime) else datetime.combine(d2_raw, datetime.min.time())
                else:
                    d2 = datetime.strptime(str(d2_raw)[:10], "%Y-%m-%d")
                if (d1 - d2).days <= 3:
                    consecutive_days += 1
                else:
                    break
            except Exception as e:
                logger.debug(f"Date parsing error: {e}, dates: {dates[i-1]}, {dates[i]}")
                break

        # Calculate averages
        avg_5d = sum(scores[:5]) / len(scores[:5]) if len(scores) >= 1 else scores[0]
        avg_20d = sum(scores[:20]) / len(scores[:20]) if len(scores) >= 1 else scores[0]

        # Today's score
        today_score = scores[0]

        # Change vs yesterday
        change_vs_yesterday = (today_score - scores[1]) if len(scores) > 1 else 0

        # Change vs 5-day average
        change_vs_5d_avg = today_score - avg_5d

        # Determine trend symbol
        if len(scores) == 1:
            trend_symbol = "🆕"  # New
        elif change_vs_5d_avg > 2:
            trend_symbol = "📈"  # Strengthening
        elif change_vs_5d_avg < -2:
            trend_symbol = "📉"  # Weakening
        else:
            trend_symbol = "➡️"  # Stable

        return {
            "consecutive_days": consecutive_days,
            "score_history": scores[:10],
            "avg_5d": avg_5d,
            "avg_20d": avg_20d,
            "change_vs_yesterday": change_vs_yesterday,
            "change_vs_5d_avg": change_vs_5d_avg,
            "trend_symbol": trend_symbol,
            "is_new": len(scores) == 1
        }

    def enrich_with_trends(self, opportunities: list[dict]) -> list[dict]:
        """Add trend data to each opportunity."""
        for opp in opportunities:
            trend = self.get_trend_data(opp["ticker"])
            opp["consecutive_days"] = trend["consecutive_days"]
            opp["trend_symbol"] = trend["trend_symbol"]
            opp["change_vs_yesterday"] = trend["change_vs_yesterday"]
            opp["change_vs_5d_avg"] = trend["change_vs_5d_avg"]
            opp["is_new"] = trend["is_new"]
            opp["avg_5d"] = trend["avg_5d"]
        return opportunities

    def categorize_opportunities(self, opportunities: list[dict]) -> dict:
        """
        Categorize opportunities by signal strength.

        Returns:
            dict with keys: strong_buy, strengthening, new_signals, persistent, other
        """
        categories = {
            "strong_buy": [],      # Score 70+ AND improving AND 3+ days
            "strengthening": [],   # Score improving vs 5d avg
            "new_signals": [],     # First appearance
            "persistent": [],      # 5+ consecutive days
            "other": []
        }

        for opp in opportunities:
            score = opp.get("composite_score", 0)
            days = opp.get("consecutive_days", 0)
            change_5d = opp.get("change_vs_5d_avg", 0)
            is_new = opp.get("is_new", False)

            # Categorize (a stock can be in multiple categories)
            if score >= 68 and change_5d >= 0 and days >= 3:
                categories["strong_buy"].append(opp)
            elif is_new:
                categories["new_signals"].append(opp)
            elif change_5d > 2:
                categories["strengthening"].append(opp)
            elif days >= 5:
                categories["persistent"].append(opp)
            else:
                categories["other"].append(opp)

        return categories

    def backfill_history(self, days: int = 42) -> int:
        """
        Backfill historical scan data by running scanner on past dates.

        Args:
            days: Number of days to backfill (default 42 = 6 weeks)

        Returns:
            Number of records created
        """
        from datetime import datetime, timedelta
        from stockpulse.data.universe import UniverseManager

        logger.info(f"Backfilling {days} days of long-term scan history...")

        universe = UniverseManager()
        tickers = universe.get_active_tickers()

        if not tickers:
            logger.warning("No tickers in universe")
            return 0

        records_created = 0
        today = datetime.now().date()

        # Get historical price data
        end_date = today
        start_date = today - timedelta(days=days + 365)  # Extra year for indicators

        logger.info(f"Fetching price data from {start_date} to {end_date}...")
        price_data = self.data_ingestion.get_daily_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )

        if price_data.empty:
            logger.warning("No price data available for backfill")
            return 0

        # Process each trading day
        trading_days = sorted(price_data["date"].unique())[-days:]
        logger.info(f"Processing {len(trading_days)} trading days...")

        for i, scan_date in enumerate(trading_days):
            # Get data up to this date
            historical_data = price_data[price_data["date"] <= scan_date].copy()

            # Score each ticker as of this date
            opportunities = []
            for ticker in tickers:
                ticker_data = historical_data[historical_data["ticker"] == ticker]
                if len(ticker_data) < 60:
                    continue

                try:
                    # Calculate scores using data available up to scan_date
                    score_data = self._score_ticker_historical(ticker, ticker_data, scan_date)
                    if score_data and score_data.get("composite_score", 0) >= 60:
                        score_data["scan_date"] = str(scan_date)[:10]
                        opportunities.append(score_data)
                except Exception as e:
                    continue  # Skip errors silently during backfill

            # Store opportunities for this date
            if opportunities:
                self._store_opportunities(opportunities)
                records_created += len(opportunities)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(trading_days)} days, {records_created} records")

        logger.info(f"Backfill complete: {records_created} records created")
        return records_created

    def get_watchlist_stats(self) -> dict:
        """Get statistics about the long-term watchlist data."""
        stats = {}

        # Total records
        result = self.db.fetchone("SELECT COUNT(*) FROM long_term_watchlist")
        stats["total_records"] = result[0] if result else 0

        # Date range
        result = self.db.fetchone("""
            SELECT MIN(scan_date), MAX(scan_date), COUNT(DISTINCT scan_date)
            FROM long_term_watchlist
        """)
        if result:
            stats["earliest_date"] = result[0]
            stats["latest_date"] = result[1]
            stats["unique_dates"] = result[2]

        # Unique tickers
        result = self.db.fetchone("SELECT COUNT(DISTINCT ticker) FROM long_term_watchlist")
        stats["unique_tickers"] = result[0] if result else 0

        # Recent days (last 7)
        result = self.db.fetchone("""
            SELECT COUNT(*) FROM long_term_watchlist
            WHERE scan_date >= date('now', '-7 days')
        """)
        stats["records_last_7_days"] = result[0] if result else 0

        return stats

    def _score_ticker_historical(self, ticker: str, price_data: pd.DataFrame, as_of_date) -> dict | None:
        """Score a ticker using historical data up to a specific date."""
        try:
            price_data = price_data.sort_values("date")
            latest = price_data.iloc[-1]
            current_price = latest["close"]

            # Calculate indicators
            close_prices = price_data["close"]

            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            # 52-week low
            low_52w = price_data["low"].rolling(min(252, len(price_data))).min().iloc[-1]
            price_vs_52w_low_pct = ((current_price - low_52w) / low_52w * 100) if low_52w > 0 else 0

            # Technical score based on RSI and distance from 52w low
            technical_score = 50
            if current_rsi < 40:
                technical_score += 25
            if price_vs_52w_low_pct < 15:
                technical_score += 25

            # Valuation score (simplified - based on price trend)
            ma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else current_price
            valuation_score = 70 if current_price < ma_50 else 50

            # Composite score (simplified for backfill)
            composite = (valuation_score * 0.25 + technical_score * 0.25 +
                        50 * 0.5)  # Other factors default to 50

            return {
                "ticker": ticker,
                "composite_score": composite,
                "valuation_score": valuation_score,
                "technical_score": technical_score,
                "dividend_score": 50,
                "quality_score": 50,
                "insider_score": 50,
                "fcf_score": 50,
                "earnings_score": 50,
                "peer_score": 50,
                "pe_percentile": 50,
                "price_vs_52w_low_pct": price_vs_52w_low_pct,
                "reasoning": f"{ticker}: historical scan as of {str(as_of_date)[:10]}"
            }

        except Exception as e:
            return None

    def send_digest(self, opportunities: list[dict]) -> bool:
        """Send long-term opportunities digest email."""
        if not opportunities:
            return False

        return self.alert_manager.send_long_term_digest(opportunities)
