"""Long-term Investment Scanner.

Identifies stocks that may be attractive for long-term investment based on:
- Valuation metrics (P/E, P/B relative to history)
- Technical accumulation signals
- Dividend yield
- Earnings quality
"""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd
import numpy as np

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion
from stockpulse.alerts.alert_manager import AlertManager

logger = get_logger(__name__)


class LongTermScanner:
    """
    Scans for long-term investment opportunities.

    Scoring components:
    - Valuation Score: P/E percentile vs history, P/B, PEG
    - Technical Score: Price near 52-week low, accumulation signals
    - Dividend Score: Yield vs history, payout sustainability
    - Quality Score: Earnings consistency, profitability metrics
    """

    def __init__(self):
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

    def run_scan(self, tickers: list[str]) -> list[dict]:
        """
        Run long-term scan on given tickers.

        Args:
            tickers: List of tickers to scan

        Returns:
            List of opportunity dictionaries, sorted by score
        """
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

        logger.info(f"Found {len(opportunities)} long-term opportunities")

        return opportunities

    def _score_ticker(self, ticker: str) -> dict | None:
        """
        Calculate composite score for a ticker.

        Returns None if insufficient data.
        """
        # Get price data
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

        # Calculate individual scores
        valuation_score = self._calculate_valuation_score(ticker, fundamentals)
        technical_score = self._calculate_technical_score(price_data)
        dividend_score = self._calculate_dividend_score(fundamentals)
        quality_score = self._calculate_quality_score(fundamentals)

        # Weighted composite
        composite_score = (
            valuation_score * 0.30 +
            technical_score * 0.30 +
            dividend_score * 0.15 +
            quality_score * 0.25
        )

        # Calculate supporting metrics
        current_price = price_data["close"].iloc[-1]
        fifty_two_week_low = price_data["low"].min()
        fifty_two_week_high = price_data["high"].max()

        price_vs_52w_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100

        # P/E percentile calculation
        pe_ratio = fundamentals[2] if len(fundamentals) > 2 else None  # pe_ratio column
        pe_percentile = self._calculate_pe_percentile(ticker, pe_ratio) if pe_ratio else 50

        # Generate reasoning
        reasoning = self._generate_reasoning(
            ticker, valuation_score, technical_score,
            dividend_score, quality_score, pe_percentile, price_vs_52w_low
        )

        return {
            "ticker": ticker,
            "scan_date": date.today(),
            "composite_score": composite_score,
            "valuation_score": valuation_score,
            "technical_score": technical_score,
            "dividend_score": dividend_score,
            "quality_score": quality_score,
            "pe_percentile": pe_percentile,
            "price_vs_52w_low_pct": price_vs_52w_low,
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

    def _calculate_pe_percentile(self, ticker: str, current_pe: float) -> float:
        """Calculate where current P/E falls in historical distribution."""
        # Get historical P/E data
        historical = self.db.fetchdf("""
            SELECT pe_ratio FROM fundamentals
            WHERE ticker = ? AND pe_ratio IS NOT NULL AND pe_ratio > 0
            ORDER BY date
        """, (ticker,))

        if historical.empty or len(historical) < 5:
            return 50  # Default to middle if no history

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
        """Store opportunities in database."""
        for opp in opportunities:
            try:
                self.db.execute("""
                    INSERT INTO long_term_watchlist (
                        ticker, scan_date, composite_score, valuation_score,
                        technical_score, dividend_score, quality_score,
                        pe_percentile, price_vs_52w_low_pct, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, scan_date) DO UPDATE SET
                        composite_score = excluded.composite_score,
                        valuation_score = excluded.valuation_score,
                        technical_score = excluded.technical_score,
                        dividend_score = excluded.dividend_score,
                        quality_score = excluded.quality_score,
                        pe_percentile = excluded.pe_percentile,
                        price_vs_52w_low_pct = excluded.price_vs_52w_low_pct,
                        reasoning = excluded.reasoning
                """, (
                    opp["ticker"],
                    opp["scan_date"],
                    opp["composite_score"],
                    opp["valuation_score"],
                    opp["technical_score"],
                    opp.get("dividend_score", 50),
                    opp["quality_score"],
                    opp["pe_percentile"],
                    opp["price_vs_52w_low_pct"],
                    opp["reasoning"]
                ))
            except Exception as e:
                logger.error(f"Error storing opportunity for {opp['ticker']}: {e}")

    def get_watchlist(self, days_back: int = 7) -> pd.DataFrame:
        """Get recent watchlist entries."""
        return self.db.fetchdf(f"""
            SELECT * FROM long_term_watchlist
            WHERE scan_date >= CURRENT_DATE - INTERVAL '{days_back} days'
            ORDER BY composite_score DESC
        """)

    def send_digest(self, opportunities: list[dict]) -> bool:
        """Send long-term opportunities digest email."""
        if not opportunities:
            return False

        return self.alert_manager.send_long_term_digest(opportunities)
