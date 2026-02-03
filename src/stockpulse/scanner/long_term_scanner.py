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

        # Get yfinance ticker for additional data
        yf_ticker = self._get_yf_ticker(ticker)

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

        # Calculate supporting metrics
        current_price = price_data["close"].iloc[-1]
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
            "scan_date": date.today(),
            "composite_score": composite_score,
            "valuation_score": valuation_score,
            "technical_score": technical_score,
            "dividend_score": dividend_score,
            "quality_score": quality_score,
            "insider_score": insider_score,
            "fcf_yield_score": fcf_yield_score,
            "earnings_momentum_score": earnings_momentum_score,
            "peer_valuation_score": peer_valuation_score,
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
        return self.db.fetchdf("""
            SELECT * FROM long_term_watchlist
            WHERE scan_date >= date('now', ?)
            ORDER BY composite_score DESC
        """, (f'-{days_back} days',))

    def send_digest(self, opportunities: list[dict]) -> bool:
        """Send long-term opportunities digest email."""
        if not opportunities:
            return False

        return self.alert_manager.send_long_term_digest(opportunities)
