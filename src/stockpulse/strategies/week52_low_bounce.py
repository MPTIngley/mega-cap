"""52-Week Low Bounce Strategy.

Identifies quality large-cap stocks trading near 52-week lows that
show signs of bottoming and reversal. Combines value with technical
bounce signals and fundamental quality filters to avoid value traps.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np
import yfinance as yf

from .base import BaseStrategy, Signal, SignalDirection
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)


class Week52LowBounceStrategy(BaseStrategy):
    """
    52-week low bounce strategy for quality stocks.

    Entry: Stock near 52-week low with reversal signals
    Exit: Target based on mean reversion or stop loss

    Best for:
    - Large-cap stocks with strong fundamentals
    - Market-wide selloffs (throwing out babies with bathwater)
    - Sector rotation plays
    """

    # Cache for fundamental data to avoid repeated API calls
    _fundamental_cache: dict[str, dict] = {}
    _cache_timestamp: datetime | None = None
    CACHE_DURATION_HOURS = 24  # Refresh fundamentals daily

    def __init__(self, config: dict[str, Any]):
        """Initialize 52-week low bounce strategy."""
        super().__init__(config)
        self.low_threshold_pct = config.get("low_threshold_pct", 10.0)  # Within 10% of 52w low
        self.bounce_threshold_pct = config.get("bounce_threshold_pct", 2.0)  # Min bounce from low
        self.volume_surge = config.get("volume_surge", 1.3)  # Volume confirmation

        # Fundamental quality filters (avoid value traps)
        self.require_profitable = config.get("require_profitable", True)
        self.max_pe_ratio = config.get("max_pe_ratio", 50.0)  # Reject if P/E > 50
        self.max_debt_to_equity = config.get("max_debt_to_equity", 2.0)  # Reject if D/E > 2
        self.min_market_cap_billions = config.get("min_market_cap_billions", 10.0)  # $10B+ only

    @property
    def name(self) -> str:
        return "week52_low_bounce"

    def _get_fundamental_data(self, ticker: str) -> dict:
        """
        Get fundamental data for a ticker with caching.

        Returns dict with: profitable, pe_ratio, debt_to_equity, market_cap_b, quality_score
        """
        # Check cache freshness
        now = datetime.now()
        if (self._cache_timestamp is None or
            (now - self._cache_timestamp).total_seconds() > self.CACHE_DURATION_HOURS * 3600):
            Week52LowBounceStrategy._fundamental_cache = {}
            Week52LowBounceStrategy._cache_timestamp = now

        # Return cached data if available
        if ticker in self._fundamental_cache:
            return self._fundamental_cache[ticker]

        # Fetch fresh data
        fundamentals = {
            "profitable": False,
            "pe_ratio": None,
            "debt_to_equity": None,
            "market_cap_b": 0,
            "quality_score": 0,  # 0-100
            "fetch_failed": False
        }

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Profitability
            net_income = info.get("netIncomeToCommon", 0) or 0
            fundamentals["profitable"] = net_income > 0

            # P/E ratio
            pe = info.get("trailingPE") or info.get("forwardPE")
            fundamentals["pe_ratio"] = pe

            # Debt to Equity
            fundamentals["debt_to_equity"] = info.get("debtToEquity")

            # Market cap in billions
            market_cap = info.get("marketCap", 0) or 0
            fundamentals["market_cap_b"] = market_cap / 1e9

            # Calculate quality score (0-100)
            score = 50  # Start neutral

            # Profitability bonus
            if fundamentals["profitable"]:
                score += 20
            else:
                score -= 30  # Big penalty for unprofitable

            # P/E assessment
            if pe is not None:
                if 5 < pe < 20:
                    score += 15  # Value territory
                elif 20 <= pe < 35:
                    score += 5   # Reasonable
                elif pe > 50:
                    score -= 20  # Expensive even at lows

            # Debt assessment
            de = fundamentals["debt_to_equity"]
            if de is not None:
                if de < 0.5:
                    score += 15  # Low debt
                elif de < 1.0:
                    score += 5   # Moderate
                elif de > 2.0:
                    score -= 15  # High debt risk

            # Market cap assessment (mega cap = more reliable)
            if fundamentals["market_cap_b"] > 100:
                score += 10  # Mega cap
            elif fundamentals["market_cap_b"] > 50:
                score += 5   # Large cap

            fundamentals["quality_score"] = max(0, min(100, score))

        except Exception as e:
            logger.debug(f"Failed to fetch fundamentals for {ticker}: {e}")
            fundamentals["fetch_failed"] = True
            fundamentals["quality_score"] = 40  # Neutral if can't fetch

        # Cache result
        Week52LowBounceStrategy._fundamental_cache[ticker] = fundamentals
        return fundamentals

    def _passes_fundamental_filters(self, ticker: str) -> tuple[bool, str, dict]:
        """
        Check if ticker passes fundamental quality filters.

        Returns: (passes, reason, fundamentals_dict)
        """
        fundamentals = self._get_fundamental_data(ticker)

        # If fetch failed, be conservative and allow with penalty
        if fundamentals["fetch_failed"]:
            return True, "fundamentals_unavailable", fundamentals

        # Check profitability
        if self.require_profitable and not fundamentals["profitable"]:
            return False, "unprofitable", fundamentals

        # Check P/E ratio
        pe = fundamentals["pe_ratio"]
        if pe is not None and pe > self.max_pe_ratio:
            return False, f"PE_too_high_{pe:.1f}", fundamentals

        # Check debt levels
        de = fundamentals["debt_to_equity"]
        if de is not None and de > self.max_debt_to_equity:
            return False, f"debt_too_high_{de:.1f}", fundamentals

        # Check market cap
        if fundamentals["market_cap_b"] < self.min_market_cap_billions:
            return False, f"market_cap_too_small_{fundamentals['market_cap_b']:.1f}B", fundamentals

        return True, "passed", fundamentals

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 52-week low related indicators."""
        df = df.copy()

        # 52-week (252 trading days) high and low
        df["week52_high"] = df["high"].rolling(252, min_periods=60).max()
        df["week52_low"] = df["low"].rolling(252, min_periods=60).min()

        # Distance from 52-week low/high
        df["pct_from_52w_low"] = (df["close"] - df["week52_low"]) / df["week52_low"] * 100
        df["pct_from_52w_high"] = (df["week52_high"] - df["close"]) / df["week52_high"] * 100

        # Recent low (20 days)
        df["recent_low"] = df["low"].rolling(20).min()
        df["bounce_from_recent_low"] = (df["close"] - df["recent_low"]) / df["recent_low"] * 100

        # Is at or near 52-week low?
        df["near_52w_low"] = df["pct_from_52w_low"] <= self.low_threshold_pct

        # Volume indicators
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)
        df["volume_surge"] = df["volume_ratio"] > self.volume_surge

        # RSI for oversold confirmation
        df["rsi"] = self.calculate_rsi(df["close"], 14)
        df["oversold"] = df["rsi"] < 35

        # Price momentum (is it bouncing?)
        df["price_change_3d"] = df["close"].pct_change(3) * 100
        df["price_change_5d"] = df["close"].pct_change(5) * 100
        df["bouncing"] = (df["price_change_3d"] > 0) & (df["close"] > df["close"].shift(1))

        # Support level (recent low acting as support)
        df["above_recent_low"] = df["close"] > df["recent_low"]

        # ATR for volatility
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Moving averages for trend context
        df["sma_50"] = self.calculate_sma(df["close"], 50)
        df["sma_200"] = self.calculate_sma(df["close"], 200)

        # Drawdown from peak
        df["drawdown"] = (df["close"] - df["week52_high"]) / df["week52_high"] * 100

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate 52-week low bounce signals with fundamental quality filters."""
        if len(df) < 252:  # Need full year of data
            return []

        df = self.calculate_indicators(df)
        signals = []

        # Get the most recent data point
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        current_price = latest["close"]

        # Check if near 52-week low
        if not latest["near_52w_low"]:
            return []

        # === FUNDAMENTAL QUALITY FILTER (avoid value traps) ===
        passes_fundamentals, reason, fundamentals = self._passes_fundamental_filters(ticker)
        if not passes_fundamentals:
            logger.debug(f"{ticker}: Rejected by fundamental filter - {reason}")
            return []

        # Check for bounce signals
        is_bouncing = latest["bouncing"]
        has_volume = latest["volume_surge"]
        is_oversold = latest["oversold"]
        bounce_size = latest["bounce_from_recent_low"]

        # Require multiple confirmation signals
        confirmations = sum([
            is_bouncing,
            has_volume,
            is_oversold or latest["rsi"] < 40,
            bounce_size >= self.bounce_threshold_pct,
        ])

        if confirmations >= 2:
            confidence = self._calculate_confidence(df, latest, confirmations, fundamentals)

            if confidence >= self.min_confidence:
                week52_low = latest["week52_low"]
                week52_high = latest["week52_high"]

                # Entry at current price
                entry_price = current_price

                # Target: 50% retracement of the drawdown
                drawdown_range = week52_high - week52_low
                target_price = current_price + (drawdown_range * 0.3)  # 30% of range

                # Stop: Below recent low
                stop_price = latest["recent_low"] * 0.98

                # Build notes with fundamental context
                notes = f"Near 52w low ({latest['pct_from_52w_low']:.1f}%), RSI={latest['rsi']:.0f}"
                if fundamentals.get("pe_ratio"):
                    notes += f", PE={fundamentals['pe_ratio']:.1f}"
                if fundamentals.get("quality_score", 0) >= 70:
                    notes += ", HIGH_QUALITY"

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_price=stop_price,
                    notes=notes
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        confirmations: int,
        fundamentals: dict | None = None
    ) -> float:
        """Calculate signal confidence with fundamental quality adjustments."""
        # Base confidence from confirmations
        base = 55 + (confirmations * 5)  # 60-75 base

        factors = {}

        # RSI level - more oversold = higher confidence
        rsi = latest.get("rsi", 50)
        if rsi < 25:
            factors["rsi"] = 1.15
        elif rsi < 30:
            factors["rsi"] = 1.10
        elif rsi < 35:
            factors["rsi"] = 1.05

        # Volume on bounce
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            factors["volume"] = 1.15
        elif volume_ratio > 1.5:
            factors["volume"] = 1.08

        # Drawdown magnitude - bigger drawdown = more upside potential
        drawdown = abs(latest.get("drawdown", 0))
        if drawdown > 40:
            factors["drawdown"] = 1.1
        elif drawdown > 30:
            factors["drawdown"] = 1.05

        # Bounce strength
        bounce = latest.get("bounce_from_recent_low", 0)
        if bounce > 5:
            factors["bounce"] = 1.1
        elif bounce > 3:
            factors["bounce"] = 1.05

        # Long-term trend context (be careful in strong downtrends)
        if latest["close"] < latest.get("sma_200", latest["close"]):
            factors["trend"] = 0.92  # Penalty for below 200 SMA

        # === FUNDAMENTAL QUALITY ADJUSTMENTS ===
        if fundamentals:
            quality_score = fundamentals.get("quality_score", 50)

            # High quality stocks get confidence boost
            if quality_score >= 80:
                factors["fundamentals"] = 1.15
            elif quality_score >= 70:
                factors["fundamentals"] = 1.10
            elif quality_score >= 60:
                factors["fundamentals"] = 1.05
            elif quality_score < 40:
                factors["fundamentals"] = 0.90  # Low quality penalty

            # Profitable company bonus
            if fundamentals.get("profitable"):
                factors["profitable"] = 1.05

            # Value P/E bonus (if P/E is reasonable)
            pe = fundamentals.get("pe_ratio")
            if pe is not None and 5 < pe < 20:
                factors["value_pe"] = 1.08

        return self.calculate_confidence(base, factors)
