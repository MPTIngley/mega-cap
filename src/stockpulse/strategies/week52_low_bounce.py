"""52-Week Low Bounce Strategy.

Identifies quality large-cap stocks trading near 52-week lows that
show signs of bottoming and reversal. Combines value with technical
bounce signals.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


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

    def __init__(self, config: dict[str, Any]):
        """Initialize 52-week low bounce strategy."""
        super().__init__(config)
        self.low_threshold_pct = config.get("low_threshold_pct", 10.0)  # Within 10% of 52w low
        self.bounce_threshold_pct = config.get("bounce_threshold_pct", 2.0)  # Min bounce from low
        self.volume_surge = config.get("volume_surge", 1.3)  # Volume confirmation

    @property
    def name(self) -> str:
        return "week52_low_bounce"

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
        """Generate 52-week low bounce signals."""
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
            confidence = self._calculate_confidence(df, latest, confirmations)

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

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_price=stop_price,
                    notes=f"Near 52w low ({latest['pct_from_52w_low']:.1f}%), RSI={latest['rsi']:.0f}, bouncing"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        confirmations: int
    ) -> float:
        """Calculate signal confidence."""
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

        return self.calculate_confidence(base, factors)
