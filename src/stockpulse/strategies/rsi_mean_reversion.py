"""RSI Mean Reversion Strategy.

Buy oversold conditions (RSI < 30), sell overbought (RSI > 70).
Works best in range-bound markets on large-cap stocks.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class RSIMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on RSI oversold/overbought conditions.

    Entry: RSI crosses below oversold threshold (buy) or above overbought (sell)
    Exit: RSI returns to neutral zone or target/stop hit

    Risk considerations:
    - False signals in trending markets
    - Need confirmation from price action
    - Better performance in range-bound conditions
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize RSI strategy."""
        super().__init__(config)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and supporting indicators."""
        df = df.copy()

        # RSI
        df["rsi"] = self.calculate_rsi(df["close"], self.rsi_period)

        # RSI slope for momentum confirmation
        df["rsi_slope"] = df["rsi"].diff(3)

        # Price momentum
        df["price_change_5d"] = df["close"].pct_change(5) * 100

        # Volume confirmation
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)

        # ATR for volatility context
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # 50-day MA for trend context
        df["sma_50"] = self.calculate_sma(df["close"], 50)
        df["above_sma_50"] = df["close"] > df["sma_50"]

        # Support/resistance zones (recent lows/highs)
        df["recent_low"] = df["low"].rolling(20).min()
        df["recent_high"] = df["high"].rolling(20).max()

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate RSI-based mean reversion signals."""
        if len(df) < 60:  # Need enough history
            return []

        df = self.calculate_indicators(df)
        signals = []

        # Get the most recent data point
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = latest["close"]
        rsi = latest["rsi"]
        rsi_prev = prev["rsi"]

        # Skip if data is invalid
        if pd.isna(rsi) or pd.isna(rsi_prev):
            return []

        # BUY SIGNAL: RSI crosses below oversold and starts recovering
        if rsi < self.rsi_oversold and rsi > rsi_prev:
            # RSI was falling and now starting to rise (potential bottom)
            confidence = self._calculate_buy_confidence(df, latest, rsi)

            if confidence >= self.min_confidence:
                entry, target, stop = self.calculate_entry_exit_prices(
                    current_price, SignalDirection.BUY
                )

                # Adjust stop to recent swing low if tighter
                recent_low = latest["recent_low"]
                if recent_low < stop and recent_low > current_price * 0.9:
                    stop = recent_low * 0.99  # Just below recent low

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=f"RSI={rsi:.1f}, oversold bounce"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        # SELL SIGNAL (for existing longs): RSI crosses above overbought
        # Note: We don't short in this strategy, just identify exit points
        elif rsi > self.rsi_overbought and rsi < rsi_prev:
            # RSI was rising and now falling (potential top)
            confidence = self._calculate_sell_confidence(df, latest, rsi)

            if confidence >= self.min_confidence:
                entry, target, stop = self.calculate_entry_exit_prices(
                    current_price, SignalDirection.SELL
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=f"RSI={rsi:.1f}, overbought reversal"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_buy_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        rsi: float
    ) -> float:
        """Calculate confidence for buy signal."""
        # Base confidence: how oversold (lower RSI = higher base confidence)
        if rsi < 20:
            base = 75
        elif rsi < 25:
            base = 70
        else:
            base = 65

        factors = {}

        # Volume confirmation (higher volume on oversold = more conviction)
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            factors["volume"] = 1.1
        elif volume_ratio > 1.2:
            factors["volume"] = 1.05
        elif volume_ratio < 0.7:
            factors["volume"] = 0.9

        # Trend context (buying in uptrend is safer)
        if latest.get("above_sma_50", False):
            factors["trend"] = 1.1
        else:
            factors["trend"] = 0.95  # Slight penalty for fighting trend

        # Price near support (near recent low = stronger support)
        recent_low = latest.get("recent_low", 0)
        if recent_low > 0:
            distance_to_low = (latest["close"] - recent_low) / latest["close"]
            if distance_to_low < 0.02:  # Within 2% of recent low
                factors["support"] = 1.1

        # Volatility penalty (very high volatility = less reliable)
        atr_pct = latest.get("atr_pct", 2.0)
        if atr_pct > 4.0:
            factors["volatility"] = 0.9
        elif atr_pct < 1.5:
            factors["volatility"] = 1.05

        return self.calculate_confidence(base, factors)

    def _calculate_sell_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        rsi: float
    ) -> float:
        """Calculate confidence for sell signal."""
        # Base confidence: how overbought
        if rsi > 80:
            base = 75
        elif rsi > 75:
            base = 70
        else:
            base = 65

        factors = {}

        # Volume on overbought condition
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            factors["volume"] = 1.1

        # Near resistance
        recent_high = latest.get("recent_high", 0)
        if recent_high > 0:
            distance_to_high = (recent_high - latest["close"]) / latest["close"]
            if distance_to_high < 0.02:
                factors["resistance"] = 1.1

        return self.calculate_confidence(base, factors)
