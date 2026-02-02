"""MACD Crossover with Volume Confirmation Strategy.

Classic trend-following strategy using MACD crossovers,
with volume filter to reduce false signals.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class MACDVolumeStrategy(BaseStrategy):
    """
    MACD Crossover with Volume Confirmation.

    Entry: MACD crosses signal line with above-average volume
    Exit: Opposite crossover or target/stop hit

    Risk considerations:
    - Lagging indicator (late entries)
    - Whipsaws in choppy markets
    - Volume filter helps but doesn't eliminate false signals
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize MACD strategy."""
        super().__init__(config)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        self.volume_threshold = config.get("volume_threshold", 1.5)

    @property
    def name(self) -> str:
        return "macd_volume"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and volume indicators."""
        df = df.copy()

        # MACD
        macd, signal, histogram = self.calculate_macd(
            df["close"],
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_histogram"] = histogram

        # MACD crossover detection
        df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))

        # MACD slope
        df["macd_slope"] = df["macd"].diff(3)
        df["histogram_slope"] = df["macd_histogram"].diff(3)

        # Volume
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)

        # Trend filters
        df["sma_50"] = self.calculate_sma(df["close"], 50)
        df["sma_200"] = self.calculate_sma(df["close"], 200)
        df["above_sma_50"] = df["close"] > df["sma_50"]
        df["above_sma_200"] = df["close"] > df["sma_200"]
        df["uptrend"] = df["sma_50"] > df["sma_200"]

        # Price momentum
        df["price_change_10d"] = df["close"].pct_change(10) * 100
        df["price_change_20d"] = df["close"].pct_change(20) * 100

        # ATR
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)

        # Zero line position (MACD above/below zero)
        df["macd_above_zero"] = df["macd"] > 0

        # Histogram divergence
        df["price_higher_high"] = df["close"] > df["close"].rolling(10).max().shift(1)
        df["macd_higher_high"] = df["macd"] > df["macd"].rolling(10).max().shift(1)

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate MACD crossover signals with volume confirmation."""
        if len(df) < 50:
            return []

        df = self.calculate_indicators(df)
        signals = []

        latest = df.iloc[-1]
        current_price = latest["close"]
        volume_ratio = latest.get("volume_ratio", 1.0)

        # Skip if insufficient volume
        if volume_ratio < self.volume_threshold:
            return []

        # Skip if data is invalid
        if pd.isna(latest["macd"]) or pd.isna(latest["macd_signal"]):
            return []

        # BULLISH CROSSOVER
        if latest.get("macd_cross_up", False):
            confidence = self._calculate_bullish_confidence(df, latest)

            if confidence >= self.min_confidence:
                entry, target, stop = self._calculate_trend_levels(
                    df, latest, SignalDirection.BUY
                )

                notes = f"MACD cross up, vol={volume_ratio:.1f}x"
                if latest.get("uptrend", False):
                    notes += ", uptrend"
                if latest.get("macd_above_zero", False):
                    notes += ", above zero"

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=notes
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        # BEARISH CROSSOVER
        elif latest.get("macd_cross_down", False):
            confidence = self._calculate_bearish_confidence(df, latest)

            if confidence >= self.min_confidence:
                entry, target, stop = self._calculate_trend_levels(
                    df, latest, SignalDirection.SELL
                )

                notes = f"MACD cross down, vol={volume_ratio:.1f}x"
                if not latest.get("uptrend", True):
                    notes += ", downtrend"
                if not latest.get("macd_above_zero", True):
                    notes += ", below zero"

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=notes
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_bullish_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series
    ) -> float:
        """Calculate confidence for bullish MACD crossover."""
        base = 60

        factors = {}

        # Volume strength
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.5:
            factors["volume"] = 1.15
        elif volume_ratio > 2.0:
            factors["volume"] = 1.1
        elif volume_ratio > 1.5:
            factors["volume"] = 1.05

        # Trend alignment
        if latest.get("uptrend", False):
            factors["trend"] = 1.1
        if latest.get("above_sma_50", False):
            factors["ma50"] = 1.05
        if latest.get("above_sma_200", False):
            factors["ma200"] = 1.05

        # Zero line position
        if latest.get("macd_above_zero", False):
            factors["zero_line"] = 1.05
        else:
            # Crossing up from below zero can be stronger
            macd = latest.get("macd", 0)
            if -0.5 < macd < 0.5:  # Near zero
                factors["zero_line"] = 1.1

        # Histogram momentum
        histogram_slope = latest.get("histogram_slope", 0)
        if histogram_slope > 0:
            factors["histogram"] = 1.05

        # Recent price action
        price_change = latest.get("price_change_10d", 0)
        if -5 < price_change < 0:
            # Slight pullback before signal is ideal
            factors["pullback"] = 1.05

        return self.calculate_confidence(base, factors)

    def _calculate_bearish_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series
    ) -> float:
        """Calculate confidence for bearish MACD crossover."""
        base = 60

        factors = {}

        # Volume strength
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.5:
            factors["volume"] = 1.15
        elif volume_ratio > 2.0:
            factors["volume"] = 1.1
        elif volume_ratio > 1.5:
            factors["volume"] = 1.05

        # Trend alignment (bearish in downtrend)
        if not latest.get("uptrend", True):
            factors["trend"] = 1.1
        if not latest.get("above_sma_50", True):
            factors["ma50"] = 1.05
        if not latest.get("above_sma_200", True):
            factors["ma200"] = 1.05

        # Zero line position
        if not latest.get("macd_above_zero", True):
            factors["zero_line"] = 1.05

        # Histogram momentum
        histogram_slope = latest.get("histogram_slope", 0)
        if histogram_slope < 0:
            factors["histogram"] = 1.05

        return self.calculate_confidence(base, factors)

    def _calculate_trend_levels(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        direction: SignalDirection
    ) -> tuple[float, float, float]:
        """Calculate entry, target, and stop for trend trade."""
        current_price = latest["close"]
        atr = latest.get("atr", current_price * 0.02)

        entry = current_price

        if direction == SignalDirection.BUY:
            # Target: use ATR-based target
            target = entry + (3 * atr)

            # Stop: below recent swing low or 2 ATR
            recent_low = df["low"].iloc[-10:].min()
            stop_swing = recent_low * 0.99
            stop_atr = entry - (2 * atr)
            stop = max(stop_swing, stop_atr)

            # Ensure minimum stop distance
            min_stop = entry * (1 - self.stop_loss_pct / 100)
            stop = max(stop, min_stop)

            # Cap target at max profit
            max_target = entry * (1 + self.take_profit_pct / 100)
            target = min(target, max_target)

        else:
            target = entry - (3 * atr)

            recent_high = df["high"].iloc[-10:].max()
            stop_swing = recent_high * 1.01
            stop_atr = entry + (2 * atr)
            stop = min(stop_swing, stop_atr)

            max_stop = entry * (1 + self.stop_loss_pct / 100)
            stop = min(stop, max_stop)

            min_target = entry * (1 - self.take_profit_pct / 100)
            target = max(target, min_target)

        return entry, target, stop
