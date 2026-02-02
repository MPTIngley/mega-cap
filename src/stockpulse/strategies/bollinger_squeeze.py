"""Bollinger Band Squeeze Breakout Strategy.

Identifies periods of low volatility (squeeze) and enters on breakout.
Squeeze occurs when Bollinger Bands contract significantly.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze Breakout Strategy.

    Entry: After a squeeze (low bandwidth), enter on breakout direction
    Exit: Target based on historical volatility expansion or stop loss

    Risk considerations:
    - False breakouts are common
    - Need volume confirmation
    - Squeeze can persist longer than expected
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Bollinger strategy."""
        super().__init__(config)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.squeeze_threshold = config.get("squeeze_threshold", 0.05)  # 5th percentile

    @property
    def name(self) -> str:
        return "bollinger_squeeze"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and squeeze indicators."""
        df = df.copy()

        # Bollinger Bands
        middle, upper, lower = self.calculate_bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        df["bb_middle"] = middle
        df["bb_upper"] = upper
        df["bb_lower"] = lower

        # Bandwidth (volatility measure)
        df["bb_bandwidth"] = (upper - lower) / middle

        # Bandwidth percentile (rolling)
        df["bb_bandwidth_pct"] = df["bb_bandwidth"].rolling(100).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50
        )

        # Squeeze detection
        df["is_squeeze"] = df["bb_bandwidth_pct"] < (self.squeeze_threshold * 100)

        # Previous squeeze state
        df["was_squeeze"] = df["is_squeeze"].shift(1)

        # Breakout detection
        df["above_upper"] = df["close"] > df["bb_upper"]
        df["below_lower"] = df["close"] < df["bb_lower"]

        # %B indicator (where price is within bands)
        df["percent_b"] = (df["close"] - lower) / (upper - lower)

        # Volume
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)

        # Momentum
        df["momentum_5d"] = df["close"].pct_change(5)

        # ATR
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)

        # Keltner Channel (for additional squeeze confirmation)
        ema_20 = self.calculate_ema(df["close"], 20)
        atr_10 = self.calculate_atr(df["high"], df["low"], df["close"], 10)
        df["kc_upper"] = ema_20 + (atr_10 * 1.5)
        df["kc_lower"] = ema_20 - (atr_10 * 1.5)

        # TTM-style squeeze: BB inside KC
        df["ttm_squeeze"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate squeeze breakout signals."""
        if len(df) < 120:  # Need enough history for percentile calculation
            return []

        df = self.calculate_indicators(df)
        signals = []

        # Look at recent bars for squeeze release
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 2 else prev

        current_price = latest["close"]

        # Check for squeeze release with breakout
        was_in_squeeze = prev.get("is_squeeze", False) or prev.get("ttm_squeeze", False)
        is_breaking_out = not latest.get("is_squeeze", True)

        if not was_in_squeeze:
            return []  # No squeeze, no signal

        # Skip if data is invalid
        if pd.isna(latest["bb_upper"]) or pd.isna(latest["bb_lower"]):
            return []

        # BULLISH BREAKOUT
        if latest["close"] > latest["bb_upper"] and prev["close"] <= prev["bb_upper"]:
            confidence = self._calculate_breakout_confidence(df, latest, "bullish")

            if confidence >= self.min_confidence:
                entry, target, stop = self._calculate_breakout_levels(
                    latest, SignalDirection.BUY
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=f"Squeeze breakout UP, bandwidth_pct={latest['bb_bandwidth_pct']:.1f}"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        # BEARISH BREAKOUT
        elif latest["close"] < latest["bb_lower"] and prev["close"] >= prev["bb_lower"]:
            confidence = self._calculate_breakout_confidence(df, latest, "bearish")

            if confidence >= self.min_confidence:
                entry, target, stop = self._calculate_breakout_levels(
                    latest, SignalDirection.SELL
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=f"Squeeze breakout DOWN, bandwidth_pct={latest['bb_bandwidth_pct']:.1f}"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_breakout_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        direction: str
    ) -> float:
        """Calculate confidence for breakout signal."""
        # Base confidence from squeeze intensity
        bandwidth_pct = latest.get("bb_bandwidth_pct", 50)
        if bandwidth_pct < 5:
            base = 75  # Extreme squeeze
        elif bandwidth_pct < 10:
            base = 70
        elif bandwidth_pct < 20:
            base = 65
        else:
            base = 55  # Mild squeeze

        factors = {}

        # Volume confirmation (crucial for breakouts)
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            factors["volume"] = 1.15
        elif volume_ratio > 1.5:
            factors["volume"] = 1.1
        elif volume_ratio > 1.2:
            factors["volume"] = 1.05
        elif volume_ratio < 0.8:
            factors["volume"] = 0.85  # Low volume breakout = suspicious

        # Momentum alignment
        momentum = latest.get("momentum_5d", 0)
        if direction == "bullish" and momentum > 0:
            factors["momentum"] = 1.05
        elif direction == "bearish" and momentum < 0:
            factors["momentum"] = 1.05
        else:
            factors["momentum"] = 0.95  # Counter-momentum

        # TTM squeeze confirmation
        if latest.get("ttm_squeeze", False) or df.iloc[-2].get("ttm_squeeze", False):
            factors["ttm"] = 1.1

        return self.calculate_confidence(base, factors)

    def _calculate_breakout_levels(
        self,
        latest: pd.Series,
        direction: SignalDirection
    ) -> tuple[float, float, float]:
        """Calculate entry, target, and stop for breakout."""
        current_price = latest["close"]
        atr = latest.get("atr", current_price * 0.02)
        bandwidth = latest.get("bb_bandwidth", 0.04)

        entry = current_price

        if direction == SignalDirection.BUY:
            # Target: expect bandwidth to expand 2-3x
            expected_expansion = bandwidth * 2.5
            target = entry * (1 + expected_expansion / 2)

            # Stop: below middle band or 2 ATR
            stop_bb = latest["bb_middle"]
            stop_atr = entry - (2 * atr)
            stop = max(stop_bb, stop_atr)

            # Ensure minimum stop distance
            min_stop = entry * (1 - self.stop_loss_pct / 100)
            stop = max(stop, min_stop)

        else:
            expected_expansion = bandwidth * 2.5
            target = entry * (1 - expected_expansion / 2)

            stop_bb = latest["bb_middle"]
            stop_atr = entry + (2 * atr)
            stop = min(stop_bb, stop_atr)

            max_stop = entry * (1 + self.stop_loss_pct / 100)
            stop = min(stop, max_stop)

        return entry, target, stop
