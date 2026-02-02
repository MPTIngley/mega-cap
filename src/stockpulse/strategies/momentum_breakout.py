"""Momentum Breakout Strategy.

Identifies breakouts above recent highs with volume confirmation.
Trend-following approach for capturing momentum moves.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy.

    Entry: Price breaks above recent high with volume confirmation
    Exit: Trailing stop or target hit

    Risk considerations:
    - Late entries (by definition, entering after move has started)
    - False breakouts common
    - Needs strong volume confirmation
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Momentum Breakout strategy."""
        super().__init__(config)
        self.lookback_days = config.get("lookback_days", 20)
        self.breakout_threshold = config.get("breakout_threshold", 0.02)  # 2%
        self.volume_confirmation = config.get("volume_confirmation", 1.3)

    @property
    def name(self) -> str:
        return "momentum_breakout"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout and momentum indicators."""
        df = df.copy()

        # Recent high/low for breakout reference
        df["recent_high"] = df["high"].rolling(self.lookback_days).max()
        df["recent_low"] = df["low"].rolling(self.lookback_days).min()

        # Previous day's recent high (for breakout detection)
        df["prev_recent_high"] = df["recent_high"].shift(1)
        df["prev_recent_low"] = df["recent_low"].shift(1)

        # Breakout detection
        df["breakout_up"] = df["close"] > df["prev_recent_high"] * (1 + self.breakout_threshold)
        df["breakout_down"] = df["close"] < df["prev_recent_low"] * (1 - self.breakout_threshold)

        # Volume
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)
        df["volume_breakout"] = df["volume_ratio"] > self.volume_confirmation

        # Momentum indicators
        df["momentum_10d"] = df["close"].pct_change(10) * 100
        df["momentum_20d"] = df["close"].pct_change(20) * 100

        # Rate of change
        df["roc_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100

        # Trend strength (ADX-like)
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        df["trend_range"] = (df["recent_high"] - df["recent_low"]) / df["close"] * 100

        # Moving averages for trend context
        df["ema_10"] = self.calculate_ema(df["close"], 10)
        df["ema_20"] = self.calculate_ema(df["close"], 20)
        df["ema_50"] = self.calculate_ema(df["close"], 50)

        df["ema_alignment"] = (
            (df["close"] > df["ema_10"]) &
            (df["ema_10"] > df["ema_20"]) &
            (df["ema_20"] > df["ema_50"])
        )

        # Gap detection
        df["gap_up"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
        df["significant_gap"] = df["gap_up"].abs() > 1.0  # > 1% gap

        # New high detection (52-week)
        df["high_52w"] = df["high"].rolling(252, min_periods=60).max()
        df["at_52w_high"] = df["close"] >= df["high_52w"] * 0.98

        # Consolidation before breakout (tight range)
        df["range_contraction"] = df["trend_range"] < df["trend_range"].rolling(20).mean()

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate momentum breakout signals."""
        if len(df) < 60:
            return []

        df = self.calculate_indicators(df)
        signals = []

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = latest["close"]

        # Skip if data is invalid
        if pd.isna(latest["recent_high"]) or pd.isna(latest["volume_ratio"]):
            return []

        # BULLISH BREAKOUT
        if latest.get("breakout_up", False) and latest.get("volume_breakout", False):
            # Confirm it's a fresh breakout (wasn't a breakout yesterday)
            if not prev.get("breakout_up", True):
                confidence = self._calculate_breakout_confidence(df, latest, "bullish")

                if confidence >= self.min_confidence:
                    entry, target, stop = self._calculate_breakout_levels(
                        df, latest, SignalDirection.BUY
                    )

                    notes = f"Breakout UP, vol={latest['volume_ratio']:.1f}x"
                    if latest.get("ema_alignment", False):
                        notes += ", EMA aligned"
                    if latest.get("at_52w_high", False):
                        notes += ", 52w high"

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

        # BEARISH BREAKOUT
        elif latest.get("breakout_down", False) and latest.get("volume_breakout", False):
            if not prev.get("breakout_down", True):
                confidence = self._calculate_breakout_confidence(df, latest, "bearish")

                if confidence >= self.min_confidence:
                    entry, target, stop = self._calculate_breakout_levels(
                        df, latest, SignalDirection.SELL
                    )

                    signal = Signal(
                        ticker=ticker,
                        strategy=self.name,
                        direction=SignalDirection.SELL,
                        confidence=confidence,
                        entry_price=entry,
                        target_price=target,
                        stop_price=stop,
                        notes=f"Breakout DOWN, vol={latest['volume_ratio']:.1f}x"
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
        base = 60

        factors = {}

        # Volume (critical for breakouts)
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.5:
            factors["volume"] = 1.2
        elif volume_ratio > 2.0:
            factors["volume"] = 1.15
        elif volume_ratio > 1.5:
            factors["volume"] = 1.1
        elif volume_ratio < 1.2:
            factors["volume"] = 0.85  # Weak volume = suspicious

        # EMA alignment (trend confirmation)
        if direction == "bullish" and latest.get("ema_alignment", False):
            factors["trend"] = 1.15
        elif direction == "bearish" and not latest.get("ema_alignment", True):
            factors["trend"] = 1.1

        # Consolidation before breakout (tighter = better)
        if latest.get("range_contraction", False):
            factors["consolidation"] = 1.1

        # 52-week high breakout (stronger signal)
        if direction == "bullish" and latest.get("at_52w_high", False):
            factors["52w"] = 1.1

        # Gap confirmation
        gap = latest.get("gap_up", 0)
        if direction == "bullish" and gap > 0.5:
            factors["gap"] = 1.05
        elif direction == "bearish" and gap < -0.5:
            factors["gap"] = 1.05

        # Recent momentum (building momentum)
        momentum = latest.get("momentum_10d", 0)
        if direction == "bullish" and momentum > 0:
            factors["momentum"] = 1.05
        elif direction == "bearish" and momentum < 0:
            factors["momentum"] = 1.05

        return self.calculate_confidence(base, factors)

    def _calculate_breakout_levels(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        direction: SignalDirection
    ) -> tuple[float, float, float]:
        """Calculate entry, target, and stop for breakout trade."""
        current_price = latest["close"]
        atr = latest.get("atr", current_price * 0.02)
        trend_range = latest.get("trend_range", 5)  # Percentage

        entry = current_price

        if direction == SignalDirection.BUY:
            # Target: project the breakout range
            breakout_magnitude = current_price - latest["prev_recent_low"]
            target = current_price + (breakout_magnitude * 0.618)  # Fibonacci projection

            # Cap at max profit
            max_target = entry * (1 + self.take_profit_pct / 100)
            target = min(target, max_target)

            # Stop: below the breakout level
            stop = latest["prev_recent_high"] * 0.98  # Just below the breakout level

            # Or use ATR-based stop
            stop_atr = entry - (2 * atr)
            stop = max(stop, stop_atr)

            # Ensure minimum stop
            min_stop = entry * (1 - self.stop_loss_pct / 100)
            stop = max(stop, min_stop)

        else:
            breakout_magnitude = latest["prev_recent_high"] - current_price
            target = current_price - (breakout_magnitude * 0.618)

            min_target = entry * (1 - self.take_profit_pct / 100)
            target = max(target, min_target)

            stop = latest["prev_recent_low"] * 1.02

            stop_atr = entry + (2 * atr)
            stop = min(stop, stop_atr)

            max_stop = entry * (1 + self.stop_loss_pct / 100)
            stop = min(stop, max_stop)

        return entry, target, stop
