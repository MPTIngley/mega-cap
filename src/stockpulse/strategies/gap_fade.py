"""Gap Fade Strategy.

Trades overnight gaps that are likely to fill during the trading day.
Large gaps often represent overreaction and tend to fade back toward
the previous close.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class GapFadeStrategy(BaseStrategy):
    """
    Gap fade strategy - trade against overnight gaps.

    Entry: Gap up/down at open that exceeds threshold
    Exit: Gap fills (returns to previous close) or time-based exit

    Best for:
    - Large-cap stocks with liquid pre-market
    - Gaps driven by sentiment rather than fundamentals
    - Non-earnings gap days
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize gap fade strategy."""
        super().__init__(config)
        self.gap_threshold_pct = config.get("gap_threshold_pct", 1.5)  # Min gap size
        self.max_gap_pct = config.get("max_gap_pct", 5.0)  # Don't fade huge gaps
        self.volume_surge_threshold = config.get("volume_surge_threshold", 1.5)

    @property
    def name(self) -> str:
        return "gap_fade"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gap-related indicators."""
        df = df.copy()

        # Previous close
        df["prev_close"] = df["close"].shift(1)

        # Gap calculations
        df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"] * 100

        # Absolute gap
        df["gap_abs"] = abs(df["gap_pct"])

        # Gap direction
        df["gap_up"] = df["gap_pct"] > 0
        df["gap_down"] = df["gap_pct"] < 0

        # Intraday range
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"] * 100

        # Did gap fill? (price crossed previous close during the day)
        df["gap_filled"] = False
        df.loc[df["gap_up"] & (df["low"] <= df["prev_close"]), "gap_filled"] = True
        df.loc[df["gap_down"] & (df["high"] >= df["prev_close"]), "gap_filled"] = True

        # Volume relative to average
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)

        # ATR for position sizing
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Recent gap fill rate (last 20 gaps)
        df["recent_gap_fill_rate"] = df["gap_filled"].rolling(20).mean()

        # Trend context (SMA)
        df["sma_20"] = self.calculate_sma(df["close"], 20)
        df["above_sma"] = df["close"] > df["sma_20"]

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate gap fade signals."""
        if len(df) < 30:
            return []

        df = self.calculate_indicators(df)
        signals = []

        # Get the most recent data point
        latest = df.iloc[-1]
        gap_pct = latest["gap_pct"]
        gap_abs = latest["gap_abs"]

        # Skip if gap is too small or too large
        if gap_abs < self.gap_threshold_pct or gap_abs > self.max_gap_pct:
            return []

        current_price = latest["close"]
        prev_close = latest["prev_close"]

        # GAP UP - Fade by selling/shorting expectation of fill
        if gap_pct > self.gap_threshold_pct:
            confidence = self._calculate_gap_up_confidence(df, latest, gap_pct)

            if confidence >= self.min_confidence:
                # Target is previous close (gap fill)
                target_price = prev_close
                # Stop is above the high of gap day
                stop_price = latest["high"] * 1.01

                entry, _, _ = self.calculate_entry_exit_prices(
                    current_price, SignalDirection.SELL
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target_price,
                    stop_price=stop_price,
                    notes=f"Gap up {gap_pct:.1f}%, fade to fill"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        # GAP DOWN - Fade by buying expectation of fill
        elif gap_pct < -self.gap_threshold_pct:
            confidence = self._calculate_gap_down_confidence(df, latest, gap_pct)

            if confidence >= self.min_confidence:
                # Target is previous close (gap fill)
                target_price = prev_close
                # Stop is below the low of gap day
                stop_price = latest["low"] * 0.99

                entry, _, _ = self.calculate_entry_exit_prices(
                    current_price, SignalDirection.BUY
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target_price,
                    stop_price=stop_price,
                    notes=f"Gap down {gap_pct:.1f}%, fade to fill"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_gap_up_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        gap_pct: float
    ) -> float:
        """Calculate confidence for fading a gap up."""
        # Base confidence - moderate gaps more likely to fill
        if 1.5 <= gap_pct <= 2.5:
            base = 70
        elif 2.5 < gap_pct <= 3.5:
            base = 65
        else:
            base = 60

        factors = {}

        # Historical gap fill rate
        fill_rate = latest.get("recent_gap_fill_rate", 0.5)
        if fill_rate > 0.7:
            factors["history"] = 1.15
        elif fill_rate > 0.5:
            factors["history"] = 1.05

        # Volume - lower volume gaps more likely to fade
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio < 1.0:
            factors["volume"] = 1.1  # Low volume = easier to fade
        elif volume_ratio > 2.0:
            factors["volume"] = 0.85  # High volume = conviction gap

        # Trend context - fading against trend is riskier
        if not latest.get("above_sma", True):
            factors["trend"] = 1.1  # Gap up in downtrend more likely to fade
        else:
            factors["trend"] = 0.95

        return self.calculate_confidence(base, factors)

    def _calculate_gap_down_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        gap_pct: float
    ) -> float:
        """Calculate confidence for fading a gap down."""
        gap_pct = abs(gap_pct)

        # Base confidence
        if 1.5 <= gap_pct <= 2.5:
            base = 70
        elif 2.5 < gap_pct <= 3.5:
            base = 65
        else:
            base = 60

        factors = {}

        # Historical gap fill rate
        fill_rate = latest.get("recent_gap_fill_rate", 0.5)
        if fill_rate > 0.7:
            factors["history"] = 1.15
        elif fill_rate > 0.5:
            factors["history"] = 1.05

        # Volume
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio < 1.0:
            factors["volume"] = 1.1
        elif volume_ratio > 2.0:
            factors["volume"] = 0.85

        # Trend context - fading with trend is safer
        if latest.get("above_sma", True):
            factors["trend"] = 1.1  # Gap down in uptrend more likely to fill
        else:
            factors["trend"] = 0.95

        return self.calculate_confidence(base, factors)
