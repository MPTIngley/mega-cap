"""Z-Score Mean Reversion Strategy.

Statistical mean reversion based on price z-score.
Enters when price deviates significantly from recent mean.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class ZScoreMeanReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.

    Entry: When price z-score exceeds threshold (statistical extreme)
    Exit: When z-score returns to zero or target/stop hit

    Risk considerations:
    - Trends can persist longer than statistical models expect
    - Works better in stable, mean-reverting environments
    - Large-cap stocks tend to mean-revert more reliably
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Z-Score strategy."""
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 20)
        self.zscore_entry = config.get("zscore_entry", -2.0)
        self.zscore_exit = config.get("zscore_exit", 0.0)

    @property
    def name(self) -> str:
        return "zscore_mean_reversion"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate z-score and supporting indicators."""
        df = df.copy()

        # Price z-score
        rolling_mean = df["close"].rolling(self.lookback_period).mean()
        rolling_std = df["close"].rolling(self.lookback_period).std()
        df["zscore"] = (df["close"] - rolling_mean) / rolling_std

        # Z-score of z-score (for momentum)
        df["zscore_slope"] = df["zscore"].diff(3)

        # Bollinger position (similar concept, for confirmation)
        df["bb_zscore"] = (df["close"] - rolling_mean) / (rolling_std * 2)

        # Volume
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)

        # Trend context
        df["sma_50"] = self.calculate_sma(df["close"], 50)
        df["trend_strength"] = (df["close"] - df["sma_50"]) / df["sma_50"] * 100

        # ATR
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Historical z-score extremes (for context)
        df["zscore_min_60d"] = df["zscore"].rolling(60).min()
        df["zscore_max_60d"] = df["zscore"].rolling(60).max()

        # Mean reversion speed (how fast does it typically revert?)
        # Higher values = faster mean reversion
        df["reversion_speed"] = df["zscore"].diff().abs().rolling(20).mean()

        # Price relative to recent range
        df["price_percentile"] = df["close"].rolling(50).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50
        )

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate z-score mean reversion signals."""
        if len(df) < 70:
            return []

        df = self.calculate_indicators(df)
        signals = []

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = latest["close"]
        zscore = latest["zscore"]
        zscore_prev = prev["zscore"]

        # Skip if data is invalid
        if pd.isna(zscore) or pd.isna(zscore_prev):
            return []

        # BUY SIGNAL: Z-score below threshold and starting to recover
        if zscore < self.zscore_entry and zscore > zscore_prev:
            # Z-score was falling and now rising (potential bottom)
            confidence = self._calculate_buy_confidence(df, latest, zscore)

            if confidence >= self.min_confidence:
                entry, target, stop = self._calculate_reversion_levels(
                    df, latest, SignalDirection.BUY, zscore
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=f"Z-score={zscore:.2f}, extreme low"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        # SELL SIGNAL: Z-score above positive threshold and starting to fall
        elif zscore > abs(self.zscore_entry) and zscore < zscore_prev:
            confidence = self._calculate_sell_confidence(df, latest, zscore)

            if confidence >= self.min_confidence:
                entry, target, stop = self._calculate_reversion_levels(
                    df, latest, SignalDirection.SELL, zscore
                )

                signal = Signal(
                    ticker=ticker,
                    strategy=self.name,
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    notes=f"Z-score={zscore:.2f}, extreme high"
                )

                if self.validate_signal(signal):
                    signals.append(signal)

        return signals

    def _calculate_buy_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        zscore: float
    ) -> float:
        """Calculate confidence for buy signal."""
        # Base confidence from z-score extremity
        if zscore < -3.0:
            base = 75
        elif zscore < -2.5:
            base = 70
        elif zscore < -2.0:
            base = 65
        else:
            base = 55

        factors = {}

        # Historical context (is this z-score unusual for this stock?)
        zscore_min = latest.get("zscore_min_60d", -3)
        if zscore < zscore_min * 0.9:
            # Near historical extreme
            factors["historical"] = 1.1

        # Volume (capitulation volume is bullish for reversal)
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            factors["volume"] = 1.1
        elif volume_ratio > 1.5:
            factors["volume"] = 1.05

        # Mean reversion speed
        reversion_speed = latest.get("reversion_speed", 0.1)
        if reversion_speed > 0.15:
            factors["reversion"] = 1.05  # Fast reverter

        # Volatility context
        atr_pct = latest.get("atr_pct", 2.0)
        if atr_pct > 4.0:
            factors["volatility"] = 0.9  # High volatility = less reliable
        elif atr_pct < 2.0:
            factors["volatility"] = 1.05  # Low volatility = more reliable

        # Price percentile
        price_pct = latest.get("price_percentile", 50)
        if price_pct < 10:
            factors["price_pct"] = 1.1  # Near 52-week low

        return self.calculate_confidence(base, factors)

    def _calculate_sell_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        zscore: float
    ) -> float:
        """Calculate confidence for sell signal."""
        if zscore > 3.0:
            base = 75
        elif zscore > 2.5:
            base = 70
        elif zscore > 2.0:
            base = 65
        else:
            base = 55

        factors = {}

        # Historical context
        zscore_max = latest.get("zscore_max_60d", 3)
        if zscore > zscore_max * 0.9:
            factors["historical"] = 1.1

        # Volume
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            factors["volume"] = 1.1

        # Mean reversion speed
        reversion_speed = latest.get("reversion_speed", 0.1)
        if reversion_speed > 0.15:
            factors["reversion"] = 1.05

        # Price percentile
        price_pct = latest.get("price_percentile", 50)
        if price_pct > 90:
            factors["price_pct"] = 1.1  # Near 52-week high

        return self.calculate_confidence(base, factors)

    def _calculate_reversion_levels(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        direction: SignalDirection,
        zscore: float
    ) -> tuple[float, float, float]:
        """Calculate entry, target, and stop based on z-score reversion."""
        current_price = latest["close"]
        atr = latest.get("atr", current_price * 0.02)

        # Calculate mean price (z-score = 0 level)
        rolling_mean = df["close"].rolling(self.lookback_period).mean().iloc[-1]
        rolling_std = df["close"].rolling(self.lookback_period).std().iloc[-1]

        entry = current_price

        if direction == SignalDirection.BUY:
            # Target: revert to mean (z-score = 0)
            target = rolling_mean

            # Stop: further deviation or ATR-based
            stop_zscore = rolling_mean + (self.zscore_entry * 1.5) * rolling_std  # More extreme z-score
            stop_atr = entry - (2 * atr)
            stop = min(stop_zscore, stop_atr)

            # Apply limits
            min_stop = entry * (1 - self.stop_loss_pct / 100)
            stop = max(stop, min_stop)

            max_target = entry * (1 + self.take_profit_pct / 100)
            target = min(target, max_target)

        else:
            target = rolling_mean

            stop_zscore = rolling_mean + (abs(self.zscore_entry) * 1.5) * rolling_std
            stop_atr = entry + (2 * atr)
            stop = max(stop_zscore, stop_atr)

            max_stop = entry * (1 + self.stop_loss_pct / 100)
            stop = min(stop, max_stop)

            min_target = entry * (1 - self.take_profit_pct / 100)
            target = max(target, min_target)

        return entry, target, stop
