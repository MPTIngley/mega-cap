"""Sector Rotation Strategy.

Identifies the strongest performing sectors and generates signals
for stocks in those sectors. Based on relative strength momentum.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalDirection


class SectorRotationStrategy(BaseStrategy):
    """
    Sector rotation strategy based on relative strength.

    Identifies strongest sectors over lookback period, then
    generates buy signals for stocks in top sectors.

    Entry: Stock in top-performing sector with good relative strength
    Exit: Sector loses relative strength or individual stock weakens
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize sector rotation strategy."""
        super().__init__(config)
        self.lookback_days = config.get("lookback_days", 20)
        self.top_sectors = config.get("top_sectors", 2)  # Number of top sectors to buy
        self.min_sector_return = config.get("min_sector_return", 2.0)  # Min % return
        self.relative_strength_threshold = config.get("relative_strength_threshold", 1.1)

    @property
    def name(self) -> str:
        return "sector_rotation"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and relative strength indicators."""
        df = df.copy()

        # Price momentum over various periods
        df["return_5d"] = df["close"].pct_change(5) * 100
        df["return_10d"] = df["close"].pct_change(10) * 100
        df["return_20d"] = df["close"].pct_change(self.lookback_days) * 100

        # Composite momentum score
        df["momentum_score"] = (
            df["return_5d"] * 0.2 +
            df["return_10d"] * 0.3 +
            df["return_20d"] * 0.5
        )

        # RSI for overbought/oversold
        df["rsi"] = self.calculate_rsi(df["close"], 14)

        # Volume trend
        df["volume_ratio"] = self.calculate_volume_ratio(df["volume"], 20)

        # Moving averages
        df["sma_20"] = self.calculate_sma(df["close"], 20)
        df["sma_50"] = self.calculate_sma(df["close"], 50)

        # Above moving averages (trend confirmation)
        df["above_sma_20"] = df["close"] > df["sma_20"]
        df["above_sma_50"] = df["close"] > df["sma_50"]

        # ATR
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Relative strength vs market (approximated by momentum rank)
        # This will be calculated in context with other stocks

        return df

    def generate_signals(self, df: pd.DataFrame, ticker: str) -> list[Signal]:
        """
        Generate sector rotation signals.

        Note: This strategy works best when called with sector context.
        Without sector data, it falls back to pure momentum.
        """
        if len(df) < 60:
            return []

        df = self.calculate_indicators(df)
        signals = []

        # Get the most recent data point
        latest = df.iloc[-1]

        current_price = latest["close"]
        momentum = latest["momentum_score"]
        return_20d = latest["return_20d"]
        rsi = latest["rsi"]

        # Basic momentum criteria
        if return_20d < self.min_sector_return:
            return []

        # Need to be in uptrend
        if not latest["above_sma_20"]:
            return []

        # Not overbought
        if rsi > 75:
            return []

        # Calculate confidence
        confidence = self._calculate_confidence(df, latest)

        if confidence >= self.min_confidence:
            entry, target, stop = self.calculate_entry_exit_prices(
                current_price, SignalDirection.BUY
            )

            signal = Signal(
                ticker=ticker,
                strategy=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                entry_price=entry,
                target_price=target,
                stop_price=stop,
                notes=f"Sector momentum: {return_20d:.1f}% (20d), RSI={rsi:.0f}"
            )

            if self.validate_signal(signal):
                signals.append(signal)

        return signals

    def generate_signals_with_sector_context(
        self,
        df: pd.DataFrame,
        ticker: str,
        sector: str,
        sector_returns: dict[str, float]
    ) -> list[Signal]:
        """
        Generate signals with full sector context.

        Args:
            df: Price data for this ticker
            ticker: Stock ticker
            sector: Sector this stock belongs to
            sector_returns: Dict mapping sector -> return over lookback period

        Returns:
            List of signals
        """
        if len(df) < 60:
            return []

        # Check if this stock's sector is in the top performers
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        top_sector_names = [s[0] for s in sorted_sectors[:self.top_sectors]]

        if sector not in top_sector_names:
            return []

        # Sector is strong - now check if this stock is strong within sector
        df = self.calculate_indicators(df)
        signals = []

        latest = df.iloc[-1]
        current_price = latest["close"]
        stock_return = latest["return_20d"]
        sector_return = sector_returns.get(sector, 0)
        rsi = latest["rsi"]

        # Stock should be at least as strong as sector
        relative_strength = stock_return / sector_return if sector_return > 0 else 0

        if relative_strength < self.relative_strength_threshold:
            return []

        # Trend confirmation
        if not latest["above_sma_20"]:
            return []

        # Not overbought
        if rsi > 75:
            return []

        confidence = self._calculate_confidence_with_sector(
            df, latest, sector, sector_return, relative_strength
        )

        if confidence >= self.min_confidence:
            entry, target, stop = self.calculate_entry_exit_prices(
                current_price, SignalDirection.BUY
            )

            signal = Signal(
                ticker=ticker,
                strategy=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                entry_price=entry,
                target_price=target,
                stop_price=stop,
                notes=f"Top sector ({sector}): {sector_return:.1f}%, RS={relative_strength:.2f}"
            )

            if self.validate_signal(signal):
                signals.append(signal)

        return signals

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        latest: pd.Series
    ) -> float:
        """Calculate confidence without sector context."""
        momentum = latest.get("momentum_score", 0)

        # Base confidence from momentum strength
        if momentum > 10:
            base = 70
        elif momentum > 5:
            base = 65
        else:
            base = 60

        factors = {}

        # Volume confirmation
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            factors["volume"] = 1.1
        elif volume_ratio > 1.2:
            factors["volume"] = 1.05

        # RSI in sweet spot (not overbought)
        rsi = latest.get("rsi", 50)
        if 50 <= rsi <= 65:
            factors["rsi"] = 1.1
        elif 40 <= rsi <= 70:
            factors["rsi"] = 1.05
        elif rsi > 70:
            factors["rsi"] = 0.9

        # Trend strength
        if latest.get("above_sma_50", False):
            factors["trend"] = 1.1

        return self.calculate_confidence(base, factors)

    def _calculate_confidence_with_sector(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        sector: str,
        sector_return: float,
        relative_strength: float
    ) -> float:
        """Calculate confidence with sector context."""
        # Higher base confidence when we have sector context
        if sector_return > 5:
            base = 72
        elif sector_return > 3:
            base = 68
        else:
            base = 65

        factors = {}

        # Relative strength bonus
        if relative_strength > 1.5:
            factors["rs"] = 1.15
        elif relative_strength > 1.2:
            factors["rs"] = 1.10

        # Volume
        volume_ratio = latest.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            factors["volume"] = 1.08

        # RSI sweet spot
        rsi = latest.get("rsi", 50)
        if 50 <= rsi <= 65:
            factors["rsi"] = 1.08
        elif rsi > 70:
            factors["rsi"] = 0.9

        # Strong uptrend
        if latest.get("above_sma_50", False):
            factors["trend"] = 1.08

        return self.calculate_confidence(base, factors)

    @staticmethod
    def calculate_sector_returns(
        price_data: pd.DataFrame,
        universe_df: pd.DataFrame,
        lookback_days: int = 20
    ) -> dict[str, float]:
        """
        Calculate sector returns for sector rotation decisions.

        Args:
            price_data: DataFrame with ticker, date, close columns
            universe_df: DataFrame with ticker, sector columns
            lookback_days: Period for return calculation

        Returns:
            Dict mapping sector name to average return
        """
        # Merge price data with sector info
        merged = price_data.merge(universe_df[["ticker", "sector"]], on="ticker", how="left")

        # Calculate returns per ticker
        ticker_returns = {}
        for ticker in merged["ticker"].unique():
            ticker_data = merged[merged["ticker"] == ticker].sort_values("date")
            if len(ticker_data) >= lookback_days:
                recent = ticker_data.iloc[-1]["close"]
                past = ticker_data.iloc[-lookback_days]["close"]
                ticker_returns[ticker] = (recent - past) / past * 100

        # Calculate sector averages
        sector_returns = {}
        ticker_sectors = universe_df.set_index("ticker")["sector"].to_dict()

        sector_ticker_returns = {}
        for ticker, ret in ticker_returns.items():
            sector = ticker_sectors.get(ticker)
            if sector:
                if sector not in sector_ticker_returns:
                    sector_ticker_returns[sector] = []
                sector_ticker_returns[sector].append(ret)

        for sector, returns in sector_ticker_returns.items():
            sector_returns[sector] = np.mean(returns) if returns else 0

        return sector_returns
