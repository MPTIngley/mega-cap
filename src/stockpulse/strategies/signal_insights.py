"""Signal Insights - Near-miss detection and blocking reason analysis.

Provides insight into why strategies aren't generating signals and
why signals might be blocked from trading.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

logger = get_logger(__name__)


# Human-readable strategy descriptions with intuition
STRATEGY_DESCRIPTIONS = {
    "rsi_mean_reversion": {
        "short": "RSI Mean Reversion: Buy oversold stocks (RSI below 30) expecting a bounce back",
        "long": """**RSI Mean Reversion** - Relative Strength Index (RSI) measures how "oversold" or
"overbought" a stock is on a scale of 0-100. When RSI drops below 30, the stock has fallen sharply
and is considered oversold - historically, these stocks tend to bounce back. We buy when RSI < 30
and sell when it recovers above 70. Think of it like buying a stock that's "on sale" after a big drop.

Our settings: Buy trigger = RSI < 30, Sell trigger = RSI > 70, Lookback = 14 days"""
    },
    "macd_volume": {
        "short": "MACD Volume: Buy when momentum shifts positive with strong trading volume",
        "long": """**MACD Volume** - MACD (Moving Average Convergence Divergence) tracks momentum by
comparing short-term vs long-term price trends. When the fast trend crosses above the slow trend,
it signals the stock is gaining momentum. We add a volume filter - only buying when trading volume
is above average, confirming real interest in the stock. Think of it like waiting for a car to
accelerate before getting in.

Our settings: MACD crossover = 12/26 day EMAs, Signal line = 9 days, Volume filter = 1.5x average"""
    },
    "zscore_mean_reversion": {
        "short": "Z-Score Mean Reversion: Buy stocks that dropped 2+ standard deviations below normal",
        "long": """**Z-Score Mean Reversion** - Z-score measures how far a stock's price is from its
recent average, in units of standard deviation. A Z-score of -2.0 means the price is 2 standard
deviations below its average - a statistically unusual drop. These extreme moves often reverse.
Think of it like a rubber band stretched too far - it tends to snap back.

Our settings: Buy trigger = Z-score < -2.0, Sell trigger = Z-score > 1.0, Lookback = 20 days"""
    },
    "momentum_breakout": {
        "short": "Momentum Breakout: Buy stocks breaking above their 20-day high with volume",
        "long": """**Momentum Breakout** - This strategy catches stocks making new highs. When a stock
breaks above its recent 20-day high price, it often signals the start of an uptrend. We confirm with
above-average volume to ensure the breakout has real buying interest behind it. Think of it like
a stock "breaking free" from a ceiling that was holding it back.

Our settings: Breakout trigger = New 20-day high, Volume filter = 1.2x average, Target = +8% gain"""
    },
    "week52_low_bounce": {
        "short": "52-Week Low Bounce: Buy quality stocks near their yearly lows",
        "long": """**52-Week Low Bounce** - Stocks trading near their 52-week (yearly) low can be
bargains if the company is fundamentally sound. This strategy buys when a stock is within 10% of
its yearly low, betting on a rebound. We focus on S&P 500 companies to ensure quality. Think of it
like buying a brand-name product at a clearance sale.

Our settings: Buy trigger = Within 10% of 52-week low, Max distance = 20% above low"""
    },
    "sector_rotation": {
        "short": "Sector Rotation: Buy stocks in sectors showing relative strength vs the market",
        "long": """**Sector Rotation** - Different sectors (tech, healthcare, energy, etc.) take turns
leading the market. This strategy identifies stocks in sectors outperforming the S&P 500 index.
We measure "relative strength" - if a stock is rising faster than the market, it has positive
momentum. Think of it like betting on the winning horse mid-race.

Our settings: Relative strength trigger > 1.1 (10% better than market), Lookback = 20 days"""
    }
}


class SignalInsights:
    """
    Analyzes stocks for near-miss conditions and blocking reasons.

    Near-miss: A stock that almost met a strategy's criteria
    Blocking: A signal that was blocked due to risk limits
    """

    def __init__(self):
        """Initialize signal insights."""
        self.db = get_db()
        self.config = get_config()
        self.data_ingestion = DataIngestion()
        self.strategy_configs = self.config.get("strategies", {})

    def get_near_misses(self, tickers: list[str], top_n: int = 3) -> dict[str, list[dict]]:
        """
        Get near-miss stocks for each strategy.

        A near-miss is a stock that almost met the entry criteria but didn't quite trigger.

        Args:
            tickers: List of tickers to analyze
            top_n: Number of near-misses to return per strategy

        Returns:
            Dict mapping strategy name to list of near-miss dicts
        """
        from datetime import date

        # Get price data
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        price_data = self.data_ingestion.get_daily_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )

        if price_data.empty:
            return {}

        near_misses = {
            "rsi_mean_reversion": [],
            "macd_volume": [],
            "zscore_mean_reversion": [],
            "momentum_breakout": [],
            "week52_low_bounce": [],
            "sector_rotation": [],
        }

        # Process each ticker
        for ticker in tickers:
            ticker_data = price_data[price_data["ticker"] == ticker].copy()

            if len(ticker_data) < 60:
                continue

            ticker_data = ticker_data.sort_values("date")

            # Calculate all indicators for near-miss detection
            indicators = self._calculate_all_indicators(ticker_data)

            if indicators is None:
                continue

            latest = indicators.iloc[-1]
            current_price = latest["close"]

            # RSI Near-Miss: RSI between 30-40 (close to oversold <30)
            rsi = latest.get("rsi", 50)
            if 30 <= rsi <= 40:
                distance = rsi - 30  # How far from trigger
                near_misses["rsi_mean_reversion"].append({
                    "ticker": ticker,
                    "price": current_price,
                    "indicator": f"RSI {rsi:.1f}",
                    "criteria": "RSI < 30 (oversold)",
                    "distance": f"{distance:.1f} pts above 30 trigger",
                    "score": 100 - (distance * 10),  # Score for sorting
                })

            # MACD Near-Miss: MACD close to signal line (within 0.3 and converging)
            macd = latest.get("macd", 0)
            macd_signal = latest.get("macd_signal", 0)
            macd_diff = macd - macd_signal
            macd_prev_diff = indicators.iloc[-2].get("macd", 0) - indicators.iloc[-2].get("macd_signal", 0)

            if -0.5 < macd_diff < 0 and macd_diff > macd_prev_diff:  # Below signal but converging
                near_misses["macd_volume"].append({
                    "ticker": ticker,
                    "price": current_price,
                    "indicator": f"MACD {abs(macd_diff):.2f} below signal",
                    "criteria": "MACD crosses above signal",
                    "distance": f"converging, {abs(macd_diff):.2f} to crossover",
                    "score": 100 - (abs(macd_diff) * 100),
                })

            # Z-Score Near-Miss: Z-score between -1.5 and -2.0
            zscore = latest.get("zscore", 0)
            if -2.0 < zscore <= -1.5:
                distance = abs(zscore - (-2.0))
                near_misses["zscore_mean_reversion"].append({
                    "ticker": ticker,
                    "price": current_price,
                    "indicator": f"Z-score {zscore:.2f}",
                    "criteria": "Z-score < -2.0",
                    "distance": f"needs {distance:.2f} more drop to hit −2.0",
                    "score": 100 - (distance * 50),
                })

            # Momentum Breakout Near-Miss: Price within 3% of 20-day high
            high_20d = latest.get("high_20d", current_price)
            if high_20d > 0:
                pct_from_high = (high_20d - current_price) / high_20d * 100
                if 0 < pct_from_high <= 3:  # Within 3% of breakout
                    near_misses["momentum_breakout"].append({
                        "ticker": ticker,
                        "price": current_price,
                        "indicator": f"${current_price:.2f} vs 20d high ${high_20d:.2f}",
                        "criteria": "Break above 20-day high",
                        "distance": f"only {pct_from_high:.1f}% below breakout",
                        "score": 100 - (pct_from_high * 20),
                    })

            # Week52 Low Bounce Near-Miss: 10-20% above 52-week low
            low_52w = latest.get("low_52w", current_price)
            if low_52w > 0:
                pct_above_low = (current_price - low_52w) / low_52w * 100
                if 10 < pct_above_low <= 20:  # Strategy triggers at <10%
                    near_misses["week52_low_bounce"].append({
                        "ticker": ticker,
                        "price": current_price,
                        "indicator": f"${current_price:.2f} vs 52w low ${low_52w:.2f}",
                        "criteria": "Within 10% of 52-week low",
                        "distance": f"{pct_above_low:.1f}% above low (need ≤10%)",
                        "score": 100 - ((pct_above_low - 10) * 10),
                    })

            # Sector Rotation Near-Miss: Relative strength close to threshold
            rel_strength = latest.get("relative_strength", 0)
            if 0.9 <= rel_strength < 1.1:  # Close to positive relative strength
                near_misses["sector_rotation"].append({
                    "ticker": ticker,
                    "price": current_price,
                    "indicator": f"Rel strength {rel_strength:.2f}",
                    "criteria": "Relative strength > 1.1",
                    "distance": f"needs +{(1.1 - rel_strength):.2f} to reach 1.1 trigger",
                    "score": 100 - ((1.1 - rel_strength) * 100),
                })

        # Sort each strategy's near-misses by score and take top N
        for strategy in near_misses:
            near_misses[strategy] = sorted(
                near_misses[strategy],
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:top_n]

        return near_misses

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Calculate all indicators needed for near-miss detection."""
        try:
            df = df.copy()

            # RSI (14-period)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

            # Z-Score (20-period)
            mean_20 = df["close"].rolling(20).mean()
            std_20 = df["close"].rolling(20).std()
            df["zscore"] = (df["close"] - mean_20) / std_20

            # 20-day high for momentum breakout
            df["high_20d"] = df["high"].rolling(20).max()

            # 52-week low (252 trading days)
            df["low_52w"] = df["low"].rolling(min(252, len(df))).min()

            # Simple relative strength vs SPY (approximated using price momentum)
            df["price_change_20d"] = df["close"].pct_change(20)
            df["relative_strength"] = 1 + df["price_change_20d"]  # Simplified

            # Volume ratio
            df["volume_sma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

            return df

        except Exception as e:
            logger.debug(f"Error calculating indicators: {e}")
            return None

    def get_blocking_reasons(
        self,
        ticker: str,
        position_manager: Any
    ) -> list[dict]:
        """
        Get detailed reasons why a ticker might be blocked from trading.

        Args:
            ticker: Ticker to check
            position_manager: PositionManager instance

        Returns:
            List of blocking reason dicts
        """
        reasons = []

        # Check cooldown
        cooldown_ok, cooldown_reason = position_manager._check_cooldown(ticker)
        if not cooldown_ok:
            days_match = cooldown_reason.split(": ")[-1].replace(" days remaining", "")
            reasons.append({
                "type": "cooldown",
                "reason": cooldown_reason,
                "severity": "medium",
                "icon": "⏱️"
            })

        # Check loss limit
        loss_ok, loss_reason = position_manager._check_loss_limit(ticker)
        if not loss_ok:
            loss_count = position_manager._loss_count_cache.get(ticker, 0)
            reasons.append({
                "type": "loss_limit",
                "reason": loss_reason,
                "severity": "high",
                "icon": "🚫",
                "detail": f"{loss_count} consecutive losses"
            })

        # Check sector concentration
        sector_ok, sector_reason = position_manager._check_sector_concentration(ticker)
        if not sector_ok:
            reasons.append({
                "type": "sector_concentration",
                "reason": sector_reason,
                "severity": "medium",
                "icon": "📊"
            })

        return reasons

    def get_strategy_status(
        self,
        position_manager: Any
    ) -> dict[str, dict]:
        """
        Get current status for each strategy including exposure and capacity.

        Args:
            position_manager: PositionManager instance

        Returns:
            Dict mapping strategy name to status dict
        """
        all_strategies = [
            "rsi_mean_reversion", "macd_volume", "zscore_mean_reversion",
            "momentum_breakout", "week52_low_bounce", "sector_rotation"
        ]

        status = {}
        max_per_strategy = position_manager.max_per_strategy_pct

        for strategy in all_strategies:
            current_exposure = position_manager.get_strategy_current_exposure_pct(strategy)
            remaining_capacity = position_manager.get_strategy_remaining_capacity_pct(strategy)

            # Get position count for this strategy
            positions = self.db.fetchone(
                "SELECT COUNT(*) FROM positions_paper WHERE status = 'open' AND strategy = ?",
                (strategy,)
            )
            position_count = positions[0] if positions else 0

            status[strategy] = {
                "current_exposure_pct": round(current_exposure, 1),
                "remaining_capacity_pct": round(remaining_capacity, 1),
                "max_allowed_pct": max_per_strategy,
                "utilization_pct": round((current_exposure / max_per_strategy) * 100, 1) if max_per_strategy > 0 else 0,
                "position_count": position_count,
                "can_open_more": remaining_capacity >= position_manager.min_position_size_pct,
            }

        return status


def format_near_misses_text(near_misses: dict[str, list[dict]]) -> str:
    """Format near-misses for console output."""
    lines = []

    for strategy, stocks in near_misses.items():
        if stocks:
            lines.append(f"  {strategy}:")
            for stock in stocks:
                lines.append(f"    • {stock['ticker']}: {stock['indicator']} ({stock['distance']})")
        else:
            lines.append(f"  {strategy}: (no near-misses)")

    return "\n".join(lines)


def format_blocking_reasons_text(blocking_details: list[tuple]) -> str:
    """Format blocking reasons for console output."""
    lines = []

    for signal, reasons in blocking_details:
        ticker = signal.ticker if hasattr(signal, 'ticker') else signal.get('ticker', 'Unknown')
        if reasons:
            reason_strs = [f"{r['icon']} {r['reason']}" for r in reasons]
            lines.append(f"  {ticker}: {'; '.join(reason_strs)}")
        else:
            lines.append(f"  {ticker}: Blocked by risk check")

    return "\n".join(lines)
