"""Market health checks and circuit breakers.

Provides safety mechanisms to pause trading during extreme market conditions.
"""

from datetime import datetime, date
from typing import Tuple
import yfinance as yf

from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Default circuit breaker thresholds
DEFAULT_CIRCUIT_BREAKER_PCT = -3.0  # Pause new entries if market drops > 3%
DEFAULT_MARKET_TICKER = "SPY"  # Use SPY as market proxy


class CircuitBreaker:
    """
    Market circuit breaker to pause trading during extreme conditions.

    Checks intraday market movement and blocks new positions if:
    - Market (SPY) drops more than threshold (default -3%)
    - Can be configured to use different thresholds or tickers
    """

    def __init__(
        self,
        threshold_pct: float = DEFAULT_CIRCUIT_BREAKER_PCT,
        market_ticker: str = DEFAULT_MARKET_TICKER
    ):
        """
        Initialize circuit breaker.

        Args:
            threshold_pct: Percentage drop that triggers circuit breaker (negative number)
            market_ticker: Ticker to use as market proxy (default SPY)
        """
        self.threshold_pct = threshold_pct
        self.market_ticker = market_ticker
        self._last_check_time: datetime | None = None
        self._last_result: Tuple[bool, float, str] | None = None
        self._cache_seconds = 60  # Cache for 60 seconds

    def check_market_health(self, force_refresh: bool = False) -> Tuple[bool, float, str]:
        """
        Check if market conditions allow new trades.

        Returns:
            Tuple of (is_healthy, market_change_pct, message)
            - is_healthy: True if OK to trade, False if circuit breaker triggered
            - market_change_pct: Intraday % change in market
            - message: Human-readable status
        """
        now = datetime.now()

        # Use cached result if fresh enough
        if (not force_refresh and
            self._last_check_time is not None and
            (now - self._last_check_time).total_seconds() < self._cache_seconds and
            self._last_result is not None):
            return self._last_result

        try:
            # Fetch current market data
            ticker = yf.Ticker(self.market_ticker)

            # Get today's data
            hist = ticker.history(period="1d", interval="1m")

            if hist.empty:
                logger.warning(f"Could not fetch {self.market_ticker} data for circuit breaker")
                return True, 0.0, "Market data unavailable - allowing trades"

            # Calculate intraday change
            open_price = hist["Open"].iloc[0]
            current_price = hist["Close"].iloc[-1]

            if open_price <= 0:
                return True, 0.0, "Invalid market data - allowing trades"

            change_pct = ((current_price - open_price) / open_price) * 100

            # Check if circuit breaker triggered
            if change_pct <= self.threshold_pct:
                message = (
                    f"🚨 CIRCUIT BREAKER TRIGGERED: {self.market_ticker} down {change_pct:.2f}% today "
                    f"(threshold: {self.threshold_pct}%). New positions blocked."
                )
                logger.warning(message)
                result = (False, change_pct, message)
            else:
                if change_pct < -1.5:
                    message = f"⚠️ Market weak: {self.market_ticker} {change_pct:+.2f}% - trading with caution"
                elif change_pct < 0:
                    message = f"Market slightly down: {self.market_ticker} {change_pct:+.2f}%"
                else:
                    message = f"Market healthy: {self.market_ticker} {change_pct:+.2f}%"
                result = (True, change_pct, message)

            # Cache result
            self._last_check_time = now
            self._last_result = result

            return result

        except Exception as e:
            logger.error(f"Error checking market health: {e}")
            # On error, allow trading but log warning
            return True, 0.0, f"Market check failed ({e}) - allowing trades"

    def is_trading_allowed(self) -> bool:
        """Simple boolean check if trading is allowed."""
        is_healthy, _, _ = self.check_market_health()
        return is_healthy

    def get_status_for_display(self) -> dict:
        """Get circuit breaker status for dashboard/console display."""
        is_healthy, change_pct, message = self.check_market_health()

        return {
            "is_healthy": is_healthy,
            "market_change_pct": change_pct,
            "message": message,
            "threshold_pct": self.threshold_pct,
            "market_ticker": self.market_ticker,
            "status": "🟢 OPEN" if is_healthy else "🔴 CIRCUIT BREAKER"
        }


# Global instance for easy access
_circuit_breaker: CircuitBreaker | None = None


def get_circuit_breaker(threshold_pct: float = DEFAULT_CIRCUIT_BREAKER_PCT) -> CircuitBreaker:
    """Get or create the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker(threshold_pct=threshold_pct)
    return _circuit_breaker


def check_circuit_breaker() -> Tuple[bool, str]:
    """
    Quick check if circuit breaker allows trading.

    Returns:
        Tuple of (is_allowed, message)
    """
    cb = get_circuit_breaker()
    is_healthy, change_pct, message = cb.check_market_health()
    return is_healthy, message
