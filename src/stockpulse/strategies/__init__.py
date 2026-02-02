"""Trading strategies for StockPulse."""

from .base import BaseStrategy, Signal, SignalDirection
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy
from .macd_volume import MACDVolumeStrategy
from .zscore_mean_reversion import ZScoreMeanReversionStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .signal_generator import SignalGenerator
from .position_manager import PositionManager
from .backtest import Backtester, BacktestResult

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalDirection",
    "RSIMeanReversionStrategy",
    "BollingerSqueezeStrategy",
    "MACDVolumeStrategy",
    "ZScoreMeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "SignalGenerator",
    "PositionManager",
    "Backtester",
    "BacktestResult",
]
