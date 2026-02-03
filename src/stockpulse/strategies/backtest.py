"""Backtesting framework for strategy evaluation."""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any
import json

import pandas as pd
import numpy as np

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

from .base import BaseStrategy, Signal, SignalDirection

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """A single trade in a backtest."""
    ticker: str
    direction: SignalDirection
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    shares: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: str = ""
    target_price: float = 0
    stop_price: float = 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy: str
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    avg_hold_days: float
    return_std_pct: float = 0.0  # Standard deviation of returns
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "strategy": self.strategy,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_hold_days": self.avg_hold_days,
            "return_std_pct": self.return_std_pct,
            "params": self.params,
        }


class Backtester:
    """
    Backtesting engine for strategy evaluation.

    Features:
    - Vectorized backtesting for speed
    - Transaction cost modeling
    - Walk-forward validation support
    - Comprehensive performance metrics
    """

    def __init__(self):
        """Initialize backtester."""
        self.db = get_db()
        self.config = get_config()
        self.backtest_config = self.config.get("backtest", {})
        self.trading_config = self.config.get("trading", {})
        self.data_ingestion = DataIngestion()

        # Transaction costs
        self.slippage_pct = self.trading_config.get("slippage_percent", 0.05)
        self.spread_pct = self.trading_config.get("spread_percent", 0.02)
        self.commission = self.trading_config.get("commission_per_trade", 0.0)

        # Default backtest parameters
        self.initial_capital = self.backtest_config.get("initial_capital", 100000.0)
        self.position_size_pct = self.backtest_config.get("position_size_pct", 5.0)

    def run_backtest(
        self,
        strategy: BaseStrategy,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        initial_capital: float | None = None
    ) -> BacktestResult:
        """
        Run a backtest for a strategy.

        Args:
            strategy: Strategy to backtest
            tickers: List of tickers to test on
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital

        Returns:
            BacktestResult with performance metrics
        """
        if start_date is None:
            start_str = self.backtest_config.get("start_date", "2023-01-01")
            start_date = date.fromisoformat(start_str)

        if end_date is None:
            end_str = self.backtest_config.get("end_date")
            end_date = date.fromisoformat(end_str) if end_str else date.today()

        if initial_capital is None:
            initial_capital = self.initial_capital

        logger.info(f"Running backtest for {strategy.name} from {start_date} to {end_date}")

        # Get price data
        price_data = self.data_ingestion.get_daily_prices(
            tickers=tickers,
            start_date=start_date - timedelta(days=90),  # Extra history for indicators
            end_date=end_date
        )

        if price_data.empty:
            logger.warning("No price data available for backtest")
            return self._empty_result(strategy.name, start_date, end_date, initial_capital)

        # Run backtest
        trades, equity_curve = self._simulate_trading(
            strategy, price_data, tickers, start_date, end_date, initial_capital
        )

        # Calculate metrics
        result = self._calculate_metrics(
            strategy.name, trades, equity_curve,
            start_date, end_date, initial_capital, strategy.config
        )

        # Store result
        self._store_result(result)

        return result

    def _simulate_trading(
        self,
        strategy: BaseStrategy,
        price_data: pd.DataFrame,
        tickers: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float
    ) -> tuple[list[BacktestTrade], pd.DataFrame]:
        """Simulate trading with the strategy."""
        trades = []
        open_positions: dict[str, BacktestTrade] = {}
        capital = initial_capital
        equity_history = []

        # Get all trading days - convert to date objects for comparison
        all_dates_raw = sorted(price_data["date"].unique())
        all_dates = []
        for d in all_dates_raw:
            try:
                if isinstance(d, date) and not isinstance(d, datetime):
                    all_dates.append(d)
                elif isinstance(d, datetime):
                    all_dates.append(d.date())
                elif hasattr(d, 'date'):
                    # pandas Timestamp
                    all_dates.append(d.date())
                else:
                    # String or numpy string - convert via isoformat
                    all_dates.append(date.fromisoformat(str(d)[:10]))
            except Exception:
                continue  # Skip invalid dates

        trading_dates = [d for d in all_dates if start_date <= d <= end_date]

        for current_date in trading_dates:
            # Get prices for this date (compare as string since DB stores strings)
            date_str = current_date.isoformat() if isinstance(current_date, date) else str(current_date)
            day_prices = price_data[price_data["date"] == date_str]
            price_dict = {row["ticker"]: row for _, row in day_prices.iterrows()}

            # Update open positions (check for exit)
            positions_to_close = []
            for ticker, position in open_positions.items():
                if ticker not in price_dict:
                    continue

                current_row = price_dict[ticker]
                current_price = current_row["close"]
                high = current_row["high"]
                low = current_row["low"]

                exit_price = None
                exit_reason = None

                # Check exit conditions
                if position.direction == SignalDirection.BUY:
                    # Check stop (use low of day)
                    if low <= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "stop"
                    # Check target (use high of day)
                    elif high >= position.target_price:
                        exit_price = position.target_price
                        exit_reason = "target"
                else:
                    if high >= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "stop"
                    elif low <= position.target_price:
                        exit_price = position.target_price
                        exit_reason = "target"

                # Check max hold days
                if exit_reason is None:
                    hold_days = (current_date - position.entry_date).days
                    if hold_days >= strategy.hold_days_max:
                        exit_price = current_price
                        exit_reason = "expired"

                if exit_price:
                    # Apply slippage
                    if position.direction == SignalDirection.BUY:
                        exit_price *= (1 - self.slippage_pct / 100)
                    else:
                        exit_price *= (1 + self.slippage_pct / 100)

                    # Calculate P&L
                    if position.direction == SignalDirection.BUY:
                        gross_pnl = (exit_price - position.entry_price) * position.shares
                    else:
                        gross_pnl = (position.entry_price - exit_price) * position.shares

                    # Transaction costs
                    exit_cost = self._calculate_trade_cost(exit_price, position.shares)
                    net_pnl = gross_pnl - exit_cost

                    position.exit_date = current_date
                    position.exit_price = exit_price
                    position.pnl = net_pnl
                    position.pnl_pct = (net_pnl / (position.entry_price * position.shares)) * 100
                    position.exit_reason = exit_reason

                    capital += (exit_price * position.shares) + net_pnl
                    positions_to_close.append(ticker)
                    trades.append(position)

            # Remove closed positions
            for ticker in positions_to_close:
                del open_positions[ticker]

            # Generate new signals
            for ticker in tickers:
                if ticker in open_positions:
                    continue  # Already have position

                # Convert current_date to string for comparison with DB data
                current_date_str = current_date.isoformat() if isinstance(current_date, date) else str(current_date)
                ticker_data = price_data[
                    (price_data["ticker"] == ticker) &
                    (price_data["date"] <= current_date_str)
                ].copy()

                if len(ticker_data) < 60:
                    continue

                ticker_data = ticker_data.sort_values("date")

                try:
                    signals = strategy.generate_signals(ticker_data, ticker)

                    for signal in signals:
                        # Check if we have capital
                        position_value = capital * (self.position_size_pct / 100)
                        if position_value < 100:  # Minimum position
                            continue

                        if len(open_positions) >= 20:  # Max positions
                            continue

                        # Apply slippage to entry
                        if signal.direction == SignalDirection.BUY:
                            entry_price = signal.entry_price * (1 + self.slippage_pct / 100)
                        else:
                            entry_price = signal.entry_price * (1 - self.slippage_pct / 100)

                        shares = position_value / entry_price
                        entry_cost = self._calculate_trade_cost(entry_price, shares)

                        # Open position
                        trade = BacktestTrade(
                            ticker=ticker,
                            direction=signal.direction,
                            entry_date=current_date,
                            entry_price=entry_price,
                            shares=shares,
                            target_price=signal.target_price,
                            stop_price=signal.stop_price
                        )

                        open_positions[ticker] = trade
                        capital -= (entry_price * shares) + entry_cost

                        break  # Only one signal per ticker per day

                except Exception as e:
                    logger.debug(f"Error generating signals for {ticker}: {e}")

            # Calculate daily equity
            positions_value = sum(
                price_dict.get(t, {}).get("close", p.entry_price) * p.shares
                for t, p in open_positions.items()
                for _ in [price_dict.get(t)]
                if _ is not None
            )

            # Fallback calculation if above fails
            positions_value = 0
            for ticker, pos in open_positions.items():
                if ticker in price_dict:
                    positions_value += price_dict[ticker]["close"] * pos.shares
                else:
                    positions_value += pos.entry_price * pos.shares

            total_equity = capital + positions_value
            equity_history.append({
                "date": current_date,
                "equity": total_equity,
                "cash": capital,
                "positions_value": positions_value,
                "open_positions": len(open_positions)
            })

        # Close any remaining positions at end
        for ticker, position in open_positions.items():
            last_price_row = price_data[price_data["ticker"] == ticker].iloc[-1]
            exit_price = last_price_row["close"] * (1 - self.slippage_pct / 100)

            if position.direction == SignalDirection.BUY:
                gross_pnl = (exit_price - position.entry_price) * position.shares
            else:
                gross_pnl = (position.entry_price - exit_price) * position.shares

            exit_cost = self._calculate_trade_cost(exit_price, position.shares)
            net_pnl = gross_pnl - exit_cost

            position.exit_date = end_date
            position.exit_price = exit_price
            position.pnl = net_pnl
            position.pnl_pct = (net_pnl / (position.entry_price * position.shares)) * 100
            position.exit_reason = "end_of_backtest"

            trades.append(position)

        equity_curve = pd.DataFrame(equity_history)
        return trades, equity_curve

    def _calculate_trade_cost(self, price: float, shares: float) -> float:
        """Calculate transaction cost for a trade."""
        trade_value = price * shares
        spread_cost = trade_value * (self.spread_pct / 100) / 2
        slippage_cost = trade_value * (self.slippage_pct / 100)
        return self.commission + spread_cost + slippage_cost

    def _calculate_metrics(
        self,
        strategy_name: str,
        trades: list[BacktestTrade],
        equity_curve: pd.DataFrame,
        start_date: date,
        end_date: date,
        initial_capital: float,
        params: dict
    ) -> BacktestResult:
        """Calculate performance metrics from trades and equity curve."""
        if not trades or equity_curve.empty:
            return self._empty_result(strategy_name, start_date, end_date, initial_capital)

        final_value = equity_curve["equity"].iloc[-1]
        total_return_pct = ((final_value - initial_capital) / initial_capital) * 100

        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        if years > 0 and final_value > 0:
            annualized_return_pct = ((final_value / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return_pct = 0

        # Daily returns for Sharpe/Sortino
        equity_curve["daily_return"] = equity_curve["equity"].pct_change()
        daily_returns = equity_curve["daily_return"].dropna()

        if len(daily_returns) > 1:
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            neg_returns = daily_returns[daily_returns < 0]
            downside_std = neg_returns.std() if len(neg_returns) > 0 else std_return

            # Annualized Sharpe (assuming 252 trading days)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

            # Annualized standard deviation of returns
            return_std_pct = std_return * np.sqrt(252) * 100
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            return_std_pct = 0

        # Max drawdown
        equity_curve["cummax"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["cummax"]) / equity_curve["cummax"]
        max_drawdown_pct = abs(equity_curve["drawdown"].min()) * 100

        # Trade statistics
        total_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in trades)
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average hold time
        hold_days = [(t.exit_date - t.entry_date).days for t in trades if t.exit_date]
        avg_hold_days = sum(hold_days) / len(hold_days) if hold_days else 0

        return BacktestResult(
            strategy=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_days=avg_hold_days,
            return_std_pct=return_std_pct,
            trades=trades,
            equity_curve=equity_curve,
            params=params
        )

    def _empty_result(
        self,
        strategy_name: str,
        start_date: date,
        end_date: date,
        initial_capital: float
    ) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            strategy=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=initial_capital,
            total_return_pct=0,
            annualized_return_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown_pct=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            avg_trade_pnl=0,
            avg_win=0,
            avg_loss=0,
            avg_hold_days=0
        )

    def _store_result(self, result: BacktestResult) -> None:
        """Store backtest result in database."""
        try:
            # Handle infinity values for JSON serialization
            profit_factor = result.profit_factor
            if profit_factor == float('inf'):
                profit_factor = 999.99  # Cap at reasonable max

            self.db.execute("""
                INSERT INTO backtest_results (
                    strategy, start_date, end_date, initial_capital,
                    final_value, total_return_pct, annualized_return_pct,
                    sharpe_ratio, sortino_ratio, max_drawdown_pct,
                    win_rate, profit_factor, total_trades, avg_trade_pnl,
                    avg_win, avg_loss, avg_hold_days, params
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.strategy,
                result.start_date,
                result.end_date,
                result.initial_capital,
                result.final_value,
                result.total_return_pct,
                result.annualized_return_pct,
                result.sharpe_ratio,
                result.sortino_ratio,
                result.max_drawdown_pct,
                result.win_rate,
                profit_factor,
                result.total_trades,
                result.avg_trade_pnl,
                result.avg_win,
                result.avg_loss,
                result.avg_hold_days,
                json.dumps(result.params)
            ))
            logger.info(f"Stored backtest result for {result.strategy}")
        except Exception as e:
            logger.error(f"Error storing backtest result: {e}")

    def get_backtest_results(self, strategy: str | None = None) -> pd.DataFrame:
        """Get stored backtest results."""
        if strategy:
            return self.db.fetchdf(
                "SELECT * FROM backtest_results WHERE strategy = ? ORDER BY run_date DESC",
                (strategy,)
            )
        return self.db.fetchdf(
            "SELECT * FROM backtest_results ORDER BY run_date DESC"
        )

    def run_walk_forward(
        self,
        strategy: BaseStrategy,
        tickers: list[str],
        start_date: date,
        end_date: date,
        train_days: int = 180,
        test_days: int = 30,
        step_days: int = 30
    ) -> list[BacktestResult]:
        """
        Run walk-forward validation.

        Splits data into rolling train/test windows to avoid overfitting.
        """
        results = []
        current_start = start_date

        while current_start + timedelta(days=train_days + test_days) <= end_date:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            # For now, just run backtest on test period
            # In a full implementation, you'd optimize params on train, test on test
            result = self.run_backtest(
                strategy, tickers,
                start_date=train_end,
                end_date=test_end
            )
            results.append(result)

            current_start += timedelta(days=step_days)

        return results

    def run_portfolio_backtest(
        self,
        strategy_names: list[str],
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        portfolio_config: dict | None = None,
        initial_capital: float | None = None
    ) -> dict:
        """
        Run a multi-strategy portfolio backtest with concentration limits.

        Args:
            strategy_names: List of strategy names to include
            tickers: List of tickers to test on
            start_date: Start date for backtest
            end_date: End date for backtest
            portfolio_config: Optional config overrides for portfolio params
            initial_capital: Starting capital

        Returns:
            Dict with portfolio-level performance metrics
        """
        from .rsi_mean_reversion import RSIMeanReversionStrategy
        from .bollinger_squeeze import BollingerSqueezeStrategy
        from .macd_volume import MACDVolumeStrategy
        from .zscore_mean_reversion import ZScoreMeanReversionStrategy
        from .momentum_breakout import MomentumBreakoutStrategy
        from .gap_fade import GapFadeStrategy
        from .week52_low_bounce import Week52LowBounceStrategy
        from .sector_rotation import SectorRotationStrategy

        strategy_classes = {
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "bollinger_squeeze": BollingerSqueezeStrategy,
            "macd_volume": MACDVolumeStrategy,
            "zscore_mean_reversion": ZScoreMeanReversionStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
            "gap_fade": GapFadeStrategy,
            "week52_low_bounce": Week52LowBounceStrategy,
            "sector_rotation": SectorRotationStrategy,
        }

        if start_date is None:
            start_str = self.backtest_config.get("start_date", "2023-01-01")
            start_date = date.fromisoformat(start_str)

        if end_date is None:
            end_str = self.backtest_config.get("end_date")
            end_date = date.fromisoformat(end_str) if end_str else date.today()

        if initial_capital is None:
            initial_capital = self.initial_capital

        # Get portfolio constraints from config or overrides
        if portfolio_config:
            conf_scaling = portfolio_config.get("confidence_scaling", {})
            risk_mgmt = portfolio_config.get("risk_management", {})
        else:
            conf_scaling = self.config.get("confidence_scaling", {})
            risk_mgmt = self.config.get("risk_management", {})

        base_size_pct = conf_scaling.get("base_size_pct", 5.0)
        max_position_pct = risk_mgmt.get("max_position_size_pct", 15.0)
        max_per_strategy_pct = risk_mgmt.get("max_per_strategy_pct", 65.0)
        max_exposure_pct = risk_mgmt.get("max_portfolio_exposure_pct", 80.0)

        # Load strategies
        strategies = []
        for name in strategy_names:
            if name in strategy_classes:
                strat_config = self.config.get("strategies", {}).get(name, {})
                strat_config["enabled"] = True
                try:
                    strategies.append(strategy_classes[name](strat_config))
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

        if not strategies:
            return {"error": "No strategies loaded", "sharpe_ratio": 0, "total_return_pct": 0, "max_drawdown_pct": 0}

        # Get price data
        price_data = self.data_ingestion.get_daily_prices(
            tickers=tickers,
            start_date=start_date - timedelta(days=90),
            end_date=end_date
        )

        if price_data.empty:
            return {"error": "No price data", "sharpe_ratio": 0, "total_return_pct": 0, "max_drawdown_pct": 0}

        # Run combined backtest
        trades = []
        open_positions = {}
        capital = initial_capital
        equity_history = []
        strategy_exposure = {s.name: 0.0 for s in strategies}

        # Get trading days
        all_dates = sorted(price_data["date"].unique())
        trading_dates = []
        for d in all_dates:
            try:
                if isinstance(d, date) and not isinstance(d, datetime):
                    dt = d
                elif hasattr(d, 'date'):
                    dt = d.date()
                else:
                    dt = date.fromisoformat(str(d)[:10])
                if start_date <= dt <= end_date:
                    trading_dates.append(dt)
            except:
                continue

        for current_date in trading_dates:
            date_str = current_date.isoformat()
            day_prices = price_data[price_data["date"] == date_str]
            price_dict = {row["ticker"]: row for _, row in day_prices.iterrows()}

            # Update open positions (check exits)
            closed_today = []
            for ticker, pos in list(open_positions.items()):
                if ticker not in price_dict:
                    continue
                row = price_dict[ticker]
                high, low, close = row["high"], row["low"], row["close"]

                exit_price = None
                exit_reason = None

                if pos["direction"] == "BUY":
                    if low <= pos["stop"]:
                        exit_price, exit_reason = pos["stop"], "stop"
                    elif high >= pos["target"]:
                        exit_price, exit_reason = pos["target"], "target"
                else:
                    if high >= pos["stop"]:
                        exit_price, exit_reason = pos["stop"], "stop"
                    elif low <= pos["target"]:
                        exit_price, exit_reason = pos["target"], "target"

                if exit_price:
                    pnl = (exit_price - pos["entry"]) * pos["shares"]
                    if pos["direction"] == "SELL":
                        pnl = -pnl
                    capital += pos["value"] + pnl
                    strategy_exposure[pos["strategy"]] -= pos["pct"]
                    trades.append({
                        "ticker": ticker, "entry": pos["entry"], "exit": exit_price,
                        "pnl": pnl, "reason": exit_reason, "strategy": pos["strategy"]
                    })
                    closed_today.append(ticker)

            for t in closed_today:
                del open_positions[t]

            # Generate signals from all strategies
            all_signals = []
            for strategy in strategies:
                for ticker in tickers:
                    if ticker in open_positions or ticker not in price_dict:
                        continue
                    ticker_data = price_data[price_data["ticker"] == ticker].copy()
                    if len(ticker_data) < 60:
                        continue
                    ticker_data = ticker_data.sort_values("date")
                    try:
                        signals = strategy.generate_signals(ticker_data, ticker)
                        for sig in signals:
                            if sig.direction == SignalDirection.BUY:
                                all_signals.append(sig)
                    except:
                        continue

            # Open positions with constraints
            current_exposure = sum(p["pct"] for p in open_positions.values())
            for signal in sorted(all_signals, key=lambda s: s.confidence, reverse=True):
                if current_exposure >= max_exposure_pct:
                    break
                if strategy_exposure.get(signal.strategy, 0) >= max_per_strategy_pct:
                    continue
                if signal.ticker in open_positions:
                    continue

                pos_pct = min(base_size_pct, max_position_pct)
                pos_value = initial_capital * (pos_pct / 100)
                if signal.ticker in price_dict:
                    entry_price = price_dict[signal.ticker]["close"]
                    shares = pos_value / entry_price

                    open_positions[signal.ticker] = {
                        "entry": entry_price, "shares": shares, "value": pos_value,
                        "pct": pos_pct, "direction": "BUY", "strategy": signal.strategy,
                        "stop": signal.stop_price, "target": signal.target_price
                    }
                    capital -= pos_value
                    current_exposure += pos_pct
                    strategy_exposure[signal.strategy] = strategy_exposure.get(signal.strategy, 0) + pos_pct

            # Record equity
            pos_value = sum(
                price_dict.get(t, {}).get("close", p["entry"]) * p["shares"]
                for t, p in open_positions.items()
                if t in price_dict
            )
            equity_history.append({"date": current_date, "equity": capital + pos_value})

        # Calculate metrics
        if not equity_history:
            return {"sharpe_ratio": 0, "total_return_pct": 0, "max_drawdown_pct": 0}

        equity_df = pd.DataFrame(equity_history)
        final_value = equity_df["equity"].iloc[-1]
        total_return_pct = ((final_value - initial_capital) / initial_capital) * 100

        # Drawdown
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
        max_drawdown_pct = equity_df["drawdown"].min()

        # Sharpe
        equity_df["daily_return"] = equity_df["equity"].pct_change()
        mean_return = equity_df["daily_return"].mean() * 252
        std_return = equity_df["daily_return"].std() * np.sqrt(252)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0

        # Win rate
        wins = len([t for t in trades if t["pnl"] > 0])
        win_rate = (wins / len(trades) * 100) if trades else 0

        return {
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "final_value": final_value,
            "total_trades": len(trades),
            "win_rate": win_rate,
        }
