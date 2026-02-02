"""
Hyperparameter Optimizer for Trading Strategies.

Uses grid search and Bayesian-like optimization to find optimal
parameters with constraints on maximum drawdown.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable
import itertools
import json

import numpy as np
import pandas as pd

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion

from .backtest import Backtester, BacktestResult
from .base import BaseStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy
from .macd_volume import MACDVolumeStrategy
from .zscore_mean_reversion import ZScoreMeanReversionStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .gap_fade import GapFadeStrategy
from .week52_low_bounce import Week52LowBounceStrategy
from .sector_rotation import SectorRotationStrategy

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization."""
    strategy_name: str
    best_params: dict[str, Any]
    best_return: float
    best_sharpe: float
    best_drawdown: float
    all_results: list[dict]
    constraint_satisfied: bool
    optimization_time_seconds: float
    final_value: float = 100000.0  # Final portfolio value
    return_std_pct: float = 0.0    # Annualized return std dev


# Define parameter search spaces for each strategy
PARAM_SEARCH_SPACES = {
    "rsi_mean_reversion": {
        "rsi_period": [10, 14, 20],
        "rsi_oversold": [20, 25, 30],
        "rsi_overbought": [70, 75, 80],
        "stop_loss_pct": [2.0, 3.0, 4.0, 5.0],
        "take_profit_pct": [6.0, 8.0, 10.0, 12.0],
        "min_confidence": [55, 60, 65, 70],
    },
    "bollinger_squeeze": {
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
        "squeeze_threshold": [0.02, 0.03, 0.04, 0.05],
        "stop_loss_pct": [2.0, 3.0, 4.0],
        "take_profit_pct": [8.0, 10.0, 12.0, 15.0],
        "min_confidence": [55, 60, 65, 70],
    },
    "macd_volume": {
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 30],
        "macd_signal": [7, 9, 11],
        "volume_threshold": [1.3, 1.5, 1.8, 2.0],
        "stop_loss_pct": [3.0, 4.0, 5.0],
        "take_profit_pct": [10.0, 12.0, 15.0],
        "min_confidence": [55, 60, 65],
    },
    "zscore_mean_reversion": {
        "lookback_period": [15, 20, 25, 30],
        "zscore_entry": [-2.0, -2.25, -2.5, -3.0],
        "zscore_exit": [0.0, 0.25, 0.5],
        "stop_loss_pct": [3.0, 4.0, 5.0, 6.0],
        "take_profit_pct": [6.0, 8.0, 10.0, 12.0],
        "min_confidence": [55, 60, 65],
    },
    "momentum_breakout": {
        "lookback_days": [10, 15, 20, 25],
        "breakout_threshold": [0.01, 0.015, 0.02, 0.025],
        "volume_confirmation": [1.3, 1.5, 1.8, 2.0],
        "stop_loss_pct": [3.0, 4.0, 5.0],
        "take_profit_pct": [8.0, 10.0, 12.0, 15.0],
        "min_confidence": [55, 60, 65],
    },
    "gap_fade": {
        "gap_threshold_pct": [1.0, 1.5, 2.0, 2.5],
        "max_gap_pct": [4.0, 5.0, 6.0],
        "volume_surge_threshold": [1.2, 1.5, 1.8],
        "stop_loss_pct": [2.0, 3.0, 4.0],
        "take_profit_pct": [3.0, 4.0, 5.0, 6.0],
        "min_confidence": [55, 60, 65],
    },
    "week52_low_bounce": {
        "low_threshold_pct": [8.0, 10.0, 12.0, 15.0],
        "bounce_threshold_pct": [1.5, 2.0, 3.0],
        "volume_surge": [1.2, 1.3, 1.5],
        "stop_loss_pct": [4.0, 5.0, 6.0, 8.0],
        "take_profit_pct": [10.0, 15.0, 20.0],
        "min_confidence": [55, 60, 65],
    },
    "sector_rotation": {
        "lookback_days": [10, 15, 20, 30],
        "top_sectors": [2, 3],
        "min_sector_return": [1.5, 2.0, 3.0],
        "relative_strength_threshold": [1.0, 1.1, 1.2],
        "stop_loss_pct": [3.0, 4.0, 5.0],
        "take_profit_pct": [8.0, 10.0, 12.0],
        "min_confidence": [55, 60, 65],
    },
}

STRATEGY_CLASSES = {
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "bollinger_squeeze": BollingerSqueezeStrategy,
    "macd_volume": MACDVolumeStrategy,
    "zscore_mean_reversion": ZScoreMeanReversionStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "gap_fade": GapFadeStrategy,
    "week52_low_bounce": Week52LowBounceStrategy,
    "sector_rotation": SectorRotationStrategy,
}


class StrategyOptimizer:
    """
    Optimize strategy parameters with drawdown constraints.

    Objective: Maximize returns (or Sharpe ratio)
    Constraint: Max drawdown <= specified limit (default 25%)
    """

    def __init__(self, max_drawdown_pct: float = 25.0):
        """
        Initialize optimizer.

        Args:
            max_drawdown_pct: Maximum allowed drawdown (default 25%)
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.db = get_db()
        self.config = get_config()
        self.backtester = Backtester()
        self.data_ingestion = DataIngestion()

    def optimize(
        self,
        strategy_name: str,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        objective: str = "sharpe",  # "sharpe", "return", or "sortino"
        max_iterations: int | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None
    ) -> OptimizationResult:
        """
        Optimize strategy parameters.

        Args:
            strategy_name: Name of strategy to optimize
            tickers: Tickers to test on
            start_date: Backtest start date
            end_date: Backtest end date
            objective: Optimization objective ("sharpe", "return", "sortino")
            max_iterations: Max parameter combinations to try (None = all)
            progress_callback: Called with (current, total, result) for progress updates

        Returns:
            OptimizationResult with best parameters
        """
        import time
        start_time = time.time()

        if strategy_name not in PARAM_SEARCH_SPACES:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        if strategy_name not in STRATEGY_CLASSES:
            raise ValueError(f"Strategy class not found: {strategy_name}")

        # Get search space
        search_space = PARAM_SEARCH_SPACES[strategy_name]
        strategy_class = STRATEGY_CLASSES[strategy_name]

        # Generate all parameter combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        # Limit iterations if specified
        if max_iterations and len(all_combinations) > max_iterations:
            # Random sample
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_iterations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]

        logger.info(f"Optimizing {strategy_name} with {len(all_combinations)} parameter combinations")

        # Set dates
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 2)  # 2 years
        if end_date is None:
            end_date = date.today()

        # Run optimizations
        all_results = []
        best_result = None
        best_score = float("-inf")

        for i, combo in enumerate(all_combinations):
            # Build params dict
            params = dict(zip(param_names, combo))
            params["enabled"] = True

            try:
                # Create strategy with these params
                strategy = strategy_class(params)

                # Run backtest
                result = self.backtester.run_backtest(
                    strategy=strategy,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date
                )

                # Calculate score based on objective
                if objective == "sharpe":
                    score = result.sharpe_ratio
                elif objective == "return":
                    score = result.total_return_pct
                elif objective == "sortino":
                    score = result.sortino_ratio
                else:
                    score = result.sharpe_ratio

                # Check drawdown constraint
                meets_constraint = abs(result.max_drawdown_pct) <= self.max_drawdown_pct

                result_dict = {
                    "params": params,
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "profit_factor": result.profit_factor,
                    "final_value": result.final_value,
                    "return_std_pct": result.return_std_pct,
                    "meets_constraint": meets_constraint,
                    "score": score if meets_constraint else float("-inf"),
                }
                all_results.append(result_dict)

                # Update best if this meets constraint and is better
                if meets_constraint and score > best_score:
                    best_score = score
                    best_result = result_dict

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(all_combinations), result_dict)

            except Exception as e:
                logger.warning(f"Error testing params {params}: {e}")
                continue

        elapsed = time.time() - start_time

        # Handle case where no params met constraint
        if best_result is None:
            # Find the one with lowest drawdown
            if all_results:
                all_results.sort(key=lambda x: abs(x["max_drawdown_pct"]))
                best_result = all_results[0]
                logger.warning(f"No params met {self.max_drawdown_pct}% drawdown constraint. "
                             f"Best drawdown: {best_result['max_drawdown_pct']:.2f}%")
            else:
                raise ValueError("No valid backtest results")

        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_result["params"],
            best_return=best_result["total_return_pct"],
            best_sharpe=best_result["sharpe_ratio"],
            best_drawdown=best_result["max_drawdown_pct"],
            all_results=all_results,
            constraint_satisfied=best_result.get("meets_constraint", False),
            optimization_time_seconds=elapsed,
            final_value=best_result.get("final_value", 100000.0),
            return_std_pct=best_result.get("return_std_pct", 0.0)
        )

    def optimize_all_strategies(
        self,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        objective: str = "sharpe",
        max_iterations_per_strategy: int = 100,
        progress_callback: Callable[[str, int, int], None] | None = None
    ) -> dict[str, OptimizationResult]:
        """
        Optimize all strategies.

        Args:
            tickers: Tickers to test on
            start_date: Backtest start date
            end_date: Backtest end date
            objective: Optimization objective
            max_iterations_per_strategy: Max combinations per strategy
            progress_callback: Called with (strategy_name, current, total)

        Returns:
            Dict mapping strategy name to OptimizationResult
        """
        results = {}

        for strategy_name in STRATEGY_CLASSES.keys():
            logger.info(f"Optimizing {strategy_name}...")

            try:
                result = self.optimize(
                    strategy_name=strategy_name,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    objective=objective,
                    max_iterations=max_iterations_per_strategy
                )
                results[strategy_name] = result

                if progress_callback:
                    progress_callback(strategy_name, len(results), len(STRATEGY_CLASSES))

            except Exception as e:
                logger.error(f"Failed to optimize {strategy_name}: {e}")
                continue

        return results

    def save_optimized_params(self, results: dict[str, OptimizationResult], config_path: str | None = None):
        """
        Save optimized parameters to config file.

        Args:
            results: Dict of optimization results
            config_path: Path to config file (default: config/config.yaml)
        """
        import yaml
        from pathlib import Path

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        # Read current config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Update strategy params
        if "strategies" not in config:
            config["strategies"] = {}

        for strategy_name, result in results.items():
            if strategy_name not in config["strategies"]:
                config["strategies"][strategy_name] = {}

            # Update params (keep enabled status)
            enabled = config["strategies"][strategy_name].get("enabled", True)
            config["strategies"][strategy_name].update(result.best_params)
            config["strategies"][strategy_name]["enabled"] = enabled

            # Add optimization metadata as comment
            config["strategies"][strategy_name]["_optimized"] = {
                "date": date.today().isoformat(),
                "return_pct": round(result.best_return, 2),
                "sharpe": round(result.best_sharpe, 2),
                "max_drawdown_pct": round(result.best_drawdown, 2),
            }

        # Write back
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved optimized params to {config_path}")

    def get_optimization_history(self) -> pd.DataFrame:
        """Get history of optimization runs from database."""
        return self.db.fetchdf("""
            SELECT * FROM optimization_runs
            ORDER BY run_date DESC
            LIMIT 100
        """)

    def store_optimization_result(self, result: OptimizationResult):
        """Store optimization result in database."""
        self.db.execute("""
            INSERT INTO optimization_runs
            (strategy, best_params, best_return_pct, best_sharpe, best_drawdown_pct,
             constraint_satisfied, optimization_time_seconds, run_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            result.strategy_name,
            json.dumps(result.best_params),
            result.best_return,
            result.best_sharpe,
            result.best_drawdown,
            result.constraint_satisfied,
            result.optimization_time_seconds
        ))


def get_param_ranges(strategy_name: str) -> dict[str, list]:
    """Get parameter search ranges for a strategy."""
    return PARAM_SEARCH_SPACES.get(strategy_name, {})


def get_strategy_params(strategy_name: str) -> dict[str, Any]:
    """Get current configured parameters for a strategy."""
    config = get_config()
    strategy_config = config.get("strategies", {}).get(strategy_name, {})
    return strategy_config
