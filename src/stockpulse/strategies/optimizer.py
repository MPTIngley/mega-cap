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

# Portfolio-level parameter search space
PORTFOLIO_PARAM_SEARCH_SPACE = {
    # Position sizing
    "base_size_pct": [3.0, 5.0, 7.5, 10.0],
    "max_position_size_pct": [10.0, 15.0, 20.0],
    "min_position_size_pct": [2.0, 3.0, 5.0],

    # Confidence multipliers
    "confidence_75_multiplier": [1.5, 2.0, 2.5],
    "confidence_85_multiplier": [2.0, 2.5, 3.0],

    # Concentration limits
    "max_per_strategy_pct": [40.0, 50.0, 65.0, 80.0],
    "max_sector_concentration_pct": [40.0, 50.0, 65.0, 80.0],
    "max_portfolio_exposure_pct": [60.0, 70.0, 80.0, 90.0],

    # Risk management
    "max_drawdown_disable_pct": [10.0, 15.0, 20.0],
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

    def save_detailed_results(self, results: dict[str, OptimizationResult], output_dir: str | None = None):
        """
        Save detailed optimization results to JSON for later analysis.

        Args:
            results: Dict of optimization results
            output_dir: Directory to save results (default: data/)
        """
        import json
        from pathlib import Path
        from datetime import datetime

        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "data"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"optimization_results_{timestamp}.json"

        # Build detailed results structure
        detailed = {
            "timestamp": datetime.now().isoformat(),
            "max_drawdown_constraint": self.max_drawdown_pct,
            "strategies": {}
        }

        for strategy_name, result in results.items():
            detailed["strategies"][strategy_name] = {
                "best_params": result.best_params,
                "best_return_pct": result.best_return,
                "best_sharpe": result.best_sharpe,
                "best_drawdown_pct": result.best_drawdown,
                "final_value": result.final_value,
                "return_std_pct": result.return_std_pct,
                "constraint_satisfied": result.constraint_satisfied,
                "optimization_time_seconds": result.optimization_time_seconds,
                "all_results": [
                    {
                        "params": r["params"],
                        "total_return_pct": r["total_return_pct"],
                        "sharpe_ratio": r["sharpe_ratio"],
                        "max_drawdown_pct": r["max_drawdown_pct"],
                        "final_value": r.get("final_value", 100000),
                        "return_std_pct": r.get("return_std_pct", 0),
                        "total_trades": r["total_trades"],
                        "win_rate": r["win_rate"],
                        "meets_constraint": r["meets_constraint"],
                        "score": r["score"] if r["score"] != float("-inf") else None
                    }
                    for r in result.all_results
                ]
            }

        with open(output_file, "w") as f:
            json.dump(detailed, f, indent=2, default=str)

        logger.info(f"Saved detailed results to {output_file}")
        return output_file

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


def get_portfolio_param_ranges() -> dict[str, list]:
    """Get portfolio-level parameter search ranges."""
    return PORTFOLIO_PARAM_SEARCH_SPACE.copy()


class PortfolioOptimizer:
    """
    Optimize portfolio-level parameters like position sizing and concentration limits.

    Uses the backtester with multiple strategies to find optimal portfolio parameters.
    """

    def __init__(self, max_drawdown_pct: float = 25.0):
        """Initialize portfolio optimizer."""
        self.max_drawdown_pct = max_drawdown_pct
        self.db = get_db()
        self.config = get_config()
        self.backtester = Backtester()
        self.data_ingestion = DataIngestion()

    def optimize_portfolio_params(
        self,
        tickers: list[str],
        strategies: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        objective: str = "sharpe",
        max_iterations: int = 50,
        progress_callback: Callable[[int, int, dict], None] | None = None
    ) -> dict:
        """
        Optimize portfolio-level parameters.

        Args:
            tickers: Tickers to test on
            strategies: Strategy names to use (None = all enabled)
            start_date: Backtest start date
            end_date: Backtest end date
            objective: Optimization objective ("sharpe", "return", "sortino")
            max_iterations: Max parameter combinations to try
            progress_callback: Progress callback function

        Returns:
            Dict with best portfolio parameters and results
        """
        import time
        start_time = time.time()

        # Set dates
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 2)
        if end_date is None:
            end_date = date.today()

        # Get strategies to use
        if strategies is None:
            strategies = [name for name, cls in STRATEGY_CLASSES.items()
                         if self.config.get("strategies", {}).get(name, {}).get("enabled", True)]

        # Generate parameter combinations
        search_space = PORTFOLIO_PARAM_SEARCH_SPACE
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        # Limit iterations
        if len(all_combinations) > max_iterations:
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_iterations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]

        logger.info(f"Optimizing portfolio params with {len(all_combinations)} combinations")

        all_results = []
        best_result = None
        best_score = float("-inf")

        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))

            try:
                # Create temporary config with these portfolio params
                test_config = {
                    "confidence_scaling": {
                        "enabled": True,
                        "base_size_pct": params["base_size_pct"],
                        "confidence_75_multiplier": params["confidence_75_multiplier"],
                        "confidence_85_multiplier": params["confidence_85_multiplier"],
                    },
                    "risk_management": {
                        "max_position_size_pct": params["max_position_size_pct"],
                        "min_position_size_pct": params["min_position_size_pct"],
                        "max_per_strategy_pct": params["max_per_strategy_pct"],
                        "max_sector_concentration_pct": params["max_sector_concentration_pct"],
                        "max_portfolio_exposure_pct": params["max_portfolio_exposure_pct"],
                        "max_drawdown_disable_pct": params["max_drawdown_disable_pct"],
                    }
                }

                # Run multi-strategy backtest with these params
                result = self.backtester.run_portfolio_backtest(
                    strategy_names=strategies,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    portfolio_config=test_config
                )

                # Calculate score
                if objective == "sharpe":
                    score = result.get("sharpe_ratio", 0)
                elif objective == "return":
                    score = result.get("total_return_pct", 0)
                else:
                    score = result.get("sharpe_ratio", 0)

                # Check constraint
                meets_constraint = abs(result.get("max_drawdown_pct", 100)) <= self.max_drawdown_pct

                result_dict = {
                    "params": params,
                    "total_return_pct": result.get("total_return_pct", 0),
                    "sharpe_ratio": result.get("sharpe_ratio", 0),
                    "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                    "final_value": result.get("final_value", 100000),
                    "total_trades": result.get("total_trades", 0),
                    "win_rate": result.get("win_rate", 0),
                    "meets_constraint": meets_constraint,
                    "score": score if meets_constraint else float("-inf"),
                }
                all_results.append(result_dict)

                if meets_constraint and score > best_score:
                    best_score = score
                    best_result = result_dict

                if progress_callback:
                    progress_callback(i + 1, len(all_combinations), result_dict)

            except Exception as e:
                logger.warning(f"Error testing portfolio params {params}: {e}")
                continue

        elapsed = time.time() - start_time

        if best_result is None and all_results:
            all_results.sort(key=lambda x: abs(x.get("max_drawdown_pct", 100)))
            best_result = all_results[0]

        return {
            "best_params": best_result["params"] if best_result else {},
            "best_return_pct": best_result.get("total_return_pct", 0) if best_result else 0,
            "best_sharpe": best_result.get("sharpe_ratio", 0) if best_result else 0,
            "best_drawdown_pct": best_result.get("max_drawdown_pct", 0) if best_result else 0,
            "all_results": all_results,
            "optimization_time_seconds": elapsed,
            "strategies_tested": strategies,
        }

    def save_portfolio_params(self, best_params: dict, config_path: str | None = None):
        """Save optimized portfolio parameters to config."""
        import yaml
        from pathlib import Path

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Update confidence_scaling
        if "confidence_scaling" not in config:
            config["confidence_scaling"] = {}
        config["confidence_scaling"]["base_size_pct"] = best_params.get("base_size_pct", 5.0)
        config["confidence_scaling"]["confidence_75_multiplier"] = best_params.get("confidence_75_multiplier", 2.0)
        config["confidence_scaling"]["confidence_85_multiplier"] = best_params.get("confidence_85_multiplier", 3.0)

        # Update risk_management
        if "risk_management" not in config:
            config["risk_management"] = {}
        config["risk_management"]["max_position_size_pct"] = best_params.get("max_position_size_pct", 15.0)
        config["risk_management"]["min_position_size_pct"] = best_params.get("min_position_size_pct", 3.0)
        config["risk_management"]["max_per_strategy_pct"] = best_params.get("max_per_strategy_pct", 65.0)
        config["risk_management"]["max_sector_concentration_pct"] = best_params.get("max_sector_concentration_pct", 65.0)
        config["risk_management"]["max_portfolio_exposure_pct"] = best_params.get("max_portfolio_exposure_pct", 80.0)
        config["risk_management"]["max_drawdown_disable_pct"] = best_params.get("max_drawdown_disable_pct", 15.0)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved optimized portfolio params to {config_path}")


def get_strategy_params(strategy_name: str) -> dict[str, Any]:
    """Get current configured parameters for a strategy."""
    config = get_config()
    strategy_config = config.get("strategies", {}).get(strategy_name, {})
    return strategy_config
