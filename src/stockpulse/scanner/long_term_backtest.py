"""Long-term Scanner Backtesting Framework.

Evaluates the long-term scanner's scoring system using historical data
with a 3-year buy-and-hold strategy.

Usage:
    python -m stockpulse.scanner.long_term_backtest
    stockpulse longterm-backtest
"""

from datetime import date, timedelta
from typing import Any
import itertools
import time

import pandas as pd
import numpy as np
import yfinance as yf

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db

logger = get_logger(__name__)


class LongTermBacktester:
    """
    Backtests long-term scanner scoring system.

    Strategy:
    - At each evaluation point (monthly), score all stocks
    - Buy stocks that meet threshold (score >= min_score)
    - Hold for 3 years (or until end of data)
    - Measure total return vs SPY buy-and-hold

    Note: This requires 5+ years of historical data to properly evaluate
    a 3-year holding period with sufficient lookback.
    """

    HOLD_PERIOD_YEARS = 3
    EVALUATION_FREQUENCY_MONTHS = 3  # Quarterly evaluation

    def __init__(self, db_path: str | None = None):
        """Initialize backtester."""
        self.config = get_config()
        self.db = get_db()

        # Historical data cache
        self._price_cache = {}
        self._fundamentals_cache = {}

    def fetch_extended_history(
        self,
        tickers: list[str],
        years: int = 6,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Fetch extended historical price data for backtesting.

        Args:
            tickers: List of tickers to fetch
            years: Number of years of history (default 6 for 3-year hold + 3-year lookback)
            progress: Show progress bar

        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, volume
        """
        logger.info(f"Fetching {years} years of history for {len(tickers)} tickers...")

        all_data = []
        start_date = date.today() - timedelta(days=years * 365)

        for i, ticker in enumerate(tickers):
            if progress and i % 10 == 0:
                print(f"  Fetching {i}/{len(tickers)}: {ticker}...", end="\r")

            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=start_date, end=date.today())

                if hist.empty:
                    continue

                hist = hist.reset_index()
                hist["ticker"] = ticker
                hist = hist.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })

                all_data.append(hist[["ticker", "date", "open", "high", "low", "close", "volume"]])

            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")

        if progress:
            print(f"  Fetched {len(all_data)} tickers with data.          ")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Cache for later use
        for ticker in df["ticker"].unique():
            self._price_cache[ticker] = df[df["ticker"] == ticker].copy()

        return df

    def calculate_historical_score(
        self,
        ticker: str,
        as_of_date: date,
        weights: dict
    ) -> float | None:
        """
        Calculate what the long-term score would have been on a historical date.

        This is a simplified version that uses available price data to estimate
        the score components that don't require live API calls.

        Args:
            ticker: Stock ticker
            as_of_date: Historical date to calculate score for
            weights: Scoring weights to use

        Returns:
            Composite score (0-100) or None if insufficient data
        """
        if ticker not in self._price_cache:
            return None

        price_df = self._price_cache[ticker]
        price_df = price_df[price_df["date"] <= as_of_date].copy()

        # Need at least 252 trading days (1 year) of history before as_of_date
        if len(price_df) < 252:
            return None

        # Get last 252 days for scoring
        price_data = price_df.tail(252).copy()

        # Calculate available scores
        valuation_score = self._historical_valuation_score(ticker, as_of_date)
        technical_score = self._historical_technical_score(price_data)
        dividend_score = 50  # Default - would need historical dividend data
        quality_score = 50  # Default - would need historical fundamentals

        # For historical backtesting, we can't reliably get insider/FCF/earnings data
        # so we use neutral scores (50) for those components
        insider_score = 50
        fcf_yield_score = 50
        earnings_momentum_score = 50
        peer_valuation_score = 50

        # Calculate composite with given weights
        w = weights
        composite = (
            valuation_score * w.get("valuation", 0.15) +
            technical_score * w.get("technical", 0.15) +
            dividend_score * w.get("dividend", 0.10) +
            quality_score * w.get("quality", 0.15) +
            insider_score * w.get("insider", 0.15) +
            fcf_yield_score * w.get("fcf_yield", 0.12) +
            earnings_momentum_score * w.get("earnings_momentum", 0.10) +
            peer_valuation_score * w.get("peer_valuation", 0.08)
        )

        return composite

    def _historical_valuation_score(self, ticker: str, as_of_date: date) -> float:
        """Estimate historical valuation score from price data."""
        # This is a simplified proxy - uses price momentum as valuation indicator
        # In a full implementation, you'd need historical P/E, P/B data
        score = 50

        if ticker not in self._price_cache:
            return score

        df = self._price_cache[ticker]
        df = df[df["date"] <= as_of_date].tail(252)

        if len(df) < 252:
            return score

        current_price = df["close"].iloc[-1]
        year_ago_price = df["close"].iloc[0]

        # Negative 1-year return often indicates value opportunity
        one_year_return = (current_price - year_ago_price) / year_ago_price

        if one_year_return < -0.30:  # Down 30%+
            score += 25
        elif one_year_return < -0.20:
            score += 15
        elif one_year_return < -0.10:
            score += 10
        elif one_year_return > 0.50:  # Up 50%+ - potentially overvalued
            score -= 15
        elif one_year_return > 0.30:
            score -= 10

        return max(0, min(100, score))

    def _historical_technical_score(self, price_data: pd.DataFrame) -> float:
        """Calculate technical score from historical price data."""
        score = 50

        if price_data.empty:
            return score

        current_price = price_data["close"].iloc[-1]

        # 52-week range position
        low_52w = price_data["low"].min()
        high_52w = price_data["high"].max()
        range_52w = high_52w - low_52w

        if range_52w > 0:
            position_in_range = (current_price - low_52w) / range_52w

            if position_in_range < 0.15:
                score += 25
            elif position_in_range < 0.25:
                score += 15
            elif position_in_range < 0.35:
                score += 5
            elif position_in_range > 0.90:
                score -= 10

        # Price vs moving averages
        sma_50 = price_data["close"].rolling(50).mean().iloc[-1]
        sma_200 = price_data["close"].rolling(200).mean().iloc[-1]

        if pd.notna(sma_200):
            if current_price < sma_200 * 0.95:
                score += 10
            elif current_price < sma_200:
                score += 5

        # RSI
        delta = price_data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if pd.notna(current_rsi):
            if current_rsi < 30:
                score += 15
            elif current_rsi < 40:
                score += 5

        return max(0, min(100, score))

    def run_backtest(
        self,
        tickers: list[str],
        weights: dict,
        min_score: float = 60,
        start_year: int = 2019,
        end_year: int = 2023,
        initial_capital: float = 100000,
        max_positions: int = 20,
        position_size_pct: float = 5.0
    ) -> dict:
        """
        Run backtest with given weights and parameters.

        Args:
            tickers: Universe of tickers to consider
            weights: Scoring weights dictionary
            min_score: Minimum score to buy (default 60)
            start_year: Year to start evaluation (default 2019)
            end_year: Year to end evaluation (default 2023)
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as % of capital

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running long-term backtest: {start_year}-{end_year}")

        # Track portfolio
        positions = []  # List of {ticker, buy_date, buy_price, sell_date, sell_price, return_pct}
        open_positions = {}  # ticker -> {buy_date, buy_price, shares}

        capital = initial_capital
        position_value = initial_capital * (position_size_pct / 100)

        # Generate evaluation dates (quarterly)
        eval_dates = []
        for year in range(start_year, end_year + 1):
            for month in [1, 4, 7, 10]:  # Quarterly
                eval_date = date(year, month, 1)
                eval_dates.append(eval_date)

        # Run through each evaluation date
        for eval_date in eval_dates:
            # Close positions held for 3+ years
            tickers_to_close = []
            for ticker, pos in open_positions.items():
                hold_days = (eval_date - pos["buy_date"]).days
                if hold_days >= self.HOLD_PERIOD_YEARS * 365:
                    tickers_to_close.append(ticker)

            for ticker in tickers_to_close:
                pos = open_positions[ticker]
                sell_price = self._get_price_on_date(ticker, eval_date)

                if sell_price:
                    return_pct = (sell_price - pos["buy_price"]) / pos["buy_price"] * 100
                    positions.append({
                        "ticker": ticker,
                        "buy_date": pos["buy_date"],
                        "buy_price": pos["buy_price"],
                        "sell_date": eval_date,
                        "sell_price": sell_price,
                        "return_pct": return_pct,
                        "hold_days": hold_days
                    })

                del open_positions[ticker]

            # Score all tickers and find new opportunities
            if len(open_positions) < max_positions:
                opportunities = []

                for ticker in tickers:
                    if ticker in open_positions:
                        continue

                    score = self.calculate_historical_score(ticker, eval_date, weights)
                    if score and score >= min_score:
                        opportunities.append((ticker, score))

                # Sort by score and buy top opportunities
                opportunities.sort(key=lambda x: x[1], reverse=True)

                for ticker, score in opportunities:
                    if len(open_positions) >= max_positions:
                        break

                    buy_price = self._get_price_on_date(ticker, eval_date)
                    if buy_price and buy_price > 0:
                        shares = position_value / buy_price
                        open_positions[ticker] = {
                            "buy_date": eval_date,
                            "buy_price": buy_price,
                            "shares": shares,
                            "score": score
                        }

        # Close any remaining positions at end
        end_date = date(end_year, 12, 31)
        for ticker, pos in open_positions.items():
            sell_price = self._get_price_on_date(ticker, end_date)
            if sell_price:
                return_pct = (sell_price - pos["buy_price"]) / pos["buy_price"] * 100
                hold_days = (end_date - pos["buy_date"]).days
                positions.append({
                    "ticker": ticker,
                    "buy_date": pos["buy_date"],
                    "buy_price": pos["buy_price"],
                    "sell_date": end_date,
                    "sell_price": sell_price,
                    "return_pct": return_pct,
                    "hold_days": hold_days
                })

        # Calculate statistics
        if not positions:
            return {
                "total_trades": 0,
                "avg_return_pct": 0,
                "win_rate": 0,
                "total_return_pct": 0,
                "spy_return_pct": 0,
                "alpha_pct": 0,
                "weights": weights,
            }

        returns = [p["return_pct"] for p in positions]
        wins = [r for r in returns if r > 0]

        # Get SPY benchmark return
        spy_start_price = self._get_price_on_date("SPY", date(start_year, 1, 1))
        spy_end_price = self._get_price_on_date("SPY", end_date)
        spy_return_pct = ((spy_end_price - spy_start_price) / spy_start_price * 100) if spy_start_price else 0

        # Calculate portfolio return (simplified - equal weight)
        avg_return = np.mean(returns)
        total_return = (1 + avg_return / 100) ** (len(eval_dates) / 4) - 1  # Rough annualization
        total_return_pct = total_return * 100

        return {
            "total_trades": len(positions),
            "avg_return_pct": round(np.mean(returns), 2),
            "median_return_pct": round(np.median(returns), 2),
            "win_rate": round(len(wins) / len(returns) * 100, 1),
            "best_trade_pct": round(max(returns), 2),
            "worst_trade_pct": round(min(returns), 2),
            "std_return_pct": round(np.std(returns), 2),
            "total_return_pct": round(total_return_pct, 2),
            "spy_return_pct": round(spy_return_pct, 2),
            "alpha_pct": round(avg_return - (spy_return_pct / (len(eval_dates) / 4)), 2),
            "avg_hold_days": round(np.mean([p["hold_days"] for p in positions]), 0),
            "weights": weights,
            "positions": positions[:10]  # Sample of trades
        }

    def _get_price_on_date(self, ticker: str, target_date: date) -> float | None:
        """Get closing price on or near a specific date."""
        if ticker not in self._price_cache:
            # Try to fetch
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(
                    start=target_date - timedelta(days=10),
                    end=target_date + timedelta(days=10)
                )
                if not hist.empty:
                    hist = hist.reset_index()
                    hist["ticker"] = ticker
                    hist = hist.rename(columns={"Date": "date", "Close": "close"})
                    hist["date"] = pd.to_datetime(hist["date"]).dt.date
                    self._price_cache[ticker] = hist
            except Exception:
                return None

        if ticker not in self._price_cache:
            return None

        df = self._price_cache[ticker]

        # Find closest date
        df = df[df["date"] <= target_date].tail(5)
        if df.empty:
            df = self._price_cache[ticker][self._price_cache[ticker]["date"] >= target_date].head(5)

        if df.empty:
            return None

        return df["close"].iloc[-1] if not df.empty else None

    def optimize_weights(
        self,
        tickers: list[str],
        max_iterations: int = 100,
        objective: str = "alpha"
    ) -> dict:
        """
        Optimize scoring weights to maximize backtest performance.

        Args:
            tickers: Universe of tickers
            max_iterations: Maximum optimization iterations
            objective: "alpha", "return", "sharpe", or "win_rate"

        Returns:
            Dictionary with best weights and results
        """
        logger.info(f"Optimizing long-term scanner weights (objective: {objective})...")

        # Weight search space
        weight_options = {
            "valuation": [0.10, 0.15, 0.20, 0.25],
            "technical": [0.10, 0.15, 0.20, 0.25],
            "dividend": [0.05, 0.10, 0.15],
            "quality": [0.10, 0.15, 0.20],
            "insider": [0.10, 0.15, 0.20, 0.25],
            "fcf_yield": [0.08, 0.12, 0.15, 0.18],
            "earnings_momentum": [0.05, 0.10, 0.15],
            "peer_valuation": [0.05, 0.08, 0.10, 0.12],
        }

        # Generate all combinations
        keys = list(weight_options.keys())
        all_values = [weight_options[k] for k in keys]
        all_combos = list(itertools.product(*all_values))

        # Filter to only those that sum to ~1.0
        valid_combos = []
        for combo in all_combos:
            total = sum(combo)
            if 0.95 <= total <= 1.05:
                valid_combos.append(combo)

        # Limit iterations
        if len(valid_combos) > max_iterations:
            np.random.seed(42)
            indices = np.random.choice(len(valid_combos), max_iterations, replace=False)
            valid_combos = [valid_combos[i] for i in indices]

        logger.info(f"Testing {len(valid_combos)} weight combinations...")

        best_result = None
        best_score = float("-inf")

        for i, combo in enumerate(valid_combos):
            weights = dict(zip(keys, combo))

            # Normalize to exactly 1.0
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            if i % 10 == 0:
                print(f"  Testing combination {i}/{len(valid_combos)}...", end="\r")

            try:
                result = self.run_backtest(tickers, weights)

                # Calculate objective score
                if objective == "alpha":
                    score = result["alpha_pct"]
                elif objective == "return":
                    score = result["avg_return_pct"]
                elif objective == "sharpe":
                    score = result["avg_return_pct"] / max(result["std_return_pct"], 1)
                elif objective == "win_rate":
                    score = result["win_rate"]
                else:
                    score = result["alpha_pct"]

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                logger.debug(f"Error testing weights: {e}")

        print(f"  Optimization complete.                              ")

        return {
            "best_weights": best_result["weights"] if best_result else {},
            "best_score": best_score,
            "best_result": best_result,
        }


def run_long_term_backtest():
    """Main entry point for long-term backtesting."""
    from stockpulse.data.universe import UniverseManager, TOP_US_STOCKS
    from stockpulse.scanner.long_term_scanner import LongTermScanner

    print("\n" + "=" * 70)
    print("  LONG-TERM SCANNER BACKTESTER")
    print("  Evaluating 3-year buy-and-hold strategy")
    print("=" * 70)

    # Get tickers
    universe = UniverseManager()
    tickers = universe.get_active_tickers()

    if not tickers:
        tickers = TOP_US_STOCKS[:50]

    print(f"\n  Universe: {len(tickers)} stocks")
    print(f"  Using: {', '.join(tickers[:10])}...")

    # Initialize backtester
    backtester = LongTermBacktester()

    # Fetch extended history
    print("\n  Fetching 6 years of historical data...")
    price_df = backtester.fetch_extended_history(tickers, years=6)
    print(f"  Loaded {len(price_df):,} price records")

    # Run backtest with default weights
    print("\n  Running backtest with default weights...")
    default_weights = LongTermScanner.DEFAULT_WEIGHTS

    result = backtester.run_backtest(
        tickers=tickers,
        weights=default_weights,
        min_score=60,
        start_year=2020,
        end_year=2023
    )

    print("\n" + "-" * 70)
    print("  DEFAULT WEIGHTS RESULTS")
    print("-" * 70)
    print(f"  Total Trades: {result['total_trades']}")
    print(f"  Avg Return/Trade: {result['avg_return_pct']:+.1f}%")
    print(f"  Win Rate: {result['win_rate']:.1f}%")
    print(f"  SPY Return (benchmark): {result['spy_return_pct']:+.1f}%")
    print(f"  Alpha: {result['alpha_pct']:+.1f}%")

    # Optimize weights
    print("\n  Optimizing weights (this may take a few minutes)...")
    opt_result = backtester.optimize_weights(
        tickers=tickers,
        max_iterations=50,
        objective="alpha"
    )

    if opt_result["best_result"]:
        print("\n" + "-" * 70)
        print("  OPTIMIZED WEIGHTS RESULTS")
        print("-" * 70)

        best = opt_result["best_result"]
        print(f"  Total Trades: {best['total_trades']}")
        print(f"  Avg Return/Trade: {best['avg_return_pct']:+.1f}%")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  SPY Return (benchmark): {best['spy_return_pct']:+.1f}%")
        print(f"  Alpha: {best['alpha_pct']:+.1f}%")

        print("\n  Optimized Weights:")
        for k, v in opt_result["best_weights"].items():
            print(f"    {k}: {v:.2f}")

        # Save to config
        print("\n  Saving optimized weights to config...")
        _save_optimized_weights(opt_result["best_weights"])

    print("\n" + "=" * 70 + "\n")


def _save_optimized_weights(weights: dict):
    """Save optimized weights to config file."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "long_term_scanner" not in config:
            config["long_term_scanner"] = {}

        config["long_term_scanner"]["weights"] = {
            k: round(v, 3) for k, v in weights.items()
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  ✓ Saved to {config_path}")

    except Exception as e:
        print(f"  ✗ Failed to save: {e}")


if __name__ == "__main__":
    run_long_term_backtest()
