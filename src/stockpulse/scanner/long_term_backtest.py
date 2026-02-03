"""Long-term Scanner Backtesting Framework.

Evaluates the long-term scanner's scoring system using historical data
with a 3-year buy-and-hold strategy.

Key constraints:
- Initial portfolio: $100,000
- Max position size at purchase: 10% of portfolio
- Position size per trade: 2% of initial capital ($2,000)
- No duplicate holdings (can't buy same stock twice until sold)
- Cooldown period: 6 months after selling before re-buying same stock
- If position appreciates above 10%, don't sell (let winners run)
- Hold period: 3 years (or until end of data)

Usage:
    python -m stockpulse.scanner.long_term_backtest
    stockpulse longterm-backtest
"""

from datetime import date, timedelta
from typing import Any
from dataclasses import dataclass, field
import itertools
import time
import json

import pandas as pd
import numpy as np
import yfinance as yf

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    ticker: str
    buy_date: date
    buy_price: float
    shares: float
    cost_basis: float
    score_at_purchase: float
    sell_date: date | None = None
    sell_price: float | None = None
    return_pct: float | None = None

    @property
    def is_open(self) -> bool:
        return self.sell_date is None

    def current_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_return_pct(self, price: float) -> float:
        return ((price - self.buy_price) / self.buy_price) * 100


@dataclass
class PortfolioState:
    """Tracks portfolio state during backtest."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)  # ticker -> Position
    closed_positions: list[Position] = field(default_factory=list)
    cooldown_until: dict[str, date] = field(default_factory=dict)  # ticker -> date can buy again
    transaction_log: list[dict] = field(default_factory=list)

    def total_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.current_value(prices.get(ticker, pos.buy_price))
            for ticker, pos in self.positions.items()
        )
        return self.cash + position_value

    def position_concentration(self, ticker: str, prices: dict[str, float]) -> float:
        """Calculate position concentration as % of total portfolio."""
        if ticker not in self.positions:
            return 0.0
        total = self.total_value(prices)
        if total <= 0:
            return 0.0
        pos_value = self.positions[ticker].current_value(prices.get(ticker, self.positions[ticker].buy_price))
        return (pos_value / total) * 100

    def can_buy(self, ticker: str, as_of_date: date) -> tuple[bool, str]:
        """Check if we can buy this ticker."""
        # Already holding?
        if ticker in self.positions:
            return False, "Already holding position"

        # In cooldown?
        if ticker in self.cooldown_until:
            if as_of_date < self.cooldown_until[ticker]:
                days_left = (self.cooldown_until[ticker] - as_of_date).days
                return False, f"Cooldown ({days_left} days left)"

        return True, ""


class LongTermBacktester:
    """
    Backtests long-term scanner scoring system.

    Strategy:
    - At each evaluation point (monthly), score all stocks
    - Buy stocks that meet threshold (score >= min_score)
    - Max 10% concentration per position at time of purchase
    - Position size: 2% of initial capital per trade
    - Hold for 3 years (or until end of data)
    - 6-month cooldown after selling before can re-buy

    Note: This requires 5+ years of historical data to properly evaluate
    a 3-year holding period with sufficient lookback.
    """

    # Backtest parameters
    INITIAL_CAPITAL = 100_000
    POSITION_SIZE_PCT = 2.0  # Each buy is 2% of initial capital
    MAX_POSITION_CONCENTRATION_PCT = 10.0  # Max 10% in any one stock at purchase
    HOLD_PERIOD_YEARS = 3
    COOLDOWN_MONTHS = 6
    EVALUATION_FREQUENCY_MONTHS = 1  # Monthly evaluation
    MIN_SCORE_DEFAULT = 60

    def __init__(self, db_path: str | None = None):
        """Initialize backtester."""
        self.config = get_config()
        self.db = get_db()

        # Historical data cache
        self._price_cache = {}
        self._fundamentals_cache = {}

        # Ensure table exists for caching historical data
        self._create_backtest_tables()

    def _create_backtest_tables(self):
        """Create tables for storing backtest historical data."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS backtest_prices (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weights TEXT,
                start_year INTEGER,
                end_year INTEGER,
                total_trades INTEGER,
                avg_return_pct REAL,
                win_rate REAL,
                total_return_pct REAL,
                spy_return_pct REAL,
                alpha_pct REAL,
                final_holdings TEXT,
                transaction_log TEXT
            )
        """)

    def fetch_extended_history(
        self,
        tickers: list[str],
        years: int = 6,
        progress: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch extended historical price data for backtesting.
        Saves to database for future use.

        Args:
            tickers: List of tickers to fetch
            years: Number of years of history
            progress: Show progress bar
            use_cache: Try to load from database first

        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, volume
        """
        start_date = date.today() - timedelta(days=years * 365)

        # Try to load from cache first
        if use_cache:
            cached_df = self._load_cached_prices(tickers, start_date)
            if not cached_df.empty:
                # Check if we have enough data
                tickers_with_data = cached_df["ticker"].unique()
                if len(tickers_with_data) >= len(tickers) * 0.8:  # 80% coverage
                    logger.info(f"Loaded {len(cached_df):,} cached price records for {len(tickers_with_data)} tickers")
                    for ticker in tickers_with_data:
                        self._price_cache[ticker] = cached_df[cached_df["ticker"] == ticker].copy()
                    return cached_df

        logger.info(f"Fetching {years} years of history for {len(tickers)} tickers...")

        all_data = []
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            if progress and i % 10 == 0:
                print(f"  Fetching {i}/{len(tickers)}: {ticker}...", end="\r")

            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=start_date, end=date.today())

                if hist.empty:
                    failed_tickers.append(ticker)
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

                df_ticker = hist[["ticker", "date", "open", "high", "low", "close", "volume"]].copy()
                all_data.append(df_ticker)

                # Cache in memory
                self._price_cache[ticker] = df_ticker

            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")
                failed_tickers.append(ticker)

        if progress:
            print(f"  Fetched {len(all_data)} tickers with data.          ")

        if failed_tickers:
            logger.info(f"Failed to fetch: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Save to database for future use
        self._save_prices_to_db(df)

        return df

    def _load_cached_prices(self, tickers: list[str], start_date: date) -> pd.DataFrame:
        """Load cached prices from database."""
        try:
            placeholders = ",".join(["?" for _ in tickers])
            df = self.db.fetchdf(f"""
                SELECT ticker, date, open, high, low, close, volume
                FROM backtest_prices
                WHERE ticker IN ({placeholders})
                AND date >= ?
                ORDER BY ticker, date
            """, (*tickers, start_date))
            return df
        except Exception as e:
            logger.debug(f"Error loading cached prices: {e}")
            return pd.DataFrame()

    def _save_prices_to_db(self, df: pd.DataFrame):
        """Save prices to database for caching."""
        try:
            records = df.to_dict("records")
            for record in records:
                self.db.execute("""
                    INSERT OR REPLACE INTO backtest_prices
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record["ticker"],
                    record["date"],
                    record.get("open"),
                    record.get("high"),
                    record.get("low"),
                    record.get("close"),
                    record.get("volume")
                ))
            logger.info(f"Saved {len(records):,} price records to database")
        except Exception as e:
            logger.warning(f"Error saving prices to database: {e}")

    def calculate_historical_score(
        self,
        ticker: str,
        as_of_date: date,
        weights: dict
    ) -> float | None:
        """
        Calculate what the long-term score would have been on a historical date.

        Uses available price data to estimate the score components.

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

        # For historical backtesting, we use neutral scores (50) for components
        # that require live API data
        dividend_score = 50
        quality_score = 50
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
        if len(price_data) >= 200:
            sma_200 = price_data["close"].rolling(200).mean().iloc[-1]
            if pd.notna(sma_200):
                if current_price < sma_200 * 0.95:
                    score += 10
                elif current_price < sma_200:
                    score += 5

        # RSI
        if len(price_data) >= 14:
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

    def _get_price_on_date(self, ticker: str, target_date: date) -> float | None:
        """Get closing price on or near a specific date."""
        if ticker not in self._price_cache:
            return None

        df = self._price_cache[ticker]

        # Find closest date on or before target
        df_before = df[df["date"] <= target_date]
        if not df_before.empty:
            return df_before["close"].iloc[-1]

        # Fallback: closest date after
        df_after = df[df["date"] >= target_date]
        if not df_after.empty:
            return df_after["close"].iloc[0]

        return None

    def _get_prices_on_date(self, tickers: list[str], target_date: date) -> dict[str, float]:
        """Get prices for multiple tickers on a date."""
        prices = {}
        for ticker in tickers:
            price = self._get_price_on_date(ticker, target_date)
            if price:
                prices[ticker] = price
        return prices

    def run_backtest(
        self,
        tickers: list[str],
        weights: dict,
        min_score: float | None = None,
        start_year: int = 2019,
        end_year: int = 2023,
    ) -> dict:
        """
        Run backtest with given weights and parameters.

        Args:
            tickers: Universe of tickers to consider
            weights: Scoring weights dictionary
            min_score: Minimum score to buy (default 60)
            start_year: Year to start evaluation
            end_year: Year to end evaluation

        Returns:
            Dictionary with backtest results including detailed holdings
        """
        if min_score is None:
            min_score = self.MIN_SCORE_DEFAULT

        logger.info(f"Running long-term backtest: {start_year}-{end_year}")

        # Initialize portfolio
        portfolio = PortfolioState(cash=self.INITIAL_CAPITAL)
        position_size_dollars = self.INITIAL_CAPITAL * (self.POSITION_SIZE_PCT / 100)

        # Generate evaluation dates (monthly)
        eval_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                try:
                    eval_date = date(year, month, 15)  # Mid-month
                    if eval_date <= date.today():
                        eval_dates.append(eval_date)
                except ValueError:
                    pass

        # Track metrics
        portfolio_values = []

        # Run through each evaluation date
        for eval_date in eval_dates:
            # Get current prices
            all_tickers = list(portfolio.positions.keys()) + tickers[:50]  # Limit for speed
            prices = self._get_prices_on_date(list(set(all_tickers)), eval_date)

            if not prices:
                continue

            # Record portfolio value
            portfolio_values.append({
                "date": eval_date,
                "value": portfolio.total_value(prices),
                "cash": portfolio.cash,
                "num_positions": len(portfolio.positions)
            })

            # === CLOSE POSITIONS held for 3+ years ===
            tickers_to_close = []
            for ticker, pos in portfolio.positions.items():
                hold_days = (eval_date - pos.buy_date).days
                if hold_days >= self.HOLD_PERIOD_YEARS * 365:
                    tickers_to_close.append(ticker)

            for ticker in tickers_to_close:
                pos = portfolio.positions[ticker]
                sell_price = prices.get(ticker)

                if sell_price:
                    # Close position
                    pos.sell_date = eval_date
                    pos.sell_price = sell_price
                    pos.return_pct = ((sell_price - pos.buy_price) / pos.buy_price) * 100

                    # Return cash
                    portfolio.cash += pos.shares * sell_price

                    # Set cooldown
                    cooldown_date = eval_date + timedelta(days=self.COOLDOWN_MONTHS * 30)
                    portfolio.cooldown_until[ticker] = cooldown_date

                    # Log transaction
                    portfolio.transaction_log.append({
                        "date": str(eval_date),
                        "action": "SELL",
                        "ticker": ticker,
                        "price": sell_price,
                        "shares": pos.shares,
                        "value": pos.shares * sell_price,
                        "return_pct": pos.return_pct,
                        "hold_days": (eval_date - pos.buy_date).days
                    })

                    portfolio.closed_positions.append(pos)
                    del portfolio.positions[ticker]

            # === OPEN NEW POSITIONS ===
            # Score all tickers and find opportunities
            opportunities = []

            for ticker in tickers:
                # Check if we can buy
                can_buy, reason = portfolio.can_buy(ticker, eval_date)
                if not can_buy:
                    continue

                # Check concentration limit BEFORE buying
                # Calculate what concentration would be after purchase
                current_total = portfolio.total_value(prices)
                if current_total > 0:
                    hypothetical_concentration = (position_size_dollars / current_total) * 100
                    if hypothetical_concentration > self.MAX_POSITION_CONCENTRATION_PCT:
                        continue

                # Get price
                price = prices.get(ticker)
                if not price or price <= 0:
                    continue

                # Calculate score
                score = self.calculate_historical_score(ticker, eval_date, weights)
                if score and score >= min_score:
                    opportunities.append((ticker, score, price))

            # Sort by score and buy top opportunities
            opportunities.sort(key=lambda x: x[1], reverse=True)

            for ticker, score, price in opportunities:
                # Check we have enough cash
                if portfolio.cash < position_size_dollars:
                    break

                # Double-check concentration (portfolio value may have changed)
                current_prices = prices.copy()
                current_prices[ticker] = price  # Add this ticker
                total_value = portfolio.total_value(current_prices)

                # Existing position check (shouldn't happen but safety)
                if ticker in portfolio.positions:
                    continue

                # Calculate shares to buy
                shares = position_size_dollars / price

                # Verify final concentration won't exceed limit
                new_position_value = shares * price
                if total_value > 0:
                    final_concentration = (new_position_value / (total_value + new_position_value)) * 100
                    if final_concentration > self.MAX_POSITION_CONCENTRATION_PCT:
                        continue

                # Execute purchase
                portfolio.cash -= new_position_value
                portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    buy_date=eval_date,
                    buy_price=price,
                    shares=shares,
                    cost_basis=new_position_value,
                    score_at_purchase=score
                )

                # Log transaction
                portfolio.transaction_log.append({
                    "date": str(eval_date),
                    "action": "BUY",
                    "ticker": ticker,
                    "price": price,
                    "shares": shares,
                    "value": new_position_value,
                    "score": score,
                    "cash_remaining": portfolio.cash
                })

        # === FINAL CLOSE: Close remaining positions at end ===
        end_date = date(end_year, 12, 31)
        final_prices = self._get_prices_on_date(list(portfolio.positions.keys()), end_date)

        for ticker, pos in list(portfolio.positions.items()):
            sell_price = final_prices.get(ticker, pos.buy_price)
            pos.sell_date = end_date
            pos.sell_price = sell_price
            pos.return_pct = ((sell_price - pos.buy_price) / pos.buy_price) * 100
            portfolio.cash += pos.shares * sell_price
            portfolio.closed_positions.append(pos)

        # === CALCULATE STATISTICS ===
        all_positions = portfolio.closed_positions

        if not all_positions:
            return self._empty_result(weights)

        returns = [p.return_pct for p in all_positions if p.return_pct is not None]
        wins = [r for r in returns if r > 0]

        # SPY benchmark
        spy_start = self._get_price_on_date("SPY", date(start_year, 1, 1))
        spy_end = self._get_price_on_date("SPY", end_date)
        spy_return_pct = ((spy_end - spy_start) / spy_start * 100) if spy_start and spy_end else 0

        # Portfolio return (actual)
        final_value = portfolio.cash
        total_return_pct = ((final_value - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL) * 100

        # Build detailed holdings report
        holdings_report = self._build_holdings_report(all_positions, portfolio_values)

        return {
            "total_trades": len(all_positions),
            "avg_return_pct": round(np.mean(returns), 2) if returns else 0,
            "median_return_pct": round(np.median(returns), 2) if returns else 0,
            "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else 0,
            "best_trade_pct": round(max(returns), 2) if returns else 0,
            "worst_trade_pct": round(min(returns), 2) if returns else 0,
            "std_return_pct": round(np.std(returns), 2) if returns else 0,
            "total_return_pct": round(total_return_pct, 2),
            "final_value": round(final_value, 2),
            "spy_return_pct": round(spy_return_pct, 2),
            "alpha_pct": round(total_return_pct - spy_return_pct, 2),
            "avg_hold_days": round(np.mean([(p.sell_date - p.buy_date).days for p in all_positions]), 0),
            "max_positions_held": max([pv["num_positions"] for pv in portfolio_values]) if portfolio_values else 0,
            "weights": weights,
            "holdings_report": holdings_report,
            "transaction_log": portfolio.transaction_log,
            "portfolio_values": portfolio_values[-12:],  # Last 12 months
        }

    def _empty_result(self, weights: dict) -> dict:
        """Return empty result structure."""
        return {
            "total_trades": 0,
            "avg_return_pct": 0,
            "win_rate": 0,
            "total_return_pct": 0,
            "final_value": self.INITIAL_CAPITAL,
            "spy_return_pct": 0,
            "alpha_pct": 0,
            "weights": weights,
            "holdings_report": {},
            "transaction_log": [],
        }

    def _build_holdings_report(
        self,
        positions: list[Position],
        portfolio_values: list[dict]
    ) -> dict:
        """Build detailed holdings report."""

        # Group by ticker
        by_ticker = {}
        for pos in positions:
            if pos.ticker not in by_ticker:
                by_ticker[pos.ticker] = []
            by_ticker[pos.ticker].append({
                "buy_date": str(pos.buy_date),
                "sell_date": str(pos.sell_date) if pos.sell_date else None,
                "buy_price": round(pos.buy_price, 2),
                "sell_price": round(pos.sell_price, 2) if pos.sell_price else None,
                "shares": round(pos.shares, 2),
                "cost_basis": round(pos.cost_basis, 2),
                "return_pct": round(pos.return_pct, 2) if pos.return_pct else None,
                "score_at_purchase": round(pos.score_at_purchase, 1),
            })

        # Best and worst performers
        sorted_by_return = sorted(positions, key=lambda p: p.return_pct or 0, reverse=True)
        best_performers = [
            {"ticker": p.ticker, "return_pct": round(p.return_pct, 1), "buy_date": str(p.buy_date)}
            for p in sorted_by_return[:5]
        ]
        worst_performers = [
            {"ticker": p.ticker, "return_pct": round(p.return_pct, 1), "buy_date": str(p.buy_date)}
            for p in sorted_by_return[-5:]
        ]

        # Portfolio growth over time
        if portfolio_values:
            growth_milestones = []
            for pv in portfolio_values[::6]:  # Every 6 months
                growth_milestones.append({
                    "date": str(pv["date"]),
                    "value": round(pv["value"], 0),
                    "positions": pv["num_positions"]
                })

        return {
            "positions_by_ticker": by_ticker,
            "best_performers": best_performers,
            "worst_performers": worst_performers,
            "total_unique_stocks": len(by_ticker),
            "growth_milestones": growth_milestones if portfolio_values else [],
        }

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
        all_results = []

        for i, combo in enumerate(valid_combos):
            weights = dict(zip(keys, combo))

            # Normalize to exactly 1.0
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            if i % 10 == 0:
                print(f"  Testing combination {i+1}/{len(valid_combos)}...", end="\r")

            try:
                result = self.run_backtest(tickers, weights)

                # Calculate objective score
                if objective == "alpha":
                    score = result["alpha_pct"]
                elif objective == "return":
                    score = result["total_return_pct"]
                elif objective == "sharpe":
                    std = result.get("std_return_pct", 1)
                    score = result["avg_return_pct"] / max(std, 1)
                elif objective == "win_rate":
                    score = result["win_rate"]
                else:
                    score = result["alpha_pct"]

                all_results.append({
                    "weights": weights,
                    "score": score,
                    "total_return_pct": result["total_return_pct"],
                    "alpha_pct": result["alpha_pct"],
                    "win_rate": result["win_rate"],
                })

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                logger.debug(f"Error testing weights: {e}")

        print(f"  Optimization complete.                              ")

        # Save best result to database
        if best_result:
            self._save_backtest_result(best_result)

        return {
            "best_weights": best_result["weights"] if best_result else {},
            "best_score": best_score,
            "best_result": best_result,
            "all_results": sorted(all_results, key=lambda x: x["score"], reverse=True)[:10],
        }

    def _save_backtest_result(self, result: dict):
        """Save backtest result to database."""
        try:
            self.db.execute("""
                INSERT INTO backtest_results
                (weights, start_year, end_year, total_trades, avg_return_pct,
                 win_rate, total_return_pct, spy_return_pct, alpha_pct,
                 final_holdings, transaction_log)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(result.get("weights", {})),
                2019,  # TODO: make dynamic
                2023,
                result.get("total_trades", 0),
                result.get("avg_return_pct", 0),
                result.get("win_rate", 0),
                result.get("total_return_pct", 0),
                result.get("spy_return_pct", 0),
                result.get("alpha_pct", 0),
                json.dumps(result.get("holdings_report", {})),
                json.dumps(result.get("transaction_log", [])[:50]),  # Limit size
            ))
        except Exception as e:
            logger.warning(f"Failed to save backtest result: {e}")


def run_long_term_backtest():
    """Main entry point for long-term backtesting."""
    from stockpulse.data.universe import UniverseManager, TOP_US_STOCKS
    from stockpulse.scanner.long_term_scanner import LongTermScanner

    print("\n" + "=" * 80)
    print("  LONG-TERM SCANNER BACKTESTER")
    print("  Strategy: 3-year buy-and-hold, 2% position size, 10% max concentration")
    print("=" * 80)

    # Get tickers
    universe = UniverseManager()
    tickers = universe.get_active_tickers()

    if not tickers:
        tickers = TOP_US_STOCKS[:50]

    print(f"\n  Universe: {len(tickers)} stocks")
    print(f"  Using: {', '.join(tickers[:10])}...")

    # Initialize backtester
    backtester = LongTermBacktester()

    # Fetch extended history (uses cache if available)
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

    _print_backtest_results("DEFAULT WEIGHTS", result)

    # Optimize weights
    print("\n  Optimizing weights (this may take a few minutes)...")
    opt_result = backtester.optimize_weights(
        tickers=tickers,
        max_iterations=50,
        objective="alpha"
    )

    if opt_result["best_result"]:
        _print_backtest_results("OPTIMIZED WEIGHTS", opt_result["best_result"])

        print("\n  Optimized Weights:")
        for k, v in opt_result["best_weights"].items():
            print(f"    {k}: {v:.3f}")

        # Print holdings detail
        _print_holdings_detail(opt_result["best_result"])

        # Save to config
        print("\n  Saving optimized weights to config...")
        _save_optimized_weights(opt_result["best_weights"])

    # Suggestions for improvement
    _print_suggestions()

    print("\n" + "=" * 80 + "\n")


def _print_backtest_results(title: str, result: dict):
    """Print formatted backtest results."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)
    print(f"  Initial Capital:     ${LongTermBacktester.INITIAL_CAPITAL:,.0f}")
    print(f"  Final Value:         ${result['final_value']:,.0f}")
    print(f"  Total Return:        {result['total_return_pct']:+.1f}%")
    print(f"  SPY Return:          {result['spy_return_pct']:+.1f}%")
    print(f"  Alpha vs SPY:        {result['alpha_pct']:+.1f}%")
    print(f"  ")
    print(f"  Total Trades:        {result['total_trades']}")
    print(f"  Win Rate:            {result['win_rate']:.1f}%")
    print(f"  Avg Return/Trade:    {result['avg_return_pct']:+.1f}%")
    print(f"  Best Trade:          {result.get('best_trade_pct', 0):+.1f}%")
    print(f"  Worst Trade:         {result.get('worst_trade_pct', 0):+.1f}%")
    print(f"  Avg Hold Days:       {result.get('avg_hold_days', 0):.0f}")
    print(f"  Max Positions Held:  {result.get('max_positions_held', 0)}")


def _print_holdings_detail(result: dict):
    """Print detailed holdings information."""
    report = result.get("holdings_report", {})

    print("\n" + "-" * 80)
    print("  HOLDINGS DETAIL")
    print("-" * 80)

    # Best performers
    print("\n  TOP 5 PERFORMERS:")
    for p in report.get("best_performers", [])[:5]:
        print(f"    {p['ticker']:6} {p['return_pct']:+6.1f}%  (bought {p['buy_date']})")

    # Worst performers
    print("\n  BOTTOM 5 PERFORMERS:")
    for p in report.get("worst_performers", [])[:5]:
        print(f"    {p['ticker']:6} {p['return_pct']:+6.1f}%  (bought {p['buy_date']})")

    # Growth milestones
    print("\n  PORTFOLIO GROWTH:")
    for m in report.get("growth_milestones", [])[:10]:
        print(f"    {m['date']}: ${m['value']:>10,.0f}  ({m['positions']} positions)")

    # Transaction sample
    txn_log = result.get("transaction_log", [])
    if txn_log:
        print("\n  RECENT TRANSACTIONS (last 10):")
        for txn in txn_log[-10:]:
            if txn["action"] == "BUY":
                print(f"    {txn['date']} BUY  {txn['ticker']:6} @ ${txn['price']:.2f} (score: {txn.get('score', 0):.0f})")
            else:
                print(f"    {txn['date']} SELL {txn['ticker']:6} @ ${txn['price']:.2f} ({txn.get('return_pct', 0):+.1f}%)")


def _print_suggestions():
    """Print suggestions for improving the backtester."""
    print("\n" + "-" * 80)
    print("  SUGGESTIONS FOR IMPROVEMENT")
    print("-" * 80)
    print("""
  1. BETTER DATA SOURCES:
     - Add historical fundamentals (P/E, P/B over time) for more accurate scoring
     - Integrate SEC 13F filings for historical institutional ownership
     - Add historical insider transactions from SEC Form 4

  2. ENHANCED SCORING:
     - Add sector momentum: rotate into sectors with relative strength
     - Add macro regime detection: risk-off in high VIX environments
     - Add earnings calendar: avoid buying before earnings

  3. PORTFOLIO OPTIMIZATION:
     - Implement rebalancing: trim winners that exceed concentration limits
     - Add stop-losses for positions down > 30%
     - Consider sector diversification constraints

  4. RISK MANAGEMENT:
     - Track drawdown and pause buying during severe drawdowns
     - Add correlation analysis to avoid concentrated sector bets
     - Consider volatility-adjusted position sizing

  5. VALIDATION:
     - Walk-forward optimization (train/test split)
     - Out-of-sample testing on different time periods
     - Monte Carlo simulation for robustness

  6. EXECUTION REALISM:
     - Add transaction costs (0.1% round-trip)
     - Model slippage on entry/exit
     - Consider market impact for large positions
    """)


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
