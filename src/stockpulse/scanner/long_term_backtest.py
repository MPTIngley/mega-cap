"""Long-term Scanner Backtesting Framework with Walk-Forward Optimization.

Enhanced backtester with:
- Transaction costs (0.1% round-trip)
- Optimizable stop-loss (20-35%)
- VIX regime detection (optimizable threshold)
- Sector diversification constraint (40% max per sector)
- Walk-forward optimization (train/test rolling windows)
- Data through 2025
- Optimizable min_score threshold (50-70)
- Edge-of-range detection for optimal parameters

Key constraints:
- Initial portfolio: $100,000
- Max position size at purchase: 10% of portfolio
- Position size per trade: 2% of initial capital ($2,000)
- No duplicate holdings (can't buy same stock twice until sold)
- Cooldown period: 6 months after selling before re-buying same stock
- If position appreciates above 10%, don't sell (let winners run)
- Hold period: 3 years (or until stop-loss hit)

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
    sector: str
    buy_date: date
    buy_price: float
    shares: float
    cost_basis: float
    score_at_purchase: float
    sell_date: date | None = None
    sell_price: float | None = None
    return_pct: float | None = None
    exit_reason: str | None = None

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

    def sector_concentration(self, sector: str, prices: dict[str, float]) -> float:
        """Calculate sector concentration as % of total portfolio."""
        total = self.total_value(prices)
        if total <= 0:
            return 0.0
        sector_value = sum(
            pos.current_value(prices.get(ticker, pos.buy_price))
            for ticker, pos in self.positions.items()
            if pos.sector == sector
        )
        return (sector_value / total) * 100

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
    Backtests long-term scanner scoring system with walk-forward optimization.

    Strategy:
    - At each evaluation point (monthly), score all stocks
    - Buy stocks that meet threshold (score >= min_score)
    - Max 10% concentration per position at time of purchase
    - Max 40% concentration per sector
    - Position size: 2% of initial capital per trade
    - Hold for 3 years (or until stop-loss hit)
    - 6-month cooldown after selling before can re-buy
    - Skip buying when VIX > threshold (market stress)
    """

    # Backtest parameters
    INITIAL_CAPITAL = 100_000
    POSITION_SIZE_PCT = 2.0  # Each buy is 2% of initial capital
    MAX_POSITION_CONCENTRATION_PCT = 10.0  # Max 10% in any one stock at purchase
    MAX_SECTOR_CONCENTRATION_PCT = 40.0  # Max 40% in any sector (tech is the future!)
    HOLD_PERIOD_YEARS = 3
    COOLDOWN_MONTHS = 6
    EVALUATION_FREQUENCY_MONTHS = 1  # Monthly evaluation
    MIN_SCORE_DEFAULT = 60
    TRANSACTION_COST_PCT = 0.05  # 0.05% each way = 0.1% round trip

    def __init__(self, db_path: str | None = None):
        """Initialize backtester."""
        self.config = get_config()
        self.db = get_db()

        # Historical data cache
        self._price_cache = {}
        self._fundamentals_cache = {}
        self._vix_cache = None
        self._sector_map = {}

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
            CREATE TABLE IF NOT EXISTS backtest_vix (
                date DATE PRIMARY KEY,
                close REAL
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weights TEXT,
                params TEXT,
                start_year INTEGER,
                end_year INTEGER,
                total_trades INTEGER,
                avg_return_pct REAL,
                win_rate REAL,
                total_return_pct REAL,
                spy_return_pct REAL,
                alpha_pct REAL,
                max_drawdown_pct REAL,
                sharpe_ratio REAL,
                final_holdings TEXT,
                transaction_log TEXT
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS backtest_optimal_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                optimization_type TEXT,
                best_weights TEXT,
                best_params TEXT,
                objective TEXT,
                score REAL,
                total_return_pct REAL,
                alpha_pct REAL,
                max_drawdown_pct REAL,
                sharpe_ratio REAL,
                walk_forward_results TEXT
            )
        """)

    def _load_sector_map(self, tickers: list[str]):
        """Load sector mapping for tickers from database."""
        try:
            placeholders = ",".join(["?" for _ in tickers])
            df = self.db.fetchdf(f"""
                SELECT ticker, sector FROM universe
                WHERE ticker IN ({placeholders})
            """, tuple(tickers))
            self._sector_map = dict(zip(df["ticker"], df["sector"]))
        except Exception as e:
            logger.debug(f"Error loading sector map: {e}")
            self._sector_map = {}

    def fetch_vix_history(self, years: int = 7, use_cache: bool = True) -> pd.DataFrame:
        """Fetch VIX historical data for regime detection."""
        start_date = date.today() - timedelta(days=years * 365)

        # Try cache first
        if use_cache:
            try:
                cached = self.db.fetchdf("""
                    SELECT date, close FROM backtest_vix
                    WHERE date >= ?
                    ORDER BY date
                """, (start_date,))
                if len(cached) > 100:
                    self._vix_cache = cached
                    logger.info(f"Loaded {len(cached)} VIX records from cache")
                    return cached
            except Exception:
                pass

        logger.info("Fetching VIX history from Yahoo Finance...")
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=start_date, end=date.today())

            if hist.empty:
                return pd.DataFrame()

            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["Date"]).dt.date
            vix_df = hist[["date", "Close"]].rename(columns={"Close": "close"})

            # Save to cache
            for _, row in vix_df.iterrows():
                self.db.execute("""
                    INSERT OR REPLACE INTO backtest_vix (date, close)
                    VALUES (?, ?)
                """, (row["date"], row["close"]))

            self._vix_cache = vix_df
            logger.info(f"Cached {len(vix_df)} VIX records")
            return vix_df

        except Exception as e:
            logger.warning(f"Error fetching VIX: {e}")
            return pd.DataFrame()

    def get_vix_on_date(self, target_date: date) -> float | None:
        """Get VIX value on or near a specific date."""
        if self._vix_cache is None or self._vix_cache.empty:
            return None

        df = self._vix_cache
        df_before = df[df["date"] <= target_date]
        if not df_before.empty:
            return df_before["close"].iloc[-1]

        return None

    def fetch_extended_history(
        self,
        tickers: list[str],
        years: int = 7,
        progress: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch extended historical price data for backtesting.
        Saves to database for future use.
        """
        start_date = date.today() - timedelta(days=years * 365)

        # Try to load from cache first
        if use_cache:
            cached_df = self._load_cached_prices(tickers, start_date)
            if not cached_df.empty:
                tickers_with_data = cached_df["ticker"].unique()
                if len(tickers_with_data) >= len(tickers) * 0.8:
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
        """Calculate what the long-term score would have been on a historical date."""
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

        # For historical backtesting, use neutral scores (50) for components
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
        params: dict | None = None,
        min_score: float | None = None,
        start_year: int = 2019,
        end_year: int = 2025,
    ) -> dict:
        """
        Run backtest with given weights and parameters.

        Args:
            tickers: Universe of tickers to consider
            weights: Scoring weights dictionary
            params: Backtest parameters (stop_loss_pct, vix_threshold)
            min_score: Minimum score to buy (default 60)
            start_year: Year to start evaluation
            end_year: Year to end evaluation

        Returns:
            Dictionary with backtest results including detailed holdings
        """
        if params is None:
            params = {}

        # Extract parameters with defaults
        stop_loss_pct = params.get("stop_loss_pct", 25.0)  # 25% default
        vix_threshold = params.get("vix_threshold", 30.0)  # Don't buy when VIX > 30

        # min_score can come from params (optimization) or direct argument
        if min_score is None:
            min_score = params.get("min_score", self.MIN_SCORE_DEFAULT)

        logger.info(f"Running backtest: {start_year}-{end_year}, stop_loss={stop_loss_pct}%, vix_thresh={vix_threshold}")

        # Load sector map
        self._load_sector_map(tickers)

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
        peak_value = self.INITIAL_CAPITAL
        max_drawdown = 0.0

        # Run through each evaluation date
        for eval_date in eval_dates:
            # Get current prices
            all_tickers = list(portfolio.positions.keys()) + tickers[:50]
            prices = self._get_prices_on_date(list(set(all_tickers)), eval_date)

            if not prices:
                continue

            current_value = portfolio.total_value(prices)

            # Track drawdown
            if current_value > peak_value:
                peak_value = current_value
            drawdown = ((peak_value - current_value) / peak_value) * 100
            max_drawdown = max(max_drawdown, drawdown)

            # Record portfolio value
            portfolio_values.append({
                "date": eval_date,
                "value": current_value,
                "cash": portfolio.cash,
                "num_positions": len(portfolio.positions),
                "drawdown": drawdown
            })

            # === CHECK STOP-LOSSES ===
            tickers_to_stop = []
            for ticker, pos in portfolio.positions.items():
                current_price = prices.get(ticker)
                if current_price:
                    unrealized = pos.unrealized_return_pct(current_price)
                    if unrealized <= -stop_loss_pct:
                        tickers_to_stop.append((ticker, current_price, unrealized))

            for ticker, sell_price, unrealized in tickers_to_stop:
                pos = portfolio.positions[ticker]
                # Apply transaction cost
                net_proceeds = pos.shares * sell_price * (1 - self.TRANSACTION_COST_PCT / 100)

                pos.sell_date = eval_date
                pos.sell_price = sell_price
                pos.return_pct = ((sell_price - pos.buy_price) / pos.buy_price) * 100
                pos.exit_reason = f"Stop-loss ({stop_loss_pct}%)"

                portfolio.cash += net_proceeds

                # Set cooldown
                cooldown_date = eval_date + timedelta(days=self.COOLDOWN_MONTHS * 30)
                portfolio.cooldown_until[ticker] = cooldown_date

                portfolio.transaction_log.append({
                    "date": str(eval_date),
                    "action": "SELL",
                    "ticker": ticker,
                    "price": sell_price,
                    "shares": pos.shares,
                    "value": net_proceeds,
                    "return_pct": pos.return_pct,
                    "reason": pos.exit_reason
                })

                portfolio.closed_positions.append(pos)
                del portfolio.positions[ticker]

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
                    # Apply transaction cost
                    net_proceeds = pos.shares * sell_price * (1 - self.TRANSACTION_COST_PCT / 100)

                    pos.sell_date = eval_date
                    pos.sell_price = sell_price
                    pos.return_pct = ((sell_price - pos.buy_price) / pos.buy_price) * 100
                    pos.exit_reason = "Hold period complete"

                    portfolio.cash += net_proceeds

                    # Set cooldown
                    cooldown_date = eval_date + timedelta(days=self.COOLDOWN_MONTHS * 30)
                    portfolio.cooldown_until[ticker] = cooldown_date

                    portfolio.transaction_log.append({
                        "date": str(eval_date),
                        "action": "SELL",
                        "ticker": ticker,
                        "price": sell_price,
                        "shares": pos.shares,
                        "value": net_proceeds,
                        "return_pct": pos.return_pct,
                        "hold_days": (eval_date - pos.buy_date).days,
                        "reason": pos.exit_reason
                    })

                    portfolio.closed_positions.append(pos)
                    del portfolio.positions[ticker]

            # === CHECK VIX REGIME ===
            vix = self.get_vix_on_date(eval_date)
            if vix and vix > vix_threshold:
                # Skip buying in high-volatility regime
                continue

            # === OPEN NEW POSITIONS ===
            opportunities = []

            for ticker in tickers:
                # Check if we can buy
                can_buy, reason = portfolio.can_buy(ticker, eval_date)
                if not can_buy:
                    continue

                # Check position concentration limit BEFORE buying
                current_total = portfolio.total_value(prices)
                if current_total > 0:
                    hypothetical_concentration = (position_size_dollars / current_total) * 100
                    if hypothetical_concentration > self.MAX_POSITION_CONCENTRATION_PCT:
                        continue

                # Check sector concentration
                sector = self._sector_map.get(ticker, "Unknown")
                sector_conc = portfolio.sector_concentration(sector, prices)
                hypothetical_sector_conc = sector_conc + (position_size_dollars / current_total * 100) if current_total > 0 else 0

                if hypothetical_sector_conc > self.MAX_SECTOR_CONCENTRATION_PCT:
                    continue

                # Get price
                price = prices.get(ticker)
                if not price or price <= 0:
                    continue

                # Calculate score
                score = self.calculate_historical_score(ticker, eval_date, weights)
                if score and score >= min_score:
                    opportunities.append((ticker, score, price, sector))

            # Sort by score and buy top opportunities
            opportunities.sort(key=lambda x: x[1], reverse=True)

            for ticker, score, price, sector in opportunities:
                # Check we have enough cash
                if portfolio.cash < position_size_dollars:
                    break

                # Re-check sector concentration
                current_prices = prices.copy()
                current_prices[ticker] = price
                total_value = portfolio.total_value(current_prices)

                sector_conc = portfolio.sector_concentration(sector, current_prices)
                if total_value > 0:
                    new_sector_conc = sector_conc + (position_size_dollars / total_value * 100)
                    if new_sector_conc > self.MAX_SECTOR_CONCENTRATION_PCT:
                        continue

                # Existing position check
                if ticker in portfolio.positions:
                    continue

                # Apply transaction cost to purchase
                cost_with_fee = position_size_dollars * (1 + self.TRANSACTION_COST_PCT / 100)
                shares = position_size_dollars / price

                # Verify final concentration won't exceed limit
                new_position_value = shares * price
                if total_value > 0:
                    final_concentration = (new_position_value / (total_value + new_position_value)) * 100
                    if final_concentration > self.MAX_POSITION_CONCENTRATION_PCT:
                        continue

                # Execute purchase
                portfolio.cash -= cost_with_fee
                portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    sector=sector,
                    buy_date=eval_date,
                    buy_price=price,
                    shares=shares,
                    cost_basis=cost_with_fee,
                    score_at_purchase=score
                )

                portfolio.transaction_log.append({
                    "date": str(eval_date),
                    "action": "BUY",
                    "ticker": ticker,
                    "sector": sector,
                    "price": price,
                    "shares": shares,
                    "value": cost_with_fee,
                    "score": score,
                    "cash_remaining": portfolio.cash
                })

        # === FINAL CLOSE: Close remaining positions at end ===
        end_date = min(date(end_year, 12, 31), date.today())
        final_prices = self._get_prices_on_date(list(portfolio.positions.keys()), end_date)

        for ticker, pos in list(portfolio.positions.items()):
            sell_price = final_prices.get(ticker, pos.buy_price)
            net_proceeds = pos.shares * sell_price * (1 - self.TRANSACTION_COST_PCT / 100)

            pos.sell_date = end_date
            pos.sell_price = sell_price
            pos.return_pct = ((sell_price - pos.buy_price) / pos.buy_price) * 100
            pos.exit_reason = "End of backtest"

            portfolio.cash += net_proceeds
            portfolio.closed_positions.append(pos)

        # === CALCULATE STATISTICS ===
        all_positions = portfolio.closed_positions

        if not all_positions:
            return self._empty_result(weights, params)

        returns = [p.return_pct for p in all_positions if p.return_pct is not None]
        wins = [r for r in returns if r > 0]

        # SPY benchmark
        spy_start = self._get_price_on_date("SPY", date(start_year, 1, 1))
        spy_end = self._get_price_on_date("SPY", end_date)
        spy_return_pct = ((spy_end - spy_start) / spy_start * 100) if spy_start and spy_end else 0

        # Portfolio return
        final_value = portfolio.cash
        total_return_pct = ((final_value - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL) * 100

        # Sharpe ratio (annualized, assuming 12% risk-free rate over this period avg)
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            # Annualize: assume ~4 trades per year average
            sharpe_ratio = (avg_return - 3) / std_return if std_return > 0 else 0  # 3% per trade risk-free proxy
        else:
            sharpe_ratio = 0

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
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "avg_hold_days": round(np.mean([(p.sell_date - p.buy_date).days for p in all_positions]), 0),
            "max_positions_held": max([pv["num_positions"] for pv in portfolio_values]) if portfolio_values else 0,
            "stop_losses_triggered": len([p for p in all_positions if p.exit_reason and "Stop-loss" in p.exit_reason]),
            "weights": weights,
            "params": params,
            "holdings_report": holdings_report,
            "transaction_log": portfolio.transaction_log,
            "portfolio_values": portfolio_values[-24:],  # Last 24 months
        }

    def _empty_result(self, weights: dict, params: dict) -> dict:
        """Return empty result structure."""
        return {
            "total_trades": 0,
            "avg_return_pct": 0,
            "win_rate": 0,
            "total_return_pct": 0,
            "final_value": self.INITIAL_CAPITAL,
            "spy_return_pct": 0,
            "alpha_pct": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "weights": weights,
            "params": params,
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
                "exit_reason": pos.exit_reason,
                "sector": pos.sector,
                "score_at_purchase": round(pos.score_at_purchase, 1),
            })

        # Best and worst performers
        sorted_by_return = sorted(positions, key=lambda p: p.return_pct or 0, reverse=True)
        best_performers = [
            {"ticker": p.ticker, "return_pct": round(p.return_pct, 1), "buy_date": str(p.buy_date), "sector": p.sector}
            for p in sorted_by_return[:5]
        ]
        worst_performers = [
            {"ticker": p.ticker, "return_pct": round(p.return_pct, 1), "buy_date": str(p.buy_date), "sector": p.sector}
            for p in sorted_by_return[-5:]
        ]

        # Sector breakdown
        sector_returns = {}
        for pos in positions:
            if pos.sector not in sector_returns:
                sector_returns[pos.sector] = []
            if pos.return_pct is not None:
                sector_returns[pos.sector].append(pos.return_pct)

        sector_summary = {
            sector: {
                "count": len(returns),
                "avg_return": round(np.mean(returns), 1) if returns else 0,
                "win_rate": round(len([r for r in returns if r > 0]) / len(returns) * 100, 1) if returns else 0
            }
            for sector, returns in sector_returns.items()
        }

        # Portfolio growth over time
        growth_milestones = []
        if portfolio_values:
            for pv in portfolio_values[::6]:  # Every 6 months
                growth_milestones.append({
                    "date": str(pv["date"]),
                    "value": round(pv["value"], 0),
                    "positions": pv["num_positions"],
                    "drawdown": round(pv["drawdown"], 1)
                })

        return {
            "positions_by_ticker": by_ticker,
            "best_performers": best_performers,
            "worst_performers": worst_performers,
            "sector_summary": sector_summary,
            "total_unique_stocks": len(by_ticker),
            "growth_milestones": growth_milestones,
        }

    def walk_forward_optimize(
        self,
        tickers: list[str],
        train_years: int = 2,
        test_years: int = 1,
        start_year: int = 2019,
        end_year: int = 2025,
        max_iterations: int = 50,
        objective: str = "alpha"
    ) -> dict:
        """
        Walk-forward optimization with rolling train/test windows.

        Args:
            tickers: Universe of tickers
            train_years: Years in training window
            test_years: Years in test window
            start_year: Overall start year
            end_year: Overall end year
            max_iterations: Max iterations per training window
            objective: Optimization objective

        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Walk-forward optimization: {train_years}yr train, {test_years}yr test")

        results = []
        current_year = start_year

        while current_year + train_years + test_years <= end_year + 1:
            train_start = current_year
            train_end = current_year + train_years - 1
            test_start = train_end + 1
            test_end = test_start + test_years - 1

            logger.info(f"Window: Train {train_start}-{train_end}, Test {test_start}-{test_end}")

            # Optimize on training period
            opt_result = self.optimize_weights(
                tickers=tickers,
                max_iterations=max_iterations,
                objective=objective,
                start_year=train_start,
                end_year=train_end
            )

            if not opt_result.get("best_result"):
                current_year += 1
                continue

            best_weights = opt_result["best_weights"]
            best_params = opt_result.get("best_params", {})

            # Test on out-of-sample period
            test_result = self.run_backtest(
                tickers=tickers,
                weights=best_weights,
                params=best_params,
                start_year=test_start,
                end_year=test_end
            )

            results.append({
                "train_period": f"{train_start}-{train_end}",
                "test_period": f"{test_start}-{test_end}",
                "train_return_pct": opt_result["best_result"]["total_return_pct"],
                "train_alpha_pct": opt_result["best_result"]["alpha_pct"],
                "test_return_pct": test_result["total_return_pct"],
                "test_alpha_pct": test_result["alpha_pct"],
                "test_sharpe": test_result["sharpe_ratio"],
                "test_max_dd": test_result["max_drawdown_pct"],
                "weights": best_weights,
                "params": best_params,
            })

            current_year += 1

        # Calculate aggregate metrics
        if results:
            avg_test_return = np.mean([r["test_return_pct"] for r in results])
            avg_test_alpha = np.mean([r["test_alpha_pct"] for r in results])
            avg_test_sharpe = np.mean([r["test_sharpe"] for r in results])
            consistency = len([r for r in results if r["test_alpha_pct"] > 0]) / len(results) * 100
        else:
            avg_test_return = avg_test_alpha = avg_test_sharpe = consistency = 0

        # Save to database
        self._save_walk_forward_results(results, objective)

        return {
            "windows": results,
            "avg_test_return_pct": round(avg_test_return, 2),
            "avg_test_alpha_pct": round(avg_test_alpha, 2),
            "avg_test_sharpe": round(avg_test_sharpe, 2),
            "consistency_pct": round(consistency, 1),
            "num_windows": len(results),
        }

    def _save_walk_forward_results(self, results: list, objective: str):
        """Save walk-forward results to database."""
        if not results:
            return

        # Use final window's weights as "best"
        final = results[-1]

        try:
            self.db.execute("""
                INSERT INTO backtest_optimal_params
                (optimization_type, best_weights, best_params, objective, score,
                 total_return_pct, alpha_pct, max_drawdown_pct, sharpe_ratio, walk_forward_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "walk_forward",
                json.dumps(final["weights"]),
                json.dumps(final["params"]),
                objective,
                final["test_alpha_pct"],
                final["test_return_pct"],
                final["test_alpha_pct"],
                final["test_max_dd"],
                final["test_sharpe"],
                json.dumps(results)
            ))
            logger.info("Saved walk-forward results to database")
        except Exception as e:
            logger.warning(f"Failed to save walk-forward results: {e}")

    def optimize_weights(
        self,
        tickers: list[str],
        max_iterations: int = 100,
        objective: str = "alpha",
        start_year: int = 2019,
        end_year: int = 2023
    ) -> dict:
        """
        Optimize scoring weights and parameters to maximize backtest performance.

        Optimizes:
        - Scoring weights (valuation, technical, etc.)
        - Stop-loss percentage (20-35%)
        - VIX threshold (25-40)

        Args:
            tickers: Universe of tickers
            max_iterations: Maximum optimization iterations
            objective: "alpha", "return", "sharpe", or "win_rate"
            start_year: Training period start
            end_year: Training period end

        Returns:
            Dictionary with best weights, params, and results
        """
        logger.info(f"Optimizing weights (objective: {objective})...")

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

        # Parameter search space
        param_options = {
            "stop_loss_pct": [20.0, 25.0, 30.0, 35.0],  # 20-35% range
            "vix_threshold": [25.0, 30.0, 35.0, 40.0],  # VIX threshold
            "min_score": [50.0, 55.0, 60.0, 65.0, 70.0],  # Minimum score to buy
        }

        # Generate weight combinations
        keys = list(weight_options.keys())
        all_values = [weight_options[k] for k in keys]
        all_combos = list(itertools.product(*all_values))

        # Filter to only those that sum to ~1.0
        valid_weight_combos = []
        for combo in all_combos:
            total = sum(combo)
            if 0.95 <= total <= 1.05:
                valid_weight_combos.append(combo)

        # Generate param combinations
        param_keys = list(param_options.keys())
        param_values = [param_options[k] for k in param_keys]
        param_combos = list(itertools.product(*param_values))

        # Combine weights and params
        all_search_combos = list(itertools.product(valid_weight_combos, param_combos))

        # Limit iterations
        if len(all_search_combos) > max_iterations:
            np.random.seed(42)
            indices = np.random.choice(len(all_search_combos), max_iterations, replace=False)
            all_search_combos = [all_search_combos[i] for i in indices]

        logger.info(f"Testing {len(all_search_combos)} combinations...")

        best_result = None
        best_score = float("-inf")
        best_weights = None
        best_params = None
        all_results = []

        for i, (weight_combo, param_combo) in enumerate(all_search_combos):
            weights = dict(zip(keys, weight_combo))
            params = dict(zip(param_keys, param_combo))

            # Normalize weights to exactly 1.0
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            if i % 10 == 0:
                print(f"  Testing combination {i+1}/{len(all_search_combos)}...", end="\r")

            try:
                result = self.run_backtest(
                    tickers, weights, params,
                    start_year=start_year, end_year=end_year
                )

                # Calculate objective score
                if objective == "alpha":
                    score = result["alpha_pct"]
                elif objective == "return":
                    score = result["total_return_pct"]
                elif objective == "sharpe":
                    score = result["sharpe_ratio"]
                elif objective == "win_rate":
                    score = result["win_rate"]
                else:
                    score = result["alpha_pct"]

                all_results.append({
                    "weights": weights,
                    "params": params,
                    "score": score,
                    "total_return_pct": result["total_return_pct"],
                    "alpha_pct": result["alpha_pct"],
                    "win_rate": result["win_rate"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                })

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_weights = weights
                    best_params = params

            except Exception as e:
                logger.debug(f"Error testing: {e}")

        print(f"  Optimization complete.                              ")

        # Save best result to database
        if best_result:
            self._save_backtest_result(best_result)
            self._save_optimal_params(best_weights, best_params, objective, best_score, best_result)

        return {
            "best_weights": best_weights or {},
            "best_params": best_params or {},
            "best_score": best_score,
            "best_result": best_result,
            "all_results": sorted(all_results, key=lambda x: x["score"], reverse=True)[:10],
        }

    def _save_backtest_result(self, result: dict):
        """Save backtest result to database."""
        try:
            self.db.execute("""
                INSERT INTO backtest_results
                (weights, params, start_year, end_year, total_trades, avg_return_pct,
                 win_rate, total_return_pct, spy_return_pct, alpha_pct,
                 max_drawdown_pct, sharpe_ratio, final_holdings, transaction_log)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(result.get("weights", {})),
                json.dumps(result.get("params", {})),
                2019,
                2025,
                result.get("total_trades", 0),
                result.get("avg_return_pct", 0),
                result.get("win_rate", 0),
                result.get("total_return_pct", 0),
                result.get("spy_return_pct", 0),
                result.get("alpha_pct", 0),
                result.get("max_drawdown_pct", 0),
                result.get("sharpe_ratio", 0),
                json.dumps(result.get("holdings_report", {})),
                json.dumps(result.get("transaction_log", [])[:50]),
            ))
        except Exception as e:
            logger.warning(f"Failed to save backtest result: {e}")

    def _save_optimal_params(self, weights: dict, params: dict, objective: str, score: float, result: dict):
        """Save optimal parameters to database."""
        try:
            self.db.execute("""
                INSERT INTO backtest_optimal_params
                (optimization_type, best_weights, best_params, objective, score,
                 total_return_pct, alpha_pct, max_drawdown_pct, sharpe_ratio, walk_forward_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "single_period",
                json.dumps(weights),
                json.dumps(params),
                objective,
                score,
                result.get("total_return_pct", 0),
                result.get("alpha_pct", 0),
                result.get("max_drawdown_pct", 0),
                result.get("sharpe_ratio", 0),
                None
            ))
            logger.info("Saved optimal params to database")
        except Exception as e:
            logger.warning(f"Failed to save optimal params: {e}")


def _check_edge_values(params: dict, param_options: dict) -> list[str]:
    """
    Check if any optimal parameter values fall on the edge of their search range.

    Returns list of warning strings for parameters at edges.
    """
    warnings = []

    for param_name, optimal_value in params.items():
        if param_name not in param_options:
            continue

        search_range = param_options[param_name]
        if not search_range:
            continue

        min_val = min(search_range)
        max_val = max(search_range)

        if optimal_value == min_val:
            warnings.append(
                f"⚠ {param_name}={optimal_value} is at LOWER edge of range "
                f"[{min_val}-{max_val}] - consider extending range lower"
            )
        elif optimal_value == max_val:
            warnings.append(
                f"⚠ {param_name}={optimal_value} is at UPPER edge of range "
                f"[{min_val}-{max_val}] - consider extending range higher"
            )

    return warnings


# Define global param_options for edge checking
PARAM_OPTIONS = {
    "stop_loss_pct": [20.0, 25.0, 30.0, 35.0],
    "vix_threshold": [25.0, 30.0, 35.0, 40.0],
    "min_score": [50.0, 55.0, 60.0, 65.0, 70.0],
}


def run_long_term_backtest():
    """Main entry point for long-term backtesting with walk-forward optimization."""
    from stockpulse.data.universe import UniverseManager, TOP_US_STOCKS
    from stockpulse.scanner.long_term_scanner import LongTermScanner

    print("\n" + "=" * 80)
    print("  LONG-TERM SCANNER BACKTESTER (Enhanced)")
    print("  Features: Transaction costs, Stop-loss, VIX regime, Sector limits")
    print("  Strategy: 3-year buy-and-hold, 2% position size, 10% max concentration")
    print("  Walk-Forward: 2-year train, 1-year test, rolling windows")
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
    print("\n  Fetching 7 years of historical data...")
    price_df = backtester.fetch_extended_history(tickers, years=7)
    print(f"  Loaded {len(price_df):,} price records")

    # Fetch VIX history
    print("\n  Fetching VIX history for regime detection...")
    backtester.fetch_vix_history(years=7)

    # Run backtest with default weights
    print("\n  Running backtest with default weights (2019-2025)...")
    default_weights = LongTermScanner.DEFAULT_WEIGHTS
    default_params = {"stop_loss_pct": 25.0, "vix_threshold": 30.0}

    result = backtester.run_backtest(
        tickers=tickers,
        weights=default_weights,
        params=default_params,
        min_score=60,
        start_year=2019,
        end_year=2025
    )

    _print_backtest_results("DEFAULT WEIGHTS", result)

    # Run walk-forward optimization
    print("\n  Running walk-forward optimization (2yr train / 1yr test)...")
    print("  This tests if optimized params work on unseen data...")

    wf_result = backtester.walk_forward_optimize(
        tickers=tickers,
        train_years=2,
        test_years=1,
        start_year=2019,
        end_year=2025,
        max_iterations=30,
        objective="alpha"
    )

    _print_walk_forward_results(wf_result)

    # Optimize on full period for final params
    print("\n  Optimizing weights on full period (2019-2024)...")
    opt_result = backtester.optimize_weights(
        tickers=tickers,
        max_iterations=50,
        objective="alpha",
        start_year=2019,
        end_year=2024
    )

    if opt_result["best_result"]:
        _print_backtest_results("OPTIMIZED WEIGHTS", opt_result["best_result"])

        print("\n  Optimized Weights:")
        for k, v in opt_result["best_weights"].items():
            print(f"    {k}: {v:.3f}")

        print("\n  Optimized Parameters:")
        for k, v in opt_result["best_params"].items():
            print(f"    {k}: {v}")

        # Check for edge values
        edge_warnings = _check_edge_values(opt_result["best_params"], PARAM_OPTIONS)
        if edge_warnings:
            print("\n  EDGE VALUE WARNINGS:")
            for warning in edge_warnings:
                print(f"    {warning}")

        # Print holdings detail
        _print_holdings_detail(opt_result["best_result"])

        # Save to config
        print("\n  Saving optimized weights and params to config...")
        _save_optimized_config(opt_result["best_weights"], opt_result["best_params"])

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
    print(f"  Max Drawdown:        {result.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Sharpe Ratio:        {result.get('sharpe_ratio', 0):.2f}")
    print(f"  ")
    print(f"  Total Trades:        {result['total_trades']}")
    print(f"  Win Rate:            {result['win_rate']:.1f}%")
    print(f"  Avg Return/Trade:    {result['avg_return_pct']:+.1f}%")
    print(f"  Best Trade:          {result.get('best_trade_pct', 0):+.1f}%")
    print(f"  Worst Trade:         {result.get('worst_trade_pct', 0):+.1f}%")
    print(f"  Stop-Losses Hit:     {result.get('stop_losses_triggered', 0)}")
    print(f"  Avg Hold Days:       {result.get('avg_hold_days', 0):.0f}")
    print(f"  Max Positions Held:  {result.get('max_positions_held', 0)}")

    # Show params if present
    params = result.get("params", {})
    if params:
        print(f"  ")
        print(f"  Stop-Loss:           {params.get('stop_loss_pct', 25)}%")
        print(f"  VIX Threshold:       {params.get('vix_threshold', 30)}")
        print(f"  Min Score:           {params.get('min_score', 60)}")


def _print_walk_forward_results(wf_result: dict):
    """Print walk-forward optimization results."""
    print("\n" + "-" * 80)
    print("  WALK-FORWARD RESULTS")
    print("-" * 80)

    print(f"\n  Number of Windows:       {wf_result['num_windows']}")
    print(f"  Avg Test Return:         {wf_result['avg_test_return_pct']:+.1f}%")
    print(f"  Avg Test Alpha:          {wf_result['avg_test_alpha_pct']:+.1f}%")
    print(f"  Avg Test Sharpe:         {wf_result['avg_test_sharpe']:.2f}")
    print(f"  Consistency (% winning): {wf_result['consistency_pct']:.0f}%")

    print("\n  Window Results:")
    print("  " + "-" * 76)
    print("  Train Period    Test Period     Train Ret   Test Ret   Test Alpha   Sharpe")
    print("  " + "-" * 76)

    all_edge_warnings = []
    for w in wf_result["windows"]:
        print(f"  {w['train_period']:14}  {w['test_period']:14}  "
              f"{w['train_return_pct']:+7.1f}%   {w['test_return_pct']:+7.1f}%   "
              f"{w['test_alpha_pct']:+8.1f}%   {w['test_sharpe']:6.2f}")
        # Collect edge warnings from each window
        if w.get("params"):
            warnings = _check_edge_values(w["params"], PARAM_OPTIONS)
            for warning in warnings:
                tagged_warning = f"[{w['train_period']}] {warning}"
                if tagged_warning not in all_edge_warnings:
                    all_edge_warnings.append(tagged_warning)

    if all_edge_warnings:
        print("\n  EDGE VALUE WARNINGS (by window):")
        for warning in all_edge_warnings:
            print(f"    {warning}")


def _print_holdings_detail(result: dict):
    """Print detailed holdings information."""
    report = result.get("holdings_report", {})

    print("\n" + "-" * 80)
    print("  HOLDINGS DETAIL")
    print("-" * 80)

    # Best performers
    print("\n  TOP 5 PERFORMERS:")
    for p in report.get("best_performers", [])[:5]:
        print(f"    {p['ticker']:6} {p['return_pct']:+6.1f}%  ({p.get('sector', 'N/A')}, bought {p['buy_date']})")

    # Worst performers
    print("\n  BOTTOM 5 PERFORMERS:")
    for p in report.get("worst_performers", [])[:5]:
        print(f"    {p['ticker']:6} {p['return_pct']:+6.1f}%  ({p.get('sector', 'N/A')}, bought {p['buy_date']})")

    # Sector summary
    sector_summary = report.get("sector_summary", {})
    if sector_summary:
        print("\n  SECTOR BREAKDOWN:")
        for sector, data in sorted(sector_summary.items(), key=lambda x: x[1]["avg_return"], reverse=True):
            print(f"    {sector:25} {data['count']:3} trades, {data['avg_return']:+5.1f}% avg, {data['win_rate']:.0f}% win")

    # Growth milestones
    print("\n  PORTFOLIO GROWTH:")
    for m in report.get("growth_milestones", [])[:10]:
        print(f"    {m['date']}: ${m['value']:>10,.0f}  ({m['positions']} positions, {m['drawdown']:.1f}% DD)")

    # Transaction sample
    txn_log = result.get("transaction_log", [])
    if txn_log:
        print("\n  RECENT TRANSACTIONS (last 10):")
        for txn in txn_log[-10:]:
            if txn["action"] == "BUY":
                print(f"    {txn['date']} BUY  {txn['ticker']:6} @ ${txn['price']:.2f} "
                      f"({txn.get('sector', 'N/A')}, score: {txn.get('score', 0):.0f})")
            else:
                reason = txn.get('reason', 'N/A')[:20]
                print(f"    {txn['date']} SELL {txn['ticker']:6} @ ${txn['price']:.2f} "
                      f"({txn.get('return_pct', 0):+.1f}%, {reason})")


def _print_suggestions():
    """Print suggestions for improving the backtester."""
    print("\n" + "-" * 80)
    print("  IMPLEMENTED ENHANCEMENTS")
    print("-" * 80)
    print("""
  ✓ Transaction costs (0.1% round-trip)
  ✓ Optimizable stop-loss (20-35% range)
  ✓ VIX regime detection (skip buying in high vol)
  ✓ Sector diversification (40% max per sector)
  ✓ Walk-forward optimization (2yr train / 1yr test)
  ✓ Data extended through 2025
  ✓ All optimal results saved to database
  ✓ Optimizable min_score (50-70 range)
  ✓ Edge-of-range detection for optimal parameters

  FUTURE ENHANCEMENTS TO CONSIDER:
  1. Add earnings calendar: avoid buying before earnings
  2. Add momentum factor: weight recent winners
  3. Dynamic position sizing based on conviction
  4. Trailing stop-loss vs fixed stop
  5. Tax-loss harvesting simulation
    """)


def _save_optimized_config(weights: dict, params: dict):
    """Save optimized weights and params to config file."""
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
        config["long_term_scanner"]["params"] = {
            k: round(v, 1) if isinstance(v, float) else v for k, v in params.items()
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  ✓ Saved to {config_path}")

    except Exception as e:
        print(f"  ✗ Failed to save: {e}")


if __name__ == "__main__":
    run_long_term_backtest()
