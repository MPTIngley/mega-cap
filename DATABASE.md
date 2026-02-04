# StockPulse Database Documentation

This document describes all SQLite databases and tables used by StockPulse.

## Database Location

- **File**: `data/stockpulse.sqlite`
- **Mode**: WAL (Write-Ahead Logging) for concurrent read/write access
- **Configuration**: Set in `config/config.yaml` under `database.path`

## Accessing the Database

### From Python
```python
from stockpulse.data.database import get_db

db = get_db()

# Execute query and return DataFrame
df = db.fetchdf("SELECT * FROM universe WHERE is_active = 1")

# Execute query and return single row
row = db.fetchone("SELECT * FROM universe WHERE ticker = ?", ("AAPL",))

# Execute query and return all rows
rows = db.fetchall("SELECT ticker FROM universe")

# Insert DataFrame
db.insert_df("prices_daily", df, on_conflict="replace")

# Raw SQL execute
db.execute("UPDATE universe SET is_active = 0 WHERE ticker = ?", ("XYZ",))
```

### From Command Line
```bash
# Open interactive SQLite shell
sqlite3 data/stockpulse.sqlite

# One-liner queries
sqlite3 data/stockpulse.sqlite "SELECT COUNT(*) FROM universe"

# Export to CSV
sqlite3 -header -csv data/stockpulse.sqlite "SELECT * FROM signals" > signals.csv
```

### Useful Utility Functions
```python
from stockpulse.data.database import get_data_summary, reset_trading_data

# Get record counts and date ranges
summary = get_data_summary()
print(summary)

# Reset trading data (keeps market data by default)
reset_trading_data()

# Reset everything including market data
reset_trading_data(keep_market_data=False)
```

---

## Core Tables (database.py)

### 1. `universe` - Stock Universe
Stocks being tracked by the system.

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Primary key - stock symbol |
| company_name | TEXT | Full company name |
| sector | TEXT | GICS sector |
| industry | TEXT | Industry classification |
| market_cap | REAL | Market capitalization |
| is_active | INTEGER | 1=active, 0=inactive |
| added_date | TEXT | Date added to universe |
| last_refreshed | TEXT | Last metadata refresh |

```sql
-- Example queries
SELECT * FROM universe WHERE is_active = 1 ORDER BY market_cap DESC;
SELECT sector, COUNT(*) FROM universe GROUP BY sector;
```

### 2. `prices_daily` - Daily OHLCV Data
Historical daily price data for all tracked stocks.

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Stock symbol (PK) |
| date | TEXT | Date YYYY-MM-DD (PK) |
| open | REAL | Opening price |
| high | REAL | High price |
| low | REAL | Low price |
| close | REAL | Closing price |
| adj_close | REAL | Adjusted close (splits/dividends) |
| volume | INTEGER | Trading volume |

```sql
-- Get latest prices
SELECT ticker, date, close FROM prices_daily
WHERE date = (SELECT MAX(date) FROM prices_daily);

-- Get price history for a ticker
SELECT * FROM prices_daily WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 30;
```

### 3. `prices_intraday` - Intraday Data (15-min bars)
Intraday price data for active trading strategies.

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Stock symbol (PK) |
| timestamp | TEXT | ISO timestamp (PK) |
| open | REAL | Opening price |
| high | REAL | High price |
| low | REAL | Low price |
| close | REAL | Closing price |
| volume | INTEGER | Trading volume |

### 4. `fundamentals` - Fundamental Data
Valuation and financial metrics.

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Stock symbol (PK) |
| date | TEXT | Date captured (PK) |
| pe_ratio | REAL | Price-to-earnings ratio |
| forward_pe | REAL | Forward P/E |
| pb_ratio | REAL | Price-to-book |
| peg_ratio | REAL | PEG ratio |
| dividend_yield | REAL | Dividend yield % |
| eps | REAL | Earnings per share |
| revenue | REAL | Annual revenue |
| profit_margin | REAL | Profit margin % |
| roe | REAL | Return on equity |
| debt_to_equity | REAL | Debt/equity ratio |
| current_ratio | REAL | Current ratio |
| fifty_two_week_high | REAL | 52-week high price |
| fifty_two_week_low | REAL | 52-week low price |
| avg_volume_10d | INTEGER | 10-day average volume |
| beta | REAL | Beta coefficient |

```sql
-- Find undervalued stocks
SELECT ticker, pe_ratio, dividend_yield, pb_ratio
FROM fundamentals
WHERE date = (SELECT MAX(date) FROM fundamentals)
AND pe_ratio < 15 AND dividend_yield > 2
ORDER BY pe_ratio;
```

### 5. `signals` - Trading Signals
Generated trading signals from all strategies.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| created_at | TEXT | Timestamp (auto) |
| ticker | TEXT | Stock symbol |
| strategy | TEXT | Strategy name |
| direction | TEXT | 'long' or 'short' |
| confidence | REAL | Signal confidence 0-100 |
| entry_price | REAL | Suggested entry price |
| target_price | REAL | Target price |
| stop_price | REAL | Stop-loss price |
| status | TEXT | 'open', 'closed', 'expired' |
| notes | TEXT | Additional notes |

```sql
-- Get recent signals
SELECT * FROM signals WHERE status = 'open' ORDER BY created_at DESC LIMIT 20;

-- Signals by strategy performance
SELECT strategy, COUNT(*) as signals,
       AVG(confidence) as avg_confidence
FROM signals GROUP BY strategy;
```

### 6. `positions_paper` - Paper Trading Positions
Simulated trading positions for strategy validation.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| signal_id | INTEGER | FK to signals |
| ticker | TEXT | Stock symbol |
| direction | TEXT | 'long' or 'short' |
| entry_price | REAL | Entry price |
| entry_date | TEXT | Entry date |
| shares | REAL | Position size |
| exit_price | REAL | Exit price (if closed) |
| exit_date | TEXT | Exit date (if closed) |
| pnl | REAL | Profit/loss $ |
| pnl_pct | REAL | Profit/loss % |
| status | TEXT | 'open' or 'closed' |
| exit_reason | TEXT | Why position was closed |
| strategy | TEXT | Strategy that generated it |

```sql
-- Open positions
SELECT * FROM positions_paper WHERE status = 'open';

-- Strategy P&L summary
SELECT strategy,
       COUNT(*) as trades,
       SUM(pnl) as total_pnl,
       AVG(pnl_pct) as avg_return
FROM positions_paper
WHERE status = 'closed'
GROUP BY strategy;
```

### 7. `positions_real` - Real Trading Positions
Actual trading positions (Phase 5 - live trading).

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| ticker | TEXT | Stock symbol |
| direction | TEXT | 'long' or 'short' |
| entry_price | REAL | Entry price |
| entry_date | TEXT | Entry date |
| shares | REAL | Position size |
| exit_price | REAL | Exit price (if closed) |
| exit_date | TEXT | Exit date (if closed) |
| pnl | REAL | Profit/loss $ |
| pnl_pct | REAL | Profit/loss % |
| status | TEXT | 'open' or 'closed' |
| exit_reason | TEXT | Why position was closed |
| strategy | TEXT | Strategy name |
| commission | REAL | Commission paid |
| notes | TEXT | Additional notes |

### 8. `strategy_state` - Strategy Runtime State
Tracks strategy performance and configuration.

| Column | Type | Description |
|--------|------|-------------|
| strategy_name | TEXT | Primary key |
| enabled | INTEGER | 1=enabled, 0=disabled |
| params | TEXT | JSON parameters |
| last_run | TEXT | Last execution time |
| total_signals | INTEGER | Total signals generated |
| win_count | INTEGER | Winning trades |
| loss_count | INTEGER | Losing trades |
| total_pnl | REAL | Cumulative P&L |
| max_drawdown | REAL | Maximum drawdown seen |
| disabled_reason | TEXT | Why disabled (if applicable) |

### 9. `alerts_log` - Alert History
Log of all alerts sent.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| created_at | TEXT | Timestamp (auto) |
| signal_id | INTEGER | Related signal |
| alert_type | TEXT | Type of alert |
| recipient | TEXT | Email/notification target |
| subject | TEXT | Alert subject |
| body | TEXT | Alert content |
| sent_successfully | INTEGER | 1=success, 0=failed |
| error_message | TEXT | Error if failed |

### 10. `long_term_watchlist` - Long-Term Scanner Results
Stocks identified by the long-term investment scanner.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| ticker | TEXT | Stock symbol |
| scan_date | TEXT | Date of scan |
| composite_score | REAL | Overall score 0-100 |
| valuation_score | REAL | Valuation component |
| technical_score | REAL | Technical component |
| dividend_score | REAL | Dividend component |
| quality_score | REAL | Quality component |
| pe_percentile | REAL | P/E percentile rank |
| price_vs_52w_low_pct | REAL | % above 52-week low |
| reasoning | TEXT | Explanation |

```sql
-- Top scoring stocks from latest scan
SELECT ticker, composite_score, valuation_score, technical_score
FROM long_term_watchlist
WHERE scan_date = (SELECT MAX(scan_date) FROM long_term_watchlist)
ORDER BY composite_score DESC LIMIT 10;
```

### 11. `backtest_results` - Basic Backtest Results
Results from strategy backtests (short-term strategies).

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| strategy | TEXT | Strategy name |
| run_date | TEXT | When backtest ran |
| start_date | TEXT | Backtest start |
| end_date | TEXT | Backtest end |
| initial_capital | REAL | Starting capital |
| final_value | REAL | Ending portfolio value |
| total_return_pct | REAL | Total return % |
| annualized_return_pct | REAL | Annualized return % |
| sharpe_ratio | REAL | Sharpe ratio |
| sortino_ratio | REAL | Sortino ratio |
| max_drawdown_pct | REAL | Maximum drawdown % |
| win_rate | REAL | Win rate % |
| profit_factor | REAL | Profit factor |
| total_trades | INTEGER | Number of trades |
| avg_trade_pnl | REAL | Average trade P&L |
| avg_win | REAL | Average winning trade |
| avg_loss | REAL | Average losing trade |
| avg_hold_days | REAL | Average holding period |
| params | TEXT | JSON parameters used |

### 12. `optimization_runs` - Strategy Optimization History
Results from strategy parameter optimization.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| strategy | TEXT | Strategy name |
| run_date | TEXT | When optimization ran |
| best_params | TEXT | JSON of best parameters |
| best_return_pct | REAL | Return with best params |
| best_sharpe | REAL | Sharpe with best params |
| best_drawdown_pct | REAL | Drawdown with best params |
| constraint_satisfied | INTEGER | Met constraints? |
| optimization_time_seconds | REAL | Time taken |

### 13. `system_state` - System Metadata
Key-value store for system state.

| Column | Type | Description |
|--------|------|-------------|
| key | TEXT | Primary key |
| value | TEXT | Value (often JSON) |
| updated_at | TEXT | Last update time |

---

## Holdings Tracker Tables (holdings_tracker.py)

### 14. `actual_holdings` - Real Purchase Records
Records of actual stock purchases (manual tracking).

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| ticker | TEXT | Stock symbol |
| strategy_type | TEXT | 'active' or 'long_term' |
| buy_date | DATE | Purchase date |
| buy_price | REAL | Purchase price |
| shares | REAL | Number of shares |
| cost_basis | REAL | Total cost |
| sell_date | DATE | Sale date (if sold) |
| sell_price | REAL | Sale price (if sold) |
| realized_pnl | REAL | Realized P&L (if sold) |
| realized_pnl_pct | REAL | Realized P&L % (if sold) |
| status | TEXT | 'open' or 'closed' |
| sector | TEXT | Stock sector |
| signal_score | REAL | Score when purchased |
| notes | TEXT | Purchase notes |
| created_at | TIMESTAMP | Record creation time |

```python
# Using the HoldingsTracker
from stockpulse.tracker.holdings_tracker import HoldingsTracker

tracker = HoldingsTracker()

# Add a holding
tracker.add_holding(
    ticker="AAPL",
    buy_date="2024-01-15",
    buy_price=185.50,
    shares=10,
    strategy_type="long_term",
    notes="Strong value score"
)

# Get open holdings with current values
holdings = tracker.get_holdings_with_current_value(strategy_type="long_term")

# Close a holding
tracker.close_holding(holding_id=1, sell_date="2024-06-15", sell_price=195.00)
```

### 15. `holdings_snapshots` - Portfolio Snapshots
Daily snapshots for equity curve tracking.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| snapshot_date | DATE | Date of snapshot |
| strategy_type | TEXT | 'active' or 'long_term' |
| total_value | REAL | Portfolio market value |
| total_cost_basis | REAL | Total cost basis |
| unrealized_pnl | REAL | Unrealized P&L |
| unrealized_pnl_pct | REAL | Unrealized P&L % |
| num_positions | INTEGER | Number of positions |
| holdings_json | TEXT | JSON of individual holdings |
| created_at | TIMESTAMP | Snapshot creation time |

```sql
-- Get equity curve
SELECT snapshot_date, total_value, unrealized_pnl_pct
FROM holdings_snapshots
WHERE strategy_type = 'long_term'
ORDER BY snapshot_date;
```

---

## Long-Term Backtester Tables (long_term_backtest.py)

### 16. `backtest_prices` - Cached Price Data
Historical prices cached for backtesting (separate from main prices_daily).

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Stock symbol (PK) |
| date | DATE | Date (PK) |
| open | REAL | Opening price |
| high | REAL | High price |
| low | REAL | Low price |
| close | REAL | Closing price |
| volume | INTEGER | Trading volume |

### 17. `backtest_vix` - VIX Data
VIX index data for market regime detection.

| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Primary key |
| close | REAL | VIX closing value |

```sql
-- High volatility periods
SELECT date, close FROM backtest_vix WHERE close > 30 ORDER BY date DESC;
```

### 18. `backtest_results` - Long-Term Backtest Results
Results from long-term strategy backtests (Note: different schema from core table).

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| run_date | TIMESTAMP | When backtest ran |
| weights | TEXT | JSON scoring weights |
| params | TEXT | JSON parameters |
| start_year | INTEGER | Backtest start year |
| end_year | INTEGER | Backtest end year |
| total_trades | INTEGER | Number of trades |
| avg_return_pct | REAL | Average return per trade |
| win_rate | REAL | Win rate % |
| total_return_pct | REAL | Total portfolio return |
| spy_return_pct | REAL | SPY benchmark return |
| alpha_pct | REAL | Alpha vs SPY |
| max_drawdown_pct | REAL | Maximum drawdown |
| sharpe_ratio | REAL | Sharpe ratio |
| final_holdings | TEXT | JSON of final positions |

### 19. `backtest_optimal_params` - Walk-Forward Optimization Results
Best parameters from walk-forward optimization.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| run_date | TIMESTAMP | When optimization ran |
| optimization_type | TEXT | Type of optimization |
| best_weights | TEXT | JSON scoring weights |
| best_params | TEXT | JSON parameters |
| objective | TEXT | Optimization objective |
| score | REAL | Objective score achieved |
| total_return_pct | REAL | Return with best params |
| alpha_pct | REAL | Alpha vs benchmark |
| max_drawdown_pct | REAL | Maximum drawdown |
| sharpe_ratio | REAL | Sharpe ratio |
| walk_forward_results | TEXT | JSON of walk-forward periods |

### 20. `backtest_earnings` - Earnings Calendar Cache
Cached earnings dates for blackout period detection.

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Stock symbol (PK) |
| earnings_date | DATE | Earnings date (PK) |

### 21. `backtest_risk_profiles` - Optimal Params by Risk Level
Best parameters for each stop-loss level.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| run_date | TIMESTAMP | When optimization ran |
| stop_loss_pct | REAL | Stop-loss level |
| best_weights | TEXT | JSON scoring weights |
| best_params | TEXT | JSON parameters |
| total_return_pct | REAL | Total return achieved |
| alpha_pct | REAL | Alpha vs benchmark |
| max_drawdown_pct | REAL | Maximum drawdown |
| sharpe_ratio | REAL | Sharpe ratio |
| win_rate | REAL | Win rate % |

```sql
-- View risk profiles
SELECT stop_loss_pct, alpha_pct, sharpe_ratio, win_rate, max_drawdown_pct
FROM backtest_risk_profiles
WHERE run_date = (SELECT MAX(run_date) FROM backtest_risk_profiles)
ORDER BY stop_loss_pct;
```

---

## Database Indexes

The following indexes are created automatically for query performance:

| Index | Table | Column(s) |
|-------|-------|-----------|
| idx_prices_daily_date | prices_daily | date |
| idx_prices_intraday_timestamp | prices_intraday | timestamp |
| idx_signals_status | signals | status |
| idx_signals_ticker | signals | ticker |
| idx_positions_paper_status | positions_paper | status |
| idx_holdings_status | actual_holdings | status, strategy_type |

---

## Common Queries

### Portfolio Performance
```sql
-- Total P&L by strategy
SELECT strategy,
       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
       SUM(pnl) as total_pnl,
       AVG(pnl_pct) as avg_return
FROM positions_paper
WHERE status = 'closed'
GROUP BY strategy
ORDER BY total_pnl DESC;
```

### Best Long-Term Candidates
```sql
-- Top scoring stocks from recent scan
SELECT w.ticker, w.composite_score, w.reasoning,
       u.sector, u.market_cap
FROM long_term_watchlist w
JOIN universe u ON w.ticker = u.ticker
WHERE w.scan_date >= date('now', '-7 days')
AND w.composite_score >= 60
ORDER BY w.composite_score DESC;
```

### Current Holdings Value
```sql
-- Using HoldingsTracker (recommended)
from stockpulse.tracker.holdings_tracker import print_holdings_summary
print_holdings_summary(strategy_type="long_term")
```

### Backtest Performance by Risk Profile
```sql
-- Compare risk profiles
SELECT stop_loss_pct,
       printf("%.1f%%", alpha_pct) as alpha,
       printf("%.2f", sharpe_ratio) as sharpe,
       printf("%.1f%%", win_rate) as win_rate,
       printf("%.1f%%", max_drawdown_pct) as max_dd
FROM backtest_risk_profiles
ORDER BY stop_loss_pct;
```

---

## Data Retention Notes

- **prices_daily**: Retained indefinitely (valuable historical data)
- **prices_intraday**: Auto-cleaned after 30 days (configurable)
- **signals**: Retained for analysis; old signals marked 'expired'
- **backtest_***: Retained for comparison; can be manually cleared

## Backup

```bash
# Simple backup
cp data/stockpulse.sqlite data/stockpulse_backup_$(date +%Y%m%d).sqlite

# With WAL checkpoint first (ensures all data written)
sqlite3 data/stockpulse.sqlite "PRAGMA wal_checkpoint(TRUNCATE)"
cp data/stockpulse.sqlite data/stockpulse_backup.sqlite
```
