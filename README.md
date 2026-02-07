# StockPulse

Automated Stock Scanning & Trading Signal System

A self-hosted system that scans the top 100 US stocks by market cap, runs trading strategies, generates signals with email alerts, and presents everything through a Streamlit dashboard.

## Features

- **8 Trading Strategies**: RSI Mean Reversion, Bollinger Squeeze, MACD+Volume, Z-Score Mean Reversion, Momentum Breakout, Gap Fade, 52-Week Low Bounce, Sector Rotation
- **Strategy Optimizer**: Hyperparameter optimization with drawdown constraints
- **Portfolio Optimizer**: Optimize position sizing, concentration limits, and confidence scaling
- **Backtesting Framework**: Test strategies with historical data including transaction costs
- **Dynamic Position Sizing**: Continuous confidence scaling with strategy weights
- **Email Alerts**: Get notified of new signals and position exits
- **Streamlit Dashboard**: Monitor signals, positions, P&L, and performance
- **Long-Term Scanner**: Identifies value investment opportunities

## Quick Start

### 1. Clone and Setup Virtual Environment

```bash
cd mega-cap

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Email (for alerts)

Copy the example and edit:

```bash
cp .env.example .env
open -e .env   # or use any text editor
```

Set these values in `.env`:

```
STOCKPULSE_EMAIL_SENDER=your-email@gmail.com
STOCKPULSE_EMAIL_RECIPIENT=your-email@gmail.com
STOCKPULSE_EMAIL_PASSWORD=your-gmail-app-password
```

**Note:** Commands auto-load `.env` - no manual sourcing required.

### 3. Initialize Database and Fetch Data

```bash
python -m stockpulse.main init
```

### 4. Run the Dashboard

```bash
python -m stockpulse.main dashboard
```

Or run the full scheduler:

```bash
python -m stockpulse.main run
```

## Commands

| Command | Description |
|---------|-------------|
| `stockpulse init` | Initialize database and fetch 2 years of historical data |
| `stockpulse run` | Run the scheduler for continuous scanning |
| `stockpulse dashboard` | Launch Streamlit dashboard |
| `stockpulse scan` | Run a single scan (generates signals, opens positions, sends email) |
| `stockpulse backtest` | Run backtests for all strategies |
| `stockpulse optimize` | Run hyperparameter optimization for all strategies |
| `stockpulse longterm-scan` | Run long-term investment scanner (8 scoring components) |
| `stockpulse longterm-backtest` | Backtest & optimize long-term scanner with 3-year hold strategy |
| `stockpulse trillion-scan` | Run Trillion+ Club mega-cap scanner + email |
| `stockpulse ai-scan` | Run AI Pulse scan (~70 AI stocks) + Claude thesis research + email |
| `stockpulse ai-backfill` | Initialize trillion club history and default theses |
| `stockpulse reset` | Reset trading data (keeps price history) |
| `stockpulse reset --clear-all` | Reset ALL data including price history |
| `stockpulse ingest` | Manually run data ingestion |
| `stockpulse test-email` | Test email configuration |
| `stockpulse digest` | Send daily portfolio digest email now |

## Daily Usage

### Start the System

```bash
cd mega-cap
source .venv/bin/activate
source .env
python -m stockpulse run
```

### Reset and Restart Fresh

Wipes all paper positions (open AND closed) while keeping market data:

```bash
cd mega-cap
source .venv/bin/activate
source .env
python -m stockpulse reset
python -m stockpulse run
```

### One-Time Scan (for testing)

```bash
python -m stockpulse scan
```

## Configuration

Edit `config/config.yaml` to customize:

- Strategy parameters
- Alert thresholds
- Risk management settings
- Scanning intervals

## API Keys & Tokens Required

### Required for Email Alerts

1. **Gmail App Password** (for sending email alerts)
   - Go to Google Account > Security > 2-Step Verification
   - Scroll to "App passwords"
   - Generate a new app password for "Mail"
   - Use this password for `STOCKPULSE_EMAIL_PASSWORD`

### No API Key Required

- **yfinance**: Stock data is fetched using yfinance (free, no API key)
- **SQLite**: Local database with WAL mode (concurrent read/write support)

## Project Structure

```
mega-cap/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   └── stockpulse.sqlite    # Database (created on init)
├── src/stockpulse/
│   ├── alerts/              # Email alert system
│   ├── dashboard/           # Streamlit dashboard
│   ├── data/                # Database and data ingestion
│   ├── scanner/             # Long-term investment scanner
│   ├── strategies/          # Trading strategies and backtesting
│   ├── tracker/             # Real trade tracker
│   └── main.py              # CLI entry point
├── tests/                   # Unit tests
├── PRD.md                   # Product Requirements Document
├── PROJECT_PLAN.md          # Implementation plan
└── pyproject.toml           # Project dependencies
```

## Strategies

| Strategy | Status | Backtest Return | Sharpe | Description |
|----------|--------|-----------------|--------|-------------|
| **Sector Rotation** | ✅ Enabled | +40.2% | 2.87 | Rotate into strongest sectors |
| **Momentum Breakout** | ✅ Enabled | +19.3% | 2.23 | Breakouts with volume confirmation |
| **Z-Score Mean Reversion** | ✅ Enabled | +14.9% | 1.80 | Statistical mean reversion |
| **MACD + Volume** | ✅ Enabled | +13.2% | 2.21 | Trend-following with MACD |
| **RSI Mean Reversion** | ✅ Enabled | +11.4% | 1.73 | Buy oversold, sell overbought |
| **52-Week Low Bounce** | ✅ Enabled | +10.4% | 3.67 | Best risk-adjusted! |
| **Bollinger Squeeze** | ❌ Disabled | +0.6% | - | Volatility breakouts |
| **Gap Fade** | ❌ Disabled | -1.3% | - | Fade overnight gaps |

## Position Sizing

Uses **continuous confidence scaling** with strategy weights:

```
confidence_mult = 1.0 + (confidence - min_confidence) / (100 - min_confidence) * (max_multiplier - 1.0)
final_size = base_size × strategy_weight × confidence_mult
final_size = min(final_size, max_position_size_pct)
```

### Example Calculations

| Signal | Confidence | Strategy Weight | Conf Mult | Raw Size | Final Size |
|--------|------------|-----------------|-----------|----------|------------|
| sector_rotation | 90% | 2.0 | 2.125 | 21.3% | **12%** (capped) |
| sector_rotation | 75% | 2.0 | 1.56 | 15.6% | **12%** (capped) |
| sector_rotation | 65% | 2.0 | 1.19 | 11.9% | **11.9%** |
| rsi_mean_reversion | 70% | 1.0 | 1.38 | 6.9% | **6.9%** |
| momentum_breakout | 80% | 1.5 | 1.75 | 13.1% | **12%** (capped) |

### Current Config (from config.yaml)

- `base_size_pct`: 5%
- `min_confidence`: 60 (below this, multiplier = 1.0)
- `max_multiplier`: 2.5 (at 100% confidence)
- `max_position_size_pct`: 12% (hard cap)
- `max_per_strategy_pct`: 70% (strategy concentration limit)
- `max_sector_concentration_pct`: 70%
- `max_portfolio_exposure_pct`: 80%

> **NOTE:** These parameters should be optimized using `stockpulse optimize` with portfolio-level optimization to find the best risk/reward balance for your strategy mix.

## Transaction Costs

The system accounts for realistic transaction costs:
- **Commission**: $0 (most brokers)
- **Slippage**: 0.05% (5 bps)
- **Spread**: 0.02% (2 bps)
- **Total round-trip**: ~0.14%

## Risk Management

Configurable limits (see `config/config.yaml`):

| Limit | Default | Description |
|-------|---------|-------------|
| `max_position_size_pct` | 12% | Hard cap per position |
| `min_position_size_pct` | 3% | Don't open if final size < 3% |
| `max_per_strategy_pct` | 70% | Max exposure to any single strategy |
| `max_sector_concentration_pct` | 70% | Max exposure to any single sector |
| `max_portfolio_exposure_pct` | 80% | Max total invested at once |
| `max_positions` | 40 | Max concurrent positions |
| `max_drawdown_disable_pct` | 15% | Stop trading if drawdown exceeds this |

### Smart Trading Features

- **Churn Prevention**: 3-day cooldown after exiting any position
- **Loss Cooldown**: 7-day cooldown after a losing trade on same ticker
- **Max Losses**: Block ticker after 3 consecutive losses
- **Dynamic Sizing**: Position sizes reduced to fit remaining capacity in strategy/portfolio limits

## Long-Term Investment Scanner

Identifies value investment opportunities using 8 scoring components:

| Component | Weight | Signal |
|-----------|--------|--------|
| **Insider Buying** | 15% | Recent insider purchases (very predictive!) |
| **Valuation** | 15% | P/E, P/B, PEG ratios vs history |
| **Technical** | 15% | Near 52-week low, RSI oversold, accumulation |
| **Quality** | 15% | Profit margin, ROE, low debt |
| **FCF Yield** | 12% | Free Cash Flow / Market Cap (better than P/E) |
| **Dividend** | 10% | Yield vs sustainability |
| **Earnings Momentum** | 10% | EPS beat streak |
| **Peer Valuation** | 8% | Cheaper than sector peers |

### Usage

```bash
# Run scanner now
stockpulse longterm-scan

# Backtest & optimize weights with 3-year hold strategy
stockpulse longterm-backtest
```

Weights can be optimized via `longterm-backtest` to maximize alpha vs SPY.

## Disclaimer

This is a personal research tool, not financial advice. All trading decisions are your own. Paper trade first before risking real money.

## License

MIT
