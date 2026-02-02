# StockPulse Command Reference

## How StockPulse Works

**You need TWO processes running for full operation:**

| Process | Terminal | Purpose |
|---------|----------|---------|
| `stockpulse run` | Terminal 1 | Scheduler - fetches data, generates signals, manages positions, sends email alerts |
| `stockpulse dashboard` | Terminal 2 | Web UI - view signals, portfolio, performance (http://localhost:8501) |

The scheduler writes to the database. The dashboard reads from it. Both can run simultaneously.

**Without the scheduler running:**
- Dashboard shows stale data
- No new signals generated
- No email alerts sent
- Positions not updated

---

## Quick Setup (One-Time)

Add this to your `~/.zshrc` for easy access from any terminal:

```bash
echo 'stockpulse() {
  cd ~/Documents/AIGames/mega-cap &&
  source venv/bin/activate &&
  python3 -m stockpulse.main "$@"
}' >> ~/.zshrc
source ~/.zshrc
```

Then use `stockpulse <command>` from anywhere.

**Note:** Commands auto-load `.env` - no manual env sourcing needed.

---

## Main Commands

| Command | What it does |
|---------|--------------|
| `stockpulse dashboard` | Launch the web dashboard (http://localhost:8501) |
| `stockpulse run` | Start the scheduler (runs scans every 15 min during market hours) |
| `stockpulse scan` | Run a single signal scan now |
| `stockpulse backtest` | Run backtests on all strategies |
| `stockpulse optimize` | **NEW** Run hyperparameter optimization for all 8 strategies |
| `stockpulse init` | Initialize DB and fetch 2 years of historical data |
| `stockpulse ingest` | Refresh universe and fetch latest price data |

---

## Strategies (8 total)

| Strategy | Type | Signal |
|----------|------|--------|
| `rsi_mean_reversion` | Mean Reversion | RSI oversold/overbought |
| `bollinger_squeeze` | Breakout | Bollinger Band squeeze |
| `macd_volume` | Momentum | MACD crossover + volume |
| `zscore_mean_reversion` | Mean Reversion | Price Z-score extremes |
| `momentum_breakout` | Momentum | Price/volume breakout |
| `gap_fade` | Mean Reversion | Overnight gap fill |
| `week52_low_bounce` | Value | Bounce from 52-week low |
| `sector_rotation` | Momentum | Buy leaders in hot sectors |

---

## Typical Workflow

**First time setup:**
```bash
stockpulse init
```

**Optimize strategy parameters:**
```bash
stockpulse optimize
```

**Daily operation (two terminals):**
```bash
stockpulse run
stockpulse dashboard
```

**Fresh start (reset all trading data):**
```bash
rm -f data/stockpulse.sqlite data/stockpulse.sqlite-wal data/stockpulse.sqlite-shm
stockpulse init
stockpulse optimize
git add config/config.yaml
git commit -m "Optimized strategy params"
git push origin main
```

**When will you see trades?**
- Signals generate during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
- Scheduler scans every 15 minutes
- Alerts sent via email when high-confidence signals found

---

## Git Workflow

### Pull Latest Changes

Always pull before starting work:
```bash
cd ~/Documents/AIGames/mega-cap
git pull origin claude/init-repo-setup-maaOL
```

### Check Current Status
```bash
git status
git log --oneline -5   # See recent commits
```

### Push to Feature Branch
```bash
git add .
git commit -m "Your commit message"
git push origin claude/init-repo-setup-maaOL
```

### Merge to Main (when ready for production)

**Option 1: Fast-forward merge (clean history)**
```bash
git checkout main
git pull origin main
git merge claude/init-repo-setup-maaOL
git push origin main
```

**Option 2: Create a Pull Request (recommended for review)**
```bash
# Push your branch first
git push origin claude/init-repo-setup-maaOL

# Then create PR on GitHub or via CLI:
gh pr create --base main --head claude/init-repo-setup-maaOL --title "Your PR title"
```

### Start Fresh Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## Manual Setup (Per Terminal)

If you don't set up the alias, run these in each new terminal:

```bash
cd ~/Documents/AIGames/mega-cap
source venv/bin/activate
python3 -m stockpulse.main <command>
```

The `.env` file is automatically loaded by the application.

---

## Database Management

**Wipe everything and start fresh:**
```bash
rm -f data/stockpulse.sqlite data/stockpulse.sqlite-wal data/stockpulse.sqlite-shm
stockpulse init
```

**Clear trades but keep market data (prices, universe):**
```bash
stockpulse reset --keep-market-data
```

Or via Python:
```bash
cd ~/Documents/AIGames/mega-cap && source venv/bin/activate
python3 -c "from stockpulse.data.database import reset_trading_data; print(reset_trading_data())"
```

**Clear everything including market data:**
```bash
cd ~/Documents/AIGames/mega-cap && source venv/bin/activate
python3 -c "from stockpulse.data.database import reset_trading_data; print(reset_trading_data(keep_market_data=False))"
```

**Get data summary:**
```bash
cd ~/Documents/AIGames/mega-cap && source venv/bin/activate
python3 -c "from stockpulse.data.database import get_data_summary; import json; print(json.dumps(get_data_summary(), indent=2, default=str))"
```

**Check for stuck Python processes:**
```bash
ps aux | grep python | grep -v grep
# Kill any stuck process:
kill <PID>
```

---

## Environment Variables (.env file)

Located at `~/Documents/AIGames/mega-cap/.env`:

```
STOCKPULSE_EMAIL_SENDER=your.email@gmail.com
STOCKPULSE_EMAIL_RECIPIENT=your.email@gmail.com
STOCKPULSE_EMAIL_PASSWORD=your-16-char-app-password
STOCKPULSE_INITIAL_CAPITAL=100000
STOCKPULSE_POSITION_SIZE_PCT=5.0
STOCKPULSE_MAX_POSITIONS=20
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'stockpulse'` | Activate venv: `source venv/bin/activate` |
| `Email not configured` | Check `.env` file has correct values |
| Dashboard shows 0 stocks | Run `stockpulse init` to fetch data |
| No signals appearing | Make sure `stockpulse run` is running during market hours |
| Stale dashboard data | Refresh browser; ensure scheduler is running |
| Want to reset trades | Use `reset_trading_data()` - see Database Management above |
| Database locked error | Use SQLite (already configured); only one writer allowed |

---

## Schedule (When Things Run)

| Job | Time (ET) | Frequency |
|-----|-----------|-----------|
| Intraday scan | 9:30 AM - 4:00 PM | Every 15 minutes |
| Daily summary | 4:30 PM | Once per day (Mon-Fri) |
| Long-term scan | 5:30 PM | Once per day (Mon-Fri) |

Outside market hours, the scheduler runs but skips intraday scans.
