# Sentiment System Review & Enhancement Plan

**Date:** 2026-02-08
**Status:** Review Complete, Implementation Ready

---

## Executive Summary

After comprehensive review of the sentiment system, I've identified several bugs, architectural concerns, and opportunities for enhancement. This document outlines findings and proposes a tiered approach to sentiment collection that balances API limits, data richness, and cost.

---

## Part 1: Bugs Found & Fixed

### Bug 1: Cache Key Truncation (FIXED)
**File:** `sentiment.py:571`
```python
# BUG: Truncates to 11 chars, doesn't create proper 5-min buckets
return f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M')[:11]}"

# FIX: Proper 5-minute bucket calculation
now = datetime.now()
bucket = now.minute // 5  # 0-11 for each 5-min window in an hour
return f"{ticker}_{now.strftime('%Y%m%d%H')}_{bucket}"
```

### Bug 2: Missing Error Handling for Haiku (FIXED)
**File:** `sentiment.py:616-623`
The `ai_analysis` variable could fail silently. Added proper try/except.

### Bug 3: No Retry Logic for API Failures
**File:** `sentiment.py` - StockTwitsFetcher and FinnhubFetcher
Added exponential backoff retry logic for transient failures.

---

## Part 2: API Limits Analysis

### Current API Usage

| API | Rate Limit | Current Delay | Daily Calls | Status |
|-----|------------|---------------|-------------|--------|
| StockTwits | ~200/hour (estimated) | 500ms | 80 (once daily) | OK |
| Finnhub | 60/min free tier | 1000ms | 80 (once daily) | OK |
| Claude Haiku | Token-based | N/A | ~6 (top movers) | OK |

### Capacity for Hourly Collection

If we collected hourly during market hours (6.5 hours):
- StockTwits: 80 tickers × 6.5 hours = 520 calls/day (within limits)
- Finnhub: 80 tickers × 6.5 hours = 520 calls/day (within limits if 1 req/sec)

**Verdict:** We have capacity for hourly collection without hitting limits.

---

## Part 3: Architecture Options

### Option A: Hourly Raw Data + Daily Analysis (RECOMMENDED)

```
Schedule:
09:30-16:00 ET (hourly): Collect raw StockTwits data, store in sentiment_hourly
17:00 ET (once daily):   Aggregate hourly data, calculate velocity/trends
17:05 ET:                Run Haiku analysis on aggregated data
17:30 ET:                AI Pulse email uses enriched sentiment
```

**Pros:**
- Detect intraday sentiment shifts and reversals
- Better message velocity calculation (real hourly volume, not estimated)
- Can alert on unusual sentiment spikes
- Still cost-effective (~$5/month Haiku)

**Cons:**
- More database storage (~7x current)
- More complex scheduler setup

### Option B: Tiered Collection (ALTERNATIVE)

```
Top 20 AI stocks (NVDA, MSFT, GOOGL, etc.): Hourly collection
Other 60 AI stocks: Daily collection
All stocks: Daily Haiku analysis
```

**Pros:**
- Focus resources on highest-value stocks
- Lower API usage
- Less database growth

**Cons:**
- Miss shifts in smaller AI stocks

### Option C: Add More Data Sources (COMPLEMENTARY)

Add these FREE data sources to enrich sentiment:

| Source | Data | Cost | API |
|--------|------|------|-----|
| **SEC EDGAR** | 8-K filings, insider trading | FREE | 10 req/sec |
| **Finnhub Analyst Ratings** | Buy/Sell/Hold changes | FREE tier | Already have |
| **Finnhub Insider Transactions** | Insider buys/sells | FREE tier | Already have |
| **FinancialModelingPrep** | Social sentiment (hourly) | FREE 250/day | New |

---

## Part 4: Alternative AI Thesis Data Sources

### Current Problem
We're only using StockTwits social sentiment, which is:
- Noisy (retail traders, not institutions)
- Limited (30 messages per ticker)
- No fundamental signals

### Recommended Additional Sources

#### 1. SEC EDGAR API (FREE - High Priority)
**What:** Real-time SEC filings (8-K, 10-K, 10-Q, Form 4 insider trading)
**Why:** Institutional-quality data, no noise
**API:** https://www.sec.gov/search-filings/edgar-application-programming-interfaces
**Rate:** 10 requests/second (very generous)
**Implementation:** Monitor 8-K filings for material events, track insider buying/selling

#### 2. Finnhub Analyst Ratings (FREE - Have API Key)
**What:** Analyst upgrades/downgrades, price target changes
**Why:** Professional analyst sentiment, actionable
**API:** https://finnhub.io/docs/api/recommendation-trends
**Implementation:** Track rating changes for AI thesis tickers

#### 3. Finnhub Insider Transactions (FREE)
**What:** Form 4 insider buys/sells
**Why:** Insiders know more than retail - insider buying is bullish signal
**API:** https://finnhub.io/docs/api/insider-transactions
**Implementation:** Flag large insider purchases in AI stocks

#### 4. Finnhub Earnings Surprises (FREE)
**What:** EPS beat/miss data
**Why:** Post-earnings drift is real alpha
**API:** https://finnhub.io/docs/api/company-earnings
**Implementation:** Track earnings surprises for thesis validation

#### 5. News Sentiment (Finnhub - FREE)
**What:** News article sentiment scores
**Why:** More signal than social media noise
**Already implemented but underutilized

---

## Part 5: Proposed Implementation Plan

### Phase 7a: Bug Fixes (TODAY)
- [x] Fix cache key truncation bug
- [x] Add retry logic for API failures
- [x] Add error handling for Haiku analysis

### Phase 7b: Hourly Collection (1-2 days)
- [ ] Create `sentiment_hourly` table for raw data
- [ ] Add hourly scheduler job (market hours only)
- [ ] Modify daily aggregation to use hourly data
- [ ] Calculate true message velocity from hourly deltas

### Phase 7c: Additional Data Sources (3-5 days)
- [ ] Add SEC EDGAR 8-K monitoring for AI stocks
- [ ] Add Finnhub analyst rating tracking
- [ ] Add Finnhub insider transaction tracking
- [ ] Create unified "thesis signal" score combining all sources

### Phase 7d: Enhanced Analysis (Future)
- [ ] Sentiment reversal detection (bullish→bearish alerts)
- [ ] Insider buying alerts for thesis stocks
- [ ] Analyst upgrade/downgrade alerts
- [ ] Integration with Trillion+ Club and Long-Term scanners

---

## Part 6: Database Schema Updates

### New Table: sentiment_hourly
```sql
CREATE TABLE sentiment_hourly (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    scan_datetime TEXT NOT NULL,  -- Full timestamp
    source TEXT DEFAULT 'stocktwits',
    bullish_count INTEGER DEFAULT 0,
    bearish_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0,
    sentiment_score REAL DEFAULT 50.0,
    raw_data TEXT,  -- JSON for debugging
    UNIQUE (ticker, scan_datetime, source)
);
```

### New Table: thesis_signals
```sql
CREATE TABLE thesis_signals (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- 'insider_buy', 'analyst_upgrade', '8k_filing', etc.
    signal_data TEXT,  -- JSON with details
    impact_score REAL DEFAULT 0,  -- -100 to +100
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE (ticker, signal_date, signal_type)
);
```

---

## Part 7: Cost Analysis

### Current Monthly Cost
| Component | Cost |
|-----------|------|
| StockTwits | $0 (free) |
| Finnhub | $0 (free tier) |
| Claude Haiku (~6 analyses/day) | ~$2 |
| **Total** | **~$2/month** |

### Proposed Monthly Cost (with enhancements)
| Component | Cost |
|-----------|------|
| StockTwits (hourly) | $0 (free) |
| Finnhub (expanded use) | $0 (free tier) |
| SEC EDGAR | $0 (free) |
| Claude Haiku (~20 analyses/day) | ~$5 |
| **Total** | **~$5/month** |

---

## Part 8: Decision Points for Martin

### Question 1: Hourly vs Daily Collection
**Recommendation:** Start with hourly for top 20 AI stocks only (Option B)
- Lower risk, can expand if valuable
- Proves concept before full rollout

### Question 2: Additional Data Sources Priority
**Recommendation:** Start with Finnhub (already have API key)
1. Analyst ratings - easiest to implement
2. Insider transactions - high signal value
3. SEC EDGAR - more complex but free and rich

### Question 3: Haiku Usage
**Current:** Only top 3 bullish + top 3 bearish
**Recommendation:** Expand to top 5 each + thesis-specific analysis
- Still only ~$5/month
- More actionable insights

---

## Summary

The sentiment system is functional but underutilized. Key improvements:

1. **Bug fixes** - Cache key, error handling, retry logic
2. **Hourly collection** - Better velocity/trend detection
3. **More data sources** - SEC, analyst ratings, insider trades
4. **Unified scoring** - Combine all signals into thesis confidence

The current system uses maybe 10% of available free API capacity. We can get significantly more data without increasing costs.
