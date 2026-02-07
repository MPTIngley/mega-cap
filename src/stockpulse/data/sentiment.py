"""Sentiment Analysis Module - StockTwits, Finnhub, and AI-powered analysis.

*** ISOLATED MODULE - CAN BE REMOVED IF NOT WORKING ***

This module provides:
1. StockTwits sentiment fetching (FREE, no API key required)
2. Finnhub social sentiment (FREE tier with API key)
3. Haiku-powered sentiment summarization
4. Aggregated sentiment scores for stock analysis
5. Database caching for daily sentiment storage

INTEGRATION STATUS:
- [x] AI Pulse Scanner (Phase 7 - current focus)
- [ ] Trillion+ Club (future integration)
- [ ] Long-Term Scanner (future integration)

DAILY WORKFLOW:
1. run_daily_sentiment_scan() fetches sentiment for AI universe stocks
2. Results stored in sentiment_daily table
3. AI Pulse Scanner reads cached sentiment
4. Haiku analyzes and summarizes for email

Usage:
    sentiment = SentimentAnalyzer()

    # Get sentiment for a single ticker
    result = sentiment.get_sentiment("AAPL")

    # Get sentiment for multiple tickers
    results = sentiment.get_bulk_sentiment(["AAPL", "NVDA", "MSFT"])

    # Run daily scan and store in DB (called by scheduler)
    run_daily_sentiment_scan()
"""

import os
import time
import json
from datetime import datetime, date, timedelta
from typing import Any
from dataclasses import dataclass

import requests

from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db

logger = get_logger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    ticker: str
    source: str
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    total_messages: int = 0
    sentiment_score: float = 50.0  # 0-100, 50 = neutral
    sentiment_label: str = "neutral"  # bullish, bearish, neutral
    trending: bool = False
    message_velocity: float = 0.0  # messages per hour
    sample_messages: list = None
    fetched_at: str = ""
    error: str = ""

    def __post_init__(self):
        if self.sample_messages is None:
            self.sample_messages = []
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "source": self.source,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "total_messages": self.total_messages,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "trending": self.trending,
            "message_velocity": self.message_velocity,
            "sample_messages": self.sample_messages,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


# ============================================================================
# STOCKTWITS INTEGRATION (FREE - NO API KEY REQUIRED)
# ============================================================================

class StockTwitsFetcher:
    """
    Fetch sentiment data from StockTwits.

    FREE API - No authentication required.
    Returns last 30 messages for any ticker with user-tagged sentiment.

    API Docs: https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
    """

    BASE_URL = "https://api.stocktwits.com/api/2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "StockPulse/1.0 (Sentiment Analyzer)"
        })
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 500ms between requests to be nice

    def _rate_limit(self):
        """Simple rate limiting to avoid hammering the API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_sentiment(self, ticker: str) -> SentimentResult:
        """
        Get sentiment for a ticker from StockTwits.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            SentimentResult with sentiment data
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/streams/symbol/{ticker.upper()}.json"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 404:
                return SentimentResult(
                    ticker=ticker,
                    source="stocktwits",
                    error="Ticker not found on StockTwits"
                )

            if response.status_code == 429:
                return SentimentResult(
                    ticker=ticker,
                    source="stocktwits",
                    error="Rate limited - try again later"
                )

            response.raise_for_status()
            data = response.json()

            # Parse the response
            messages = data.get("messages", [])
            symbol_info = data.get("symbol", {})

            # Count sentiment
            bullish = 0
            bearish = 0
            neutral = 0
            sample_messages = []

            for msg in messages:
                entities = msg.get("entities", {})
                sentiment = entities.get("sentiment", {})
                sentiment_basic = sentiment.get("basic") if sentiment else None

                if sentiment_basic == "Bullish":
                    bullish += 1
                elif sentiment_basic == "Bearish":
                    bearish += 1
                else:
                    neutral += 1

                # Collect sample messages (first 5 with sentiment)
                if len(sample_messages) < 5 and sentiment_basic:
                    sample_messages.append({
                        "text": msg.get("body", "")[:200],
                        "sentiment": sentiment_basic,
                        "created_at": msg.get("created_at", ""),
                        "likes": msg.get("likes", {}).get("total", 0),
                    })

            total = len(messages)

            # Calculate sentiment score (0-100)
            # 50 = neutral, >50 = bullish, <50 = bearish
            if total > 0:
                # Weight bullish/bearish, ignore neutral for score
                sentiment_total = bullish + bearish
                if sentiment_total > 0:
                    sentiment_score = (bullish / sentiment_total) * 100
                else:
                    sentiment_score = 50.0
            else:
                sentiment_score = 50.0

            # Determine label
            if sentiment_score >= 65:
                sentiment_label = "bullish"
            elif sentiment_score <= 35:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            # Check if trending
            is_trending = symbol_info.get("is_following", False) or total >= 25

            # Calculate message velocity (approximate)
            # StockTwits returns last 30 messages, estimate time span
            if messages and len(messages) >= 2:
                try:
                    first_time = datetime.fromisoformat(messages[0]["created_at"].replace("Z", "+00:00"))
                    last_time = datetime.fromisoformat(messages[-1]["created_at"].replace("Z", "+00:00"))
                    hours_span = max((first_time - last_time).total_seconds() / 3600, 0.1)
                    velocity = len(messages) / hours_span
                except Exception:
                    velocity = 0.0
            else:
                velocity = 0.0

            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                bullish_count=bullish,
                bearish_count=bearish,
                neutral_count=neutral,
                total_messages=total,
                sentiment_score=round(sentiment_score, 1),
                sentiment_label=sentiment_label,
                trending=is_trending,
                message_velocity=round(velocity, 1),
                sample_messages=sample_messages,
            )

        except requests.exceptions.Timeout:
            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                error="Request timed out"
            )
        except requests.exceptions.RequestException as e:
            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                error=f"Request error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"StockTwits error for {ticker}: {e}")
            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                error=f"Error: {str(e)}"
            )


# ============================================================================
# FINNHUB INTEGRATION (FREE TIER - API KEY REQUIRED)
# ============================================================================

class FinnhubFetcher:
    """
    Fetch sentiment data from Finnhub.

    FREE TIER: 60 requests/minute
    Requires API key - sign up at https://finnhub.io/register

    Provides:
    - Social sentiment scores (Reddit, Twitter mentions)
    - News sentiment
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.is_configured = bool(self.api_key)

        if not self.is_configured:
            logger.info(
                "FINNHUB_API_KEY not set. Finnhub sentiment disabled. "
                "Get a free key at https://finnhub.io/register"
            )

        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 1.0  # 1 second between requests (60/min limit)

    def _rate_limit(self):
        """Rate limiting for Finnhub's 60 req/min limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_social_sentiment(self, ticker: str) -> SentimentResult:
        """
        Get social sentiment from Finnhub.

        Note: Finnhub's social sentiment endpoint may require premium.
        This uses the free news sentiment as fallback.
        """
        if not self.is_configured:
            return SentimentResult(
                ticker=ticker,
                source="finnhub",
                error="FINNHUB_API_KEY not configured"
            )

        self._rate_limit()

        # Try news sentiment (available on free tier)
        url = f"{self.BASE_URL}/news-sentiment"
        params = {
            "symbol": ticker.upper(),
            "token": self.api_key
        }

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 401:
                return SentimentResult(
                    ticker=ticker,
                    source="finnhub",
                    error="Invalid API key"
                )

            if response.status_code == 429:
                return SentimentResult(
                    ticker=ticker,
                    source="finnhub",
                    error="Rate limited"
                )

            response.raise_for_status()
            data = response.json()

            # Parse Finnhub sentiment data
            sentiment_data = data.get("sentiment", {})
            buzz = data.get("buzz", {})

            # Finnhub returns sentiment on -1 to 1 scale
            # Convert to 0-100
            raw_score = sentiment_data.get("bullishPercent", 0.5)  # 0-1
            sentiment_score = raw_score * 100

            # Article counts
            articles_week = buzz.get("articlesInLastWeek", 0)

            # Determine label
            if sentiment_score >= 60:
                sentiment_label = "bullish"
            elif sentiment_score <= 40:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            return SentimentResult(
                ticker=ticker,
                source="finnhub",
                bullish_count=int(raw_score * articles_week),
                bearish_count=int((1 - raw_score) * articles_week),
                total_messages=articles_week,
                sentiment_score=round(sentiment_score, 1),
                sentiment_label=sentiment_label,
                trending=articles_week > 50,
                message_velocity=articles_week / 168,  # per hour over a week
            )

        except Exception as e:
            logger.debug(f"Finnhub error for {ticker}: {e}")
            return SentimentResult(
                ticker=ticker,
                source="finnhub",
                error=f"Error: {str(e)}"
            )


# ============================================================================
# HAIKU-POWERED SENTIMENT ANALYSIS
# ============================================================================

class HaikuSentimentAnalyzer:
    """
    Use Claude Haiku to analyze and summarize sentiment data.

    Cost-effective AI analysis of raw sentiment data.
    """

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.is_configured = bool(self.api_key)

        if not self.is_configured:
            logger.debug("ANTHROPIC_API_KEY not set. AI sentiment analysis disabled.")

    def analyze_sentiment(
        self,
        ticker: str,
        stocktwits_result: SentimentResult | None = None,
        finnhub_result: SentimentResult | None = None,
        price_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        Use Haiku to analyze sentiment data and generate insights.

        Args:
            ticker: Stock ticker
            stocktwits_result: StockTwits sentiment data
            finnhub_result: Finnhub sentiment data
            price_context: Optional dict with price/performance data

        Returns:
            Dict with AI-generated sentiment analysis
        """
        if not self.is_configured:
            return {
                "ticker": ticker,
                "summary": "AI analysis not available (ANTHROPIC_API_KEY not set)",
                "recommendation": "neutral",
                "confidence": 0,
            }

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            # Build context from sentiment data
            context_parts = []

            if stocktwits_result and not stocktwits_result.error:
                st = stocktwits_result
                context_parts.append(f"""
StockTwits Data:
- Total messages (last 30): {st.total_messages}
- Bullish: {st.bullish_count}, Bearish: {st.bearish_count}, Neutral: {st.neutral_count}
- Sentiment Score: {st.sentiment_score}/100 ({st.sentiment_label})
- Message velocity: {st.message_velocity:.1f} msgs/hour
- Trending: {st.trending}

Sample messages:
{chr(10).join([f"  - [{m['sentiment']}] {m['text'][:100]}..." for m in st.sample_messages[:3]])}
""")

            if finnhub_result and not finnhub_result.error:
                fh = finnhub_result
                context_parts.append(f"""
Finnhub News Sentiment:
- Articles this week: {fh.total_messages}
- Sentiment Score: {fh.sentiment_score}/100 ({fh.sentiment_label})
""")

            if price_context:
                context_parts.append(f"""
Price Context:
- Current Price: ${price_context.get('price', 'N/A')}
- 30-day change: {price_context.get('pct_30d', 0):+.1f}%
- RSI: {price_context.get('rsi', 50):.0f}
""")

            if not context_parts:
                return {
                    "ticker": ticker,
                    "summary": "No sentiment data available",
                    "recommendation": "neutral",
                    "confidence": 0,
                }

            prompt = f"""Analyze the social sentiment data for {ticker} and provide a brief, actionable summary.

{chr(10).join(context_parts)}

Provide:
1. SENTIMENT SUMMARY (2-3 sentences): What is retail/social sentiment saying?
2. KEY SIGNALS: Any notable patterns or warnings in the chatter?
3. RECOMMENDATION: bullish/bearish/neutral based on sentiment
4. CONFIDENCE: 0-100 based on data quality and signal strength

Be concise. Focus on actionable insights. Under 150 words total."""

            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = message.content[0].text

            # Parse recommendation
            analysis_lower = analysis.lower()
            if "recommendation: bullish" in analysis_lower or "bullish" in analysis_lower.split("recommendation")[-1][:50]:
                recommendation = "bullish"
            elif "recommendation: bearish" in analysis_lower or "bearish" in analysis_lower.split("recommendation")[-1][:50]:
                recommendation = "bearish"
            else:
                recommendation = "neutral"

            # Parse confidence (rough extraction)
            confidence = 50
            if "confidence:" in analysis_lower:
                try:
                    conf_part = analysis_lower.split("confidence:")[-1][:20]
                    import re
                    conf_match = re.search(r'(\d+)', conf_part)
                    if conf_match:
                        confidence = min(100, max(0, int(conf_match.group(1))))
                except Exception:
                    pass

            return {
                "ticker": ticker,
                "summary": analysis,
                "recommendation": recommendation,
                "confidence": confidence,
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Haiku analysis error for {ticker}: {e}")
            return {
                "ticker": ticker,
                "summary": f"Analysis error: {str(e)}",
                "recommendation": "neutral",
                "confidence": 0,
            }


# ============================================================================
# MAIN SENTIMENT ANALYZER (COMBINES ALL SOURCES)
# ============================================================================

class SentimentAnalyzer:
    """
    Unified sentiment analyzer combining multiple data sources.

    Sources:
    1. StockTwits (FREE, no API key) - Retail trader sentiment
    2. Finnhub (FREE tier, API key required) - News sentiment
    3. Haiku AI (requires ANTHROPIC_API_KEY) - Intelligent summarization

    Usage:
        analyzer = SentimentAnalyzer()

        # Get sentiment for one ticker
        result = analyzer.get_sentiment("NVDA")

        # Get sentiment for multiple tickers
        results = analyzer.get_bulk_sentiment(["NVDA", "AAPL", "MSFT"])
    """

    def __init__(self):
        self.stocktwits = StockTwitsFetcher()
        self.finnhub = FinnhubFetcher()
        self.haiku = HaikuSentimentAnalyzer()

        # Cache results to avoid hammering APIs
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def _cache_key(self, ticker: str) -> str:
        return f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M')[:11]}"  # 5-min buckets

    def get_sentiment(
        self,
        ticker: str,
        include_ai_analysis: bool = True,
        price_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        Get comprehensive sentiment for a ticker.

        Args:
            ticker: Stock ticker symbol
            include_ai_analysis: Whether to run Haiku analysis
            price_context: Optional price data for context

        Returns:
            Dict with all sentiment data and AI analysis
        """
        cache_key = self._cache_key(ticker)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch from all sources
        stocktwits_result = self.stocktwits.get_sentiment(ticker)
        finnhub_result = self.finnhub.get_social_sentiment(ticker) if self.finnhub.is_configured else None

        # Calculate aggregate score
        scores = []
        if stocktwits_result and not stocktwits_result.error:
            scores.append(stocktwits_result.sentiment_score)
        if finnhub_result and not finnhub_result.error:
            scores.append(finnhub_result.sentiment_score)

        aggregate_score = sum(scores) / len(scores) if scores else 50.0

        # Determine aggregate label
        if aggregate_score >= 60:
            aggregate_label = "bullish"
        elif aggregate_score <= 40:
            aggregate_label = "bearish"
        else:
            aggregate_label = "neutral"

        # Get AI analysis if requested
        ai_analysis = None
        if include_ai_analysis and self.haiku.is_configured:
            ai_analysis = self.haiku.analyze_sentiment(
                ticker=ticker,
                stocktwits_result=stocktwits_result,
                finnhub_result=finnhub_result,
                price_context=price_context,
            )

        result = {
            "ticker": ticker,
            "aggregate_score": round(aggregate_score, 1),
            "aggregate_label": aggregate_label,
            "stocktwits": stocktwits_result.to_dict() if stocktwits_result else None,
            "finnhub": finnhub_result.to_dict() if finnhub_result else None,
            "ai_analysis": ai_analysis,
            "fetched_at": datetime.now().isoformat(),
        }

        # Cache result
        self._cache[cache_key] = result
        return result

    def get_bulk_sentiment(
        self,
        tickers: list[str],
        include_ai_analysis: bool = False,  # Default off for bulk to save API calls
        max_tickers: int = 20,
    ) -> dict[str, dict]:
        """
        Get sentiment for multiple tickers.

        Args:
            tickers: List of ticker symbols
            include_ai_analysis: Whether to run Haiku analysis (default off for bulk)
            max_tickers: Maximum tickers to process (to avoid rate limits)

        Returns:
            Dict mapping ticker -> sentiment data
        """
        results = {}

        for ticker in tickers[:max_tickers]:
            try:
                results[ticker] = self.get_sentiment(
                    ticker,
                    include_ai_analysis=include_ai_analysis
                )
                # Small delay between tickers to be nice to APIs
                time.sleep(0.3)
            except Exception as e:
                logger.debug(f"Error getting sentiment for {ticker}: {e}")
                results[ticker] = {
                    "ticker": ticker,
                    "error": str(e),
                    "aggregate_score": 50.0,
                    "aggregate_label": "neutral",
                }

        return results

    def get_sector_sentiment(
        self,
        tickers_by_sector: dict[str, list[str]],
    ) -> dict[str, dict]:
        """
        Get aggregated sentiment by sector.

        Args:
            tickers_by_sector: Dict mapping sector name to list of tickers

        Returns:
            Dict with sector-level sentiment aggregation
        """
        sector_results = {}

        for sector, tickers in tickers_by_sector.items():
            # Get sentiment for sector tickers (limit to 5 per sector)
            ticker_sentiments = self.get_bulk_sentiment(tickers[:5], include_ai_analysis=False)

            # Aggregate
            scores = [
                s["aggregate_score"]
                for s in ticker_sentiments.values()
                if "aggregate_score" in s
            ]

            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= 60:
                    label = "bullish"
                elif avg_score <= 40:
                    label = "bearish"
                else:
                    label = "neutral"
            else:
                avg_score = 50.0
                label = "neutral"

            sector_results[sector] = {
                "sector": sector,
                "avg_sentiment_score": round(avg_score, 1),
                "sentiment_label": label,
                "tickers_analyzed": len(scores),
                "ticker_sentiments": ticker_sentiments,
            }

        return sector_results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_quick_sentiment(ticker: str) -> dict:
    """
    Quick sentiment check for a single ticker (StockTwits only).

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with basic sentiment data
    """
    fetcher = StockTwitsFetcher()
    result = fetcher.get_sentiment(ticker)
    return result.to_dict()


def format_sentiment_for_email(sentiment_data: dict) -> str:
    """
    Format sentiment data for inclusion in email digests.

    Args:
        sentiment_data: Result from SentimentAnalyzer.get_sentiment()

    Returns:
        HTML-formatted sentiment section
    """
    ticker = sentiment_data.get("ticker", "N/A")
    score = sentiment_data.get("aggregate_score", 50)
    label = sentiment_data.get("aggregate_label", "neutral")

    # Color based on sentiment
    if label == "bullish":
        color = "#4ade80"
        emoji = "🟢"
    elif label == "bearish":
        color = "#f87171"
        emoji = "🔴"
    else:
        color = "#fbbf24"
        emoji = "🟡"

    # Get StockTwits details
    st = sentiment_data.get("stocktwits", {})
    st_total = st.get("total_messages", 0)
    st_bullish = st.get("bullish_count", 0)
    st_bearish = st.get("bearish_count", 0)

    html = f"""
    <div style="background: #1e293b; border-radius: 8px; padding: 12px; margin: 8px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-weight: bold; color: #e2e8f0;">{ticker} Sentiment</span>
            <span style="color: {color}; font-weight: bold;">{emoji} {score:.0f}/100 ({label.upper()})</span>
        </div>
        <div style="color: #94a3b8; font-size: 11px; margin-top: 6px;">
            StockTwits: {st_total} messages | 🟢 {st_bullish} bullish | 🔴 {st_bearish} bearish
        </div>
    """

    # Add AI analysis if available
    ai = sentiment_data.get("ai_analysis", {})
    if ai and ai.get("summary"):
        summary = ai["summary"][:300]
        html += f"""
        <div style="color: #cbd5e1; font-size: 12px; margin-top: 8px; padding-top: 8px; border-top: 1px solid #334155;">
            <strong>AI Analysis:</strong> {summary}...
        </div>
        """

    html += "</div>"
    return html


# ============================================================================
# DATABASE STORAGE FOR DAILY SENTIMENT CACHING
# ============================================================================

class SentimentStorage:
    """
    Database storage for sentiment data.

    Stores daily sentiment scans so the AI Pulse scanner can read
    cached results without hitting APIs repeatedly.
    """

    def __init__(self):
        self.db = get_db()
        self._init_tables()

    def _init_tables(self):
        """Create sentiment tables if they don't exist."""
        cursor = self.db.conn.cursor()

        # Daily sentiment snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                source TEXT DEFAULT 'stocktwits',
                bullish_count INTEGER DEFAULT 0,
                bearish_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                sentiment_score REAL DEFAULT 50.0,
                sentiment_label TEXT DEFAULT 'neutral',
                trending INTEGER DEFAULT 0,
                message_velocity REAL DEFAULT 0.0,
                sample_messages TEXT,
                ai_summary TEXT,
                ai_recommendation TEXT,
                ai_confidence REAL DEFAULT 0,
                raw_data TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (ticker, scan_date, source)
            )
        """)

        # Category-level sentiment aggregation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_category_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                avg_sentiment_score REAL DEFAULT 50.0,
                sentiment_label TEXT DEFAULT 'neutral',
                tickers_analyzed INTEGER DEFAULT 0,
                top_bullish_ticker TEXT,
                top_bearish_ticker TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (category, scan_date)
            )
        """)

        self.db.conn.commit()
        logger.debug("Sentiment tables initialized")

    def store_sentiment(self, ticker: str, sentiment_data: dict) -> bool:
        """Store sentiment data for a ticker."""
        try:
            today = date.today().isoformat()

            # Extract StockTwits data
            st = sentiment_data.get("stocktwits", {})
            ai = sentiment_data.get("ai_analysis", {})

            self.db.execute("""
                INSERT INTO sentiment_daily (
                    ticker, scan_date, source,
                    bullish_count, bearish_count, neutral_count, total_messages,
                    sentiment_score, sentiment_label, trending, message_velocity,
                    sample_messages, ai_summary, ai_recommendation, ai_confidence,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, scan_date, source) DO UPDATE SET
                    bullish_count = excluded.bullish_count,
                    bearish_count = excluded.bearish_count,
                    neutral_count = excluded.neutral_count,
                    total_messages = excluded.total_messages,
                    sentiment_score = excluded.sentiment_score,
                    sentiment_label = excluded.sentiment_label,
                    trending = excluded.trending,
                    message_velocity = excluded.message_velocity,
                    sample_messages = excluded.sample_messages,
                    ai_summary = excluded.ai_summary,
                    ai_recommendation = excluded.ai_recommendation,
                    ai_confidence = excluded.ai_confidence,
                    raw_data = excluded.raw_data
            """, (
                ticker,
                today,
                "stocktwits",
                st.get("bullish_count", 0),
                st.get("bearish_count", 0),
                st.get("neutral_count", 0),
                st.get("total_messages", 0),
                sentiment_data.get("aggregate_score", 50.0),
                sentiment_data.get("aggregate_label", "neutral"),
                1 if st.get("trending", False) else 0,
                st.get("message_velocity", 0.0),
                json.dumps(st.get("sample_messages", [])),
                ai.get("summary", "") if ai else "",
                ai.get("recommendation", "neutral") if ai else "neutral",
                ai.get("confidence", 0) if ai else 0,
                json.dumps(sentiment_data),
            ))
            return True

        except Exception as e:
            logger.error(f"Error storing sentiment for {ticker}: {e}")
            return False

    def get_cached_sentiment(self, ticker: str, max_age_hours: int = 24) -> dict | None:
        """
        Get cached sentiment for a ticker.

        Args:
            ticker: Stock ticker
            max_age_hours: Maximum age of cached data to return

        Returns:
            Cached sentiment data or None if not found/expired
        """
        try:
            cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

            row = self.db.fetchone("""
                SELECT raw_data FROM sentiment_daily
                WHERE ticker = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ticker, cutoff))

            if row and row[0]:
                return json.loads(row[0])
            return None

        except Exception as e:
            logger.debug(f"Error getting cached sentiment for {ticker}: {e}")
            return None

    def get_todays_sentiment(self, tickers: list[str] | None = None) -> dict[str, dict]:
        """
        Get all sentiment data from today's scan.

        Args:
            tickers: Optional list to filter by

        Returns:
            Dict mapping ticker -> sentiment data
        """
        try:
            today = date.today().isoformat()

            if tickers:
                placeholders = ",".join(["?" for _ in tickers])
                query = f"""
                    SELECT ticker, raw_data FROM sentiment_daily
                    WHERE scan_date = ? AND ticker IN ({placeholders})
                """
                rows = self.db.fetchdf(query, (today, *tickers))
            else:
                rows = self.db.fetchdf("""
                    SELECT ticker, raw_data FROM sentiment_daily
                    WHERE scan_date = ?
                """, (today,))

            results = {}
            if not rows.empty:
                for _, row in rows.iterrows():
                    try:
                        results[row["ticker"]] = json.loads(row["raw_data"])
                    except Exception:
                        pass
            return results

        except Exception as e:
            logger.error(f"Error getting today's sentiment: {e}")
            return {}

    def store_category_sentiment(self, category: str, data: dict) -> bool:
        """Store aggregated category sentiment."""
        try:
            today = date.today().isoformat()

            self.db.execute("""
                INSERT INTO sentiment_category_daily (
                    category, scan_date, avg_sentiment_score, sentiment_label,
                    tickers_analyzed, top_bullish_ticker, top_bearish_ticker
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (category, scan_date) DO UPDATE SET
                    avg_sentiment_score = excluded.avg_sentiment_score,
                    sentiment_label = excluded.sentiment_label,
                    tickers_analyzed = excluded.tickers_analyzed,
                    top_bullish_ticker = excluded.top_bullish_ticker,
                    top_bearish_ticker = excluded.top_bearish_ticker
            """, (
                category,
                today,
                data.get("avg_sentiment_score", 50.0),
                data.get("sentiment_label", "neutral"),
                data.get("tickers_analyzed", 0),
                data.get("top_bullish_ticker", ""),
                data.get("top_bearish_ticker", ""),
            ))
            return True

        except Exception as e:
            logger.error(f"Error storing category sentiment for {category}: {e}")
            return False


# ============================================================================
# DAILY SENTIMENT SCAN (CALLED BY SCHEDULER)
# ============================================================================

def run_daily_sentiment_scan(
    tickers: list[str] | None = None,
    include_ai_analysis: bool = True,
    max_tickers: int = 50,
) -> dict[str, Any]:
    """
    Run daily sentiment scan for AI universe stocks.

    This function:
    1. Fetches sentiment from StockTwits for all AI universe tickers
    2. Runs Haiku analysis on top movers
    3. Stores results in database for AI Pulse scanner to use

    Args:
        tickers: List of tickers to scan (defaults to AI universe)
        include_ai_analysis: Run Haiku on top movers
        max_tickers: Maximum tickers to scan

    Returns:
        Summary of scan results
    """
    from stockpulse.scanner.ai_pulse import AI_UNIVERSE

    logger.info("Starting daily sentiment scan...")

    # Use provided tickers or default to AI universe
    target_tickers = tickers or list(AI_UNIVERSE)
    target_tickers = target_tickers[:max_tickers]

    # Initialize
    analyzer = SentimentAnalyzer()
    storage = SentimentStorage()

    results = {
        "scan_date": date.today().isoformat(),
        "tickers_scanned": 0,
        "successful": 0,
        "failed": 0,
        "bullish": [],
        "bearish": [],
        "trending": [],
        "errors": [],
    }

    # Fetch sentiment for each ticker
    for i, ticker in enumerate(target_tickers):
        try:
            # Get sentiment (without AI analysis for bulk - save API calls)
            sentiment = analyzer.get_sentiment(ticker, include_ai_analysis=False)

            # Store in database
            if storage.store_sentiment(ticker, sentiment):
                results["successful"] += 1

                # Track notable sentiment
                score = sentiment.get("aggregate_score", 50)
                label = sentiment.get("aggregate_label", "neutral")
                st = sentiment.get("stocktwits", {})

                if score >= 65:
                    results["bullish"].append({"ticker": ticker, "score": score})
                elif score <= 35:
                    results["bearish"].append({"ticker": ticker, "score": score})

                if st.get("trending"):
                    results["trending"].append(ticker)
            else:
                results["failed"] += 1

            results["tickers_scanned"] += 1

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"  Scanned {i + 1}/{len(target_tickers)} tickers...")

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{ticker}: {str(e)}")
            logger.debug(f"Error scanning {ticker}: {e}")

    # Run AI analysis on most extreme sentiment stocks
    if include_ai_analysis and analyzer.haiku.is_configured:
        logger.info("Running Haiku analysis on notable stocks...")

        # Analyze top bullish and bearish
        notable_tickers = (
            [t["ticker"] for t in results["bullish"][:3]] +
            [t["ticker"] for t in results["bearish"][:3]]
        )

        for ticker in notable_tickers:
            try:
                # Get fresh sentiment with AI analysis
                sentiment = analyzer.get_sentiment(ticker, include_ai_analysis=True)
                storage.store_sentiment(ticker, sentiment)
            except Exception as e:
                logger.debug(f"AI analysis error for {ticker}: {e}")

    # Sort results
    results["bullish"].sort(key=lambda x: x["score"], reverse=True)
    results["bearish"].sort(key=lambda x: x["score"])

    logger.info(
        f"Daily sentiment scan complete: {results['successful']}/{results['tickers_scanned']} successful, "
        f"{len(results['bullish'])} bullish, {len(results['bearish'])} bearish, "
        f"{len(results['trending'])} trending"
    )

    return results


def get_sentiment_summary_for_email(tickers: list[str], max_display: int = 10) -> str:
    """
    Get formatted sentiment summary for email inclusion.

    Reads from cached daily sentiment data.

    Args:
        tickers: Tickers to include
        max_display: Max tickers to show

    Returns:
        HTML formatted sentiment section
    """
    storage = SentimentStorage()
    cached = storage.get_todays_sentiment(tickers)

    if not cached:
        return """
        <div style="background: #1e293b; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h3 style="color: #94a3b8; margin: 0;">Social Sentiment</h3>
            <p style="color: #64748b; font-size: 12px;">No sentiment data available. Run: stockpulse sentiment-scan</p>
        </div>
        """

    # Sort by sentiment score
    sorted_tickers = sorted(
        [(t, d) for t, d in cached.items()],
        key=lambda x: x[1].get("aggregate_score", 50),
        reverse=True
    )

    html_parts = ["""
    <div style="background: #0f3460; padding: 15px; border-radius: 8px; margin: 15px 0;">
        <h3 style="color: #00d9ff; margin: 0 0 15px 0; border-bottom: 1px solid #1e3a5f; padding-bottom: 10px;">
            📊 Social Sentiment (StockTwits)
        </h3>
        <table style="width: 100%; font-size: 12px;">
            <tr>
                <th style="text-align: left; color: #94a3b8; padding: 5px;">Ticker</th>
                <th style="text-align: center; color: #94a3b8; padding: 5px;">Score</th>
                <th style="text-align: center; color: #94a3b8; padding: 5px;">Sentiment</th>
                <th style="text-align: right; color: #94a3b8; padding: 5px;">Messages</th>
            </tr>
    """]

    for ticker, data in sorted_tickers[:max_display]:
        score = data.get("aggregate_score", 50)
        label = data.get("aggregate_label", "neutral")
        st = data.get("stocktwits", {})
        total = st.get("total_messages", 0)

        # Color and emoji based on sentiment
        if label == "bullish":
            color = "#4ade80"
            emoji = "🟢"
        elif label == "bearish":
            color = "#f87171"
            emoji = "🔴"
        else:
            color = "#fbbf24"
            emoji = "🟡"

        html_parts.append(f"""
            <tr>
                <td style="padding: 5px; color: #e2e8f0;">{ticker}</td>
                <td style="padding: 5px; text-align: center; color: {color}; font-weight: bold;">{score:.0f}</td>
                <td style="padding: 5px; text-align: center;">{emoji} {label.upper()}</td>
                <td style="padding: 5px; text-align: right; color: #94a3b8;">{total}</td>
            </tr>
        """)

    html_parts.append("""
        </table>
        <p style="color: #64748b; font-size: 10px; margin: 10px 0 0 0; text-align: right;">
            Data from StockTwits • Score: 0-100 (50=neutral)
        </p>
    </div>
    """)

    return "".join(html_parts)
