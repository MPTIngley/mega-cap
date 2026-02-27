"""Sentiment Analysis Module - Multi-source sentiment aggregation.

*** ISOLATED MODULE - CAN BE REMOVED IF NOT WORKING ***

This module provides:
1. StockTwits sentiment fetching (FREE, no API key required)
2. Finnhub social sentiment (FREE tier with API key)
3. Reddit sentiment via PRAW (FREE, requires Reddit app credentials)
4. Google News RSS headlines (FREE, no auth)
5. Google Trends search interest (FREE, no auth)
6. CNN Fear & Greed Index (FREE, no auth)
7. Options put/call ratio via yfinance (FREE, existing dep)
8. Wikipedia page view attention (FREE, no auth)
9. Haiku-powered sentiment summarization
10. Aggregated sentiment scores with configurable weights
11. Database caching for daily sentiment storage

INTEGRATION STATUS:
- [x] AI Pulse Scanner
- [x] Trillion+ Club
- [x] Long-Term Scanner

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

    def __init__(self, max_retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "StockPulse/1.0 (Sentiment Analyzer)"
        })
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 500ms between requests to be nice
        self._max_retries = max_retries

    def _rate_limit(self):
        """Simple rate limiting to avoid hammering the API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_with_retry(self, url: str) -> requests.Response | None:
        """Fetch URL with exponential backoff retry logic."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=10)
                # Don't retry on 404 or 429 - those are valid responses
                if response.status_code in (404, 429) or response.ok:
                    return response
                # Retry on 5xx errors
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self._max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        continue
                return response
            except requests.exceptions.Timeout:
                last_error = "timeout"
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
        logger.debug(f"All {self._max_retries} retries failed for {url}: {last_error}")
        return None

    def get_sentiment(self, ticker: str) -> SentimentResult:
        """
        Get sentiment for a ticker from StockTwits.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            SentimentResult with sentiment data
        """
        url = f"{self.BASE_URL}/streams/symbol/{ticker.upper()}.json"

        # Use retry logic for resilient fetching
        response = self._fetch_with_retry(url)

        if response is None:
            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                error="All retries failed - network error"
            )

        try:
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

        except requests.exceptions.HTTPError as e:
            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                error=f"HTTP error: {str(e)}"
            )
        except json.JSONDecodeError:
            return SentimentResult(
                ticker=ticker,
                source="stocktwits",
                error="Invalid JSON response"
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

    def __init__(self, api_key: str | None = None, max_retries: int = 3):
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
        self._max_retries = max_retries

    def _rate_limit(self):
        """Rate limiting for Finnhub's 60 req/min limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_with_retry(self, url: str, params: dict) -> requests.Response | None:
        """Fetch URL with exponential backoff retry logic."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=10)
                # Don't retry on auth errors or rate limits
                if response.status_code in (401, 429) or response.ok:
                    return response
                # Retry on 5xx errors
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self._max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                return response
            except requests.exceptions.Timeout:
                last_error = "timeout"
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
        logger.debug(f"All {self._max_retries} retries failed for Finnhub {url}: {last_error}")
        return None

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

        # Try news sentiment (available on free tier)
        url = f"{self.BASE_URL}/news-sentiment"
        params = {
            "symbol": ticker.upper(),
            "token": self.api_key
        }

        # Use retry logic for resilient fetching
        response = self._fetch_with_retry(url, params)

        if response is None:
            return SentimentResult(
                ticker=ticker,
                source="finnhub",
                error="All retries failed - network error"
            )

        try:
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
# REDDIT SENTIMENT VIA PRAW (FREE - REQUIRES REDDIT APP CREDENTIALS)
# ============================================================================

# Common words that look like tickers but aren't
TICKER_BLACKLIST = {
    "DD", "CEO", "IPO", "YOLO", "HODL", "FOMO", "FUD", "ATH", "ATL",
    "EPS", "PE", "ETF", "SEC", "FDA", "FED", "GDP", "CPI", "IMO",
    "USA", "UK", "EU", "USD", "EUR", "GBP", "AI", "ML", "API",
    "CEO", "CFO", "COO", "CTO", "VP", "PM", "ER", "IV", "OI",
    "ITM", "OTM", "ATM", "DTE", "RSI", "EMA", "SMA", "MACD",
    "PT", "SP", "DOW", "DJIA", "OTC", "YOY", "MOM", "QOQ",
    "TLDR", "PSA", "LMAO", "IMHO", "BTW", "TIL", "OP", "TL",
    "RH", "TD", "WS", "THE", "AND", "FOR", "ARE", "NOT", "YOU",
    "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "HAD",
    "HAS", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
    "WAY", "WHO", "BOY", "DID", "GET", "LET", "SAY", "SHE", "TOO",
    "USE", "DAD", "MOM", "RUN", "BIG", "RED", "TWO", "ANY", "FEW",
    "GOT", "HIM", "MAN", "OWN", "TOP", "END",
}


class RedditFetcher:
    """
    Fetch sentiment data from Reddit via PRAW.

    FREE - Requires Reddit app credentials (create at reddit.com/prefs/apps).
    Rate limit: 100 req/min (generous).

    Searches across wallstreetbets, stocks, investing, options, stockmarket.

    Environment variables:
        REDDIT_CLIENT_ID: Reddit app client ID
        REDDIT_CLIENT_SECRET: Reddit app secret
        REDDIT_USER_AGENT: User agent string (default: StockPulse/1.0)
    """

    DEFAULT_SUBREDDITS = "wallstreetbets+stocks+investing+options+stockmarket"

    def __init__(self, config: dict | None = None):
        self.client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        self.client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        self.user_agent = os.environ.get("REDDIT_USER_AGENT", "StockPulse/1.0 (sentiment analysis)")
        self.is_configured = bool(self.client_id and self.client_secret)

        self._reddit = None
        self._config = config or {}
        self._subreddits = self._config.get("subreddits", self.DEFAULT_SUBREDDITS)
        self._posts_per_ticker = self._config.get("posts_per_ticker", 20)
        self._lookback_hours = self._config.get("lookback_hours", 24)

        if not self.is_configured:
            logger.info(
                "REDDIT_CLIENT_ID/SECRET not set. Reddit sentiment disabled. "
                "Create a free app at reddit.com/prefs/apps"
            )

    def _get_reddit(self):
        """Lazy-init PRAW Reddit instance."""
        if self._reddit is None:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            except Exception as e:
                logger.error(f"Failed to initialize PRAW: {e}")
                self.is_configured = False
        return self._reddit

    def get_sentiment(self, ticker: str) -> SentimentResult:
        """
        Get sentiment for a ticker from Reddit.

        Searches subreddits for posts mentioning the ticker,
        classifies sentiment based on upvote ratios and keyword analysis.
        """
        if not self.is_configured:
            return SentimentResult(
                ticker=ticker,
                source="reddit",
                error="Reddit credentials not configured"
            )

        reddit = self._get_reddit()
        if reddit is None:
            return SentimentResult(ticker=ticker, source="reddit", error="PRAW init failed")

        try:
            subreddit = reddit.subreddit(self._subreddits)
            query = f"${ticker} OR {ticker}"

            posts = []
            sample_messages = []
            bullish = 0
            bearish = 0
            neutral = 0

            for post in subreddit.search(query, sort="hot", time_filter="day",
                                         limit=self._posts_per_ticker):
                # Verify ticker actually appears in title or selftext
                text = f"{post.title} {post.selftext[:500]}".upper()
                if f"${ticker}" not in text and ticker not in text.split():
                    continue

                posts.append(post)

                # Classify sentiment from upvote ratio and score
                # High upvote ratio + positive keywords = bullish signal
                post_sentiment = self._classify_post(post, ticker)
                if post_sentiment == "bullish":
                    bullish += 1
                elif post_sentiment == "bearish":
                    bearish += 1
                else:
                    neutral += 1

                if len(sample_messages) < 5:
                    sample_messages.append({
                        "text": post.title[:200],
                        "sentiment": post_sentiment,
                        "created_at": datetime.fromtimestamp(post.created_utc).isoformat(),
                        "likes": post.score,
                        "subreddit": str(post.subreddit),
                        "comments": post.num_comments,
                    })

            total = len(posts)

            if total == 0:
                return SentimentResult(
                    ticker=ticker,
                    source="reddit",
                    sentiment_score=50.0,
                    sentiment_label="neutral",
                    total_messages=0,
                )

            # Calculate sentiment score
            sentiment_total = bullish + bearish
            if sentiment_total > 0:
                sentiment_score = (bullish / sentiment_total) * 100
            else:
                sentiment_score = 50.0

            if sentiment_score >= 65:
                sentiment_label = "bullish"
            elif sentiment_score <= 35:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            # Message velocity (posts per hour in last 24h)
            velocity = total / max(self._lookback_hours, 1)

            # Trending if mention count is high for a single day
            trending = total >= 10

            return SentimentResult(
                ticker=ticker,
                source="reddit",
                bullish_count=bullish,
                bearish_count=bearish,
                neutral_count=neutral,
                total_messages=total,
                sentiment_score=round(sentiment_score, 1),
                sentiment_label=sentiment_label,
                trending=trending,
                message_velocity=round(velocity, 2),
                sample_messages=sample_messages,
            )

        except Exception as e:
            logger.debug(f"Reddit error for {ticker}: {e}")
            return SentimentResult(ticker=ticker, source="reddit", error=str(e))

    def _classify_post(self, post, ticker: str) -> str:
        """Classify a Reddit post as bullish/bearish/neutral."""
        title_lower = post.title.lower()

        bullish_keywords = [
            "buy", "calls", "moon", "rocket", "bull", "long", "upgrade",
            "breakout", "undervalued", "squeeze", "gains", "profit",
            "surge", "soar", "rally", "beat", "strong",
        ]
        bearish_keywords = [
            "sell", "puts", "crash", "bear", "short", "downgrade",
            "overvalued", "dump", "loss", "drop", "fall", "tank",
            "plunge", "miss", "weak", "warning", "bubble",
        ]

        bull_hits = sum(1 for kw in bullish_keywords if kw in title_lower)
        bear_hits = sum(1 for kw in bearish_keywords if kw in title_lower)

        # Upvote ratio > 0.8 with decent score suggests agreement
        if post.upvote_ratio > 0.8 and post.score > 10:
            if bull_hits > bear_hits:
                return "bullish"
            elif bear_hits > bull_hits:
                return "bearish"

        if bull_hits > bear_hits + 1:
            return "bullish"
        elif bear_hits > bull_hits + 1:
            return "bearish"

        return "neutral"

    def get_trending_tickers(self, limit: int = 20) -> list[dict]:
        """
        Get most-discussed tickers across Reddit today.

        Returns list of {ticker, mention_count, avg_score, top_subreddit}.
        """
        if not self.is_configured:
            return []

        reddit = self._get_reddit()
        if reddit is None:
            return []

        try:
            import re
            ticker_counts = {}
            subreddit = reddit.subreddit(self._subreddits)

            for post in subreddit.hot(limit=100):
                text = f"{post.title} {post.selftext[:200]}"
                # Find $TICKER patterns and standalone uppercase words
                found_tickers = set()
                for match in re.findall(r'\$([A-Z]{1,5})\b', text):
                    if match not in TICKER_BLACKLIST:
                        found_tickers.add(match)
                for word in text.split():
                    if (re.match(r'^[A-Z]{2,5}$', word) and
                            word not in TICKER_BLACKLIST and len(word) >= 2):
                        found_tickers.add(word)

                for t in found_tickers:
                    if t not in ticker_counts:
                        ticker_counts[t] = {
                            "ticker": t, "mention_count": 0,
                            "total_score": 0, "subreddits": set(),
                        }
                    ticker_counts[t]["mention_count"] += 1
                    ticker_counts[t]["total_score"] += post.score
                    ticker_counts[t]["subreddits"].add(str(post.subreddit))

            # Sort by mention count
            results = []
            for t, data in sorted(ticker_counts.items(),
                                  key=lambda x: x[1]["mention_count"], reverse=True)[:limit]:
                results.append({
                    "ticker": data["ticker"],
                    "mention_count": data["mention_count"],
                    "avg_score": data["total_score"] / max(data["mention_count"], 1),
                    "top_subreddit": max(data["subreddits"], key=lambda s: s) if data["subreddits"] else "",
                })

            return results

        except Exception as e:
            logger.debug(f"Reddit trending error: {e}")
            return []


# ============================================================================
# GOOGLE NEWS RSS SENTIMENT (FREE - NO AUTH)
# ============================================================================

class GoogleNewsFetcher:
    """
    Fetch headline sentiment from Google News RSS.

    FREE - No API key, no rate limit concerns.
    Parses RSS feed for stock-related headlines.

    URL: https://news.google.com/rss/search?q={TICKER}+stock&hl=en-US
    """

    def __init__(self, max_headlines: int = 15):
        self._max_headlines = max_headlines
        self.is_configured = True  # Always available, no auth needed

    def get_sentiment(self, ticker: str) -> SentimentResult:
        """Get sentiment from Google News headlines for a ticker."""
        try:
            import feedparser

            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)

            if not feed.entries:
                return SentimentResult(
                    ticker=ticker, source="google_news",
                    sentiment_score=50.0, sentiment_label="neutral",
                )

            headlines = []
            bullish = 0
            bearish = 0
            neutral = 0
            sample_messages = []

            for entry in feed.entries[:self._max_headlines]:
                title = entry.get("title", "")
                source = entry.get("source", {}).get("title", "") if hasattr(entry, "source") else ""
                published = entry.get("published", "")

                headlines.append(title)
                sentiment = self._classify_headline(title)

                if sentiment == "bullish":
                    bullish += 1
                elif sentiment == "bearish":
                    bearish += 1
                else:
                    neutral += 1

                if len(sample_messages) < 5:
                    sample_messages.append({
                        "text": title[:200],
                        "sentiment": sentiment,
                        "created_at": published,
                        "source": source,
                    })

            total = len(headlines)
            sentiment_total = bullish + bearish
            if sentiment_total > 0:
                sentiment_score = (bullish / sentiment_total) * 100
            else:
                sentiment_score = 50.0

            if sentiment_score >= 60:
                sentiment_label = "bullish"
            elif sentiment_score <= 40:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            return SentimentResult(
                ticker=ticker,
                source="google_news",
                bullish_count=bullish,
                bearish_count=bearish,
                neutral_count=neutral,
                total_messages=total,
                sentiment_score=round(sentiment_score, 1),
                sentiment_label=sentiment_label,
                trending=total >= 10,
                sample_messages=sample_messages,
            )

        except Exception as e:
            logger.debug(f"Google News error for {ticker}: {e}")
            return SentimentResult(ticker=ticker, source="google_news", error=str(e))

    def _classify_headline(self, headline: str) -> str:
        """Keyword-based headline sentiment classification."""
        h = headline.lower()

        bullish_keywords = [
            "surge", "soar", "rally", "jump", "gain", "rise", "climb",
            "upgrade", "beat", "strong", "record", "high", "boost",
            "outperform", "buy", "bullish", "profit", "growth", "upbeat",
            "breakout", "momentum", "optimistic", "exceed", "positive",
        ]
        bearish_keywords = [
            "crash", "plunge", "drop", "fall", "decline", "sink", "tumble",
            "downgrade", "miss", "weak", "low", "cut", "loss", "sell",
            "bearish", "warning", "concern", "risk", "negative", "slump",
            "fear", "worry", "plummet", "disappoint", "trouble",
        ]

        bull_hits = sum(1 for kw in bullish_keywords if kw in h)
        bear_hits = sum(1 for kw in bearish_keywords if kw in h)

        if bull_hits > bear_hits:
            return "bullish"
        elif bear_hits > bull_hits:
            return "bearish"
        return "neutral"


# ============================================================================
# GOOGLE TRENDS SEARCH INTEREST (FREE - NO AUTH)
# ============================================================================

class GoogleTrendsFetcher:
    """
    Fetch search interest data from Google Trends.

    FREE - No API key required.
    Uses pytrends library (unofficial wrapper).
    Rate limit: ~10 req/min (use delays).

    Query: "{TICKER} stock" with 7-day timeframe.
    Signal: Current interest vs 7-day average → spike detection.
    """

    def __init__(self):
        self._pytrends = None
        self.is_configured = True

    def _get_pytrends(self):
        """Lazy-init pytrends."""
        if self._pytrends is None:
            try:
                from pytrends.request import TrendReq
                self._pytrends = TrendReq(hl='en-US', tz=300)
            except Exception as e:
                logger.debug(f"pytrends init failed: {e}")
                self.is_configured = False
        return self._pytrends

    def get_search_interest(self, ticker: str) -> dict:
        """
        Get Google Trends search interest for a ticker.

        Returns:
            Dict with interest_now, interest_avg, spike_ratio, is_trending
        """
        pt = self._get_pytrends()
        if pt is None:
            return {"error": "pytrends not available", "spike_ratio": 1.0, "is_trending": False}

        try:
            keyword = f"{ticker} stock"
            pt.build_payload([keyword], timeframe='now 7-d')
            df = pt.interest_over_time()

            if df.empty:
                return {"interest_now": 0, "interest_avg": 0, "spike_ratio": 1.0, "is_trending": False}

            values = df[keyword].values
            interest_now = float(values[-1]) if len(values) > 0 else 0
            interest_avg = float(values.mean()) if len(values) > 0 else 0

            spike_ratio = interest_now / max(interest_avg, 1)
            is_trending = spike_ratio > 2.0

            # Convert to 0-100 score
            # Spike ratio of 1.0 = neutral (50), 2.0+ = bullish (75+), 0.5- = bearish (25-)
            score = min(100, max(0, 50 + (spike_ratio - 1.0) * 25))

            return {
                "interest_now": interest_now,
                "interest_avg": round(interest_avg, 1),
                "spike_ratio": round(spike_ratio, 2),
                "is_trending": is_trending,
                "score": round(score, 1),
            }

        except Exception as e:
            logger.debug(f"Google Trends error for {ticker}: {e}")
            return {"error": str(e), "spike_ratio": 1.0, "is_trending": False, "score": 50.0}


# ============================================================================
# CNN FEAR & GREED INDEX (FREE - NO AUTH)
# ============================================================================

def get_fear_greed_index() -> dict:
    """
    Fetch CNN Fear & Greed Index.

    One API call gives overall market mood (0-100).
    0 = Extreme Fear, 100 = Extreme Greed.

    Returns:
        Dict with score, rating, timestamp
    """
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "StockPulse/1.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if not response.ok:
            return {"error": f"HTTP {response.status_code}", "score": 50, "rating": "Neutral"}

        data = response.json()
        fg = data.get("fear_and_greed", {})
        score = fg.get("score", 50)
        rating = fg.get("rating", "Neutral")

        return {
            "score": round(score, 1),
            "rating": rating,
            "previous_close": fg.get("previous_close", score),
            "previous_1_week": fg.get("previous_1_week", score),
            "previous_1_month": fg.get("previous_1_month", score),
            "timestamp": fg.get("timestamp", datetime.now().isoformat()),
        }

    except Exception as e:
        logger.debug(f"Fear & Greed error: {e}")
        return {"error": str(e), "score": 50, "rating": "Neutral"}


# ============================================================================
# OPTIONS PUT/CALL RATIO VIA YFINANCE (FREE - EXISTING DEPENDENCY)
# ============================================================================

class OptionsSentimentFetcher:
    """
    Fetch put/call ratio from options data via yfinance.

    FREE - Uses existing yfinance dependency.
    Put/call ratio: <0.7 = bullish, 0.7-1.0 = neutral, >1.0 = bearish.

    Only run for top 20 tickers (options data is heavy).
    """

    def __init__(self):
        self.is_configured = True

    def get_put_call_ratio(self, ticker: str) -> dict:
        """
        Get put/call ratio for a ticker.

        Returns:
            Dict with ratio, score (0-100), label
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            # Get nearest expiration
            expirations = stock.options
            if not expirations:
                return {"error": "No options data", "ratio": 1.0, "score": 50.0, "label": "neutral"}

            chain = stock.option_chain(expirations[0])
            put_volume = chain.puts["volume"].sum()
            call_volume = chain.calls["volume"].sum()

            if call_volume == 0:
                ratio = 2.0  # Very bearish if no call volume
            else:
                ratio = put_volume / call_volume

            # Score: 0.5 ratio = very bullish (80), 1.0 = neutral (50), 1.5 = very bearish (20)
            score = max(0, min(100, 100 - (ratio * 50)))

            if ratio < 0.7:
                label = "bullish"
            elif ratio > 1.0:
                label = "bearish"
            else:
                label = "neutral"

            return {
                "ratio": round(ratio, 3),
                "score": round(score, 1),
                "label": label,
                "put_volume": int(put_volume) if put_volume == put_volume else 0,
                "call_volume": int(call_volume) if call_volume == call_volume else 0,
                "expiration": expirations[0],
            }

        except Exception as e:
            logger.debug(f"Options P/C error for {ticker}: {e}")
            return {"error": str(e), "ratio": 1.0, "score": 50.0, "label": "neutral"}


# ============================================================================
# WIKIPEDIA PAGE VIEWS (FREE - NO AUTH)
# ============================================================================

class WikipediaFetcher:
    """
    Fetch Wikipedia page view data for attention spike detection.

    FREE - No auth, 100 req/sec rate limit (no concern).
    Page view spikes = retail attention.

    API: Wikimedia REST API
    """

    # Map tickers to Wikipedia article names
    TICKER_TO_WIKI = {
        "AAPL": "Apple_Inc.", "MSFT": "Microsoft", "GOOGL": "Alphabet_Inc.",
        "AMZN": "Amazon_(company)", "META": "Meta_Platforms", "NVDA": "Nvidia",
        "TSM": "TSMC", "AVGO": "Broadcom_Inc.", "ORCL": "Oracle_Corporation",
        "CRM": "Salesforce", "AMD": "AMD", "PLTR": "Palantir_Technologies",
        "NOW": "ServiceNow", "ADBE": "Adobe_Inc.", "IBM": "IBM",
        "INTC": "Intel", "MU": "Micron_Technology", "QCOM": "Qualcomm",
        "ARM": "Arm_Holdings", "TSLA": "Tesla,_Inc.", "ASML": "ASML",
        "SNOW": "Snowflake_Inc.", "MRVL": "Marvell_Technology",
    }

    def __init__(self):
        self.is_configured = True

    def get_page_views(self, ticker: str) -> dict:
        """
        Get Wikipedia page views for a ticker's company.

        Returns:
            Dict with views_today, views_avg_30d, spike_ratio, is_trending, score
        """
        article = self.TICKER_TO_WIKI.get(ticker)
        if not article:
            return {"error": f"No wiki mapping for {ticker}", "score": 50.0, "is_trending": False}

        try:
            end = date.today()
            start_30d = end - timedelta(days=30)

            url = (
                f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                f"en.wikipedia/all-access/all-agents/{article}/daily/"
                f"{start_30d.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
            )
            headers = {"User-Agent": "StockPulse/1.0 (sentiment analysis)"}
            response = requests.get(url, headers=headers, timeout=10)

            if not response.ok:
                return {"error": f"HTTP {response.status_code}", "score": 50.0, "is_trending": False}

            data = response.json()
            items = data.get("items", [])

            if not items:
                return {"error": "No data", "score": 50.0, "is_trending": False}

            views = [item.get("views", 0) for item in items]
            views_today = views[-1] if views else 0
            views_avg = sum(views) / len(views) if views else 0

            spike_ratio = views_today / max(views_avg, 1)
            is_trending = spike_ratio > 2.0

            # Score: spike_ratio 1.0 = neutral (50), 2.0+ = high interest (75+)
            score = min(100, max(0, 50 + (spike_ratio - 1.0) * 25))

            return {
                "views_today": views_today,
                "views_avg_30d": round(views_avg, 0),
                "spike_ratio": round(spike_ratio, 2),
                "is_trending": is_trending,
                "score": round(score, 1),
                "article": article,
            }

        except Exception as e:
            logger.debug(f"Wikipedia error for {ticker}: {e}")
            return {"error": str(e), "score": 50.0, "is_trending": False}


# ============================================================================
# FINNHUB ANALYST RATINGS (FREE TIER)
# ============================================================================

@dataclass
class AnalystRating:
    """Container for analyst rating data."""
    ticker: str
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_buy: int = 0
    strong_sell: int = 0
    total_analysts: int = 0
    consensus: str = "hold"  # strong_buy, buy, hold, sell, strong_sell
    consensus_score: float = 50.0  # 0-100, higher = more bullish
    period: str = ""  # e.g., "2026-02-01"
    fetched_at: str = ""
    error: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "buy": self.buy,
            "hold": self.hold,
            "sell": self.sell,
            "strong_buy": self.strong_buy,
            "strong_sell": self.strong_sell,
            "total_analysts": self.total_analysts,
            "consensus": self.consensus,
            "consensus_score": self.consensus_score,
            "period": self.period,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


class FinnhubAnalystRatings:
    """
    Fetch analyst recommendation trends from Finnhub.

    FREE TIER: 60 requests/minute
    API Docs: https://finnhub.io/docs/api/recommendation-trends

    Returns Buy/Hold/Sell counts from analysts covering a stock.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None, max_retries: int = 3):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.is_configured = bool(self.api_key)
        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 1.0
        self._max_retries = max_retries

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_with_retry(self, url: str, params: dict) -> requests.Response | None:
        """Fetch with retry logic."""
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code in (401, 429) or response.ok:
                    return response
                if response.status_code >= 500 and attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return response
            except (requests.exceptions.Timeout, requests.exceptions.RequestException):
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def get_ratings(self, ticker: str) -> AnalystRating:
        """
        Get analyst recommendation trends for a ticker.

        Returns the most recent recommendation data.
        """
        if not self.is_configured:
            return AnalystRating(ticker=ticker, error="FINNHUB_API_KEY not configured")

        url = f"{self.BASE_URL}/stock/recommendation"
        params = {"symbol": ticker.upper(), "token": self.api_key}

        response = self._fetch_with_retry(url, params)
        if response is None:
            return AnalystRating(ticker=ticker, error="All retries failed")

        try:
            if response.status_code == 401:
                return AnalystRating(ticker=ticker, error="Invalid API key")
            if response.status_code == 429:
                return AnalystRating(ticker=ticker, error="Rate limited")

            response.raise_for_status()
            data = response.json()

            if not data:
                return AnalystRating(ticker=ticker, error="No analyst data available")

            # Get the most recent recommendation
            latest = data[0] if data else {}

            buy = latest.get("buy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0)
            strong_buy = latest.get("strongBuy", 0)
            strong_sell = latest.get("strongSell", 0)
            period = latest.get("period", "")

            total = buy + hold + sell + strong_buy + strong_sell

            # Calculate consensus score (0-100)
            # Weight: strong_buy=5, buy=4, hold=3, sell=2, strong_sell=1
            if total > 0:
                weighted_sum = (strong_buy * 5 + buy * 4 + hold * 3 + sell * 2 + strong_sell * 1)
                consensus_score = ((weighted_sum / total) - 1) / 4 * 100  # Normalize to 0-100
            else:
                consensus_score = 50.0

            # Determine consensus label
            if consensus_score >= 75:
                consensus = "strong_buy"
            elif consensus_score >= 60:
                consensus = "buy"
            elif consensus_score >= 40:
                consensus = "hold"
            elif consensus_score >= 25:
                consensus = "sell"
            else:
                consensus = "strong_sell"

            return AnalystRating(
                ticker=ticker,
                buy=buy,
                hold=hold,
                sell=sell,
                strong_buy=strong_buy,
                strong_sell=strong_sell,
                total_analysts=total,
                consensus=consensus,
                consensus_score=round(consensus_score, 1),
                period=period,
            )

        except Exception as e:
            logger.debug(f"Finnhub analyst rating error for {ticker}: {e}")
            return AnalystRating(ticker=ticker, error=str(e))


# ============================================================================
# FINNHUB INSIDER TRANSACTIONS (FREE TIER)
# ============================================================================

@dataclass
class InsiderActivity:
    """Container for insider transaction data."""
    ticker: str
    net_shares_30d: int = 0  # Net shares bought/sold in last 30 days
    buy_transactions: int = 0
    sell_transactions: int = 0
    total_transactions: int = 0
    net_value_30d: float = 0.0  # Approximate $ value
    insider_sentiment: str = "neutral"  # bullish (net buying), bearish (net selling), neutral
    insider_score: float = 50.0  # 0-100
    notable_insiders: list = None  # List of recent insider names/titles
    fetched_at: str = ""
    error: str = ""

    def __post_init__(self):
        if self.notable_insiders is None:
            self.notable_insiders = []
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "net_shares_30d": self.net_shares_30d,
            "buy_transactions": self.buy_transactions,
            "sell_transactions": self.sell_transactions,
            "total_transactions": self.total_transactions,
            "net_value_30d": self.net_value_30d,
            "insider_sentiment": self.insider_sentiment,
            "insider_score": self.insider_score,
            "notable_insiders": self.notable_insiders,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


class FinnhubInsiderTransactions:
    """
    Fetch insider transaction data from Finnhub.

    FREE TIER: 60 requests/minute
    API Docs: https://finnhub.io/docs/api/insider-transactions

    Returns Form 4 filings showing insider buys/sells.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None, max_retries: int = 3):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.is_configured = bool(self.api_key)
        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 1.0
        self._max_retries = max_retries

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_with_retry(self, url: str, params: dict) -> requests.Response | None:
        """Fetch with retry logic."""
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code in (401, 429) or response.ok:
                    return response
                if response.status_code >= 500 and attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return response
            except (requests.exceptions.Timeout, requests.exceptions.RequestException):
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def get_insider_activity(self, ticker: str) -> InsiderActivity:
        """
        Get insider transaction activity for a ticker.

        Analyzes recent Form 4 filings to determine insider sentiment.
        """
        if not self.is_configured:
            return InsiderActivity(ticker=ticker, error="FINNHUB_API_KEY not configured")

        url = f"{self.BASE_URL}/stock/insider-transactions"
        params = {"symbol": ticker.upper(), "token": self.api_key}

        response = self._fetch_with_retry(url, params)
        if response is None:
            return InsiderActivity(ticker=ticker, error="All retries failed")

        try:
            if response.status_code == 401:
                return InsiderActivity(ticker=ticker, error="Invalid API key")
            if response.status_code == 429:
                return InsiderActivity(ticker=ticker, error="Rate limited")

            response.raise_for_status()
            data = response.json()
            transactions = data.get("data", [])

            if not transactions:
                return InsiderActivity(ticker=ticker, error="No insider data available")

            # Filter to last 30 days
            cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            recent = [t for t in transactions if t.get("transactionDate", "") >= cutoff]

            # Analyze transactions
            net_shares = 0
            buy_count = 0
            sell_count = 0
            notable = []

            for txn in recent[:20]:  # Limit to 20 most recent
                change = txn.get("change", 0)
                txn_type = txn.get("transactionCode", "")

                # P = Purchase, S = Sale, A = Award/Grant
                if txn_type == "P" or change > 0:
                    net_shares += abs(change)
                    buy_count += 1
                elif txn_type == "S" or change < 0:
                    net_shares -= abs(change)
                    sell_count += 1

                # Collect notable insiders
                name = txn.get("name", "")
                if name and len(notable) < 3:
                    notable.append(name)

            total = buy_count + sell_count

            # Calculate insider sentiment
            if total == 0:
                insider_score = 50.0
                sentiment = "neutral"
            elif buy_count > sell_count * 2:
                insider_score = 80.0
                sentiment = "bullish"
            elif sell_count > buy_count * 2:
                insider_score = 20.0
                sentiment = "bearish"
            elif net_shares > 0:
                insider_score = 60.0
                sentiment = "bullish"
            elif net_shares < 0:
                insider_score = 40.0
                sentiment = "bearish"
            else:
                insider_score = 50.0
                sentiment = "neutral"

            return InsiderActivity(
                ticker=ticker,
                net_shares_30d=net_shares,
                buy_transactions=buy_count,
                sell_transactions=sell_count,
                total_transactions=total,
                insider_sentiment=sentiment,
                insider_score=insider_score,
                notable_insiders=notable,
            )

        except Exception as e:
            logger.debug(f"Finnhub insider transaction error for {ticker}: {e}")
            return InsiderActivity(ticker=ticker, error=str(e))


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
                model="claude-haiku-4-5-20251001",
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

    Sources (all FREE except Haiku):
    1. StockTwits (FREE, no API key) - Retail trader sentiment
    2. Finnhub News (FREE tier, API key) - News sentiment
    3. Reddit via PRAW (FREE, app credentials) - Retail discussion sentiment
    4. Google News RSS (FREE, no auth) - Headline sentiment
    5. Finnhub Analyst Ratings (FREE tier) - Professional analyst consensus
    6. Finnhub Insider Transactions (FREE tier) - Form 4 insider trading
    7. Google Trends (FREE, no auth) - Search interest spikes
    8. Options Put/Call (FREE, yfinance) - Options positioning
    9. Wikipedia page views (FREE, no auth) - Attention spikes
    10. Haiku AI (PAID - use sparingly) - Intelligent summarization

    Usage:
        analyzer = SentimentAnalyzer()

        # Get sentiment for one ticker
        result = analyzer.get_sentiment("NVDA")

        # Get full data including analyst/insider (slower, more API calls)
        result = analyzer.get_full_sentiment("NVDA")

        # Get sentiment for multiple tickers
        results = analyzer.get_bulk_sentiment(["NVDA", "AAPL", "MSFT"])
    """

    # Default scoring weights (configurable via config.yaml)
    DEFAULT_WEIGHTS = {
        "stocktwits": 0.10,
        "reddit": 0.15,
        "finnhub_news": 0.05,
        "google_news": 0.15,
        "analyst": 0.25,
        "insider": 0.15,
        "google_trends": 0.05,
        "put_call": 0.03,
        "wiki": 0.02,
    }

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        sentiment_config = self._config.get("sentiment", {})

        self.stocktwits = StockTwitsFetcher()
        self.finnhub = FinnhubFetcher()
        self.reddit = RedditFetcher(sentiment_config.get("reddit", {}))
        self.google_news = GoogleNewsFetcher()
        self.analyst_ratings = FinnhubAnalystRatings()
        self.insider_txns = FinnhubInsiderTransactions()
        self.google_trends = GoogleTrendsFetcher()
        self.options_sentiment = OptionsSentimentFetcher()
        self.wikipedia = WikipediaFetcher()
        self.haiku = HaikuSentimentAnalyzer()

        # Load weights from config or use defaults
        self._weights = sentiment_config.get("weights", self.DEFAULT_WEIGHTS)

        # Cache results to avoid hammering APIs
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def _cache_key(self, ticker: str) -> str:
        """Generate cache key with proper 5-minute buckets."""
        now = datetime.now()
        bucket = now.minute // 5  # 0-11 for each 5-min window in an hour
        return f"{ticker}_{now.strftime('%Y%m%d%H')}_{bucket}"

    def _weighted_score(self, source_scores: dict[str, float]) -> float:
        """
        Calculate weighted aggregate score from available sources.

        Sources that aren't available get their weight redistributed
        proportionally among the sources that are present.

        Args:
            source_scores: Dict mapping source name to score (0-100)

        Returns:
            Weighted aggregate score (0-100)
        """
        if not source_scores:
            return 50.0

        total_weight = 0
        weighted_sum = 0

        for source, score in source_scores.items():
            weight = self._weights.get(source, 0)
            if weight > 0:
                weighted_sum += score * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return sum(source_scores.values()) / len(source_scores)

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

        # Fetch from all social sources
        stocktwits_result = self.stocktwits.get_sentiment(ticker)
        finnhub_result = self.finnhub.get_social_sentiment(ticker) if self.finnhub.is_configured else None
        reddit_result = self.reddit.get_sentiment(ticker) if self.reddit.is_configured else None
        google_news_result = self.google_news.get_sentiment(ticker)

        # Calculate weighted aggregate score
        source_scores = {}
        if stocktwits_result and not stocktwits_result.error:
            source_scores["stocktwits"] = stocktwits_result.sentiment_score
        if finnhub_result and not finnhub_result.error:
            source_scores["finnhub_news"] = finnhub_result.sentiment_score
        if reddit_result and not reddit_result.error:
            source_scores["reddit"] = reddit_result.sentiment_score
        if google_news_result and not google_news_result.error:
            source_scores["google_news"] = google_news_result.sentiment_score

        aggregate_score = self._weighted_score(source_scores)

        # Determine aggregate label
        if aggregate_score >= 60:
            aggregate_label = "bullish"
        elif aggregate_score <= 40:
            aggregate_label = "bearish"
        else:
            aggregate_label = "neutral"

        # Get AI analysis if requested (with error handling)
        ai_analysis = None
        if include_ai_analysis and self.haiku.is_configured:
            try:
                ai_analysis = self.haiku.analyze_sentiment(
                    ticker=ticker,
                    stocktwits_result=stocktwits_result,
                    finnhub_result=finnhub_result,
                    price_context=price_context,
                )
            except Exception as e:
                logger.debug(f"Haiku analysis error for {ticker}: {e}")
                ai_analysis = {
                    "ticker": ticker,
                    "summary": f"Analysis unavailable: {str(e)}",
                    "recommendation": "neutral",
                    "confidence": 0,
                }

        result = {
            "ticker": ticker,
            "aggregate_score": round(aggregate_score, 1),
            "aggregate_label": aggregate_label,
            "source_scores": source_scores,
            "stocktwits": stocktwits_result.to_dict() if stocktwits_result else None,
            "finnhub": finnhub_result.to_dict() if finnhub_result else None,
            "reddit": reddit_result.to_dict() if reddit_result else None,
            "google_news": google_news_result.to_dict() if google_news_result else None,
            "ai_analysis": ai_analysis,
            "fetched_at": datetime.now().isoformat(),
        }

        # Cache result
        self._cache[cache_key] = result
        return result

    def get_full_sentiment(
        self,
        ticker: str,
        include_ai_analysis: bool = False,  # Default OFF to save Haiku costs
        price_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        Get comprehensive sentiment including analyst ratings and insider data.

        This is slower (more API calls) but provides richer data.
        Use for daily scans, not hourly.

        Args:
            ticker: Stock ticker symbol
            include_ai_analysis: Whether to run Haiku (COSTLY - default OFF)
            price_context: Optional price data for context

        Returns:
            Dict with all sentiment data including analyst/insider signals
        """
        # Get basic sentiment first (includes social + news sources)
        result = self.get_sentiment(ticker, include_ai_analysis=False, price_context=price_context)

        # Add analyst ratings (FREE via Finnhub)
        analyst_data = None
        if self.analyst_ratings.is_configured:
            try:
                analyst = self.analyst_ratings.get_ratings(ticker)
                if not analyst.error:
                    analyst_data = analyst.to_dict()
            except Exception as e:
                logger.debug(f"Analyst rating error for {ticker}: {e}")

        # Add insider transactions (FREE via Finnhub)
        insider_data = None
        if self.insider_txns.is_configured:
            try:
                insider = self.insider_txns.get_insider_activity(ticker)
                if not insider.error:
                    insider_data = insider.to_dict()
            except Exception as e:
                logger.debug(f"Insider transaction error for {ticker}: {e}")

        # Add alternative signals
        trends_data = None
        try:
            trends_data = self.google_trends.get_search_interest(ticker)
            if trends_data.get("error"):
                trends_data = None
        except Exception as e:
            logger.debug(f"Google Trends error for {ticker}: {e}")

        putcall_data = None
        try:
            putcall_data = self.options_sentiment.get_put_call_ratio(ticker)
            if putcall_data.get("error"):
                putcall_data = None
        except Exception as e:
            logger.debug(f"Put/call error for {ticker}: {e}")

        wiki_data = None
        try:
            wiki_data = self.wikipedia.get_page_views(ticker)
            if wiki_data.get("error"):
                wiki_data = None
        except Exception as e:
            logger.debug(f"Wikipedia error for {ticker}: {e}")

        # Build full source scores for enhanced weighted scoring
        source_scores = dict(result.get("source_scores", {}))

        if analyst_data and analyst_data.get("total_analysts", 0) > 0:
            source_scores["analyst"] = analyst_data.get("consensus_score", 50)

        if insider_data and insider_data.get("total_transactions", 0) > 0:
            source_scores["insider"] = insider_data.get("insider_score", 50)

        if trends_data:
            source_scores["google_trends"] = trends_data.get("score", 50)

        if putcall_data:
            source_scores["put_call"] = putcall_data.get("score", 50)

        if wiki_data:
            source_scores["wiki"] = wiki_data.get("score", 50)

        enhanced_score = self._weighted_score(source_scores)

        # Update result with enhanced data
        result["analyst_ratings"] = analyst_data
        result["insider_activity"] = insider_data
        result["google_trends"] = trends_data
        result["put_call_ratio"] = putcall_data
        result["wikipedia"] = wiki_data
        result["source_scores"] = source_scores
        result["enhanced_score"] = round(enhanced_score, 1)

        # Determine enhanced label
        if enhanced_score >= 60:
            result["enhanced_label"] = "bullish"
        elif enhanced_score <= 40:
            result["enhanced_label"] = "bearish"
        else:
            result["enhanced_label"] = "neutral"

        # Only run Haiku if explicitly requested (SAVE COSTS)
        if include_ai_analysis and self.haiku.is_configured:
            try:
                result["ai_analysis"] = self.haiku.analyze_sentiment(
                    ticker=ticker,
                    stocktwits_result=self.stocktwits.get_sentiment(ticker) if not result.get("stocktwits") else None,
                    finnhub_result=None,
                    price_context=price_context,
                )
            except Exception as e:
                logger.debug(f"Haiku analysis error for {ticker}: {e}")

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

        # Hourly sentiment snapshots (for top 20 AI stocks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_hourly (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_datetime TEXT NOT NULL,
                source TEXT DEFAULT 'stocktwits',
                bullish_count INTEGER DEFAULT 0,
                bearish_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                sentiment_score REAL DEFAULT 50.0,
                sentiment_label TEXT DEFAULT 'neutral',
                message_velocity REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (ticker, scan_datetime, source)
            )
        """)

        # Enhanced signals (analyst ratings, insider transactions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_data TEXT,
                signal_score REAL DEFAULT 50.0,
                signal_label TEXT DEFAULT 'neutral',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (ticker, scan_date, signal_type)
            )
        """)

        self.db.conn.commit()
        logger.debug("Sentiment tables initialized")

    def store_sentiment(self, ticker: str, sentiment_data: dict) -> bool:
        """Store sentiment data for a ticker (all sources)."""
        try:
            today = date.today().isoformat()
            ai = sentiment_data.get("ai_analysis", {})

            # Store each source separately in sentiment_daily
            sources_to_store = [
                ("stocktwits", sentiment_data.get("stocktwits", {})),
                ("reddit", sentiment_data.get("reddit", {})),
                ("google_news", sentiment_data.get("google_news", {})),
                ("finnhub", sentiment_data.get("finnhub", {})),
            ]

            for source_name, source_data in sources_to_store:
                if not source_data or source_data.get("error"):
                    continue

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
                    source_name,
                    source_data.get("bullish_count", 0),
                    source_data.get("bearish_count", 0),
                    source_data.get("neutral_count", 0),
                    source_data.get("total_messages", 0),
                    source_data.get("sentiment_score", 50.0),
                    source_data.get("sentiment_label", "neutral"),
                    1 if source_data.get("trending", False) else 0,
                    source_data.get("message_velocity", 0.0),
                    json.dumps(source_data.get("sample_messages", [])),
                    ai.get("summary", "") if ai and source_name == "stocktwits" else "",
                    ai.get("recommendation", "neutral") if ai and source_name == "stocktwits" else "neutral",
                    ai.get("confidence", 0) if ai and source_name == "stocktwits" else 0,
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

    def has_fresh_sentiment(self, tickers: list[str], min_coverage: float = 0.5) -> bool:
        """
        Check if we have fresh sentiment data for today.

        Args:
            tickers: List of tickers to check
            min_coverage: Minimum fraction of tickers that must have data (0.5 = 50%)

        Returns:
            True if sufficient fresh data exists, False otherwise
        """
        try:
            if not tickers:
                return False

            today = date.today().isoformat()
            placeholders = ",".join(["?" for _ in tickers])

            row = self.db.fetchone(f"""
                SELECT COUNT(DISTINCT ticker) FROM sentiment_daily
                WHERE scan_date = ? AND ticker IN ({placeholders})
            """, (today, *tickers))

            count = row[0] if row else 0
            coverage = count / len(tickers)

            logger.debug(f"Sentiment freshness check: {count}/{len(tickers)} tickers ({coverage:.0%})")
            return coverage >= min_coverage

        except Exception as e:
            logger.error(f"Error checking sentiment freshness: {e}")
            return False

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

    def store_hourly_sentiment(self, ticker: str, sentiment_data: dict) -> bool:
        """Store hourly sentiment snapshot for a ticker."""
        try:
            scan_time = datetime.now().strftime("%Y-%m-%d %H:00:00")
            st = sentiment_data.get("stocktwits", {})

            self.db.execute("""
                INSERT INTO sentiment_hourly (
                    ticker, scan_datetime, source,
                    bullish_count, bearish_count, neutral_count, total_messages,
                    sentiment_score, sentiment_label, message_velocity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, scan_datetime, source) DO UPDATE SET
                    bullish_count = excluded.bullish_count,
                    bearish_count = excluded.bearish_count,
                    neutral_count = excluded.neutral_count,
                    total_messages = excluded.total_messages,
                    sentiment_score = excluded.sentiment_score,
                    sentiment_label = excluded.sentiment_label,
                    message_velocity = excluded.message_velocity
            """, (
                ticker,
                scan_time,
                "stocktwits",
                st.get("bullish_count", 0),
                st.get("bearish_count", 0),
                st.get("neutral_count", 0),
                st.get("total_messages", 0),
                sentiment_data.get("aggregate_score", 50.0),
                sentiment_data.get("aggregate_label", "neutral"),
                st.get("message_velocity", 0.0),
            ))
            return True

        except Exception as e:
            logger.debug(f"Error storing hourly sentiment for {ticker}: {e}")
            return False

    def store_signal(self, ticker: str, signal_type: str, signal_data: dict) -> bool:
        """Store enhanced signal (analyst ratings, insider transactions)."""
        try:
            today = date.today().isoformat()

            self.db.execute("""
                INSERT INTO sentiment_signals (
                    ticker, scan_date, signal_type,
                    signal_data, signal_score, signal_label
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, scan_date, signal_type) DO UPDATE SET
                    signal_data = excluded.signal_data,
                    signal_score = excluded.signal_score,
                    signal_label = excluded.signal_label
            """, (
                ticker,
                today,
                signal_type,
                json.dumps(signal_data),
                signal_data.get("consensus_score", signal_data.get("insider_score", 50.0)),
                signal_data.get("consensus", signal_data.get("insider_sentiment", "neutral")),
            ))
            return True

        except Exception as e:
            logger.debug(f"Error storing {signal_type} for {ticker}: {e}")
            return False

    def get_hourly_trend(self, ticker: str, hours: int = 6) -> list[dict]:
        """Get hourly sentiment trend for a ticker."""
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:00:00")

            rows = self.db.fetchdf("""
                SELECT scan_datetime, sentiment_score, sentiment_label, total_messages
                FROM sentiment_hourly
                WHERE ticker = ? AND scan_datetime >= ?
                ORDER BY scan_datetime ASC
            """, (ticker, cutoff))

            if rows.empty:
                return []

            return rows.to_dict("records")

        except Exception as e:
            logger.debug(f"Error getting hourly trend for {ticker}: {e}")
            return []

    def get_signals(self, tickers: list[str]) -> dict[str, dict]:
        """Get today's signals for tickers (analyst + insider)."""
        try:
            today = date.today().isoformat()
            placeholders = ",".join(["?" for _ in tickers])

            rows = self.db.fetchdf(f"""
                SELECT ticker, signal_type, signal_data, signal_score, signal_label
                FROM sentiment_signals
                WHERE scan_date = ? AND ticker IN ({placeholders})
            """, (today, *tickers))

            results = {}
            if not rows.empty:
                for _, row in rows.iterrows():
                    ticker = row["ticker"]
                    if ticker not in results:
                        results[ticker] = {}
                    try:
                        results[ticker][row["signal_type"]] = {
                            "data": json.loads(row["signal_data"]) if row["signal_data"] else {},
                            "score": row["signal_score"],
                            "label": row["signal_label"],
                        }
                    except Exception:
                        pass
            return results

        except Exception as e:
            logger.debug(f"Error getting signals: {e}")
            return {}


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

    # Fetch enhanced signals for all scanned tickers (FREE via Finnhub)
    logger.info("Fetching analyst ratings and insider data...")
    for ticker in target_tickers[:max_tickers]:
        try:
            # Analyst ratings
            if analyzer.analyst_ratings.is_configured:
                analyst = analyzer.analyst_ratings.get_ratings(ticker)
                if not analyst.error:
                    storage.store_signal(ticker, "analyst_rating", analyst.to_dict())

            # Insider transactions
            if analyzer.insider_txns.is_configured:
                insider = analyzer.insider_txns.get_insider_activity(ticker)
                if not insider.error:
                    storage.store_signal(ticker, "insider_txn", insider.to_dict())

        except Exception as e:
            logger.debug(f"Enhanced signal error for {ticker}: {e}")

    # Fetch alternative signals (Google Trends, Put/Call, Wikipedia)
    logger.info("Fetching alternative signals (trends, options, wiki)...")
    for ticker in target_tickers[:20]:  # Limit to top 20 for heavier sources
        try:
            # Google Trends (rate limited, go slow)
            trends = analyzer.google_trends.get_search_interest(ticker)
            if not trends.get("error"):
                storage.store_signal(ticker, "google_trends", trends)
            time.sleep(1)  # Be nice to Google

            # Options put/call ratio
            putcall = analyzer.options_sentiment.get_put_call_ratio(ticker)
            if not putcall.get("error"):
                storage.store_signal(ticker, "put_call_ratio", putcall)

            # Wikipedia page views
            wiki = analyzer.wikipedia.get_page_views(ticker)
            if not wiki.get("error"):
                storage.store_signal(ticker, "wiki_attention", wiki)

        except Exception as e:
            logger.debug(f"Alternative signal error for {ticker}: {e}")

    # Fetch Fear & Greed Index (one global call)
    logger.info("Fetching CNN Fear & Greed Index...")
    try:
        fg = get_fear_greed_index()
        if not fg.get("error"):
            storage.store_signal("_MARKET", "fear_greed", fg)
    except Exception as e:
        logger.debug(f"Fear & Greed error: {e}")

    # Run AI analysis on most extreme sentiment stocks (LIMIT TO SAVE COSTS)
    # Only analyze top 3 bullish + top 3 bearish = 6 Haiku calls max
    if include_ai_analysis and analyzer.haiku.is_configured:
        logger.info("Running Haiku analysis on top 3 bullish + top 3 bearish stocks...")

        # Analyze top bullish and bearish only
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


# Top 20 AI stocks for hourly monitoring (highest market cap / most important)
TOP_20_AI_STOCKS = [
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN",
    "META", "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "PLTR", "SNOW", "NOW", "ADBE",
    "IBM", "INTC", "MU", "QCOM", "ARM",
]


def run_hourly_sentiment_scan() -> dict[str, Any]:
    """
    Run hourly sentiment scan for top 20 AI stocks.

    This is a lightweight scan that only collects StockTwits data.
    NO Haiku analysis (saves costs) - just raw social sentiment.

    Called by scheduler every hour during market hours.

    Returns:
        Summary of scan results
    """
    logger.info("Starting hourly sentiment scan for top 20 AI stocks...")

    analyzer = SentimentAnalyzer()
    storage = SentimentStorage()

    results = {
        "scan_time": datetime.now().isoformat(),
        "tickers_scanned": 0,
        "successful": 0,
        "failed": 0,
    }

    for ticker in TOP_20_AI_STOCKS:
        try:
            # Get sentiment WITHOUT AI analysis (save costs)
            sentiment = analyzer.get_sentiment(ticker, include_ai_analysis=False)

            # Store in hourly table
            if storage.store_hourly_sentiment(ticker, sentiment):
                results["successful"] += 1
            else:
                results["failed"] += 1

            results["tickers_scanned"] += 1

        except Exception as e:
            results["failed"] += 1
            logger.debug(f"Hourly scan error for {ticker}: {e}")

    logger.info(
        f"Hourly sentiment scan complete: {results['successful']}/{results['tickers_scanned']} successful"
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
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr>
                <th style="text-align: left; color: #94a3b8; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">Ticker</th>
                <th style="text-align: center; color: #94a3b8; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">Score</th>
                <th style="text-align: center; color: #94a3b8; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">Sentiment</th>
                <th style="text-align: center; color: #4ade80; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">🟢 Bull</th>
                <th style="text-align: center; color: #f87171; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">🔴 Bear</th>
                <th style="text-align: center; color: #94a3b8; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">⚪ Neut</th>
                <th style="text-align: right; color: #94a3b8; padding: 8px 5px; border-bottom: 1px solid #1e3a5f;">Total</th>
            </tr>
    """]

    for ticker, data in sorted_tickers[:max_display]:
        score = data.get("aggregate_score", 50)
        label = data.get("aggregate_label", "neutral")
        st = data.get("stocktwits", {})
        total = st.get("total_messages", 0)
        bullish = st.get("bullish_count", 0)
        bearish = st.get("bearish_count", 0)
        neutral = st.get("neutral_count", 0)

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
                <td style="padding: 6px 5px; color: #e2e8f0; font-weight: bold;">{ticker}</td>
                <td style="padding: 6px 5px; text-align: center; color: {color}; font-weight: bold;">{score:.0f}</td>
                <td style="padding: 6px 5px; text-align: center; color: {color};">{emoji} {label.upper()}</td>
                <td style="padding: 6px 5px; text-align: center; color: #4ade80;">{bullish}</td>
                <td style="padding: 6px 5px; text-align: center; color: #f87171;">{bearish}</td>
                <td style="padding: 6px 5px; text-align: center; color: #94a3b8;">{neutral}</td>
                <td style="padding: 6px 5px; text-align: right; color: #cbd5e1;">{total}</td>
            </tr>
        """)

    html_parts.append("""
        </table>
        <p style="color: #64748b; font-size: 10px; margin: 10px 0 0 0; text-align: right;">
            Data from StockTwits • Score: 0-100 (50=neutral) • Bull/Bear/Neut = post counts with tagged sentiment
        </p>
    </div>
    """)

    return "".join(html_parts)
