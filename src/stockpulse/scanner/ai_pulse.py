"""AI Pulse Scanner - Mega-Cap and AI Investment Research System.

This module provides:
1. Trillion+ Club tracking (stocks that have been over $1T at any point in last 30 days)
2. Category classification (Hyperscalers vs Neoclouds)
3. AI thesis research and tracking with Claude API
4. Time-series entry point detection for mega-caps

The core thesis: mega-cap tech companies will continue to dominate,
so we're looking for optimal long-term entry points.
"""

import os
from datetime import datetime, date, timedelta
from typing import Any
import json

import pandas as pd
import numpy as np
import yfinance as yf

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.ingestion import DataIngestion
from stockpulse.alerts.alert_manager import AlertManager

logger = get_logger(__name__)


# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================

# Hyperscalers: Established cloud giants with massive revenue and infrastructure
HYPERSCALERS = {
    "MSFT": {"name": "Microsoft", "cloud": "Azure", "notes": "Enterprise cloud leader"},
    "AMZN": {"name": "Amazon", "cloud": "AWS", "notes": "Cloud market share leader"},
    "GOOGL": {"name": "Alphabet", "cloud": "GCP", "notes": "AI/ML cloud innovation"},
    "ORCL": {"name": "Oracle", "cloud": "OCI", "notes": "Enterprise database + cloud"},
    "IBM": {"name": "IBM", "cloud": "IBM Cloud", "notes": "Hybrid cloud focus"},
}

# Neoclouds: New AI-native cloud providers, heavily leveraged, high growth potential
NEOCLOUDS = {
    "CRWV": {"name": "CoreWeave", "notes": "GPU cloud for AI, IPO 2024"},
    # Lambda Labs - private
    # Together AI - private
    # Cerebras - private (filed S-1)
}

# AI Infrastructure plays (GPU, chips, data centers)
AI_INFRASTRUCTURE = {
    "NVDA": {"name": "NVIDIA", "category": "GPU/AI chips", "notes": "Dominant AI chip maker"},
    "AMD": {"name": "AMD", "category": "GPU/chips", "notes": "GPU competition, MI300"},
    "AVGO": {"name": "Broadcom", "category": "Custom chips", "notes": "Google TPU partner"},
    "MRVL": {"name": "Marvell", "category": "Custom chips", "notes": "Custom AI silicon"},
    "TSM": {"name": "TSMC", "category": "Foundry", "notes": "Makes all advanced chips"},
    "ASML": {"name": "ASML", "category": "Equipment", "notes": "EUV lithography monopoly"},
}

# AI Software/Platform leaders
AI_SOFTWARE = {
    "MSFT": {"name": "Microsoft", "notes": "Copilot, OpenAI partnership"},
    "GOOGL": {"name": "Alphabet", "notes": "Gemini, DeepMind, Waymo"},
    "META": {"name": "Meta", "notes": "Llama open source, AI ads"},
    "CRM": {"name": "Salesforce", "notes": "Einstein AI, enterprise"},
    "PLTR": {"name": "Palantir", "notes": "AIP platform, government"},
    "NOW": {"name": "ServiceNow", "notes": "Enterprise AI workflows"},
}

# Robotics / Physical AI thesis
ROBOTICS_THESIS = {
    "TSLA": {"name": "Tesla", "notes": "Optimus robot, FSD, Dojo"},
    "GOOGL": {"name": "Alphabet", "notes": "Waymo autonomous driving"},
    "AMZN": {"name": "Amazon", "notes": "Warehouse robotics, delivery"},
    "ISRG": {"name": "Intuitive Surgical", "notes": "Da Vinci surgical robots"},
    "FANUY": {"name": "Fanuc", "notes": "Industrial robotics leader"},
}

# Known trillion dollar club members (current and recent)
TRILLION_CLUB_SEED = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "TSM", "V", "JPM", "WMT", "XOM", "UNH", "MA", "LLY", "AVGO"
]


class ClaudeResearch:
    """Claude API integration for AI thesis research."""

    def __init__(self):
        """Initialize Claude API client."""
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.is_configured = bool(self.api_key)

        if not self.is_configured:
            logger.warning(
                "ANTHROPIC_API_KEY not set. AI research features will be limited. "
                "Add ANTHROPIC_API_KEY to .env to enable Claude research."
            )

    def research_thesis(
        self,
        thesis_name: str,
        context: str,
        tickers: list[str],
        max_tokens: int = 1024
    ) -> dict[str, Any]:
        """
        Research an investment thesis using Claude.

        Args:
            thesis_name: Name of the thesis (e.g., "Tesla Robot Thesis")
            context: Current context about the thesis
            tickers: Related tickers to research
            max_tokens: Maximum response length

        Returns:
            Research findings dict with analysis, signals, and recommendations
        """
        if not self.is_configured:
            return {
                "thesis": thesis_name,
                "analysis": "Claude API not configured. Add ANTHROPIC_API_KEY to .env.",
                "signals": [],
                "recommendation": "neutral",
                "confidence": 0,
            }

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""You are an expert AI and technology investment analyst.
Analyze the following investment thesis and provide actionable insights.

THESIS: {thesis_name}

CONTEXT: {context}

RELATED TICKERS: {', '.join(tickers)}

TODAY'S DATE: {date.today().isoformat()}

Provide a concise analysis with:
1. THESIS STATUS: Is this thesis playing out? What's the current evidence?
2. KEY SIGNALS: What recent developments support or contradict this thesis?
3. ENTRY POINTS: Are any of these stocks at attractive entry points right now?
4. RISKS: What could invalidate this thesis?
5. RECOMMENDATION: bullish/neutral/bearish on this thesis overall
6. TOP PICK: Which single stock best captures this thesis right now?

Be specific, cite recent developments if known, and focus on actionable insights.
Keep response under 500 words."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = message.content[0].text

            # Parse recommendation from response
            recommendation = "neutral"
            if "bullish" in analysis.lower():
                recommendation = "bullish"
            elif "bearish" in analysis.lower():
                recommendation = "bearish"

            return {
                "thesis": thesis_name,
                "analysis": analysis,
                "signals": [],  # Could parse signals from response
                "recommendation": recommendation,
                "confidence": 70 if recommendation != "neutral" else 50,
                "researched_at": datetime.now().isoformat(),
            }

        except ImportError:
            logger.warning("anthropic package not installed. Run: pip install anthropic")
            return {
                "thesis": thesis_name,
                "analysis": "anthropic package not installed",
                "signals": [],
                "recommendation": "neutral",
                "confidence": 0,
            }
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "thesis": thesis_name,
                "analysis": f"Error: {str(e)}",
                "signals": [],
                "recommendation": "neutral",
                "confidence": 0,
            }

    def generate_market_pulse(self, market_data: dict[str, Any]) -> str:
        """
        Generate a daily AI market pulse summary.

        Args:
            market_data: Dict with market metrics, top movers, news

        Returns:
            Formatted market pulse summary
        """
        if not self.is_configured:
            return "Claude API not configured. Add ANTHROPIC_API_KEY to .env for AI-powered summaries."

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""You are an AI market analyst providing a daily briefing.
Generate a concise, actionable market pulse for AI/tech investors.

MARKET DATA:
{json.dumps(market_data, indent=2, default=str)}

TODAY'S DATE: {date.today().isoformat()}

Provide a brief (3-5 bullet points) market pulse covering:
- AI sector sentiment today
- Notable moves in AI stocks
- Any important news/developments
- What to watch tomorrow

Keep it punchy and actionable. Under 200 words."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text

        except Exception as e:
            logger.error(f"Market pulse generation error: {e}")
            return f"Unable to generate market pulse: {str(e)}"


class AIPulseScanner:
    """
    AI Pulse Scanner for mega-cap and AI investment tracking.

    Combines:
    - Trillion+ Club tracking (market cap monitoring)
    - Category classification (Hyperscalers, Neoclouds, AI Infra)
    - Thesis tracking and validation
    - Entry point detection using time-series analysis
    """

    def __init__(self):
        """Initialize AI Pulse scanner."""
        self.db = get_db()
        self.config = get_config()
        self.ai_config = self.config.get("ai_pulse", {})
        self.data_ingestion = DataIngestion()
        self.alert_manager = AlertManager()
        self.claude = ClaudeResearch()

        # Initialize database tables
        self._init_tables()

        # Cache for yfinance data
        self._yf_cache = {}

    def _init_tables(self) -> None:
        """Initialize AI Pulse specific database tables."""
        cursor = self.db.conn.cursor()

        # Trillion+ Club tracking with market cap history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trillion_club (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                market_cap REAL,
                market_cap_category TEXT,
                peak_market_cap_30d REAL,
                current_price REAL,
                price_vs_30d_high_pct REAL,
                entry_score REAL,
                category TEXT,
                subcategory TEXT,
                reasoning TEXT,
                UNIQUE (ticker, scan_date)
            )
        """)

        # AI thesis tracker
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_theses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thesis_name TEXT NOT NULL UNIQUE,
                description TEXT,
                tickers TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT,
                status TEXT DEFAULT 'active',
                last_research TEXT,
                recommendation TEXT DEFAULT 'neutral',
                confidence REAL DEFAULT 50,
                notes TEXT
            )
        """)

        # Thesis research history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thesis_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thesis_id INTEGER,
                research_date TEXT NOT NULL,
                analysis TEXT,
                recommendation TEXT,
                confidence REAL,
                key_signals TEXT,
                FOREIGN KEY (thesis_id) REFERENCES ai_theses(id)
            )
        """)

        # AI Pulse daily snapshots for time-series
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_pulse_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                trillion_club_count INTEGER,
                total_trillion_market_cap REAL,
                avg_entry_score REAL,
                top_opportunity TEXT,
                market_pulse TEXT,
                UNIQUE (snapshot_date)
            )
        """)

        self.db.conn.commit()

        # Seed default theses if table is empty
        self._seed_default_theses()

    def _seed_default_theses(self) -> None:
        """Seed default investment theses if none exist."""
        result = self.db.fetchone("SELECT COUNT(*) FROM ai_theses")
        if result and result[0] > 0:
            return

        default_theses = [
            {
                "name": "Tesla Robot Thesis",
                "description": "Tesla's Optimus humanoid robot could be bigger than cars. "
                              "Physical AI + manufacturing scale + FSD learnings = moat.",
                "tickers": "TSLA,NVDA,ISRG",
            },
            {
                "name": "AI Infrastructure Buildout",
                "description": "Massive capex cycle for AI data centers. GPU demand, power, "
                              "cooling, and custom silicon will be multi-year tailwinds.",
                "tickers": "NVDA,AMD,AVGO,MRVL,TSM,ASML",
            },
            {
                "name": "Hyperscaler Dominance",
                "description": "Big 3 cloud providers (AWS, Azure, GCP) will capture most "
                              "AI inference revenue as enterprises deploy AI.",
                "tickers": "AMZN,MSFT,GOOGL",
            },
            {
                "name": "AI Software Monetization",
                "description": "Enterprise AI tools (Copilot, Einstein, etc.) will drive "
                              "massive revenue growth for software leaders.",
                "tickers": "MSFT,CRM,NOW,PLTR",
            },
            {
                "name": "Neocloud Disruption",
                "description": "New AI-native cloud providers could take share from "
                              "hyperscalers for GPU-intensive AI workloads.",
                "tickers": "CRWV",
            },
        ]

        for thesis in default_theses:
            try:
                self.db.execute("""
                    INSERT INTO ai_theses (thesis_name, description, tickers)
                    VALUES (?, ?, ?)
                """, (thesis["name"], thesis["description"], thesis["tickers"]))
            except Exception as e:
                logger.debug(f"Thesis already exists or error: {e}")

    def _get_yf_ticker(self, ticker: str) -> yf.Ticker:
        """Get cached yfinance Ticker object."""
        if ticker not in self._yf_cache:
            self._yf_cache[ticker] = yf.Ticker(ticker)
        return self._yf_cache[ticker]

    def get_trillion_club_members(self) -> list[dict]:
        """
        Identify stocks that have been over $1T market cap in the last 30 days.

        Returns:
            List of trillion club member dicts with market cap data
        """
        logger.info("Scanning for Trillion+ Club members...")

        trillion_threshold = 1_000_000_000_000  # $1 trillion
        members = []

        # Start with seed list + any from database
        candidates = set(TRILLION_CLUB_SEED)

        # Add any previously tracked
        existing = self.db.fetchdf("""
            SELECT DISTINCT ticker FROM trillion_club
            WHERE scan_date >= date('now', '-60 days')
        """)
        if not existing.empty:
            candidates.update(existing["ticker"].tolist())

        for ticker in candidates:
            try:
                yf_ticker = self._get_yf_ticker(ticker)
                info = yf_ticker.info

                current_market_cap = info.get("marketCap", 0)
                if not current_market_cap:
                    continue

                # Get 30-day high market cap (approximate from price history)
                # Market cap = shares * price, so we look at price high
                hist = yf_ticker.history(period="1mo")
                if hist.empty:
                    continue

                current_price = info.get("regularMarketPrice") or info.get("currentPrice") or hist["Close"].iloc[-1]
                high_price_30d = hist["High"].max()
                shares_outstanding = info.get("sharesOutstanding", 0)

                if shares_outstanding:
                    peak_market_cap_30d = shares_outstanding * high_price_30d
                else:
                    # Estimate from current ratio
                    if current_price > 0:
                        peak_market_cap_30d = current_market_cap * (high_price_30d / current_price)
                    else:
                        peak_market_cap_30d = current_market_cap

                # Check if ever hit $1T in last 30 days
                if peak_market_cap_30d >= trillion_threshold:
                    # Determine category
                    category, subcategory = self._categorize_stock(ticker)

                    # Calculate entry score with breakdown
                    entry_score, score_breakdown = self._calculate_entry_score(
                        ticker, current_price, hist, info, return_breakdown=True
                    )

                    # Price vs 30d high
                    price_vs_30d_high_pct = ((current_price / high_price_30d) - 1) * 100 if high_price_30d > 0 else 0

                    member = {
                        "ticker": ticker,
                        "company_name": info.get("shortName", info.get("longName", ticker)),
                        "market_cap": current_market_cap,
                        "market_cap_b": current_market_cap / 1_000_000_000,
                        "market_cap_category": self._market_cap_category(current_market_cap),
                        "peak_market_cap_30d": peak_market_cap_30d,
                        "peak_market_cap_30d_b": peak_market_cap_30d / 1_000_000_000,
                        "current_price": current_price,
                        "price_vs_30d_high_pct": price_vs_30d_high_pct,
                        "entry_score": entry_score,
                        "score_breakdown": score_breakdown,
                        "category": category,
                        "subcategory": subcategory,
                        "sector": info.get("sector", "Unknown"),
                        "scan_date": date.today(),
                    }
                    members.append(member)

                    logger.debug(
                        f"{ticker}: ${current_market_cap/1e12:.2f}T current, "
                        f"${peak_market_cap_30d/1e12:.2f}T peak, score {entry_score:.0f}"
                    )

            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                continue

        # Sort by entry score (best opportunities first)
        members.sort(key=lambda x: x["entry_score"], reverse=True)

        logger.info(f"Found {len(members)} Trillion+ Club members")
        return members

    def _categorize_stock(self, ticker: str) -> tuple[str, str]:
        """
        Categorize a stock into our classification system.

        Returns:
            Tuple of (category, subcategory)
        """
        if ticker in HYPERSCALERS:
            return "Hyperscaler", HYPERSCALERS[ticker].get("cloud", "Cloud")

        if ticker in NEOCLOUDS:
            return "Neocloud", "AI Cloud"

        if ticker in AI_INFRASTRUCTURE:
            return "AI Infrastructure", AI_INFRASTRUCTURE[ticker].get("category", "Chips")

        if ticker in AI_SOFTWARE:
            return "AI Software", "Platform"

        if ticker in ROBOTICS_THESIS:
            return "Robotics/Physical AI", ROBOTICS_THESIS[ticker].get("notes", "")

        # Default categorization based on sector
        try:
            info = self._get_yf_ticker(ticker).info
            sector = info.get("sector", "")
            industry = info.get("industry", "")

            if "Technology" in sector:
                if "Semiconductor" in industry:
                    return "AI Infrastructure", "Chips"
                elif "Software" in industry:
                    return "AI Software", industry
                else:
                    return "Technology", industry
            elif "Financial" in sector:
                return "Fintech", industry
            elif "Consumer" in sector:
                return "Consumer Tech", industry
            elif "Health" in sector:
                return "Healthcare", industry
            else:
                return "Other", sector

        except Exception:
            return "Unknown", ""

    def _market_cap_category(self, market_cap: float) -> str:
        """Categorize market cap size."""
        if market_cap >= 2_000_000_000_000:  # $2T+
            return "Super Mega Cap"
        elif market_cap >= 1_000_000_000_000:  # $1T+
            return "Trillion Club"
        elif market_cap >= 500_000_000_000:  # $500B+
            return "Near Trillion"
        elif market_cap >= 200_000_000_000:  # $200B+
            return "Mega Cap"
        else:
            return "Large Cap"

    def _calculate_entry_score(
        self,
        ticker: str,
        current_price: float,
        history: pd.DataFrame,
        info: dict,
        return_breakdown: bool = False
    ) -> float | tuple[float, dict]:
        """
        Calculate an entry point score for a mega-cap stock.

        Higher score = better entry point.
        Considers: valuation, technical position, sentiment.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            history: Price history DataFrame
            info: yfinance info dict
            return_breakdown: If True, return (score, breakdown_dict)

        Returns:
            Score (0-100), or tuple of (score, breakdown) if return_breakdown=True
        """
        score = 50  # Base score
        breakdown = {
            "base": {"points": 50, "label": "Base Score", "raw_value": "-"},
        }

        # === TECHNICAL FACTORS ===

        # Distance from 30-day high (lower = better entry)
        high_30d = history["High"].max()
        pct_from_high = 0.0
        distance_pts = 0
        if high_30d > 0:
            pct_from_high = ((current_price / high_30d) - 1) * 100
            if pct_from_high <= -15:
                distance_pts = 20  # 15%+ pullback
            elif pct_from_high <= -10:
                distance_pts = 15
            elif pct_from_high <= -5:
                distance_pts = 10
            elif pct_from_high <= -2:
                distance_pts = 5
            elif pct_from_high >= 0:
                distance_pts = -5  # At or near highs
        score += distance_pts
        breakdown["distance_from_high"] = {
            "points": distance_pts,
            "label": "Distance from 30d High",
            "raw_value": f"{pct_from_high:+.1f}%",
        }

        # RSI (oversold = better entry)
        current_rsi = 50.0
        rsi_pts = 0
        if len(history) >= 14:
            delta = history["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            if current_rsi < 30:
                rsi_pts = 15
            elif current_rsi < 40:
                rsi_pts = 10
            elif current_rsi > 80:
                rsi_pts = -15
            elif current_rsi > 70:
                rsi_pts = -10
        score += rsi_pts
        breakdown["rsi"] = {
            "points": rsi_pts,
            "label": "RSI (14)",
            "raw_value": f"{current_rsi:.1f}",
        }

        # 50-day MA position
        ma_50_pts = 0
        ma_50_pct = 0.0
        if len(history) >= 50:
            ma_50 = history["Close"].rolling(50).mean().iloc[-1]
            ma_50_pct = ((current_price / ma_50) - 1) * 100 if ma_50 > 0 else 0
            if current_price < ma_50 * 0.95:
                ma_50_pts = 10  # 5%+ below 50 MA
            elif current_price < ma_50:
                ma_50_pts = 5
        score += ma_50_pts
        breakdown["ma_50"] = {
            "points": ma_50_pts,
            "label": "50-Day MA Position",
            "raw_value": f"{ma_50_pct:+.1f}% vs MA",
        }

        # === VALUATION FACTORS ===

        pe_ratio = info.get("trailingPE", info.get("forwardPE", 0))
        pe_pts = 0
        if pe_ratio:
            # For mega-caps, PE < 20 is attractive
            if pe_ratio < 15:
                pe_pts = 15
            elif pe_ratio < 20:
                pe_pts = 10
            elif pe_ratio < 25:
                pe_pts = 5
            elif pe_ratio > 60:
                pe_pts = -15
            elif pe_ratio > 40:
                pe_pts = -10
        score += pe_pts
        breakdown["pe_ratio"] = {
            "points": pe_pts,
            "label": "P/E Ratio",
            "raw_value": f"{pe_ratio:.1f}" if pe_ratio else "N/A",
        }

        # Forward PE vs trailing (growth expectation)
        forward_pe = info.get("forwardPE", 0)
        trailing_pe = info.get("trailingPE", 0)
        growth_pts = 0
        if forward_pe and trailing_pe and forward_pe < trailing_pe:
            growth_pts = 5  # Earnings growth expected
        score += growth_pts
        breakdown["earnings_growth"] = {
            "points": growth_pts,
            "label": "Earnings Growth Expected",
            "raw_value": f"Fwd {forward_pe:.1f} vs Trail {trailing_pe:.1f}" if forward_pe and trailing_pe else "N/A",
        }

        # === MOMENTUM FACTORS ===

        # Recent price momentum (not too hot, not too cold)
        momentum_pts = 0
        pct_change_20d = 0.0
        if len(history) >= 20:
            pct_change_20d = (current_price / history["Close"].iloc[-20] - 1) * 100
            if -10 <= pct_change_20d <= 0:
                momentum_pts = 5  # Healthy consolidation
            elif pct_change_20d < -15:
                momentum_pts = 10  # Potential oversold bounce
            elif pct_change_20d > 20:
                momentum_pts = -10  # Extended
        score += momentum_pts
        breakdown["momentum_20d"] = {
            "points": momentum_pts,
            "label": "20-Day Momentum",
            "raw_value": f"{pct_change_20d:+.1f}%",
        }

        final_score = max(0, min(100, score))

        if return_breakdown:
            # Add total to breakdown
            breakdown["total"] = {
                "points": final_score,
                "label": "TOTAL SCORE",
                "raw_value": "-",
            }
            return final_score, breakdown

        return final_score

    def run_scan(self) -> dict[str, Any]:
        """
        Run full AI Pulse scan.

        Returns:
            Dict containing all scan results:
            - trillion_club: List of trillion+ stocks with entry scores
            - categories: Breakdown by category
            - theses: Thesis research updates
            - market_pulse: AI-generated market summary
        """
        logger.info("Running AI Pulse scan...")

        # 1. Get Trillion+ Club members
        trillion_club = self.get_trillion_club_members()

        # 2. Categorize members
        categories = {
            "Hyperscaler": [],
            "Neocloud": [],
            "AI Infrastructure": [],
            "AI Software": [],
            "Robotics/Physical AI": [],
            "Other": [],
        }
        for member in trillion_club:
            cat = member.get("category", "Other")
            if cat in categories:
                categories[cat].append(member)
            else:
                categories["Other"].append(member)

        # 3. Research active theses
        theses_results = self._research_theses()

        # 4. Generate market pulse
        market_data = {
            "trillion_club_count": len(trillion_club),
            "total_market_cap_t": sum(m["market_cap"] for m in trillion_club) / 1e12,
            "top_entry_opportunities": [
                {"ticker": m["ticker"], "score": m["entry_score"], "pct_from_high": m["price_vs_30d_high_pct"]}
                for m in trillion_club[:5]
            ],
            "categories": {cat: len(stocks) for cat, stocks in categories.items()},
            "best_entry_score": trillion_club[0]["entry_score"] if trillion_club else 0,
        }
        market_pulse = self.claude.generate_market_pulse(market_data)

        # 5. Store results
        self._store_scan_results(trillion_club)
        self._store_snapshot(trillion_club, market_pulse)

        # 6. Enrich with trend data
        trillion_club = self._enrich_with_trends(trillion_club)

        results = {
            "scan_date": date.today().isoformat(),
            "trillion_club": trillion_club,
            "trillion_club_count": len(trillion_club),
            "categories": categories,
            "theses": theses_results,
            "market_pulse": market_pulse,
            "best_opportunities": [m for m in trillion_club if m["entry_score"] >= 70][:5],
        }

        logger.info(f"AI Pulse scan complete: {len(trillion_club)} trillion club members")
        return results

    def _research_theses(self) -> list[dict]:
        """Research all active theses."""
        theses_df = self.db.fetchdf("""
            SELECT * FROM ai_theses WHERE status = 'active'
        """)

        if theses_df.empty:
            return []

        results = []
        for _, thesis in theses_df.iterrows():
            thesis_name = thesis["thesis_name"]
            description = thesis["description"]
            tickers = thesis["tickers"].split(",") if thesis["tickers"] else []

            # Only research if Claude is configured
            if self.claude.is_configured:
                research = self.claude.research_thesis(
                    thesis_name=thesis_name,
                    context=description,
                    tickers=tickers
                )

                # Store research
                try:
                    self.db.execute("""
                        INSERT INTO thesis_research (thesis_id, research_date, analysis, recommendation, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        thesis["id"],
                        date.today().isoformat(),
                        research.get("analysis", ""),
                        research.get("recommendation", "neutral"),
                        research.get("confidence", 50)
                    ))

                    # Update thesis with latest
                    self.db.execute("""
                        UPDATE ai_theses
                        SET last_research = ?, recommendation = ?, confidence = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        research.get("analysis", "")[:500],
                        research.get("recommendation", "neutral"),
                        research.get("confidence", 50),
                        datetime.now().isoformat(),
                        thesis["id"]
                    ))
                except Exception as e:
                    logger.debug(f"Error storing research: {e}")

                results.append({
                    "thesis_name": thesis_name,
                    "tickers": tickers,
                    **research
                })
            else:
                # Return basic info without research
                results.append({
                    "thesis_name": thesis_name,
                    "tickers": tickers,
                    "description": description,
                    "recommendation": thesis.get("recommendation", "neutral"),
                    "confidence": thesis.get("confidence", 50),
                    "analysis": "Claude API not configured for live research.",
                })

        return results

    def _store_scan_results(self, members: list[dict]) -> None:
        """Store trillion club scan results."""
        for member in members:
            try:
                self.db.execute("""
                    INSERT INTO trillion_club (
                        ticker, scan_date, market_cap, market_cap_category,
                        peak_market_cap_30d, current_price, price_vs_30d_high_pct,
                        entry_score, category, subcategory, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, scan_date) DO UPDATE SET
                        market_cap = excluded.market_cap,
                        market_cap_category = excluded.market_cap_category,
                        peak_market_cap_30d = excluded.peak_market_cap_30d,
                        current_price = excluded.current_price,
                        price_vs_30d_high_pct = excluded.price_vs_30d_high_pct,
                        entry_score = excluded.entry_score,
                        category = excluded.category,
                        subcategory = excluded.subcategory
                """, (
                    member["ticker"],
                    member["scan_date"],
                    member["market_cap"],
                    member["market_cap_category"],
                    member["peak_market_cap_30d"],
                    member["current_price"],
                    member["price_vs_30d_high_pct"],
                    member["entry_score"],
                    member["category"],
                    member["subcategory"],
                    member.get("reasoning", ""),
                ))
            except Exception as e:
                logger.debug(f"Error storing {member['ticker']}: {e}")

    def _store_snapshot(self, members: list[dict], market_pulse: str) -> None:
        """Store daily snapshot for time-series tracking."""
        try:
            total_market_cap = sum(m["market_cap"] for m in members)
            avg_entry_score = sum(m["entry_score"] for m in members) / len(members) if members else 0
            top_opportunity = members[0]["ticker"] if members else ""

            self.db.execute("""
                INSERT INTO ai_pulse_snapshots (
                    snapshot_date, trillion_club_count, total_trillion_market_cap,
                    avg_entry_score, top_opportunity, market_pulse
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    trillion_club_count = excluded.trillion_club_count,
                    total_trillion_market_cap = excluded.total_trillion_market_cap,
                    avg_entry_score = excluded.avg_entry_score,
                    top_opportunity = excluded.top_opportunity,
                    market_pulse = excluded.market_pulse
            """, (
                date.today().isoformat(),
                len(members),
                total_market_cap,
                avg_entry_score,
                top_opportunity,
                market_pulse[:2000] if market_pulse else "",
            ))
        except Exception as e:
            logger.debug(f"Error storing snapshot: {e}")

    def _enrich_with_trends(self, members: list[dict]) -> list[dict]:
        """Add trend data to each member."""
        for member in members:
            trend = self.get_trend_data(member["ticker"])
            member["consecutive_days"] = trend["consecutive_days"]
            member["trend_symbol"] = trend["trend_symbol"]
            member["score_change_5d"] = trend["score_change_5d"]
            member["is_new"] = trend["is_new"]
        return members

    def get_trend_data(self, ticker: str, days_back: int = 30) -> dict:
        """Get trend data for a trillion club member."""
        history = self.db.fetchdf("""
            SELECT scan_date, entry_score
            FROM trillion_club
            WHERE ticker = ? AND scan_date >= date('now', ?)
            ORDER BY scan_date DESC
        """, (ticker, f'-{days_back} days'))

        if history.empty:
            return {
                "consecutive_days": 0,
                "score_history": [],
                "score_change_5d": 0,
                "trend_symbol": "🆕",
                "is_new": True
            }

        scores = history["entry_score"].tolist()
        dates = history["scan_date"].tolist()

        # Count consecutive days
        consecutive_days = len(scores)

        # Score change vs 5 days ago
        score_change_5d = (scores[0] - scores[min(4, len(scores)-1)]) if len(scores) > 1 else 0

        # Trend symbol
        if len(scores) == 1:
            trend_symbol = "🆕"
        elif score_change_5d > 5:
            trend_symbol = "📈"
        elif score_change_5d < -5:
            trend_symbol = "📉"
        else:
            trend_symbol = "➡️"

        return {
            "consecutive_days": consecutive_days,
            "score_history": scores[:10],
            "score_change_5d": score_change_5d,
            "trend_symbol": trend_symbol,
            "is_new": len(scores) == 1
        }

    def check_data_quality(self) -> dict[str, Any]:
        """
        Check if database has sufficient data for clean signals.

        Returns:
            Dict with data quality status and any issues
        """
        issues = []
        status = "ok"

        # Check price data freshness
        result = self.db.fetchone("""
            SELECT MAX(date) FROM prices_daily
        """)
        if result and result[0]:
            latest_date = datetime.strptime(str(result[0])[:10], "%Y-%m-%d").date()
            days_stale = (date.today() - latest_date).days
            if days_stale > 3:
                issues.append(f"Price data is {days_stale} days old")
                status = "warning"
        else:
            issues.append("No price data in database")
            status = "error"

        # Check trillion club history
        result = self.db.fetchone("""
            SELECT COUNT(DISTINCT scan_date) FROM trillion_club
            WHERE scan_date >= date('now', '-14 days')
        """)
        tc_dates = result[0] if result else 0
        if tc_dates < 5:
            issues.append(f"Only {tc_dates} days of trillion club history (need 5+)")
            if status == "ok":
                status = "warning"

        # Check fundamentals
        result = self.db.fetchone("""
            SELECT COUNT(*) FROM fundamentals
            WHERE date >= date('now', '-7 days')
        """)
        fund_count = result[0] if result else 0
        if fund_count < 50:
            issues.append(f"Only {fund_count} recent fundamentals records")

        return {
            "status": status,
            "issues": issues,
            "ok": status == "ok",
            "checked_at": datetime.now().isoformat()
        }

    def get_theses(self) -> list[dict]:
        """Get all tracked theses."""
        df = self.db.fetchdf("""
            SELECT * FROM ai_theses ORDER BY updated_at DESC
        """)
        return df.to_dict("records") if not df.empty else []

    def add_thesis(self, name: str, description: str, tickers: list[str]) -> int:
        """Add a new investment thesis to track."""
        cursor = self.db.execute("""
            INSERT INTO ai_theses (thesis_name, description, tickers)
            VALUES (?, ?, ?)
        """, (name, description, ",".join(tickers)))
        return cursor.lastrowid
