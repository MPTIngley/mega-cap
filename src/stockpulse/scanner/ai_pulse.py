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
# CATEGORY DEFINITIONS - Expanded AI Universe
# ============================================================================

# Hyperscalers: Established cloud giants with massive revenue and infrastructure
HYPERSCALERS = {
    "MSFT": {"name": "Microsoft", "cloud": "Azure", "notes": "Enterprise cloud leader, OpenAI partnership"},
    "AMZN": {"name": "Amazon", "cloud": "AWS", "notes": "Cloud market share leader, Bedrock AI"},
    "GOOGL": {"name": "Alphabet", "cloud": "GCP", "notes": "Gemini, DeepMind, Vertex AI"},
    "ORCL": {"name": "Oracle", "cloud": "OCI", "notes": "Enterprise database + cloud, NVIDIA partnership"},
    "IBM": {"name": "IBM", "cloud": "IBM Cloud", "notes": "watsonx AI, hybrid cloud focus"},
    "BABA": {"name": "Alibaba", "cloud": "Aliyun", "notes": "China cloud leader, Qwen AI models"},
}

# Neoclouds: New AI-native cloud providers, heavily leveraged, high growth potential
NEOCLOUDS = {
    "CRWV": {"name": "CoreWeave", "notes": "GPU cloud for AI, NVIDIA-backed"},
    # Lambda Labs - private
    # Together AI - private
    # Cerebras - IPO pending
}

# AI Infrastructure plays (GPU, chips, data centers, networking)
AI_INFRASTRUCTURE = {
    # GPU/AI Chips
    "NVDA": {"name": "NVIDIA", "category": "GPU/AI chips", "notes": "Dominant AI chip maker, Blackwell"},
    "AMD": {"name": "AMD", "category": "GPU/chips", "notes": "MI300X competition, data center growth"},
    "INTC": {"name": "Intel", "category": "CPU/AI chips", "notes": "Gaudi accelerators, foundry ambitions"},
    "QCOM": {"name": "Qualcomm", "category": "Edge AI chips", "notes": "Snapdragon X, on-device AI"},
    # Custom Silicon
    "AVGO": {"name": "Broadcom", "category": "Custom chips", "notes": "Google TPU partner, VMware"},
    "MRVL": {"name": "Marvell", "category": "Custom chips", "notes": "Custom AI silicon for hyperscalers"},
    "ARM": {"name": "Arm Holdings", "category": "CPU architecture", "notes": "AI chip architecture, edge compute"},
    # Foundry & Equipment
    "TSM": {"name": "TSMC", "category": "Foundry", "notes": "Makes all advanced AI chips"},
    "ASML": {"name": "ASML", "category": "Equipment", "notes": "EUV lithography monopoly"},
    "AMAT": {"name": "Applied Materials", "category": "Equipment", "notes": "Chip manufacturing equipment"},
    "LRCX": {"name": "Lam Research", "category": "Equipment", "notes": "Etch and deposition equipment"},
    "KLAC": {"name": "KLA Corp", "category": "Equipment", "notes": "Process control equipment"},
    # Memory
    "MU": {"name": "Micron", "category": "Memory", "notes": "HBM memory for AI, DRAM leader"},
    "WDC": {"name": "Western Digital", "category": "Storage", "notes": "Data storage for AI workloads"},
    "SMCI": {"name": "Super Micro", "category": "AI Servers", "notes": "AI server systems, GPU integration"},
    # Networking
    "ANET": {"name": "Arista Networks", "category": "Networking", "notes": "AI data center networking"},
    "CSCO": {"name": "Cisco", "category": "Networking", "notes": "Enterprise networking, AI infrastructure"},
    # Data Centers
    "EQIX": {"name": "Equinix", "category": "Data Centers", "notes": "Largest data center REIT"},
    "DLR": {"name": "Digital Realty", "category": "Data Centers", "notes": "AI-ready data centers"},
    # Power/Utilities for AI
    "VST": {"name": "Vistra", "category": "Power", "notes": "Power for AI data centers"},
    "CEG": {"name": "Constellation Energy", "category": "Power", "notes": "Nuclear for AI data centers"},
    "NRG": {"name": "NRG Energy", "category": "Power", "notes": "Power generation for data centers"},
}

# AI Software/Platform leaders
AI_SOFTWARE = {
    # Major Platforms
    "MSFT": {"name": "Microsoft", "notes": "Copilot, OpenAI, GitHub, Azure AI"},
    "GOOGL": {"name": "Alphabet", "notes": "Gemini, DeepMind, Waymo, Cloud AI"},
    "META": {"name": "Meta", "notes": "Llama open source, AI ads, Reality Labs"},
    "AAPL": {"name": "Apple", "notes": "Apple Intelligence, on-device AI, Siri"},
    # Enterprise AI
    "CRM": {"name": "Salesforce", "notes": "Einstein AI, Agentforce, enterprise AI"},
    "NOW": {"name": "ServiceNow", "notes": "AI workflows, enterprise automation"},
    "ADBE": {"name": "Adobe", "notes": "Firefly generative AI, Creative Cloud"},
    "WDAY": {"name": "Workday", "notes": "AI for HR and finance"},
    "SAP": {"name": "SAP", "notes": "Enterprise AI, Joule assistant"},
    "INTU": {"name": "Intuit", "notes": "AI for small business, TurboTax AI"},
    # AI-Native Software
    "PLTR": {"name": "Palantir", "notes": "AIP platform, government & enterprise"},
    "AI": {"name": "C3.ai", "notes": "Enterprise AI applications"},
    "PATH": {"name": "UiPath", "notes": "Robotic process automation, AI agents"},
    "SNOW": {"name": "Snowflake", "notes": "AI data cloud, Cortex AI"},
    "MDB": {"name": "MongoDB", "notes": "Database for AI applications"},
    "DDOG": {"name": "Datadog", "notes": "AI observability, LLM monitoring"},
    "CFLT": {"name": "Confluent", "notes": "Real-time data streaming for AI"},
    "ESTC": {"name": "Elastic", "notes": "Search and AI, vector search"},
    "GTLB": {"name": "GitLab", "notes": "DevSecOps with AI, code generation"},
    # Cybersecurity AI
    "CRWD": {"name": "CrowdStrike", "notes": "AI-powered cybersecurity"},
    "PANW": {"name": "Palo Alto Networks", "notes": "AI security platform"},
    "ZS": {"name": "Zscaler", "notes": "Zero trust AI security"},
    "S": {"name": "SentinelOne", "notes": "AI endpoint protection"},
    "FTNT": {"name": "Fortinet", "notes": "AI-driven network security"},
    # Communication/Collaboration AI
    "ZM": {"name": "Zoom", "notes": "AI companion, meeting intelligence"},
    "DOCN": {"name": "DigitalOcean", "notes": "Cloud for developers, AI/ML tools"},
    # Search & AI
    "RDDT": {"name": "Reddit", "notes": "AI training data, search partnerships"},
    "PINS": {"name": "Pinterest", "notes": "Visual AI, recommendation engine"},
}

# Robotics / Physical AI thesis
ROBOTICS_THESIS = {
    "TSLA": {"name": "Tesla", "notes": "Optimus humanoid robot, FSD, Dojo supercomputer"},
    "GOOGL": {"name": "Alphabet", "notes": "Waymo autonomous driving, robotics research"},
    "AMZN": {"name": "Amazon", "notes": "Warehouse robotics, Zoox, drone delivery"},
    "ISRG": {"name": "Intuitive Surgical", "notes": "Da Vinci surgical robots, market leader"},
    "FANUY": {"name": "Fanuc", "notes": "Industrial robotics, CNC systems"},
    "ABB": {"name": "ABB Ltd", "notes": "Industrial automation and robotics"},
    "ROK": {"name": "Rockwell Automation", "notes": "Industrial AI and automation"},
    "TER": {"name": "Teradyne", "notes": "Universal Robots, test equipment"},
    "IRBT": {"name": "iRobot", "notes": "Consumer robotics (Roomba)"},
    "DE": {"name": "Deere & Co", "notes": "Autonomous farming equipment, AgTech AI"},
    "HON": {"name": "Honeywell", "notes": "Industrial automation, quantum computing"},
    "EMR": {"name": "Emerson Electric", "notes": "Industrial automation software"},
}

# AI Healthcare - emerging category
AI_HEALTHCARE = {
    "ISRG": {"name": "Intuitive Surgical", "notes": "Surgical robotics, Da Vinci"},
    "VEEV": {"name": "Veeva Systems", "notes": "Life sciences cloud, AI for pharma"},
    "DXCM": {"name": "DexCom", "notes": "AI-powered glucose monitoring"},
    "ILMN": {"name": "Illumina", "notes": "Genomics, AI for drug discovery"},
    "TMO": {"name": "Thermo Fisher", "notes": "Lab equipment, AI analytics"},
    "DHR": {"name": "Danaher", "notes": "Life sciences tools, diagnostics AI"},
    "RXRX": {"name": "Recursion Pharma", "notes": "AI drug discovery platform"},
    "EXAI": {"name": "Exscientia", "notes": "AI-designed drug candidates"},
    "SDGR": {"name": "Schrodinger", "notes": "Computational drug discovery"},
}

# AI Edge/Consumer - new category
AI_EDGE_CONSUMER = {
    "AAPL": {"name": "Apple", "notes": "On-device AI, Apple Intelligence"},
    "QCOM": {"name": "Qualcomm", "notes": "Snapdragon AI, edge NPUs"},
    "AMD": {"name": "AMD", "notes": "Ryzen AI, edge compute"},
    "SONY": {"name": "Sony", "notes": "AI in gaming, sensors, entertainment"},
    "LOGI": {"name": "Logitech", "notes": "AI peripherals, collaboration devices"},
}

# AI-Adjacent: Crypto/Blockchain - exponential industry accelerated by AI
AI_ADJACENT_CRYPTO = {
    "COIN": {"name": "Coinbase", "notes": "Largest US crypto exchange, institutional onramp"},
    "MSTR": {"name": "MicroStrategy", "notes": "Bitcoin treasury company, leveraged BTC play"},
    "MARA": {"name": "Marathon Digital", "notes": "Bitcoin mining at scale, energy arbitrage"},
    "RIOT": {"name": "Riot Platforms", "notes": "Bitcoin mining, data center infrastructure"},
    "SQ": {"name": "Block (Square)", "notes": "Bitcoin payments, Cash App, TBD web5"},
    "HOOD": {"name": "Robinhood", "notes": "Retail crypto trading, AI-powered fintech"},
}

# AI-Adjacent: Biotech - AI-accelerated drug discovery and genomics
AI_ADJACENT_BIOTECH = {
    "MRNA": {"name": "Moderna", "notes": "mRNA platform, AI-designed vaccines and therapeutics"},
    "CRSP": {"name": "CRISPR Therapeutics", "notes": "Gene editing, AI-guided CRISPR targets"},
    "NTLA": {"name": "Intellia Therapeutics", "notes": "In vivo CRISPR gene editing"},
    "BEAM": {"name": "Beam Therapeutics", "notes": "Base editing, precision gene modification"},
    "RXRX": {"name": "Recursion Pharma", "notes": "AI drug discovery platform, massive bio data"},
    "EXAI": {"name": "Exscientia", "notes": "AI-designed drug candidates, clinical trials"},
    "SDGR": {"name": "Schrodinger", "notes": "Computational drug discovery, physics-based AI"},
    "TWST": {"name": "Twist Bioscience", "notes": "Synthetic biology, DNA data storage"},
    "DNLI": {"name": "Denali Therapeutics", "notes": "Neurodegeneration, AI-driven biomarkers"},
    "ABBV": {"name": "AbbVie", "notes": "AI pharma R&D, Humira successor pipeline"},
    "LLY": {"name": "Eli Lilly", "notes": "AI drug discovery, GLP-1 dominance, largest pharma"},
    "NVO": {"name": "Novo Nordisk", "notes": "GLP-1 leader (Ozempic/Wegovy), AI manufacturing"},
}

# AI-Adjacent: Space - AI-enabled space economy
AI_ADJACENT_SPACE = {
    "RKLB": {"name": "Rocket Lab", "notes": "Small launch vehicle, Neutron rocket, space systems"},
    "LUNR": {"name": "Intuitive Machines", "notes": "Lunar landers, NASA commercial lunar program"},
    "ASTS": {"name": "AST SpaceMobile", "notes": "Space-based cellular broadband, direct-to-phone"},
    "RDW": {"name": "Redwire", "notes": "Space manufacturing, 3D printing in orbit"},
    "MNTS": {"name": "Momentus", "notes": "In-space transportation, last-mile delivery"},
    "BA": {"name": "Boeing", "notes": "Space launch (SLS/Starliner), defense AI"},
    "LMT": {"name": "Lockheed Martin", "notes": "Space systems, AI defense, satellite tech"},
    "NOC": {"name": "Northrop Grumman", "notes": "Space systems, autonomous defense, JWST"},
    "IRDM": {"name": "Iridium", "notes": "Satellite communications, IoT connectivity"},
    "SPCE": {"name": "Virgin Galactic", "notes": "Space tourism, suborbital flights"},
}

# AI Supply Chain: Compute, Energy, Materials - the picks and shovels
AI_SUPPLY_CHAIN = {
    "VST": {"name": "Vistra", "notes": "Power generation for AI data centers, nuclear fleet"},
    "CEG": {"name": "Constellation Energy", "notes": "Nuclear power for hyperscaler data centers"},
    "NRG": {"name": "NRG Energy", "notes": "Power generation, data center energy supplier"},
    "NEE": {"name": "NextEra Energy", "notes": "Largest renewable energy, wind/solar for data centers"},
    "SMR": {"name": "NuScale Power", "notes": "Small modular nuclear reactors for AI data centers"},
    "OKLO": {"name": "Oklo", "notes": "Advanced fission power, Sam Altman backed, data center power"},
    "FSLR": {"name": "First Solar", "notes": "US solar manufacturing, data center renewable power"},
    "UEC": {"name": "Uranium Energy Corp", "notes": "Uranium mining, nuclear fuel for AI power demand"},
    "CCJ": {"name": "Cameco", "notes": "Uranium production, nuclear fuel supply chain"},
    "EQIX": {"name": "Equinix", "notes": "Largest data center REIT, AI colocation"},
    "DLR": {"name": "Digital Realty", "notes": "AI-ready data centers, hyperscaler partnerships"},
    "ALB": {"name": "Albemarle", "notes": "Lithium production, battery materials for AI/EV"},
    "MP": {"name": "MP Materials", "notes": "Rare earth mining, magnets for EVs and robotics"},
    "FCX": {"name": "Freeport-McMoRan", "notes": "Copper mining, essential for electrification"},
}

# All AI Universe tickers (for scanning)
AI_UNIVERSE = set(
    list(HYPERSCALERS.keys()) +
    list(NEOCLOUDS.keys()) +
    list(AI_INFRASTRUCTURE.keys()) +
    list(AI_SOFTWARE.keys()) +
    list(ROBOTICS_THESIS.keys()) +
    list(AI_HEALTHCARE.keys()) +
    list(AI_EDGE_CONSUMER.keys()) +
    list(AI_ADJACENT_CRYPTO.keys()) +
    list(AI_ADJACENT_BIOTECH.keys()) +
    list(AI_ADJACENT_SPACE.keys()) +
    list(AI_SUPPLY_CHAIN.keys())
)

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
        max_tokens: int = 1024,
        ticker_performance: dict[str, dict] | None = None
    ) -> dict[str, Any]:
        """
        Research an investment thesis using Claude.

        Args:
            thesis_name: Name of the thesis (e.g., "Tesla Robot Thesis")
            context: Current context about the thesis
            tickers: Related tickers to research
            max_tokens: Maximum response length
            ticker_performance: Dict of ticker -> {price, pct_30d, pct_90d, rsi, signal}

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

            # Build performance data section
            perf_section = ""
            if ticker_performance:
                perf_lines = []
                for ticker, data in ticker_performance.items():
                    perf_lines.append(
                        f"  {ticker}: ${data['price']:.2f} | "
                        f"30d: {data['pct_30d']:+.1f}% | "
                        f"90d: {data['pct_90d']:+.1f}% | "
                        f"RSI: {data['rsi']:.0f} ({data['signal']})"
                    )
                perf_section = "\n\nACTUAL PRICE PERFORMANCE (CRITICAL DATA):\n" + "\n".join(perf_lines)
                perf_section += "\n\nIMPORTANT: Base your analysis on this ACTUAL price data. If stocks are down 10%+, acknowledge the pullback. If RSI is oversold (<30), note the opportunity. Be objective about price action."

            prompt = f"""You are a SKEPTICAL AI and technology investment analyst who prioritizes capital preservation.
Analyze the following investment thesis based on ACTUAL PRICE DATA.

THESIS: {thesis_name}

CONTEXT: {context}

RELATED TICKERS: {', '.join(tickers)}
{perf_section}

TODAY'S DATE: {date.today().isoformat()}

CRITICAL RULES:
- If ANY stock is down >10% in 30 days, you MUST acknowledge this is concerning and affects the thesis
- If multiple stocks are down >15% in 30 days, the thesis is likely BEARISH or NEUTRAL, not bullish
- Pullbacks are only opportunities if fundamentals are intact - don't assume they are
- Extended rallies (+20% or more) are RISKS, not positives

Provide analysis with:
1. PRICE REALITY CHECK: Summarize the ACTUAL 30d and 90d performance. Are stocks mostly up or down? By how much?
2. THESIS STATUS: Given the price action, is the market validating or rejecting this thesis?
3. HONEST ASSESSMENT: What is the price action telling us? Significant drawdowns suggest caution.
4. ENTRY POINTS: Only suggest entries for stocks that are down AND have strong fundamentals. Acknowledge risk.
5. RISKS: Why might these stocks be selling off? Is the market seeing something we're not?
6. RECOMMENDATION:
   - BEARISH if most stocks down >10% in 30d or thesis appears to be failing
   - NEUTRAL if mixed signals or significant uncertainty
   - BULLISH only if stocks are UP or minor pullback with clear bullish catalysts
7. TOP PICK: Best risk/reward given ACTUAL price levels and risk

BE BRUTALLY HONEST. If stocks are getting crushed, say so. The user loses money from false optimism.
Keep response under 500 words."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = message.content[0].text

            # Parse recommendation from response - look at the explicit recommendation section
            recommendation = "neutral"
            analysis_lower = analysis.lower()

            # Look for explicit recommendation markers
            if "recommendation: bullish" in analysis_lower or "recommendation:\nbullish" in analysis_lower:
                recommendation = "bullish"
            elif "recommendation: bearish" in analysis_lower or "recommendation:\nbearish" in analysis_lower:
                recommendation = "bearish"
            elif "recommendation: neutral" in analysis_lower or "recommendation:\nneutral" in analysis_lower:
                recommendation = "neutral"
            else:
                # Count bullish vs bearish keywords for fallback
                bullish_words = ["bullish", "opportunity", "upside", "buy", "undervalued", "attractive entry"]
                bearish_words = ["bearish", "downside", "sell", "overvalued", "caution", "risk", "selloff", "decline", "concerning"]

                bullish_count = sum(1 for word in bullish_words if word in analysis_lower)
                bearish_count = sum(1 for word in bearish_words if word in analysis_lower)

                if bearish_count > bullish_count + 2:
                    recommendation = "bearish"
                elif bullish_count > bearish_count + 2:
                    recommendation = "bullish"

            # Calculate confidence based on price performance too
            confidence = 50
            if ticker_performance:
                avg_30d = sum(d['pct_30d'] for d in ticker_performance.values()) / len(ticker_performance)
                # If stocks are down big, lower confidence in bullish calls
                if recommendation == "bullish" and avg_30d < -10:
                    confidence = 40  # Low confidence - bullish call but stocks are down
                elif recommendation == "bearish" and avg_30d < -10:
                    confidence = 75  # High confidence - bearish matches price action
                elif recommendation == "bullish" and avg_30d > 5:
                    confidence = 70  # Bullish matches price action
                else:
                    confidence = 55

            return {
                "thesis": thesis_name,
                "analysis": analysis,
                "signals": [],  # Could parse signals from response
                "recommendation": recommendation,
                "confidence": confidence,
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

            prompt = f"""You are a SKEPTICAL AI market analyst who prioritizes capital preservation.
Generate a concise, HONEST market pulse for AI/tech investors.

MARKET DATA:
{json.dumps(market_data, indent=2, default=str)}

TODAY'S DATE: {date.today().isoformat()}

CRITICAL: Look at the avg_30d_performance and worst_30d_performers in the data.
- If avg_30d_performance is negative, the AI sector is SELLING OFF - acknowledge this
- If many stocks are down >10%, this is concerning - don't sugarcoat it
- Pullbacks are only opportunities if there's clear reason to believe fundamentals are intact

Provide a brief (3-5 bullet points) market pulse covering:
- AI sector sentiment (BE HONEST about weakness if present)
- Notable moves in AI stocks (especially BIG losers)
- Risk factors to watch
- Actionable guidance (sometimes "stay cautious" is the right call)

DO NOT be blindly bullish. If the data shows a selloff, call it out.
Keep it punchy and honest. Under 200 words."""

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

        # AI stocks daily cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_stocks_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                company_name TEXT,
                current_price REAL,
                market_cap REAL,
                pct_30d REAL,
                pct_90d REAL,
                ai_score REAL,
                score_breakdown TEXT,
                category TEXT,
                subcategory TEXT,
                sector TEXT,
                created_at TEXT DEFAULT (datetime('now')),
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
            {
                "name": "AI Proliferation Direct",
                "description": "Invest directly in the companies building and deploying AI. "
                              "The imminent takeoff thesis: AI capabilities are accelerating "
                              "faster than markets price in. These are the primary beneficiaries.",
                "tickers": "TSLA,NVDA,PLTR,GOOGL,MSFT,META",
            },
            {
                "name": "AI-Adjacent Exponentials",
                "description": "AI accelerates adjacent exponential industries: robotics, "
                              "biotech/genomics, crypto/blockchain, and space. These sectors "
                              "compound with AI progress - genomics gets AI-designed drugs, "
                              "crypto gets AI agents, space gets autonomous systems.",
                "tickers": "TSLA,MRNA,CRSP,COIN,RKLB,LLY",
            },
            {
                "name": "AI Supply Chain (Picks & Shovels)",
                "description": "The AI buildout requires massive compute, energy, and materials. "
                              "Nuclear/power for data centers, uranium for reactors, copper for "
                              "electrification, rare earths for robotics. Infrastructure always "
                              "gets paid regardless of which AI company wins.",
                "tickers": "CEG,OKLO,SMR,CCJ,FCX,MP,NEE",
            },
            {
                "name": "Liquidity Over Lockup",
                "description": "In an AI-accelerated world, capital locked in 30-40 year retirement "
                              "vehicles (401k, Roth IRA) carries opportunity cost risk. These vehicles "
                              "assume a predictable world with long time horizons. Prefer liquid, "
                              "accessible positions in high-conviction AI plays over tax-deferred "
                              "lockups that may be irrelevant when they mature. This thesis tracks "
                              "liquid AI positions vs broad market index benchmarks.",
                "tickers": "NVDA,TSLA,GOOGL,COIN,PLTR,MSFT",
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

        # Ensure we have enough historical data for trend analysis
        self._ensure_trillion_history()

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
            return "AI Software", AI_SOFTWARE[ticker].get("notes", "Platform")

        if ticker in ROBOTICS_THESIS:
            return "Robotics/Physical AI", ROBOTICS_THESIS[ticker].get("notes", "")

        if ticker in AI_HEALTHCARE:
            return "AI Healthcare", AI_HEALTHCARE[ticker].get("notes", "Healthcare")

        if ticker in AI_EDGE_CONSUMER:
            return "AI Edge/Consumer", AI_EDGE_CONSUMER[ticker].get("notes", "Edge AI")

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

    def _calculate_ai_score(
        self,
        ticker: str,
        return_breakdown: bool = False
    ) -> float | tuple[float, dict]:
        """
        Calculate AI Opportunity Score for a stock.

        This score measures how attractive a stock is as an AI investment based on:
        - Price performance (30d, 90d momentum)
        - AI category positioning (infrastructure vs software)
        - Technical setup (RSI, MA position)
        - Valuation relative to growth
        - Market cap tier

        Args:
            ticker: Stock ticker
            return_breakdown: If True, return (score, breakdown_dict)

        Returns:
            Score (0-100), or tuple of (score, breakdown) if return_breakdown=True
        """
        try:
            yf_ticker = self._get_yf_ticker(ticker)
            info = yf_ticker.info
            hist = yf_ticker.history(period="6mo")

            if hist.empty or not info:
                return (0, {}) if return_breakdown else 0

            current_price = info.get("regularMarketPrice") or info.get("currentPrice") or hist["Close"].iloc[-1]
        except Exception as e:
            logger.debug(f"Error getting data for {ticker}: {e}")
            return (0, {}) if return_breakdown else 0

        score = 50  # Base score
        breakdown = {
            "base": {"points": 50, "label": "Base Score", "raw_value": "-"},
        }

        # === PRICE PERFORMANCE (most critical) ===

        # 30-day performance
        pct_30d = 0.0
        perf_30d_pts = 0
        if len(hist) >= 22:
            pct_30d = (current_price / hist["Close"].iloc[-22] - 1) * 100
            # Pullbacks are opportunities, extended runs are risky
            if pct_30d <= -20:
                perf_30d_pts = 20  # Deep pullback - strong opportunity
            elif pct_30d <= -10:
                perf_30d_pts = 15  # Meaningful pullback
            elif pct_30d <= -5:
                perf_30d_pts = 10  # Healthy consolidation
            elif pct_30d <= 0:
                perf_30d_pts = 5   # Flat/slight dip
            elif pct_30d >= 30:
                perf_30d_pts = -15  # Way overextended
            elif pct_30d >= 20:
                perf_30d_pts = -10  # Overextended
            elif pct_30d >= 10:
                perf_30d_pts = -5   # Running hot
        score += perf_30d_pts
        breakdown["perf_30d"] = {
            "points": perf_30d_pts,
            "label": "30-Day Performance",
            "raw_value": f"{pct_30d:+.1f}%",
        }

        # 90-day performance (longer-term trend)
        pct_90d = 0.0
        perf_90d_pts = 0
        if len(hist) >= 63:
            pct_90d = (current_price / hist["Close"].iloc[-63] - 1) * 100
            # Similar logic but less weight
            if pct_90d <= -30:
                perf_90d_pts = 15  # Major pullback from 90d ago
            elif pct_90d <= -15:
                perf_90d_pts = 10
            elif pct_90d <= 0:
                perf_90d_pts = 5
            elif pct_90d >= 50:
                perf_90d_pts = -10  # Massively extended
            elif pct_90d >= 30:
                perf_90d_pts = -5
        score += perf_90d_pts
        breakdown["perf_90d"] = {
            "points": perf_90d_pts,
            "label": "90-Day Performance",
            "raw_value": f"{pct_90d:+.1f}%",
        }

        # === AI CATEGORY POSITIONING ===

        category_pts = 0
        category_name = "Other"
        if ticker in AI_INFRASTRUCTURE:
            category_pts = 10  # AI infrastructure is core thesis
            category_name = "AI Infrastructure"
        elif ticker in HYPERSCALERS:
            category_pts = 8  # Hyperscalers = proven AI winners
            category_name = "Hyperscaler"
        elif ticker in AI_SOFTWARE:
            category_pts = 7  # AI software is growth area
            category_name = "AI Software"
        elif ticker in AI_SUPPLY_CHAIN:
            category_pts = 7  # Supply chain (energy, materials) = essential infrastructure
            category_name = "AI Supply Chain"
        elif ticker in ROBOTICS_THESIS:
            category_pts = 6  # Robotics is emerging
            category_name = "Robotics/Physical AI"
        elif ticker in AI_EDGE_CONSUMER:
            category_pts = 6  # Edge AI is growing
            category_name = "AI Edge/Consumer"
        elif ticker in AI_ADJACENT_BIOTECH:
            category_pts = 6  # AI-accelerated biotech
            category_name = "AI-Adjacent: Biotech"
        elif ticker in AI_ADJACENT_SPACE:
            category_pts = 5  # AI-enabled space economy
            category_name = "AI-Adjacent: Space"
        elif ticker in AI_ADJACENT_CRYPTO:
            category_pts = 5  # AI + crypto convergence
            category_name = "AI-Adjacent: Crypto"
        elif ticker in AI_HEALTHCARE:
            category_pts = 5
            category_name = "AI Healthcare"
        elif ticker in NEOCLOUDS:
            category_pts = 5  # Neoclouds are speculative
            category_name = "Neocloud"
        score += category_pts
        breakdown["ai_category"] = {
            "points": category_pts,
            "label": "AI Category",
            "raw_value": category_name,
        }

        # === TECHNICAL FACTORS ===

        # RSI
        current_rsi = 50.0
        rsi_pts = 0
        if len(hist) >= 14:
            delta = hist["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50

            if current_rsi < 30:
                rsi_pts = 15  # Deeply oversold
            elif current_rsi < 40:
                rsi_pts = 10  # Oversold
            elif current_rsi < 50:
                rsi_pts = 5   # Below average
            elif current_rsi > 80:
                rsi_pts = -10  # Overbought
            elif current_rsi > 70:
                rsi_pts = -5   # Getting hot
        score += rsi_pts
        breakdown["rsi"] = {
            "points": rsi_pts,
            "label": "RSI (14)",
            "raw_value": f"{current_rsi:.1f}",
        }

        # 50-day MA position
        ma_50_pts = 0
        ma_50_pct = 0.0
        if len(hist) >= 50:
            ma_50 = hist["Close"].rolling(50).mean().iloc[-1]
            ma_50_pct = ((current_price / ma_50) - 1) * 100 if ma_50 > 0 else 0
            if ma_50_pct <= -10:
                ma_50_pts = 10  # Well below MA - potential support
            elif ma_50_pct <= -5:
                ma_50_pts = 7
            elif ma_50_pct <= 0:
                ma_50_pts = 5  # At or below MA
            elif ma_50_pct >= 15:
                ma_50_pts = -5  # Extended above MA
        score += ma_50_pts
        breakdown["ma_50"] = {
            "points": ma_50_pts,
            "label": "50-Day MA Position",
            "raw_value": f"{ma_50_pct:+.1f}% vs MA",
        }

        # === VALUATION ===

        pe_ratio = info.get("trailingPE", info.get("forwardPE", 0)) or 0
        peg = info.get("pegRatio", 0) or 0
        val_pts = 0

        if peg and 0 < peg < 1:
            val_pts = 10  # PEG < 1 = growth at reasonable price
        elif peg and 1 <= peg < 1.5:
            val_pts = 5
        elif pe_ratio:
            if pe_ratio < 20:
                val_pts = 8
            elif pe_ratio < 30:
                val_pts = 4
            elif pe_ratio > 80:
                val_pts = -10
            elif pe_ratio > 50:
                val_pts = -5
        score += val_pts
        breakdown["valuation"] = {
            "points": val_pts,
            "label": "Valuation (PEG/PE)",
            "raw_value": f"PEG: {peg:.2f}" if peg else f"P/E: {pe_ratio:.1f}" if pe_ratio else "N/A",
        }

        # === MARKET CAP TIER ===

        market_cap = info.get("marketCap", 0)
        cap_pts = 0
        cap_label = "Unknown"
        if market_cap >= 1_000_000_000_000:  # $1T+
            cap_pts = 5  # Mega-cap stability
            cap_label = "Mega Cap ($1T+)"
        elif market_cap >= 200_000_000_000:  # $200B+
            cap_pts = 4
            cap_label = "Large Cap ($200B+)"
        elif market_cap >= 50_000_000_000:  # $50B+
            cap_pts = 3
            cap_label = "Mid-Large ($50B+)"
        elif market_cap >= 10_000_000_000:  # $10B+
            cap_pts = 2
            cap_label = "Mid Cap ($10B+)"
        else:
            cap_pts = 0
            cap_label = "Small Cap"
        score += cap_pts
        breakdown["market_cap"] = {
            "points": cap_pts,
            "label": "Market Cap Tier",
            "raw_value": cap_label,
        }

        final_score = max(0, min(100, score))

        if return_breakdown:
            breakdown["total"] = {
                "points": final_score,
                "label": "TOTAL AI SCORE",
                "raw_value": "-",
            }
            return final_score, breakdown

        return final_score

    def get_ai_stocks(self) -> list[dict]:
        """
        Scan all AI universe stocks and calculate AI opportunity scores.

        Returns:
            List of AI stocks with scores, sorted by score descending
        """
        logger.info(f"Scanning {len(AI_UNIVERSE)} AI universe stocks...")

        stocks = []
        for ticker in AI_UNIVERSE:
            try:
                yf_ticker = self._get_yf_ticker(ticker)
                info = yf_ticker.info
                hist = yf_ticker.history(period="6mo")

                if hist.empty or not info:
                    continue

                current_price = info.get("regularMarketPrice") or info.get("currentPrice")
                if not current_price:
                    continue

                # Calculate AI score with breakdown
                ai_score, score_breakdown = self._calculate_ai_score(ticker, return_breakdown=True)

                # Determine category
                category, subcategory = self._categorize_stock(ticker)

                # Calculate performance metrics
                pct_30d = (current_price / hist["Close"].iloc[-22] - 1) * 100 if len(hist) >= 22 else 0
                pct_90d = (current_price / hist["Close"].iloc[-63] - 1) * 100 if len(hist) >= 63 else 0

                # Get market cap
                market_cap = info.get("marketCap", 0)

                stock = {
                    "ticker": ticker,
                    "company_name": info.get("shortName", info.get("longName", ticker)),
                    "current_price": current_price,
                    "market_cap": market_cap,
                    "market_cap_b": market_cap / 1_000_000_000 if market_cap else 0,
                    "pct_30d": pct_30d,
                    "pct_90d": pct_90d,
                    "ai_score": ai_score,
                    "score_breakdown": score_breakdown,
                    "category": category,
                    "subcategory": subcategory,
                    "sector": info.get("sector", "Unknown"),
                    "scan_date": date.today(),
                }
                stocks.append(stock)

                logger.debug(f"{ticker}: AI Score {ai_score:.0f}, 30d: {pct_30d:+.1f}%, 90d: {pct_90d:+.1f}%")

            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                continue

        # Sort by AI score (best opportunities first)
        stocks.sort(key=lambda x: x["ai_score"], reverse=True)

        logger.info(f"Scanned {len(stocks)} AI stocks")
        return stocks

    def save_ai_stocks(self, stocks: list[dict]) -> int:
        """
        Save AI stocks scan results to database.

        Args:
            stocks: List of AI stock dicts from get_ai_stocks()

        Returns:
            Number of stocks saved
        """
        saved = 0
        scan_date = date.today().isoformat()

        for stock in stocks:
            try:
                # Convert score_breakdown to JSON string
                breakdown_json = json.dumps(stock.get("score_breakdown", {}))

                self.db.execute("""
                    INSERT INTO ai_stocks_daily (
                        ticker, scan_date, company_name, current_price, market_cap,
                        pct_30d, pct_90d, ai_score, score_breakdown, category, subcategory, sector
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, scan_date) DO UPDATE SET
                        company_name = excluded.company_name,
                        current_price = excluded.current_price,
                        market_cap = excluded.market_cap,
                        pct_30d = excluded.pct_30d,
                        pct_90d = excluded.pct_90d,
                        ai_score = excluded.ai_score,
                        score_breakdown = excluded.score_breakdown,
                        category = excluded.category,
                        subcategory = excluded.subcategory,
                        sector = excluded.sector,
                        created_at = datetime('now')
                """, (
                    stock["ticker"],
                    scan_date,
                    stock.get("company_name", ""),
                    stock.get("current_price", 0),
                    stock.get("market_cap", 0),
                    stock.get("pct_30d", 0),
                    stock.get("pct_90d", 0),
                    stock.get("ai_score", 0),
                    breakdown_json,
                    stock.get("category", ""),
                    stock.get("subcategory", ""),
                    stock.get("sector", ""),
                ))
                saved += 1
            except Exception as e:
                logger.debug(f"Error saving AI stock {stock.get('ticker')}: {e}")

        logger.info(f"Saved {saved} AI stocks to database")
        return saved

    def get_cached_ai_stocks(self, max_age_hours: int = 24) -> tuple[list[dict], str | None]:
        """
        Get cached AI stocks from database.

        Args:
            max_age_hours: Maximum age of cached data to return

        Returns:
            Tuple of (list of AI stocks, scan timestamp or None if no data)
        """
        try:
            # Get the most recent scan
            row = self.db.fetchone("""
                SELECT scan_date, created_at FROM ai_stocks_daily
                ORDER BY created_at DESC
                LIMIT 1
            """)

            if not row:
                return [], None

            scan_date, created_at = row

            # Check if data is fresh enough
            from datetime import datetime, timedelta
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else None
            if created_dt:
                age = datetime.now() - created_dt.replace(tzinfo=None)
                if age > timedelta(hours=max_age_hours):
                    return [], None

            # Get all stocks from the most recent scan
            df = self.db.fetchdf("""
                SELECT * FROM ai_stocks_daily
                WHERE scan_date = ?
                ORDER BY ai_score DESC
            """, (scan_date,))

            if df.empty:
                return [], None

            stocks = []
            for _, row in df.iterrows():
                stock = {
                    "ticker": row["ticker"],
                    "company_name": row.get("company_name", ""),
                    "current_price": row.get("current_price", 0),
                    "market_cap": row.get("market_cap", 0),
                    "market_cap_b": row.get("market_cap", 0) / 1_000_000_000 if row.get("market_cap") else 0,
                    "pct_30d": row.get("pct_30d", 0),
                    "pct_90d": row.get("pct_90d", 0),
                    "ai_score": row.get("ai_score", 0),
                    "category": row.get("category", ""),
                    "subcategory": row.get("subcategory", ""),
                    "sector": row.get("sector", ""),
                }

                # Parse score breakdown
                try:
                    if row.get("score_breakdown"):
                        stock["score_breakdown"] = json.loads(row["score_breakdown"])
                except Exception:
                    stock["score_breakdown"] = {}

                stocks.append(stock)

            return stocks, created_at

        except Exception as e:
            logger.error(f"Error getting cached AI stocks: {e}")
            return [], None

    def _get_ticker_performance(self, tickers: list[str]) -> dict[str, dict]:
        """
        Get price performance data for a list of tickers.

        Returns dict with 30d, 90d performance and current RSI for each ticker.
        """
        results = {}
        for ticker in tickers:
            try:
                yf_ticker = self._get_yf_ticker(ticker)
                hist = yf_ticker.history(period="6mo")
                info = yf_ticker.info

                if hist.empty:
                    continue

                current_price = info.get("regularMarketPrice") or info.get("currentPrice") or hist["Close"].iloc[-1]

                pct_30d = (current_price / hist["Close"].iloc[-22] - 1) * 100 if len(hist) >= 22 else 0
                pct_90d = (current_price / hist["Close"].iloc[-63] - 1) * 100 if len(hist) >= 63 else 0

                # RSI
                current_rsi = 50.0
                if len(hist) >= 14:
                    delta = hist["Close"].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50

                results[ticker] = {
                    "price": current_price,
                    "pct_30d": pct_30d,
                    "pct_90d": pct_90d,
                    "rsi": current_rsi,
                    "signal": "OVERSOLD" if current_rsi < 30 else ("OVERBOUGHT" if current_rsi > 70 else "NEUTRAL"),
                }
            except Exception as e:
                logger.debug(f"Error getting performance for {ticker}: {e}")
                continue

        return results

    def run_scan(self) -> dict[str, Any]:
        """
        Run full AI Pulse scan.

        Returns:
            Dict containing all scan results:
            - ai_stocks: List of AI universe stocks with scores
            - categories: Breakdown by category
            - theses: Thesis research updates
            - market_pulse: AI-generated market summary
            - top_picks: Best AI opportunities with breakdowns
        """
        logger.info("Running AI Pulse scan...")

        # 1. Scan all AI universe stocks
        ai_stocks = self.get_ai_stocks()

        # 1.5 Save to database for dashboard caching
        self.save_ai_stocks(ai_stocks)

        # 2. Categorize by AI category
        categories = {
            "Hyperscaler": [],
            "AI Infrastructure": [],
            "AI Software": [],
            "AI Supply Chain": [],
            "Robotics/Physical AI": [],
            "AI Edge/Consumer": [],
            "AI-Adjacent: Biotech": [],
            "AI-Adjacent: Space": [],
            "AI-Adjacent: Crypto": [],
            "AI Healthcare": [],
            "Neocloud": [],
            "Other": [],
        }
        for stock in ai_stocks:
            cat = stock.get("category", "Other")
            if cat in categories:
                categories[cat].append(stock)
            else:
                categories["Other"].append(stock)

        # 3. Research active theses (now with real price data)
        theses_results = self._research_theses()

        # 4. Generate market pulse with AI stock data
        top_picks = [s for s in ai_stocks if s["ai_score"] >= 70][:10]
        worst_performers = sorted(ai_stocks, key=lambda x: x["pct_30d"])[:5]

        market_data = {
            "ai_stocks_scanned": len(ai_stocks),
            "top_ai_opportunities": [
                {
                    "ticker": s["ticker"],
                    "score": s["ai_score"],
                    "pct_30d": s["pct_30d"],
                    "pct_90d": s["pct_90d"],
                    "category": s["category"]
                }
                for s in ai_stocks[:10]
            ],
            "category_summary": {cat: len(stocks) for cat, stocks in categories.items()},
            "worst_30d_performers": [
                {"ticker": s["ticker"], "pct_30d": s["pct_30d"]}
                for s in worst_performers
            ],
            "avg_30d_performance": sum(s["pct_30d"] for s in ai_stocks) / len(ai_stocks) if ai_stocks else 0,
        }
        market_pulse = self.claude.generate_market_pulse(market_data)

        # 5. Calculate category scores (avg AI score per category)
        category_scores = {}
        for cat, stocks in categories.items():
            if stocks:
                avg_score = sum(s["ai_score"] for s in stocks) / len(stocks)
                avg_30d = sum(s["pct_30d"] for s in stocks) / len(stocks)
                category_scores[cat] = {
                    "avg_score": avg_score,
                    "avg_30d": avg_30d,
                    "count": len(stocks),
                    "top_pick": stocks[0]["ticker"] if stocks else None,
                }

        results = {
            "scan_date": date.today().isoformat(),
            "ai_stocks": ai_stocks,
            "ai_stocks_count": len(ai_stocks),
            "categories": categories,
            "category_scores": category_scores,
            "theses": theses_results,
            "market_pulse": market_pulse,
            "top_picks": top_picks,
            "worst_performers": worst_performers,
        }

        logger.info(f"AI Pulse scan complete: {len(ai_stocks)} AI stocks scanned")
        return results

    def _research_theses(self) -> list[dict]:
        """Research all active theses with real price performance data."""
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

            # Get actual price performance for thesis tickers
            ticker_performance = self._get_ticker_performance(tickers)

            # Only research if Claude is configured
            if self.claude.is_configured:
                research = self.claude.research_thesis(
                    thesis_name=thesis_name,
                    context=description,
                    tickers=tickers,
                    ticker_performance=ticker_performance  # Pass performance data
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

    def _ensure_trillion_history(self, min_days: int = 5) -> None:
        """
        Check if trillion_club has enough historical data for trend analysis.

        If not enough history exists OR there are gaps in recent data, runs backfill.

        Args:
            min_days: Minimum number of unique scan dates required (default 5)
        """
        # Check how many unique dates we have in trillion_club (last 14 days)
        result = self.db.fetchone("""
            SELECT COUNT(DISTINCT scan_date) FROM trillion_club
            WHERE scan_date >= date('now', '-14 days')
        """)
        tc_dates_14d = result[0] if result and result[0] else 0

        # Check how many trading days exist in prices_daily (last 14 days)
        result = self.db.fetchone("""
            SELECT COUNT(DISTINCT date) FROM prices_daily
            WHERE date >= date('now', '-14 days')
        """)
        price_dates_14d = result[0] if result and result[0] else 0

        # If trillion_club has fewer dates than prices, we have gaps to fill
        needs_backfill = False
        missing_days = price_dates_14d - tc_dates_14d

        if missing_days >= 2:
            needs_backfill = True
            logger.info(
                f"Trillion Club missing {missing_days} days in last 14 days "
                f"(trillion_club: {tc_dates_14d}, prices: {price_dates_14d}). Running backfill..."
            )

        if tc_dates_14d < min_days:
            needs_backfill = True
            logger.info(
                f"Trillion Club history insufficient ({tc_dates_14d} days in last 14d, need {min_days}). "
                f"Running backfill..."
            )

        if needs_backfill:
            try:
                # Backfill 3 weeks to cover gaps
                records = self.backfill_trillion_history(days=21)
                logger.info(f"Trillion Club backfill complete: {records} records created")
            except Exception as e:
                logger.error(f"Failed to backfill trillion club history: {e}")
        else:
            logger.info(f"Trillion Club history sufficient ({tc_dates_14d} days in last 14d, no gaps)")

    def backfill_trillion_history(self, days: int = 21) -> int:
        """
        Backfill historical trillion club data by calculating scores on past dates.

        Uses stored price data to calculate historical entry scores, enabling
        trend detection and consecutive day tracking.

        Args:
            days: Number of days to backfill (default 21 = 3 weeks)

        Returns:
            Number of records created
        """
        logger.info(f"Backfilling {days} days of Trillion+ Club history...")

        records_created = 0
        today = date.today()

        # Get all trading days from prices_daily in the backfill range
        trading_days = self.db.fetchdf("""
            SELECT DISTINCT date FROM prices_daily
            WHERE date >= date('now', ?) AND date < date('now')
            ORDER BY date
        """, (f'-{days} days',))

        if trading_days.empty:
            logger.warning("No price data available for backfill")
            return 0

        trading_dates = trading_days["date"].tolist()
        logger.info(f"Processing {len(trading_dates)} trading days...")

        # Get trillion club candidates
        candidates = set(TRILLION_CLUB_SEED)

        # Add any previously tracked
        existing = self.db.fetchdf("""
            SELECT DISTINCT ticker FROM trillion_club
        """)
        if not existing.empty:
            candidates.update(existing["ticker"].tolist())

        for i, scan_date_raw in enumerate(trading_dates):
            # Parse date
            try:
                if hasattr(scan_date_raw, 'strftime'):
                    scan_date = scan_date_raw if isinstance(scan_date_raw, date) else scan_date_raw.date()
                else:
                    scan_date = datetime.strptime(str(scan_date_raw)[:10], "%Y-%m-%d").date()
            except Exception as e:
                logger.debug(f"Date parsing error: {e}")
                continue

            # Skip if we already have data for this date
            existing_check = self.db.fetchone("""
                SELECT COUNT(*) FROM trillion_club WHERE scan_date = ?
            """, (scan_date.isoformat(),))
            if existing_check and existing_check[0] > 0:
                continue

            # Calculate scores for each candidate as of this date
            day_records = 0
            for ticker in candidates:
                try:
                    score_data = self._calculate_historical_entry_score(ticker, scan_date)
                    if score_data:
                        # Store the result
                        self.db.execute("""
                            INSERT INTO trillion_club (
                                ticker, scan_date, market_cap, market_cap_category,
                                peak_market_cap_30d, current_price, price_vs_30d_high_pct,
                                entry_score, category, subcategory, reasoning
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (ticker, scan_date) DO UPDATE SET
                                entry_score = excluded.entry_score,
                                current_price = excluded.current_price,
                                price_vs_30d_high_pct = excluded.price_vs_30d_high_pct
                        """, (
                            ticker,
                            scan_date.isoformat(),
                            score_data.get("market_cap", 0),
                            score_data.get("market_cap_category", "Mega Cap"),
                            score_data.get("peak_market_cap_30d", 0),
                            score_data.get("current_price", 0),
                            score_data.get("price_vs_30d_high_pct", 0),
                            score_data.get("entry_score", 50),
                            score_data.get("category", "Unknown"),
                            score_data.get("subcategory", ""),
                            f"Historical backfill for {scan_date.isoformat()}",
                        ))
                        day_records += 1
                        records_created += 1
                except Exception as e:
                    logger.debug(f"Error scoring {ticker} for {scan_date}: {e}")
                    continue

            if (i + 1) % 5 == 0:
                logger.info(f"  Processed {i + 1}/{len(trading_dates)} days, {records_created} records")

        logger.info(f"Trillion Club backfill complete: {records_created} records created")
        return records_created

    def _calculate_historical_entry_score(self, ticker: str, as_of_date: date) -> dict | None:
        """
        Calculate entry score for a ticker as of a historical date.

        Uses stored price data to simulate what the entry score would have been
        on that date.

        Args:
            ticker: Stock ticker
            as_of_date: Date to calculate score for

        Returns:
            Dict with score data, or None if insufficient data
        """
        # Get price history up to and including as_of_date
        price_data = self.db.fetchdf("""
            SELECT date, open, high, low, close, volume
            FROM prices_daily
            WHERE ticker = ? AND date <= ? AND date >= date(?, '-60 days')
            ORDER BY date
        """, (ticker, as_of_date.isoformat(), as_of_date.isoformat()))

        if price_data.empty or len(price_data) < 20:
            return None

        # Get the closing price on as_of_date
        current_price = price_data["close"].iloc[-1]

        # Calculate 30-day high
        recent_30d = price_data.tail(22) if len(price_data) >= 22 else price_data
        high_30d = recent_30d["high"].max()

        # Calculate price vs 30d high
        pct_from_high = ((current_price / high_30d) - 1) * 100 if high_30d > 0 else 0

        # Calculate RSI
        close_prices = price_data["close"]
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50

        # Calculate 50-day MA
        ma_50 = close_prices.rolling(min(50, len(close_prices))).mean().iloc[-1]
        ma_50_pct = ((current_price / ma_50) - 1) * 100 if ma_50 > 0 else 0

        # Calculate 20-day momentum
        pct_20d = 0.0
        if len(close_prices) >= 20:
            pct_20d = (current_price / close_prices.iloc[-20] - 1) * 100

        # Calculate entry score (simplified version of _calculate_entry_score)
        score = 50  # Base score

        # Distance from 30d high
        if pct_from_high <= -15:
            score += 20
        elif pct_from_high <= -10:
            score += 15
        elif pct_from_high <= -5:
            score += 10
        elif pct_from_high <= -2:
            score += 5
        elif pct_from_high >= 0:
            score -= 5

        # RSI
        if current_rsi < 30:
            score += 15
        elif current_rsi < 40:
            score += 10
        elif current_rsi > 80:
            score -= 15
        elif current_rsi > 70:
            score -= 10

        # 50-day MA position
        if ma_50_pct <= -5:
            score += 10
        elif ma_50_pct <= 0:
            score += 5

        # 20-day momentum
        if -10 <= pct_20d <= 0:
            score += 5
        elif pct_20d < -15:
            score += 10
        elif pct_20d > 20:
            score -= 10

        # Get category
        category, subcategory = self._categorize_stock(ticker)

        return {
            "ticker": ticker,
            "entry_score": max(0, min(100, score)),
            "current_price": current_price,
            "price_vs_30d_high_pct": pct_from_high,
            "market_cap": 0,  # Not available historically without API call
            "market_cap_category": "Mega Cap",
            "peak_market_cap_30d": 0,
            "category": category,
            "subcategory": subcategory,
        }
