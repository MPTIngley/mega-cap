"""
StockPulse Dashboard - Streamlit Application

A comprehensive trading dashboard with:
- Live Signals
- Paper Portfolio
- Performance Analytics
- Backtests
- Long-Term Watchlist
- Settings & Debug
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date, timedelta
import traceback

# Auto-load .env file before anything else
def _load_env():
    """Automatically load .env file from project root."""
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent.parent / ".env",
        Path.home() / "mega-cap" / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
            break

_load_env()

import streamlit as st
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stockpulse.utils.config import load_config, get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.data.universe import UniverseManager
from stockpulse.data.ingestion import DataIngestion
from stockpulse.strategies.signal_generator import SignalGenerator
from stockpulse.strategies.position_manager import PositionManager
from stockpulse.strategies.backtest import Backtester
from stockpulse.dashboard.charts import (
    ChartTheme, create_price_chart, create_equity_curve,
    create_performance_chart, create_win_rate_chart, create_pnl_distribution
)

# Page config
st.set_page_config(
    page_title="StockPulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode Theme with WCAG 2.1 AA compliant contrast
st.markdown("""
<style>
    /* ========================================
       DARK THEME - Slate color palette
       ======================================== */

    /* Main app background */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #0f172a !important;
    }

    /* Main content area */
    .main .block-container {
        background-color: #0f172a !important;
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }

    /* ========================================
       SIDEBAR - Darker slate
       ======================================== */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #1e293b !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }

    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] .stRadio label span {
        color: #e2e8f0 !important;
    }

    /* ========================================
       MAIN CONTENT TEXT
       ======================================== */

    /* All text defaults */
    .main p, .main span, .main label, .main div {
        color: #e2e8f0 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }

    h1 { font-size: 2rem !important; font-weight: 700 !important; }
    h2 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #3b82f6 !important;
        padding-bottom: 0.5rem !important;
        margin-top: 1.5rem !important;
    }
    h3 { font-size: 1.25rem !important; font-weight: 600 !important; }

    /* ========================================
       METRICS - High visibility
       ======================================== */
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* Positive delta - bright green */
    [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] + div,
    [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) {
        color: #22c55e !important;
    }

    /* Negative delta - bright red */
    [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] + div,
    [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {
        color: #ef4444 !important;
    }

    /* ========================================
       DATA TABLES - Dark styling
       ======================================== */
    .stDataFrame {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Table container */
    [data-testid="stDataFrame"] > div {
        background-color: #1e293b !important;
        border-radius: 8px !important;
    }

    /* Table headers */
    .stDataFrame thead th {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #475569 !important;
    }

    /* Table cells */
    .stDataFrame tbody td {
        color: #e2e8f0 !important;
        background-color: #1e293b !important;
        border-bottom: 1px solid #334155 !important;
    }

    /* Alternating rows */
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: #273548 !important;
    }

    /* Table hover */
    .stDataFrame tbody tr:hover td {
        background-color: #334155 !important;
    }

    /* ========================================
       FORM ELEMENTS - Dark inputs
       ======================================== */

    /* Select boxes */
    .stSelectbox label, .stMultiSelect label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        background-color: #1e293b !important;
        border-color: #475569 !important;
    }

    .stSelectbox [data-baseweb="select"] *,
    .stMultiSelect [data-baseweb="select"] * {
        color: #e2e8f0 !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] {
        background-color: #1e293b !important;
    }

    [data-baseweb="menu"] {
        background-color: #1e293b !important;
    }

    [data-baseweb="menu"] li {
        color: #e2e8f0 !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: #334155 !important;
    }

    /* Text inputs */
    .stTextInput label, .stNumberInput label {
        color: #e2e8f0 !important;
    }

    .stTextInput input, .stNumberInput input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Sliders */
    .stSlider label {
        color: #e2e8f0 !important;
    }

    /* ========================================
       BUTTONS - Blue accent
       ======================================== */
    .stButton button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border: none !important;
        transition: background-color 0.2s ease !important;
    }

    .stButton button:hover {
        background-color: #2563eb !important;
    }

    .stButton button:active {
        background-color: #1d4ed8 !important;
    }

    /* Secondary/outline buttons */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid #475569 !important;
        color: #e2e8f0 !important;
    }

    .stButton button[kind="secondary"]:hover {
        background-color: #334155 !important;
    }

    /* ========================================
       ALERTS / INFO BOXES - Dark variants
       ======================================== */
    .stAlert {
        border-radius: 8px !important;
    }

    /* Info box - blue */
    [data-testid="stAlert"][data-baseweb="notification"],
    .stInfo {
        background-color: #1e3a5f !important;
        color: #93c5fd !important;
        border-left: 4px solid #3b82f6 !important;
    }

    /* Success box - green */
    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background-color: #14532d !important;
        color: #86efac !important;
        border-left: 4px solid #22c55e !important;
    }

    /* Warning box - amber */
    .stWarning, [data-baseweb="notification"][kind="warning"] {
        background-color: #713f12 !important;
        color: #fcd34d !important;
        border-left: 4px solid #f59e0b !important;
    }

    /* Error box - red */
    .stError, [data-baseweb="notification"][kind="negative"] {
        background-color: #7f1d1d !important;
        color: #fca5a5 !important;
        border-left: 4px solid #ef4444 !important;
    }

    /* ========================================
       TABS - Dark styling
       ======================================== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        gap: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: #f1f5f9 !important;
        background-color: #334155 !important;
        border-radius: 6px !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
    }

    /* Tab content area */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: transparent !important;
    }

    /* ========================================
       EXPANDERS - Dark styling
       ======================================== */
    .streamlit-expanderHeader {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        background-color: #1e293b !important;
        border-radius: 8px !important;
    }

    .streamlit-expanderContent {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 0 0 8px 8px !important;
    }

    [data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* ========================================
       CODE BLOCKS - Dark styling
       ======================================== */
    code {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    pre {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* ========================================
       CAPTIONS & SMALL TEXT
       ======================================== */
    .stCaption, small, .st-emotion-cache-1gulkj5 {
        color: #94a3b8 !important;
    }

    /* ========================================
       DIVIDERS
       ======================================== */
    hr {
        border-color: #334155 !important;
    }

    /* ========================================
       PLOTLY CHARTS - Dark theme
       ======================================== */
    .js-plotly-plot .plotly .modebar {
        background-color: transparent !important;
    }

    .js-plotly-plot .plotly .modebar-btn path {
        fill: #94a3b8 !important;
    }

    .js-plotly-plot .plotly .modebar-btn:hover path {
        fill: #e2e8f0 !important;
    }

    /* ========================================
       CARDS / CONTAINERS
       ======================================== */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
    }

    /* ========================================
       SCROLLBAR - Dark styling
       ======================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e293b;
    }

    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }

    /* ========================================
       POSITIVE/NEGATIVE VALUE COLORS
       ======================================== */
    .positive, .profit {
        color: #22c55e !important;
    }

    .negative, .loss {
        color: #ef4444 !important;
    }

    /* ========================================
       STATUS BADGES
       ======================================== */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-success {
        background-color: #14532d;
        color: #86efac;
    }

    .badge-warning {
        background-color: #713f12;
        color: #fcd34d;
    }

    .badge-danger {
        background-color: #7f1d1d;
        color: #fca5a5;
    }

    .badge-info {
        background-color: #1e3a5f;
        color: #93c5fd;
    }
</style>
""", unsafe_allow_html=True)


logger = get_logger(__name__)

# Global debug log
_debug_log = []

def debug_print(message: str, level: str = "INFO"):
    """Print to console and store in debug log."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted = f"[{timestamp}] [{level}] {message}"
    try:
        print(formatted)
    except OSError:
        pass  # Ignore if stdout not available (e.g., detached terminal)
    _debug_log.append({"time": timestamp, "level": level, "message": message})


def format_sentiment(sentiment_data: dict, ticker: str) -> str:
    """Format sentiment data for display in dashboard tables.

    Args:
        sentiment_data: Dict of ticker -> sentiment info from SentimentStorage
        ticker: The ticker to get sentiment for

    Returns:
        Formatted string like "🟢 72" or "—" if no data
    """
    sent = sentiment_data.get(ticker, {})
    sent_score = sent.get("aggregate_score", 0)
    sent_label = sent.get("aggregate_label", "")
    if sent_score > 0:
        if sent_label == "bullish":
            return f"🟢 {sent_score:.0f}"
        elif sent_label == "bearish":
            return f"🔴 {sent_score:.0f}"
        else:
            return f"🟡 {sent_score:.0f}"
    return "—"


@st.cache_resource
def init_services():
    """Initialize services (cached)."""
    global _debug_log
    _debug_log = []  # Reset debug log

    # SQLite with WAL mode supports concurrent access - no special setup needed

    debug_print("=" * 60)
    debug_print("STOCKPULSE DASHBOARD STARTUP")
    debug_print("=" * 60)
    debug_print(f"Python version: {sys.version}")
    debug_print(f"Working directory: {Path.cwd()}")

    # Find and load config
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    debug_print(f"Primary config path: {config_path}")
    debug_print(f"Config exists: {config_path.exists()}")

    if not config_path.exists():
        alt_paths = [
            Path.cwd() / "config" / "config.yaml",
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
        ]
        for alt in alt_paths:
            debug_print(f"Trying alternative: {alt}")
            if alt.exists():
                config_path = alt
                debug_print(f"Using alternative config: {alt}")
                break

    try:
        load_config(config_path)
        debug_print("Config loaded successfully", "OK")
    except Exception as e:
        debug_print(f"Config load error: {e}", "ERROR")
        raise

    # Environment variables
    debug_print("")
    debug_print("ENVIRONMENT VARIABLES:")
    env_status = {}
    env_vars = {
        "STOCKPULSE_EMAIL_SENDER": "Email sender address",
        "STOCKPULSE_EMAIL_RECIPIENT": "Primary recipient",
        "STOCKPULSE_EMAIL_PASSWORD": "Gmail app password",
        "STOCKPULSE_EMAIL_RECIPIENTS_CC": "CC recipients (optional)",
    }
    for var, desc in env_vars.items():
        val = os.environ.get(var, "")
        if var == "STOCKPULSE_EMAIL_PASSWORD":
            display = "****" if val else "(NOT SET)"
            env_status[var] = "OK" if val else "MISSING"
        else:
            display = val if val else "(NOT SET)"
            if var == "STOCKPULSE_EMAIL_RECIPIENTS_CC":
                env_status[var] = "OK" if val else "OPTIONAL"
            else:
                env_status[var] = "OK" if val else "MISSING"
        debug_print(f"  {var}: {display} [{env_status[var]}]")

    # Database (SQLite with WAL mode supports concurrent access)
    debug_print("")
    debug_print("DATABASE:")
    try:
        db = get_db()
        debug_print(f"  Path: {db.db_path}")
        debug_print(f"  Exists: {db.db_path.exists()}")
        debug_print(f"  Size: {db.db_path.stat().st_size / 1024:.1f} KB" if db.db_path.exists() else "  Size: N/A")

        # Table counts
        tables = ["universe", "prices_daily", "prices_intraday", "signals",
                  "positions_paper", "positions_real", "alerts_log", "backtest_results"]
        debug_print("  Table counts:")
        for table in tables:
            try:
                count = db.fetchone(f"SELECT COUNT(*) FROM {table}")[0]
                debug_print(f"    {table}: {count}")
            except Exception as e:
                debug_print(f"    {table}: ERROR - {e}", "WARN")

    except Exception as e:
        debug_print(f"Database error: {e}", "ERROR")
        raise

    # Initialize services
    debug_print("")
    debug_print("INITIALIZING SERVICES:")
    services = {}

    try:
        services["db"] = db
        debug_print("  Database: OK", "OK")
    except Exception as e:
        debug_print(f"  Database: FAILED - {e}", "ERROR")

    try:
        services["universe"] = UniverseManager()
        tickers = services["universe"].get_active_tickers()
        debug_print(f"  Universe: OK ({len(tickers)} tickers)", "OK")
        if tickers:
            debug_print(f"    Sample: {tickers[:5]}")
    except Exception as e:
        debug_print(f"  Universe: FAILED - {e}", "ERROR")

    try:
        services["ingestion"] = DataIngestion()
        staleness = services["ingestion"].check_data_staleness()
        debug_print(f"  Ingestion: OK", "OK")
        debug_print(f"    Last daily: {staleness.get('last_daily', 'Never')}")
        debug_print(f"    Last intraday: {staleness.get('last_intraday', 'Never')}")
        debug_print(f"    Data stale: {staleness.get('is_stale', 'Unknown')}")

        # Auto-ingest if daily data is stale (older than 1 day)
        if staleness.get('is_stale', False) or staleness.get('hours_since_daily', 0) > 24:
            debug_print("  Auto-refreshing stale data...", "WARN")
            try:
                tickers = services.get("universe", UniverseManager()).get_active_tickers()
                if tickers:
                    # Fetch fresh daily prices
                    services["ingestion"].run_daily_ingestion(tickers)
                    debug_print(f"  Auto-ingest complete for {len(tickers)} tickers", "OK")
            except Exception as ingest_err:
                debug_print(f"  Auto-ingest failed: {ingest_err}", "WARN")
    except Exception as e:
        debug_print(f"  Ingestion: FAILED - {e}", "ERROR")

    try:
        services["signals"] = SignalGenerator()
        debug_print("  SignalGenerator: OK", "OK")
    except Exception as e:
        debug_print(f"  SignalGenerator: FAILED - {e}", "ERROR")

    try:
        services["positions"] = PositionManager()
        debug_print("  PositionManager: OK", "OK")
    except Exception as e:
        debug_print(f"  PositionManager: FAILED - {e}", "ERROR")

    try:
        services["backtester"] = Backtester()
        debug_print("  Backtester: OK", "OK")
    except Exception as e:
        debug_print(f"  Backtester: FAILED - {e}", "ERROR")

    # Email status
    debug_print("")
    debug_print("EMAIL STATUS:")
    try:
        from stockpulse.alerts.email_sender import EmailSender
        email = EmailSender()
        if email.is_configured:
            debug_print(f"  Configured: YES", "OK")
            debug_print(f"  Sender: {email.sender}")
            debug_print(f"  Recipient: {email.recipient}")
            debug_print(f"  CC: {email.cc_recipients if email.cc_recipients else 'None'}")
        else:
            debug_print("  Configured: NO - Set environment variables!", "WARN")
    except Exception as e:
        debug_print(f"  Email error: {e}", "ERROR")

    debug_print("")
    debug_print("=" * 60)
    debug_print("STARTUP COMPLETE")
    debug_print("=" * 60)

    # Store debug log in services for UI access
    services["_debug_log"] = _debug_log.copy()
    services["_startup_time"] = datetime.now()

    return services


def main():
    """Main dashboard application."""
    # Initialize services
    try:
        services = init_services()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        st.code(traceback.format_exc())
        st.info("Check the terminal for detailed debug output.")
        return

    # Sidebar
    st.sidebar.title("📊 StockPulse")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Live Signals", "Paper Portfolio", "Long-Term Holdings", "Performance", "Backtests", "Long-Term Watchlist", "AI Stocks", "Trillion Club", "AI Theses", "Universe", "Settings", "Debug"],
        label_visibility="collapsed"
    )

    # Status indicators in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")

    # Email status with helpful info
    try:
        from stockpulse.alerts.email_sender import EmailSender
        email = EmailSender()
        if email.is_configured:
            st.sidebar.success("✅ Email: Ready")
        else:
            st.sidebar.warning("⚠️ Email: Not configured")
            st.sidebar.caption("Run: source .env")
    except Exception as e:
        st.sidebar.error("❌ Email: Error")

    # Universe status
    universe_mgr = services.get("universe")
    if universe_mgr and hasattr(universe_mgr, 'get_active_tickers'):
        tickers = universe_mgr.get_active_tickers()
        count = len(tickers) if tickers else 0
        if count > 0:
            st.sidebar.success(f"📈 Universe: {count} stocks")
        else:
            st.sidebar.warning("📈 Universe: Empty")
            st.sidebar.caption("Click Initialize Data")
    else:
        st.sidebar.warning("📈 Universe: Not loaded")

    # Data status - check for actual price data
    try:
        db = services.get("db") or get_db()
        price_count = db.fetchone("SELECT COUNT(*) FROM prices_daily")[0] or 0
        if price_count > 0:
            last_date = db.fetchone("SELECT MAX(date) FROM prices_daily")[0]
            st.sidebar.success(f"📊 Data: {price_count:,} prices (to {last_date})")
        else:
            st.sidebar.warning("📊 Data: No data yet")
    except Exception:
        st.sidebar.warning("📊 Data: Unknown")

    # Portfolio status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Portfolio Status**")
    try:
        position_mgr = services.get("positions")
        config = get_config()
        initial_capital = config.get("portfolio", {}).get("initial_capital", 100000.0)
        max_positions = config.get("portfolio", {}).get("max_positions", 40)
        max_exposure_pct = config.get("risk_management", {}).get("max_portfolio_exposure_pct", 80.0)
        max_per_strategy_pct = config.get("risk_management", {}).get("max_per_strategy_pct", 70.0)

        # Get open positions
        open_positions = position_mgr.get_open_positions() if position_mgr else pd.DataFrame()
        num_positions = len(open_positions) if not open_positions.empty else 0

        # Calculate exposure
        if not open_positions.empty:
            total_invested = (open_positions["entry_price"] * open_positions["shares"]).sum()
            exposure_pct = (total_invested / initial_capital) * 100
            cash_remaining = initial_capital - total_invested
        else:
            total_invested = 0
            exposure_pct = 0
            cash_remaining = initial_capital

        remaining_exposure = max_exposure_pct - exposure_pct

        # Display portfolio metrics
        st.sidebar.metric("Exposure", f"{exposure_pct:.0f}% / {max_exposure_pct:.0f}%",
                         delta=f"{remaining_exposure:.0f}% available", delta_color="normal")
        st.sidebar.metric("Positions", f"{num_positions} / {max_positions}",
                         delta=f"{max_positions - num_positions} slots free", delta_color="normal")
        st.sidebar.caption(f"💵 Cash: ${cash_remaining:,.0f}")

        # Strategy capacity summary
        if position_mgr and not open_positions.empty:
            strategies_used = open_positions["strategy"].unique() if "strategy" in open_positions.columns else []
            if len(strategies_used) > 0:
                st.sidebar.markdown("**Strategy Capacity**")
                for strat in sorted(strategies_used):
                    strat_pct = position_mgr.get_strategy_current_exposure_pct(strat)
                    remaining = max_per_strategy_pct - strat_pct
                    short_name = strat.replace("_", " ").title()[:12]
                    if remaining < 5:
                        st.sidebar.warning(f"⚠️ {short_name}: {strat_pct:.0f}%/{max_per_strategy_pct:.0f}%")
                    else:
                        st.sidebar.caption(f"✓ {short_name}: {strat_pct:.0f}%/{max_per_strategy_pct:.0f}%")
    except Exception as e:
        st.sidebar.warning(f"📊 Portfolio: Error")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")

    if st.sidebar.button("🔄 Refresh Data"):
        # Clear both data and resource caches to get fresh data
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Route to page
    if page == "Live Signals":
        render_signals_page(services)
    elif page == "Paper Portfolio":
        render_portfolio_page(services)
    elif page == "Long-Term Holdings":
        render_longterm_holdings_page(services)
    elif page == "Performance":
        render_performance_page(services)
    elif page == "Backtests":
        render_backtests_page(services)
    elif page == "Long-Term Watchlist":
        render_watchlist_page(services)
    elif page == "AI Stocks":
        render_ai_stocks_page(services)
    elif page == "Trillion Club":
        render_trillion_club_page(services)
    elif page == "AI Theses":
        render_ai_theses_page(services)
    elif page == "Universe":
        render_universe_page(services)
    elif page == "Settings":
        render_settings_page(services)
    elif page == "Debug":
        render_debug_page(services)


def render_universe_page(services: dict):
    """Render Universe page showing all tracked stocks."""
    st.title("🌍 Stock Universe")

    st.markdown("The top US stocks by market cap that StockPulse tracks and analyzes.")

    # Get universe data
    try:
        db = services.get("db") or get_db()
        universe_df = db.fetchdf("""
            SELECT
                ticker,
                company_name,
                sector,
                industry,
                market_cap,
                is_active
            FROM universe
            ORDER BY market_cap DESC
        """)
    except Exception as e:
        st.error(f"Error loading universe: {e}")
        return

    if universe_df.empty:
        st.warning("No stocks in universe. Run `stockpulse init` to populate.")
        return

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    active_count = len(universe_df[universe_df["is_active"] == 1]) if "is_active" in universe_df.columns else len(universe_df)
    total_market_cap = universe_df["market_cap"].sum() if "market_cap" in universe_df.columns else 0

    with col1:
        st.metric("Total Stocks", len(universe_df))
    with col2:
        st.metric("Active Stocks", active_count)
    with col3:
        st.metric("Total Market Cap", f"${total_market_cap/1e12:.1f}T")
    with col4:
        sectors = universe_df["sector"].nunique() if "sector" in universe_df.columns else 0
        st.metric("Sectors", sectors)

    # Sector breakdown
    st.markdown("---")
    st.subheader("Sector Breakdown")

    if "sector" in universe_df.columns and "market_cap" in universe_df.columns:
        sector_df = universe_df.groupby("sector").agg({
            "ticker": "count",
            "market_cap": "sum"
        }).reset_index()
        sector_df.columns = ["Sector", "Count", "Market Cap"]
        sector_df = sector_df.sort_values("Market Cap", ascending=False)
        sector_df["Market Cap"] = sector_df["Market Cap"].apply(lambda x: f"${x/1e12:.2f}T")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)

        with col2:
            # Pie chart of sector allocation
            import plotly.express as px
            sector_for_chart = universe_df.groupby("sector")["market_cap"].sum().reset_index()
            fig = px.pie(
                sector_for_chart,
                values="market_cap",
                names="sector",
                title="Market Cap by Sector"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Full stock list
    st.markdown("---")
    st.subheader("All Stocks")

    # Format for display
    display_df = universe_df.copy()
    if "market_cap" in display_df.columns:
        display_df["market_cap"] = display_df["market_cap"].apply(
            lambda x: f"${x/1e9:.1f}B" if x and x >= 1e9 else (f"${x/1e6:.0f}M" if x else "N/A")
        )
    if "is_active" in display_df.columns:
        display_df["is_active"] = display_df["is_active"].apply(lambda x: "✓" if x else "✗")

    # Rename columns for display
    display_df = display_df.rename(columns={
        "ticker": "Ticker",
        "company_name": "Company",
        "sector": "Sector",
        "industry": "Industry",
        "market_cap": "Market Cap",
        "is_active": "Active"
    })

    # Filter options
    col1, col2 = st.columns(2)
    selected_sector = "All"  # Initialize with default
    with col1:
        search = st.text_input("Search ticker or company", "")
    with col2:
        if "Sector" in display_df.columns:
            sectors = ["All"] + sorted(display_df["Sector"].dropna().unique().tolist())
            selected_sector = st.selectbox("Filter by sector", sectors)

    # Apply filters
    filtered_df = display_df
    if search:
        search_lower = search.lower()
        filtered_df = filtered_df[
            filtered_df["Ticker"].str.lower().str.contains(search_lower, na=False) |
            filtered_df["Company"].str.lower().str.contains(search_lower, na=False)
        ]
    if "Sector" in display_df.columns and selected_sector != "All":
        filtered_df = filtered_df[filtered_df["Sector"] == selected_sector]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=500)

    st.caption(f"Showing {len(filtered_df)} of {len(display_df)} stocks")


def render_debug_page(services: dict):
    """Render Debug page with verbose system information."""
    st.title("🔧 Debug Information")

    # Startup time
    startup_time = services.get("_startup_time", datetime.now())
    st.caption(f"Dashboard started: {startup_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Quick diagnosis
    st.markdown("---")
    st.subheader("🩺 Quick Diagnosis")

    issues = []
    fixes = []

    # Check email
    email_sender = os.environ.get("STOCKPULSE_EMAIL_SENDER", "")
    email_pass = os.environ.get("STOCKPULSE_EMAIL_PASSWORD", "")
    if not email_sender or not email_pass:
        issues.append("❌ Email not configured in environment")
        fixes.append("Your `.env` file has the values, but they're not exported. Run:\n```bash\nset -a && source .env && set +a\n```\nOr add `export` before each line in your `.env` file.")

    # Check universe
    universe_mgr = services.get("universe")
    if universe_mgr:
        tickers = universe_mgr.get_active_tickers()
        if not tickers:
            issues.append("❌ Stock universe is empty")
            fixes.append("Go to **Live Signals** tab and click **Initialize Data**")

    # Check data
    try:
        staleness = services["ingestion"].check_data_staleness()
        if not staleness.get("last_daily"):
            issues.append("❌ No price data loaded")
            fixes.append("Click **Initialize Data** to fetch historical prices")
    except:
        pass

    if issues:
        for i, (issue, fix) in enumerate(zip(issues, fixes)):
            st.error(issue)
            st.markdown(fix)
            if i < len(issues) - 1:
                st.markdown("---")
    else:
        st.success("✅ All systems operational!")

    # Environment Variables
    st.markdown("---")
    st.subheader("Environment Variables")

    env_vars = {
        "STOCKPULSE_EMAIL_SENDER": os.environ.get("STOCKPULSE_EMAIL_SENDER", ""),
        "STOCKPULSE_EMAIL_RECIPIENT": os.environ.get("STOCKPULSE_EMAIL_RECIPIENT", ""),
        "STOCKPULSE_EMAIL_PASSWORD": "****" if os.environ.get("STOCKPULSE_EMAIL_PASSWORD") else "(NOT SET)",
        "STOCKPULSE_EMAIL_RECIPIENTS_CC": os.environ.get("STOCKPULSE_EMAIL_RECIPIENTS_CC", "(not set)"),
    }

    col1, col2 = st.columns(2)
    for i, (key, val) in enumerate(env_vars.items()):
        with col1 if i % 2 == 0 else col2:
            status = "✅" if val and val != "(NOT SET)" and val != "(not set)" else "❌"
            st.text(f"{status} {key}")
            st.code(val or "(empty)")

    # Database Status
    st.markdown("---")
    st.subheader("Database Status")

    db = services.get("db")
    if db:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Path", str(db.db_path.name))
        with col2:
            size = db.db_path.stat().st_size / 1024 if db.db_path.exists() else 0
            st.metric("Size", f"{size:.1f} KB")
        with col3:
            st.metric("Exists", "Yes" if db.db_path.exists() else "No")

        # Table details
        st.markdown("**Table Row Counts:**")
        tables = ["universe", "prices_daily", "prices_intraday", "signals",
                  "positions_paper", "positions_real", "alerts_log",
                  "long_term_watchlist", "backtest_results", "system_state"]

        table_data = []
        for table in tables:
            try:
                count = db.fetchone(f"SELECT COUNT(*) FROM {table}")[0]
                table_data.append({"Table": table, "Rows": count, "Status": "✅"})
            except Exception as e:
                table_data.append({"Table": table, "Rows": "Error", "Status": "❌"})

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Services Status
    st.markdown("---")
    st.subheader("Services Status")

    service_status = []
    for name in ["db", "universe", "ingestion", "signals", "positions", "backtester"]:
        svc = services.get(name)
        service_status.append({
            "Service": name,
            "Status": "✅ Loaded" if svc else "❌ Not loaded",
            "Type": type(svc).__name__ if svc else "N/A"
        })

    st.dataframe(pd.DataFrame(service_status), use_container_width=True, hide_index=True)

    # Email Test
    st.markdown("---")
    st.subheader("Email Configuration")

    try:
        from stockpulse.alerts.email_sender import EmailSender
        email = EmailSender()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Configured", "Yes" if email.is_configured else "No")
            st.text(f"SMTP Server: {email.smtp_server}:{email.smtp_port}")
            st.text(f"Sender: {email.sender or '(not set)'}")
        with col2:
            st.text(f"Recipient: {email.recipient or '(not set)'}")
            st.text(f"CC: {', '.join(email.cc_recipients) if email.cc_recipients else '(none)'}")

        if not email.is_configured:
            st.warning("""
            **Email not configured!** Set these environment variables:
            ```
            export STOCKPULSE_EMAIL_SENDER="your-email@gmail.com"
            export STOCKPULSE_EMAIL_RECIPIENT="your-email@gmail.com"
            export STOCKPULSE_EMAIL_PASSWORD="your-app-password"
            ```
            """)
    except Exception as e:
        st.error(f"Email module error: {e}")

    # Startup Log
    st.markdown("---")
    st.subheader("Startup Log")

    debug_log = services.get("_debug_log", [])
    if debug_log:
        log_text = "\n".join([f"[{l['time']}] [{l['level']}] {l['message']}" for l in debug_log])
        st.code(log_text, language="")
    else:
        st.info("No debug log available. Restart dashboard to capture startup logs.")

    # System Info
    st.markdown("---")
    st.subheader("System Information")

    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Python: {sys.version.split()[0]}")
        st.text(f"Platform: {sys.platform}")
        st.text(f"Working Dir: {Path.cwd()}")
    with col2:
        st.text(f"Streamlit: {st.__version__}")
        try:
            import duckdb
            st.text(f"DuckDB: {duckdb.__version__}")
        except:
            st.text("DuckDB: (error)")
        try:
            import yfinance
            st.text(f"yfinance: {yfinance.__version__}")
        except:
            st.text("yfinance: (error)")


def render_signals_page(services: dict):
    """Render Live Signals page."""
    import time
    from datetime import datetime
    st.title("📡 Live Signals")

    # Auto-refresh (default ON, triggers when scan completes)
    col_refresh1, col_refresh2 = st.columns([3, 1])
    with col_refresh1:
        auto_refresh = st.checkbox("Auto-refresh on scan completion", value=True)
    with col_refresh2:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_dash_refresh = datetime.now().isoformat()
            st.session_state.last_poll_time = time.time()
            st.rerun()

    # Initialize session state
    if "last_dash_refresh" not in st.session_state:
        st.session_state.last_dash_refresh = datetime.now().isoformat()
    if "last_poll_time" not in st.session_state:
        st.session_state.last_poll_time = time.time()

    # Check for scan completion (poll every 30 seconds)
    if auto_refresh:
        poll_elapsed = time.time() - st.session_state.last_poll_time
        if poll_elapsed > 30:  # Check every 30 seconds
            st.session_state.last_poll_time = time.time()
            try:
                db = services.get("db")
                if db:
                    row = db.fetchone("SELECT value FROM system_state WHERE key = 'last_scan_completed'")
                    if row and row[0]:
                        last_scan = row[0]
                        if last_scan > st.session_state.last_dash_refresh:
                            st.session_state.last_dash_refresh = datetime.now().isoformat()
                            st.rerun()
            except Exception:
                pass
            st.rerun()  # Rerun to check again in 30s

        # Show status
        try:
            db = services.get("db")
            if db:
                row = db.fetchone("SELECT value FROM system_state WHERE key = 'last_scan_completed'")
                if row and row[0]:
                    st.caption(f"🔗 Last scan: {row[0][:19]} | checking in {int(30 - poll_elapsed)}s")
        except Exception:
            st.caption("⚠️ Could not check scan status")

    # Get open signals
    try:
        signals_df = services["signals"].get_open_signals()
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        signals_df = pd.DataFrame()

    # Available strategies (always show all 5)
    all_strategies = [
        "rsi_mean_reversion",
        "bollinger_squeeze",
        "macd_volume",
        "mean_reversion_zscore",
        "momentum_breakout"
    ]

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_confidence = st.slider("Min Confidence", 0, 100, 60)

    with col2:
        strategies = ["All"] + all_strategies
        selected_strategy = st.selectbox("Strategy", strategies)

    with col3:
        directions = ["All", "BUY", "SELL"]
        selected_direction = st.selectbox("Direction", directions)

    # Apply filters
    filtered = signals_df.copy() if not signals_df.empty else pd.DataFrame()

    if not filtered.empty:
        filtered = filtered[filtered["confidence"] >= min_confidence]

        if selected_strategy != "All":
            filtered = filtered[filtered["strategy"] == selected_strategy]

        if selected_direction != "All":
            filtered = filtered[filtered["direction"] == selected_direction]

    # Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Signals", len(filtered))
    with col2:
        buy_count = len(filtered[filtered["direction"] == "BUY"]) if not filtered.empty else 0
        st.metric("Buy Signals", buy_count)
    with col3:
        sell_count = len(filtered[filtered["direction"] == "SELL"]) if not filtered.empty else 0
        st.metric("Sell Signals", sell_count)
    with col4:
        avg_conf = filtered["confidence"].mean() if not filtered.empty else 0
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")

    # Get portfolio tickers and cooldown tickers to mark in signals
    try:
        positions_df = services["positions"].get_open_positions()
        portfolio_tickers = set(positions_df["ticker"].tolist()) if not positions_df.empty else set()
    except Exception:
        portfolio_tickers = set()

    try:
        blocked_list = services["positions"].get_blocked_tickers()
        cooldown_tickers = {b["ticker"] for b in blocked_list}
    except Exception:
        blocked_list = []
        cooldown_tickers = set()

    # Helper to format signals dataframe
    def format_signals_df(df, portfolio_tickers, cooldown_tickers):
        """Format signals dataframe for display, marking portfolio and cooldown tickers."""
        if df.empty:
            return df

        # Only select columns that exist
        desired_cols = ["ticker", "strategy", "confidence",
                       "entry_price", "target_price", "stop_price", "created_at", "notes"]
        available_cols = [c for c in desired_cols if c in df.columns]
        display_df = df[available_cols].copy()

        # Mark portfolio and cooldown tickers
        if "ticker" in display_df.columns:
            def mark_ticker(x):
                if x in portfolio_tickers:
                    return f"📌 {x}"
                elif x in cooldown_tickers:
                    return f"⏱️ {x}"
                return x
            display_df["ticker"] = display_df["ticker"].apply(mark_ticker)

        if "confidence" in display_df.columns:
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0f}%")
        if "entry_price" in display_df.columns:
            display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
        if "target_price" in display_df.columns:
            display_df["target_price"] = display_df["target_price"].apply(lambda x: f"${x:.2f}")
        if "stop_price" in display_df.columns:
            display_df["stop_price"] = display_df["stop_price"].apply(lambda x: f"${x:.2f}")

        return display_df

    # Split signals into BUY and SELL
    st.markdown("---")

    if not filtered.empty:
        buy_signals = filtered[filtered["direction"] == "BUY"]
        sell_signals = filtered[filtered["direction"] == "SELL"]

        # BUY Signals section
        st.subheader(f"🟢 BUY Signals ({len(buy_signals)})")
        if not buy_signals.empty:
            buy_display = format_signals_df(buy_signals, portfolio_tickers, cooldown_tickers)
            st.dataframe(buy_display, use_container_width=True, hide_index=True)
        else:
            st.info("No BUY signals matching filters.")

        st.markdown("---")

        # SELL Signals section
        st.subheader(f"🔴 SELL Signals ({len(sell_signals)})")
        if not sell_signals.empty:
            sell_display = format_signals_df(sell_signals, portfolio_tickers, cooldown_tickers)
            st.dataframe(sell_display, use_container_width=True, hide_index=True)
        else:
            st.info("No SELL signals matching filters.")

        # Legend
        legends = []
        if portfolio_tickers:
            legends.append("📌 = in portfolio")
        if cooldown_tickers:
            legends.append("⏱️ = in cooldown (blocked)")
        if legends:
            st.caption(" | ".join(legends))

        # === SIGNAL ACTION ANALYSIS ===
        st.markdown("---")
        st.subheader("📋 Signal Action Analysis")
        st.caption("Why we are/aren't acting on each BUY signal")

        # Get comprehensive blocking reasons for each signal
        action_analysis = []
        for _, sig in buy_signals.iterrows():
            ticker = sig["ticker"]
            strategy = sig["strategy"]

            try:
                # Get all blocking reasons from position manager (pass confidence for accurate sizing)
                confidence = sig["confidence"] if "confidence" in sig else 70
                block_info = services["positions"].get_signal_blocking_reasons(ticker, strategy, confidence)
                status = block_info.get("status", "❓ UNKNOWN")
                reason = block_info.get("reason", "Unknown")

                # If there are multiple reasons, show them
                all_reasons = block_info.get("all_reasons", [])
                if len(all_reasons) > 1:
                    reason = "; ".join(all_reasons[:2])  # Show up to 2 reasons
            except Exception as e:
                status = "❓ UNKNOWN"
                reason = f"Error: {str(e)[:40]}"

            action_analysis.append({
                "Ticker": ticker,
                "Signal": f"{sig['confidence']:.0f}% {strategy}",
                "Status": status,
                "Reason": reason
            })

        if action_analysis:
            st.dataframe(pd.DataFrame(action_analysis), use_container_width=True, hide_index=True)

        # === TOP SIGNALS BY STRATEGY ===
        st.markdown("---")
        st.subheader("🎯 Top Signals by Strategy")

        # Group signals by strategy
        all_strategies = filtered["strategy"].unique() if not filtered.empty else []

        if len(all_strategies) > 0:
            for strategy in sorted(all_strategies):
                strategy_signals = filtered[filtered["strategy"] == strategy].head(5)
                if not strategy_signals.empty:
                    with st.expander(f"**{strategy}** ({len(strategy_signals)} signals)", expanded=False):
                        display_data = []
                        for _, sig in strategy_signals.iterrows():
                            ticker = sig["ticker"]

                            # Get comprehensive status from position manager
                            try:
                                confidence = sig["confidence"] if "confidence" in sig else 70
                                block_info = services["positions"].get_signal_blocking_reasons(ticker, strategy, confidence)
                                status_str = block_info.get("status", "❓")
                                status = status_str.split()[0] if status_str else "❓"  # Just the emoji
                                reason_str = block_info.get("reason", "Unknown")
                                reason = reason_str[:25] + "..." if len(reason_str) > 25 else reason_str
                            except Exception:
                                status = "❓"
                                reason = "Unknown"

                            display_data.append({
                                "Status": status,
                                "Ticker": ticker,
                                "Conf": f"{sig['confidence']:.0f}%",
                                "Entry": f"${sig['entry_price']:.2f}",
                                "Target": f"${sig['target_price']:.2f}",
                                "Reason": reason
                            })
                        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        else:
            st.info("No signals to group by strategy.")

        # === NEAR MISSES - Stocks close to triggering ===
        st.markdown("---")
        st.subheader("🎯 Near Misses - Almost Triggering")
        st.caption("Stocks close to meeting signal criteria but not quite there yet")

        try:
            from stockpulse.strategies.signal_insights import SignalInsights
            signal_insights = SignalInsights()
            tickers = services["universe"].get_active_tickers() if services.get("universe") else []

            if tickers:
                near_misses = signal_insights.get_near_misses(tickers, top_n=3)

                # Display near misses for each strategy
                strategies_with_near_misses = [s for s in near_misses if near_misses[s]]

                if strategies_with_near_misses:
                    for strat_name in strategies_with_near_misses:
                        strat_near_misses = near_misses[strat_name]
                        if strat_near_misses:
                            strat_display = strat_name.replace("_", " ").title()
                            with st.expander(f"**{strat_display}** - {len(strat_near_misses)} stocks close to trigger", expanded=False):
                                near_miss_data = []
                                for nm in strat_near_misses:
                                        near_miss_data.append({
                                        "Ticker": nm.get("ticker", ""),
                                        "Price": f"${nm.get('price', 0):.2f}",
                                        "Indicator": nm.get("indicator", ""),
                                        "Distance": nm.get("distance", ""),
                                        "Updated": nm.get("updated", "?"),
                                    })
                                if near_miss_data:
                                    st.dataframe(pd.DataFrame(near_miss_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No stocks currently close to triggering any strategy thresholds.")
            else:
                st.info("Load universe to see near-misses.")
        except Exception as e:
            st.warning(f"Could not load near-misses: {str(e)[:50]}")

    else:
        st.info("📊 **No signals from latest scan** - Market conditions don't currently meet any strategy thresholds.")

        # Show near-misses for each strategy
        st.markdown("---")
        st.subheader("🎯 Near Misses by Strategy")
        st.caption("Stocks closest to triggering each strategy and what's needed")

        try:
            from stockpulse.strategies.signal_insights import SignalInsights, STRATEGY_DESCRIPTIONS
            signal_insights = SignalInsights()
            tickers = services["universe"].get_active_tickers() if services.get("universe") else []

            if tickers:
                near_misses = signal_insights.get_near_misses(tickers, top_n=5)

                # Get strategy configs for threshold info
                config = get_config()
                strategy_configs = config.get("strategies", {})

                # All strategies to show
                all_strategies = [
                    "rsi_mean_reversion",
                    "macd_volume",
                    "zscore_mean_reversion",
                    "momentum_breakout",
                    "week52_low_bounce",
                    "sector_rotation"
                ]

                for strat_name in all_strategies:
                    strat_near_misses = near_misses.get(strat_name, [])
                    strat_desc = STRATEGY_DESCRIPTIONS.get(strat_name, {}).get("short", strat_name.replace("_", " ").title())
                    strat_config = strategy_configs.get(strat_name, {})

                    # Get threshold info for this strategy
                    threshold_info = ""
                    if strat_name == "rsi_mean_reversion":
                        threshold_info = f"Threshold: RSI < {strat_config.get('rsi_oversold', 25)}"
                    elif strat_name == "zscore_mean_reversion":
                        threshold_info = f"Threshold: Z-score < {strat_config.get('zscore_entry', -2.25)}"
                    elif strat_name == "sector_rotation":
                        threshold_info = f"Threshold: Relative Strength > {strat_config.get('relative_strength_threshold', 1.2)}"
                    elif strat_name == "week52_low_bounce":
                        threshold_info = f"Threshold: Within {strat_config.get('low_threshold_pct', 12)}% of 52-week low"
                    elif strat_name == "momentum_breakout":
                        threshold_info = "Threshold: Break above 20-day high with volume"
                    elif strat_name == "macd_volume":
                        threshold_info = "Threshold: MACD crosses above signal line"

                    with st.expander(f"**{strat_desc}** ({len(strat_near_misses)} close)", expanded=len(strat_near_misses) > 0):
                        if threshold_info:
                            st.caption(threshold_info)

                        if strat_near_misses:
                            near_miss_data = []
                            for nm in strat_near_misses:
                                near_miss_data.append({
                                    "Ticker": nm.get("ticker", ""),
                                    "Price": f"${nm.get('price', 0):.2f}",
                                    "Current": nm.get("indicator", ""),
                                    "Gap": nm.get("distance", ""),
                                    "Updated": nm.get("updated", "?"),
                                })
                            st.dataframe(pd.DataFrame(near_miss_data), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No stocks close to this threshold")
            else:
                st.warning("Load universe to see near-misses.")
        except Exception as e:
            st.warning(f"Could not load near-misses: {str(e)[:100]}")

        # Show Initialize Data button only if truly no data
        if signals_df.empty:
            st.markdown("---")
            st.markdown("**First time setup?** Initialize data below:")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Initialize Data", type="primary"):
                    with st.spinner("Loading stock universe and fetching data..."):
                        try:
                            # Initialize universe
                            services["universe"].refresh_universe()
                            st.success("Universe loaded!")

                            # Fetch some initial data
                            tickers = services["universe"].get_active_tickers()[:10]  # Start with 10
                            if tickers:
                                services["ingestion"].run_daily_ingestion(tickers)
                                st.success(f"Loaded data for {len(tickers)} stocks!")

                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            with col2:
                if st.button("🔍 Run Signal Scan"):
                    with st.spinner("Running signal scan..."):
                        try:
                            tickers = services["universe"].get_active_tickers()
                            if not tickers:
                                st.error("No tickers in universe. Initialize data first.")
                            else:
                                signals = services["signals"].scan_universe(tickers[:20])
                                st.success(f"Scan complete! Found {len(signals)} signals.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Scan error: {e}")

    # Signal details
    if not filtered.empty:
        st.markdown("---")
        st.subheader("Signal Details")

        selected_ticker = st.selectbox("Select ticker to view chart", filtered["ticker"].unique())

        if selected_ticker:
            price_data = services["ingestion"].get_daily_prices(
                [selected_ticker],
                start_date=date.today() - timedelta(days=90)
            )

            if not price_data.empty:
                ticker_signals = filtered[filtered["ticker"] == selected_ticker].to_dict("records")
                fig = create_price_chart(price_data, selected_ticker, signals=ticker_signals)
                st.plotly_chart(fig, use_container_width=True)


def render_portfolio_page(services: dict):
    """Render Paper Portfolio page."""
    st.title("📈 Paper Portfolio")

    # Quick actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("🔍 Run Manual Scan"):
            with st.spinner("Running scan and opening positions..."):
                try:
                    from stockpulse.strategies.signal_generator import SignalGenerator
                    from stockpulse.strategies.base import SignalDirection

                    signal_gen = SignalGenerator()
                    position_mgr = services["positions"]
                    tickers = services["universe"].get_active_tickers()[:50]

                    signals = signal_gen.generate_signals(tickers)
                    buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]

                    opened = 0
                    for signal in sorted(buy_signals, key=lambda s: s.confidence, reverse=True)[:10]:
                        pos_id = position_mgr.open_position_from_signal(signal)
                        if pos_id:
                            opened += 1

                    st.success(f"Scan complete! {len(signals)} signals, {opened} positions opened.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    with col3:
        # Show database counts
        try:
            db = services.get("db") or get_db()
            sig_count = db.fetchone("SELECT COUNT(*) FROM signals")[0]
            pos_count = db.fetchone("SELECT COUNT(*) FROM positions_paper")[0]
            st.caption(f"DB: {sig_count} signals, {pos_count} positions")
        except:
            st.caption("DB: error reading")

    # Get portfolio data
    positions_df = services["positions"].get_open_positions()
    closed_df = services["positions"].get_closed_positions(start_date=date.today() - timedelta(days=90))
    performance = services["positions"].get_performance_summary()

    # Get initial capital from config
    config = get_config()
    initial_capital = config.get("portfolio", {}).get("initial_capital", 100000.0)

    # Calculate invested amount
    invested_amount = 0.0
    if not positions_df.empty:
        invested_amount = (positions_df["entry_price"] * positions_df["shares"]).sum()

    # Get current prices for unrealized P&L (use LIVE prices, not stale daily)
    unrealized_pnl = 0.0
    positions_with_pnl = []
    if not positions_df.empty:
        try:
            tickers = positions_df["ticker"].tolist()
            # Fetch live prices from yfinance
            ingestion = services.get("ingestion")
            if ingestion:
                current_prices = ingestion.fetch_current_prices(tickers)
            else:
                current_prices = {}

            # Fallback to daily prices if live fetch failed
            if not current_prices:
                db = services.get("db") or get_db()
                prices_df = db.fetchdf(f"""
                    SELECT ticker, close FROM prices_daily
                    WHERE ticker IN ({','.join(['?']*len(tickers))})
                    AND date = (SELECT MAX(date) FROM prices_daily)
                """, tuple(tickers))
                current_prices = dict(zip(prices_df["ticker"], prices_df["close"])) if not prices_df.empty else {}

            for _, pos in positions_df.iterrows():
                ticker = pos["ticker"]
                entry_price = pos["entry_price"]
                shares = pos["shares"]
                current_price = current_prices.get(ticker, entry_price)
                pos_value = current_price * shares
                cost_basis = entry_price * shares
                pos_unrealized = pos_value - cost_basis
                pos_unrealized_pct = (pos_unrealized / cost_basis * 100) if cost_basis > 0 else 0
                unrealized_pnl += pos_unrealized

                positions_with_pnl.append({
                    "status": "🟢",
                    "ticker": ticker,
                    "strategy": pos.get("strategy", ""),
                    "entry": entry_price,
                    "current": current_price,
                    "shares": shares,
                    "cost": cost_basis,
                    "value": pos_value,
                    "unrealized": pos_unrealized,
                    "unrealized_pct": pos_unrealized_pct,
                    "entry_date": pos.get("entry_date", ""),
                })
        except Exception as e:
            st.warning(f"Error fetching current prices: {e}")

    # Calculate realized P&L from closed positions
    realized_pnl = performance.get("total_pnl", 0)

    # Calculate totals
    cash_available = initial_capital - invested_amount + realized_pnl
    total_portfolio_value = cash_available + invested_amount + unrealized_pnl
    total_return = total_portfolio_value - initial_capital
    total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0

    # === PORTFOLIO SUMMARY ===
    st.markdown("---")
    st.subheader("💰 Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Capital", f"${initial_capital:,.0f}")
    with col2:
        st.metric("Cash Available", f"${cash_available:,.2f}")
    with col3:
        st.metric("Invested", f"${invested_amount:,.2f}")
    with col4:
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric("Total Value", f"${total_portfolio_value:,.2f}", delta=f"{total_return_pct:+.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Open Positions", len(positions_df))
    with col2:
        color = "normal" if unrealized_pnl >= 0 else "inverse"
        st.metric("Unrealized P&L", f"${unrealized_pnl:+,.2f}")
    with col3:
        st.metric("Realized P&L", f"${realized_pnl:+,.2f}")
    with col4:
        win_rate = performance.get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")

    # === OPEN POSITIONS WITH P&L ===
    st.markdown("---")
    st.subheader("📈 Open Positions")

    if positions_with_pnl:
        # Group by strategy
        strategies = sorted(set(p["strategy"] for p in positions_with_pnl))

        for strategy in strategies:
            strategy_positions = [p for p in positions_with_pnl if p["strategy"] == strategy]
            strategy_unrealized = sum(p["unrealized"] for p in strategy_positions)

            with st.expander(f"**{strategy}** ({len(strategy_positions)} positions, ${strategy_unrealized:+,.2f})", expanded=True):
                display_data = []
                for p in strategy_positions:
                    # Format entry date (just show date part)
                    entry_date = p.get("entry_date", "")
                    if entry_date:
                        entry_date = str(entry_date)[:10]

                    display_data.append({
                        "Status": p["status"],
                        "Ticker": p["ticker"],
                        "Bought": entry_date,
                        "Entry": f"${p['entry']:.2f}",
                        "Current": f"${p['current']:.2f}",
                        "Shares": f"{p['shares']:.2f}",
                        "Value": f"${p['value']:,.2f}",
                        "P&L": f"${p['unrealized']:+,.2f}",
                        "P&L %": f"{p['unrealized_pct']:+.2f}%",
                    })
                st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No open positions. Click 'Run Manual Scan' to generate signals and open positions.")

    # === CLOSED POSITIONS ===
    st.markdown("---")
    st.subheader("📉 Closed Positions")

    if not closed_df.empty:
        # Add status indicator and sort
        closed_df = closed_df.copy()
        closed_df["status"] = closed_df["pnl"].apply(lambda x: "✅" if x and x > 0 else "❌" if x and x < 0 else "⚪")

        # Group by strategy
        strategies = closed_df["strategy"].unique() if "strategy" in closed_df.columns else ["Unknown"]

        for strategy in strategies:
            strategy_closed = closed_df[closed_df["strategy"] == strategy] if "strategy" in closed_df.columns else closed_df
            strategy_pnl = strategy_closed["pnl"].sum() if "pnl" in strategy_closed.columns else 0
            wins = len(strategy_closed[strategy_closed["pnl"] > 0]) if "pnl" in strategy_closed.columns else 0
            losses = len(strategy_closed[strategy_closed["pnl"] <= 0]) if "pnl" in strategy_closed.columns else 0

            with st.expander(f"**{strategy}** ({wins}W/{losses}L, ${strategy_pnl:+,.2f})", expanded=False):
                display_cols = ["status", "ticker", "entry_date", "exit_date", "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason"]
                available_cols = [c for c in display_cols if c in strategy_closed.columns]
                display_df = strategy_closed[available_cols].copy()

                # Format dates (just show date part)
                if "entry_date" in display_df.columns:
                    display_df["entry_date"] = display_df["entry_date"].apply(lambda x: str(x)[:10] if x else "N/A")
                if "exit_date" in display_df.columns:
                    display_df["exit_date"] = display_df["exit_date"].apply(lambda x: str(x)[:10] if x else "N/A")
                if "entry_price" in display_df.columns:
                    display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
                if "exit_price" in display_df.columns:
                    display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:.2f}" if x else "N/A")
                if "pnl" in display_df.columns:
                    display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:+,.2f}" if x else "N/A")
                if "pnl_pct" in display_df.columns:
                    display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%" if x else "N/A")

                display_df = display_df.rename(columns={
                    "status": "W/L", "entry_date": "Bought", "exit_date": "Sold",
                    "entry_price": "Entry", "exit_price": "Exit",
                    "pnl": "P&L", "pnl_pct": "P&L %", "exit_reason": "Reason"
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No closed positions yet.")

    # Show blocked tickers (cooldowns)
    st.markdown("---")
    st.subheader("🚫 Trading Cooldowns")

    blocked = services["positions"].get_blocked_tickers()
    if blocked:
        blocked_df = pd.DataFrame(blocked)
        st.dataframe(blocked_df, use_container_width=True, hide_index=True)
        st.caption("These tickers are temporarily blocked from new positions due to cooldown rules.")
    else:
        st.success("No tickers in cooldown. All clear for trading!")


def render_performance_page(services: dict):
    """Render Performance Analytics page."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.title("📊 Performance Analytics")

    performance = services["positions"].get_performance_summary()
    config = get_config()
    initial_capital = config.get("portfolio", {}).get("initial_capital", 100000.0)

    # === PORTFOLIO VALUE TIME SERIES ===
    st.markdown("---")
    st.subheader("📈 Portfolio Value Over Time")

    # Build comprehensive time series from trade history
    try:
        db = services.get("db") or get_db()

        # Get all closed trades with dates
        closed_trades = db.fetchdf("""
            SELECT
                ticker, entry_date, exit_date, entry_price, exit_price,
                shares, pnl, strategy
            FROM positions_paper
            WHERE status = 'closed'
            ORDER BY exit_date
        """)

        # Get all open trades
        open_trades = db.fetchdf("""
            SELECT ticker, entry_date, entry_price, shares, strategy
            FROM positions_paper
            WHERE status = 'open'
        """)

        # Get current prices for open positions
        positions_df = services["positions"].get_open_positions()
        tickers = positions_df["ticker"].tolist() if not positions_df.empty else []

        ingestion = services.get("ingestion")
        current_prices = {}
        if ingestion and tickers:
            current_prices = ingestion.fetch_current_prices(tickers)

        if not current_prices and tickers:
            prices_df = db.fetchdf("""
                SELECT ticker, close FROM prices_daily
                WHERE date = (SELECT MAX(date) FROM prices_daily)
            """)
            current_prices = dict(zip(prices_df["ticker"], prices_df["close"])) if not prices_df.empty else {}

        # Build daily time series
        if not closed_trades.empty or not open_trades.empty:
            # Determine date range
            all_dates = []
            if not closed_trades.empty:
                all_dates.extend(pd.to_datetime(closed_trades["entry_date"]).tolist())
                all_dates.extend(pd.to_datetime(closed_trades["exit_date"]).tolist())
            if not open_trades.empty:
                all_dates.extend(pd.to_datetime(open_trades["entry_date"]).tolist())

            if all_dates:
                min_date = min(all_dates).date()
                max_date = date.today()

                # Create date range
                date_range = pd.date_range(start=min_date, end=max_date, freq='D')

                portfolio_values = []
                cash_values = []
                invested_values = []

                for d in date_range:
                    d_date = d.date()

                    # Calculate realized P&L up to this date
                    realized_pnl = 0.0
                    if not closed_trades.empty:
                        closed_by_date = closed_trades[pd.to_datetime(closed_trades["exit_date"]).dt.date <= d_date]
                        realized_pnl = closed_by_date["pnl"].sum() if not closed_by_date.empty else 0.0

                    # Calculate invested amount (positions open on this date)
                    invested = 0.0

                    # From closed trades: positions that were open on this date
                    if not closed_trades.empty:
                        for _, trade in closed_trades.iterrows():
                            entry_d = pd.to_datetime(trade["entry_date"]).date()
                            exit_d = pd.to_datetime(trade["exit_date"]).date()
                            if entry_d <= d_date < exit_d:
                                invested += trade["entry_price"] * trade["shares"]

                    # From open trades: positions opened before this date
                    if not open_trades.empty:
                        for _, trade in open_trades.iterrows():
                            entry_d = pd.to_datetime(trade["entry_date"]).date()
                            if entry_d <= d_date:
                                invested += trade["entry_price"] * trade["shares"]

                    # Cash = initial + realized - invested
                    cash = initial_capital + realized_pnl - invested

                    # For current day, add unrealized P&L
                    unrealized = 0.0
                    if d_date == date.today() and not open_trades.empty:
                        for _, trade in open_trades.iterrows():
                            ticker = trade["ticker"]
                            if ticker in current_prices:
                                current_price = current_prices[ticker]
                                entry_price = trade["entry_price"]
                                shares = trade["shares"]
                                unrealized += (current_price - entry_price) * shares

                    portfolio_value = cash + invested + unrealized

                    portfolio_values.append(portfolio_value)
                    cash_values.append(cash)
                    invested_values.append(invested)

                # Create time series dataframe
                ts_df = pd.DataFrame({
                    "date": date_range,
                    "portfolio_value": portfolio_values,
                    "cash": cash_values,
                    "invested": invested_values
                })

                # Plot portfolio value and cash
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.6, 0.4],
                    subplot_titles=("Total Portfolio Value", "Cash vs Invested")
                )

                # Portfolio Value line
                fig.add_trace(
                    go.Scatter(
                        x=ts_df["date"],
                        y=ts_df["portfolio_value"],
                        mode="lines",
                        name="Portfolio Value",
                        line=dict(color="#3b82f6", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(59, 130, 246, 0.1)"
                    ),
                    row=1, col=1
                )

                # Initial capital reference line
                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="#64748b",
                    annotation_text=f"Initial: ${initial_capital:,.0f}",
                    row=1, col=1
                )

                # Cash line
                fig.add_trace(
                    go.Scatter(
                        x=ts_df["date"],
                        y=ts_df["cash"],
                        mode="lines",
                        name="Cash Available",
                        line=dict(color="#22c55e", width=2)
                    ),
                    row=2, col=1
                )

                # Invested line
                fig.add_trace(
                    go.Scatter(
                        x=ts_df["date"],
                        y=ts_df["invested"],
                        mode="lines",
                        name="Invested",
                        line=dict(color="#f59e0b", width=2)
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    height=500,
                    paper_bgcolor="#1e293b",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=60, r=40, t=80, b=40)
                )

                fig.update_xaxes(gridcolor="#334155", linecolor="#64748b")
                fig.update_yaxes(gridcolor="#334155", linecolor="#64748b", tickformat="$,.0f")

                st.plotly_chart(fig, use_container_width=True)

                # Key metrics (with empty check)
                if not ts_df.empty:
                    latest_value = ts_df["portfolio_value"].iloc[-1]
                    total_return = ((latest_value / initial_capital) - 1) * 100
                    latest_cash = ts_df["cash"].iloc[-1]
                    latest_invested = ts_df["invested"].iloc[-1]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Portfolio Value", f"${latest_value:,.2f}", delta=f"{total_return:+.2f}%")
                    with col2:
                        st.metric("Cash Available", f"${latest_cash:,.2f}")
                    with col3:
                        st.metric("Invested", f"${latest_invested:,.2f}")
                    with col4:
                        pct_invested = (latest_invested / latest_value * 100) if latest_value > 0 else 0
                        st.metric("% Invested", f"{pct_invested:.1f}%")
            else:
                st.info("No trading history yet. Start trading to see performance charts!")
        else:
            st.info("No trading history yet. Start trading to see performance charts!")

    except Exception as e:
        st.error(f"Error generating charts: {e}")
        import traceback
        st.code(traceback.format_exc())

    # === PERFORMANCE SUMMARY ===
    st.markdown("---")
    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", performance.get("total_trades", 0))
    with col2:
        st.metric("Total P&L", f"${performance.get('total_pnl', 0):+,.2f}")
    with col3:
        st.metric("Win Rate", f"{performance.get('win_rate', 0):.1f}%")
    with col4:
        st.metric("Profit Factor", f"{performance.get('profit_factor', 0):.2f}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Wins", performance.get("total_wins", 0))
    with col2:
        st.metric("Total Losses", performance.get("total_losses", 0))
    with col3:
        st.metric("Avg Win", f"${performance.get('avg_win', 0):+,.2f}")
    with col4:
        st.metric("Avg Loss", f"${performance.get('avg_loss', 0):,.2f}")

    # === STRATEGY PERFORMANCE ===
    st.markdown("---")
    st.subheader("🎯 Strategy Performance")

    strategy_perf = services["positions"].get_strategy_performance()

    if not strategy_perf.empty:
        # Create strategy comparison bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=strategy_perf["strategy"],
            y=strategy_perf["total_pnl"],
            name="Total P&L",
            marker_color=strategy_perf["total_pnl"].apply(
                lambda x: "#22c55e" if pd.notna(x) and x >= 0 else "#ef4444"
            ).tolist()
        ))

        fig.update_layout(
            title="P&L by Strategy",
            height=350,
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            yaxis=dict(tickformat="$,.0f", gridcolor="#334155"),
            xaxis=dict(gridcolor="#334155")
        )

        st.plotly_chart(fig, use_container_width=True)

        # Strategy table
        display_cols = ["strategy", "total_trades", "wins", "losses", "win_rate", "total_pnl", "avg_pnl"]
        available_cols = [c for c in display_cols if c in strategy_perf.columns]
        display_df = strategy_perf[available_cols].copy()

        if "total_pnl" in display_df.columns:
            display_df["total_pnl"] = display_df["total_pnl"].apply(lambda x: f"${x:+,.2f}")
        if "avg_pnl" in display_df.columns:
            display_df["avg_pnl"] = display_df["avg_pnl"].apply(lambda x: f"${x:+,.2f}")
        if "win_rate" in display_df.columns:
            display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No strategy performance data available yet.")

    # === P&L DISTRIBUTION ===
    st.markdown("---")
    st.subheader("📉 P&L Distribution")

    closed_df = services["positions"].get_closed_positions()

    if not closed_df.empty and "pnl" in closed_df.columns:
        fig = create_pnl_distribution(closed_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for P&L distribution.")


def render_backtests_page(services: dict):
    """Render Backtests page with interactive parameter tuning and optimization."""
    st.title("🔬 Backtests & Optimization")

    # Import optimizer
    try:
        from stockpulse.strategies.optimizer import (
            StrategyOptimizer, get_param_ranges, get_strategy_params,
            STRATEGY_CLASSES
        )
        optimizer_available = True
    except ImportError:
        optimizer_available = False

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Results", "🎛️ Run Backtest", "🎯 Optimize"])

    # TAB 1: Results History
    with tab1:
        results_df = services["backtester"].get_backtest_results()

        if not results_df.empty:
            st.subheader("Backtest Results History")

            display_cols = [
                "strategy", "total_return_pct", "annualized_return_pct",
                "sharpe_ratio", "max_drawdown_pct", "win_rate",
                "profit_factor", "total_trades", "run_date"
            ]
            available_cols = [c for c in display_cols if c in results_df.columns]
            display_df = results_df[available_cols].copy()

            pct_cols = ["total_return_pct", "annualized_return_pct", "max_drawdown_pct", "win_rate"]
            for col in pct_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")

            ratio_cols = ["sharpe_ratio", "profit_factor"]
            for col in ratio_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Strategy Comparison")

            fig = create_performance_chart(
                results_df[["strategy", "total_return_pct"]].rename(columns={"total_return_pct": "total_pnl"}),
                "total_pnl"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No backtest results yet. Use the 'Run Backtest' tab to test strategies.")

    # TAB 2: Run Backtest with Custom Params
    with tab2:
        st.subheader("Run Backtest with Custom Parameters")

        # Strategy selection
        strategy_options = [
            "rsi_mean_reversion", "bollinger_squeeze", "macd_volume",
            "zscore_mean_reversion", "momentum_breakout",
            "gap_fade", "week52_low_bounce", "sector_rotation"
        ]
        selected_strategy = st.selectbox("Select Strategy", strategy_options, key="bt_strategy")

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date(2024, 1, 1), key="bt_start")
        with col2:
            end_date = st.date_input("End Date", date.today(), key="bt_end")

        st.markdown("---")
        st.subheader("Strategy Parameters")

        # Get current params and param ranges
        current_params = get_strategy_params(selected_strategy) if optimizer_available else {}
        param_ranges = get_param_ranges(selected_strategy) if optimizer_available else {}

        # Create parameter inputs based on strategy
        custom_params = {"enabled": True}

        if selected_strategy == "rsi_mean_reversion":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["rsi_period"] = st.slider("RSI Period", 5, 30, current_params.get("rsi_period", 14))
                custom_params["rsi_oversold"] = st.slider("RSI Oversold", 15, 40, current_params.get("rsi_oversold", 25))
            with col2:
                custom_params["rsi_overbought"] = st.slider("RSI Overbought", 60, 85, current_params.get("rsi_overbought", 75))
                custom_params["min_confidence"] = st.slider("Min Confidence", 50, 80, current_params.get("min_confidence", 65))
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 1.0, 10.0, current_params.get("stop_loss_pct", 3.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 3.0, 20.0, current_params.get("take_profit_pct", 8.0), 0.5)

        elif selected_strategy == "bollinger_squeeze":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["bb_period"] = st.slider("BB Period", 10, 30, current_params.get("bb_period", 20))
                custom_params["bb_std"] = st.slider("BB Std Dev", 1.0, 3.0, current_params.get("bb_std", 2.0), 0.1)
            with col2:
                custom_params["squeeze_threshold"] = st.slider("Squeeze Threshold", 0.01, 0.10, current_params.get("squeeze_threshold", 0.03), 0.01)
                custom_params["min_confidence"] = st.slider("Min Confidence", 50, 80, current_params.get("min_confidence", 65))
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 1.0, 10.0, current_params.get("stop_loss_pct", 3.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 5.0, 25.0, current_params.get("take_profit_pct", 12.0), 0.5)

        elif selected_strategy == "macd_volume":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["macd_fast"] = st.slider("MACD Fast", 5, 20, current_params.get("macd_fast", 12))
                custom_params["macd_slow"] = st.slider("MACD Slow", 15, 35, current_params.get("macd_slow", 26))
            with col2:
                custom_params["macd_signal"] = st.slider("MACD Signal", 5, 15, current_params.get("macd_signal", 9))
                custom_params["volume_threshold"] = st.slider("Volume Threshold", 1.0, 3.0, current_params.get("volume_threshold", 2.0), 0.1)
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 1.0, 10.0, current_params.get("stop_loss_pct", 4.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 5.0, 25.0, current_params.get("take_profit_pct", 15.0), 0.5)

        elif selected_strategy == "zscore_mean_reversion":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["lookback_period"] = st.slider("Lookback Period", 10, 40, current_params.get("lookback_period", 20))
                custom_params["zscore_entry"] = st.slider("Z-Score Entry", -4.0, -1.0, current_params.get("zscore_entry", -2.5), 0.1)
            with col2:
                custom_params["zscore_exit"] = st.slider("Z-Score Exit", -0.5, 1.0, current_params.get("zscore_exit", 0.5), 0.1)
                custom_params["min_confidence"] = st.slider("Min Confidence", 50, 80, current_params.get("min_confidence", 60))
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 1.0, 10.0, current_params.get("stop_loss_pct", 4.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 3.0, 20.0, current_params.get("take_profit_pct", 10.0), 0.5)

        elif selected_strategy == "momentum_breakout":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["lookback_days"] = st.slider("Lookback Days", 5, 30, current_params.get("lookback_days", 10))
                custom_params["breakout_threshold"] = st.slider("Breakout Threshold", 0.005, 0.05, current_params.get("breakout_threshold", 0.01), 0.005)
            with col2:
                custom_params["volume_confirmation"] = st.slider("Volume Confirmation", 1.0, 3.0, current_params.get("volume_confirmation", 1.8), 0.1)
                custom_params["min_confidence"] = st.slider("Min Confidence", 50, 80, current_params.get("min_confidence", 60))
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 1.0, 10.0, current_params.get("stop_loss_pct", 3.5), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 5.0, 25.0, current_params.get("take_profit_pct", 12.0), 0.5)

        elif selected_strategy == "gap_fade":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["gap_threshold_pct"] = st.slider("Gap Threshold %", 0.5, 3.0, current_params.get("gap_threshold_pct", 1.5), 0.1)
                custom_params["max_gap_pct"] = st.slider("Max Gap %", 3.0, 8.0, current_params.get("max_gap_pct", 5.0), 0.5)
            with col2:
                custom_params["volume_surge_threshold"] = st.slider("Volume Surge", 1.0, 2.5, current_params.get("volume_surge_threshold", 1.5), 0.1)
                custom_params["min_confidence"] = st.slider("Min Confidence", 50, 80, current_params.get("min_confidence", 60))
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 1.0, 6.0, current_params.get("stop_loss_pct", 3.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 2.0, 8.0, current_params.get("take_profit_pct", 4.0), 0.5)

        elif selected_strategy == "week52_low_bounce":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["low_threshold_pct"] = st.slider("Low Threshold %", 5.0, 20.0, current_params.get("low_threshold_pct", 10.0), 1.0)
                custom_params["bounce_threshold_pct"] = st.slider("Bounce Threshold %", 1.0, 5.0, current_params.get("bounce_threshold_pct", 2.0), 0.5)
            with col2:
                custom_params["volume_surge"] = st.slider("Volume Surge", 1.0, 2.0, current_params.get("volume_surge", 1.3), 0.1)
                custom_params["min_confidence"] = st.slider("Min Confidence", 50, 80, current_params.get("min_confidence", 60))
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 3.0, 12.0, current_params.get("stop_loss_pct", 6.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 8.0, 30.0, current_params.get("take_profit_pct", 15.0), 1.0)

        elif selected_strategy == "sector_rotation":
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_params["lookback_days"] = st.slider("Lookback Days", 10, 40, current_params.get("lookback_days", 20))
                custom_params["top_sectors"] = st.slider("Top Sectors", 1, 4, current_params.get("top_sectors", 2))
            with col2:
                custom_params["min_sector_return"] = st.slider("Min Sector Return %", 1.0, 5.0, current_params.get("min_sector_return", 2.0), 0.5)
                custom_params["relative_strength_threshold"] = st.slider("RS Threshold", 0.9, 1.5, current_params.get("relative_strength_threshold", 1.1), 0.05)
            with col3:
                custom_params["stop_loss_pct"] = st.slider("Stop Loss %", 2.0, 8.0, current_params.get("stop_loss_pct", 4.0), 0.5)
                custom_params["take_profit_pct"] = st.slider("Take Profit %", 5.0, 20.0, current_params.get("take_profit_pct", 10.0), 1.0)

        st.markdown("---")

        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner(f"Running backtest for {selected_strategy}..."):
                try:
                    # Get strategy class
                    strategy_class = STRATEGY_CLASSES[selected_strategy]
                    strategy = strategy_class(custom_params)

                    # Get tickers
                    universe_df = services["db"].fetchdf("SELECT ticker FROM universe WHERE is_active = 1 LIMIT 50")
                    tickers = universe_df["ticker"].tolist() if not universe_df.empty else ["AAPL", "MSFT", "GOOGL"]

                    # Run backtest
                    result = services["backtester"].run_backtest(
                        strategy=strategy,
                        tickers=tickers,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # Display results
                    st.success("Backtest complete!")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{result.total_return_pct:.2f}%")
                    col2.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                    col3.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
                    col4.metric("Win Rate", f"{result.win_rate:.1f}%")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Trades", result.total_trades)
                    col2.metric("Profit Factor", f"{result.profit_factor:.2f}")
                    col3.metric("Avg Trade P&L", f"${result.avg_trade_pnl:.2f}")
                    col4.metric("Avg Hold Days", f"{result.avg_hold_days:.1f}")

                except Exception as e:
                    st.error(f"Backtest failed: {e}")

    # TAB 3: Optimization
    with tab3:
        st.subheader("🎯 Hyperparameter Optimization")
        st.markdown("""
        Find optimal parameters that **maximize returns** while keeping **max drawdown below your limit**.
        """)

        if not optimizer_available:
            st.error("Optimizer not available. Check installation.")
            return

        # Settings
        col1, col2 = st.columns(2)
        with col1:
            opt_strategy = st.selectbox("Strategy to Optimize", strategy_options, key="opt_strategy")
            max_drawdown = st.slider("Max Allowed Drawdown %", 10.0, 50.0, 25.0, 1.0)

        with col2:
            objective = st.selectbox("Optimize For", ["sharpe", "return", "sortino"])
            max_iterations = st.slider("Max Iterations", 20, 200, 50, 10)

        col1, col2 = st.columns(2)
        with col1:
            opt_start = st.date_input("Backtest Start", date(2023, 1, 1), key="opt_start")
        with col2:
            opt_end = st.date_input("Backtest End", date.today(), key="opt_end")

        st.markdown("---")

        if st.button("🔍 Run Optimization", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()

            try:
                optimizer = StrategyOptimizer(max_drawdown_pct=max_drawdown)

                # Get tickers
                universe_df = services["db"].fetchdf("SELECT ticker FROM universe WHERE is_active = 1 LIMIT 30")
                tickers = universe_df["ticker"].tolist() if not universe_df.empty else ["AAPL", "MSFT", "GOOGL"]

                def progress_callback(current, total, result):
                    progress_bar.progress(current / total)
                    status_text.text(f"Testing combination {current}/{total} - Return: {result['total_return_pct']:.2f}%, Drawdown: {result['max_drawdown_pct']:.2f}%")

                result = optimizer.optimize(
                    strategy_name=opt_strategy,
                    tickers=tickers,
                    start_date=opt_start,
                    end_date=opt_end,
                    objective=objective,
                    max_iterations=max_iterations,
                    progress_callback=progress_callback
                )

                progress_bar.progress(1.0)
                status_text.text("Optimization complete!")

                # Display results
                st.success(f"✅ Found optimal parameters in {result.optimization_time_seconds:.1f} seconds")

                if result.constraint_satisfied:
                    st.info(f"Drawdown constraint satisfied: {abs(result.best_drawdown):.2f}% <= {max_drawdown}%")
                else:
                    st.warning(f"Could not satisfy {max_drawdown}% drawdown constraint. Best found: {abs(result.best_drawdown):.2f}%")

                col1, col2, col3 = st.columns(3)
                col1.metric("Best Return", f"{result.best_return:.2f}%")
                col2.metric("Best Sharpe", f"{result.best_sharpe:.2f}")
                col3.metric("Max Drawdown", f"{result.best_drawdown:.2f}%")

                st.markdown("### Optimal Parameters")
                params_df = pd.DataFrame([result.best_params]).T
                params_df.columns = ["Value"]
                st.dataframe(params_df, use_container_width=True)

                # Save button
                st.markdown("---")
                if st.button("💾 Save as Default Config"):
                    try:
                        optimizer.save_optimized_params({opt_strategy: result})
                        st.success(f"Saved optimized params to config/config.yaml")
                        st.info("Restart the app to use new parameters.")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

                # Show top 10 results
                st.markdown("### Top 10 Parameter Combinations")
                valid_results = [r for r in result.all_results if r.get("meets_constraint", False)]
                if valid_results:
                    valid_results.sort(key=lambda x: x["score"], reverse=True)
                    top_results = valid_results[:10]

                    top_df = pd.DataFrame([{
                        "Return %": r["total_return_pct"],
                        "Sharpe": r["sharpe_ratio"],
                        "Drawdown %": r["max_drawdown_pct"],
                        "Win Rate %": r["win_rate"],
                        "Trades": r["total_trades"],
                    } for r in top_results])
                    st.dataframe(top_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_watchlist_page(services: dict):
    """Render Long-Term Watchlist page."""
    st.title("📋 Long-Term Watchlist")

    # Scoring Methodology Section
    with st.expander("📊 Scoring Methodology (click to expand)"):
        st.markdown("""
        ### How Stocks Are Scored

        Each stock receives a **Composite Score (0-100)** based on weighted factors:

        | Factor | Weight | Description |
        |--------|--------|-------------|
        | **Valuation** | 18% | P/E and P/B percentile vs history |
        | **Technical** | 10% | RSI, price vs moving averages |
        | **Dividend** | 10% | Yield and payout sustainability |
        | **Quality** | 15% | ROE, debt ratios, margins |
        | **Insider Activity** | 15% | Recent insider buying/selling |
        | **FCF Yield** | 12% | Free cash flow yield vs market |
        | **Earnings Momentum** | 10% | EPS beat streak |
        | **Peer Valuation** | 10% | Valuation vs sector peers |

        **Score Interpretation:**
        - 🔥 **70+** = Strong Buy candidate
        - ✅ **60-69** = Good opportunity
        - ➡️ **50-59** = Neutral / watchlist
        - ⚠️ **<50** = Needs improvement
        """)

    try:
        # Get only the latest entry per ticker (dedupe)
        watchlist_df = services["db"].fetchdf("""
            SELECT w.* FROM long_term_watchlist w
            INNER JOIN (
                SELECT ticker, MAX(scan_date) as max_date
                FROM long_term_watchlist
                WHERE scan_date >= date('now', '-7 days')
                GROUP BY ticker
            ) latest ON w.ticker = latest.ticker AND w.scan_date = latest.max_date
            ORDER BY w.composite_score DESC
        """)
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        watchlist_df = pd.DataFrame()

    if not watchlist_df.empty:
        st.markdown("---")
        st.subheader("Top Long-Term Opportunities")

        # Add trend symbols by getting trend data for each ticker
        from stockpulse.scanner.long_term_scanner import LongTermScanner
        scanner = LongTermScanner()

        # Convert to list of dicts, enrich with trends, convert back
        opps = watchlist_df.to_dict('records')
        opps = scanner.enrich_with_trends(opps)

        # Load sentiment data
        sentiment_data = {}
        try:
            from stockpulse.data.sentiment import SentimentStorage
            storage = SentimentStorage()
            lt_tickers = [o.get("ticker") for o in opps if o.get("ticker")]
            sentiment_data = storage.get_todays_sentiment(lt_tickers)
        except Exception:
            pass  # Sentiment optional

        # Add sentiment to each opportunity
        for opp in opps:
            opp["sentiment"] = format_sentiment(sentiment_data, opp.get("ticker", ""))

        watchlist_df = pd.DataFrame(opps)

        # Create formatted trend column matching email format: "📈 11d (+2.4)"
        def format_trend(row):
            if row.get('is_new', False):
                return "🆕 New"
            trend = row.get('trend_symbol', '➡️')
            days = row.get('consecutive_days', 0)
            change_5d = row.get('change_vs_5d_avg', 0)
            sign = "+" if change_5d >= 0 else ""
            return f"{trend} {days}d ({sign}{change_5d:.1f})"

        watchlist_df['trend_formatted'] = watchlist_df.apply(format_trend, axis=1)

        # Better display with trend, company name, sector, sentiment, and price info
        display_cols = [
            "trend_formatted", "ticker", "company_name", "sector", "composite_score",
            "sentiment", "current_price", "week52_low", "week52_high", "price_vs_52w_low_pct"
        ]
        available_cols = [c for c in display_cols if c in watchlist_df.columns]
        display_df = watchlist_df[available_cols].copy()

        # Format numeric columns
        if "composite_score" in display_df.columns:
            display_df["composite_score"] = display_df["composite_score"].round(0).astype(int)
        if "current_price" in display_df.columns:
            display_df["current_price"] = display_df["current_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
        if "week52_low" in display_df.columns:
            display_df["week52_low"] = display_df["week52_low"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
        if "week52_high" in display_df.columns:
            display_df["week52_high"] = display_df["week52_high"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
        if "price_vs_52w_low_pct" in display_df.columns:
            display_df["price_vs_52w_low_pct"] = display_df["price_vs_52w_low_pct"].round(1).astype(str) + "%"

        # Rename columns for display
        col_rename = {
            "trend_formatted": "Trend",
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "composite_score": "Score",
            "sentiment": "Sentiment",
            "current_price": "Price",
            "week52_low": "52W Low",
            "week52_high": "52W High",
            "price_vs_52w_low_pct": "vs Low"
        }
        display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Trend legend
        st.caption("**Trend:** 📈 Strengthening (vs 5d avg) | 📉 Weakening | ➡️ Stable | 🆕 New • **Xd** = days on list • **(+X.X)** = score vs 5-day average")

        st.markdown("---")
        selected_ticker = st.selectbox("Select ticker for details", watchlist_df["ticker"].unique())

        if selected_ticker:
            ticker_data = watchlist_df[watchlist_df["ticker"] == selected_ticker].iloc[0]

            # Company header
            company_name = ticker_data.get('company_name', selected_ticker)
            sector = ticker_data.get('sector', 'Unknown')
            st.subheader(f"{selected_ticker} - {company_name}")
            st.caption(f"Sector: {sector}")

            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score = ticker_data.get('composite_score', 0)
                emoji = "🔥" if score >= 70 else "✅" if score >= 60 else "➡️"
                st.metric("Composite Score", f"{emoji} {score:.0f}")
            with col2:
                st.metric("P/E Percentile", f"{ticker_data.get('pe_percentile', 0):.0f}%")
            with col3:
                st.metric("vs 52W Low", f"{ticker_data.get('price_vs_52w_low_pct', 0):.1f}%")
            with col4:
                st.metric("Scan Date", str(ticker_data.get('scan_date', 'N/A'))[:10])

            # Score breakdown table
            st.markdown("#### Score Breakdown")
            score_data = {
                "Factor": ["Valuation", "Technical", "Dividend", "Quality", "Insider", "FCF Yield", "Earnings", "Peer Val"],
                "Weight": ["18%", "10%", "10%", "15%", "15%", "12%", "10%", "10%"],
                "Score": [
                    f"{ticker_data.get('valuation_score', 0):.0f}",
                    f"{ticker_data.get('technical_score', 0):.0f}",
                    f"{ticker_data.get('dividend_score', 0):.0f}",
                    f"{ticker_data.get('quality_score', 0):.0f}",
                    f"{ticker_data.get('insider_score', 0):.0f}",
                    f"{ticker_data.get('fcf_score', 0):.0f}",
                    f"{ticker_data.get('earnings_score', 0):.0f}",
                    f"{ticker_data.get('peer_score', 0):.0f}",
                ]
            }
            score_df = pd.DataFrame(score_data)
            st.dataframe(score_df, use_container_width=True, hide_index=True)

            # Reasoning
            st.markdown("#### Analysis")
            st.info(ticker_data.get('reasoning', 'N/A'))

            # Price chart
            st.markdown("#### Price History (1 Year)")
            price_data = services["ingestion"].get_daily_prices(
                [selected_ticker],
                start_date=date.today() - timedelta(days=365)
            )

            if not price_data.empty:
                fig = create_price_chart(price_data, selected_ticker, show_volume=True)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No long-term opportunities identified yet. The scanner runs daily after market close.")


def render_longterm_holdings_page(services: dict):
    """Render Long-Term Holdings page for tracking actual purchases."""
    st.title("📊 Long-Term Holdings Tracker")
    st.markdown("Track your actual long-term investments and compare to signals.")

    # Initialize tracker
    try:
        from stockpulse.tracker.holdings_tracker import HoldingsTracker
        tracker = HoldingsTracker()
    except Exception as e:
        st.error(f"Error loading holdings tracker: {e}")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Current Holdings", "➕ Add Position", "📜 Closed Positions"])

    # TAB 1: Current Holdings
    with tab1:
        st.subheader("Portfolio Summary")

        holdings_df = tracker.get_holdings_with_current_value("long_term")

        if not holdings_df.empty:
            # Calculate totals
            total_cost = holdings_df["cost_basis"].sum()
            total_value = holdings_df["current_value"].sum()
            total_pnl = holdings_df["unrealized_pnl"].sum()
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Positions", len(holdings_df))
            with col2:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col3:
                st.metric("Current Value", f"${total_value:,.2f}")
            with col4:
                delta_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Unrealized P&L", f"${total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.1f}%")

            # Holdings table
            st.markdown("---")
            st.subheader("Individual Holdings")

            display_data = []
            for _, row in holdings_df.iterrows():
                pnl_emoji = "🟢" if row["unrealized_pnl_pct"] >= 0 else "🔴"
                display_data.append({
                    "Status": pnl_emoji,
                    "Ticker": row["ticker"],
                    "Shares": f"{row['shares']:.2f}",
                    "Buy Price": f"${row['buy_price']:.2f}",
                    "Current": f"${row['current_price']:.2f}",
                    "Cost": f"${row['cost_basis']:,.2f}",
                    "Value": f"${row['current_value']:,.2f}",
                    "P&L": f"${row['unrealized_pnl']:+,.2f}",
                    "P&L %": f"{row['unrealized_pnl_pct']:+.1f}%",
                    "Buy Date": str(row["buy_date"])[:10] if row.get("buy_date") else "N/A",
                })

            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            # Close position section
            st.markdown("---")
            st.subheader("Close a Position")

            col1, col2, col3 = st.columns(3)
            with col1:
                close_id = st.selectbox(
                    "Select Position",
                    options=holdings_df["id"].tolist(),
                    format_func=lambda x: f"#{x} - {holdings_df[holdings_df['id']==x]['ticker'].values[0]}"
                )
            with col2:
                sell_price = st.number_input("Sell Price", min_value=0.01, step=0.01)
            with col3:
                sell_date = st.date_input("Sell Date", date.today())

            if st.button("Close Position", type="secondary"):
                if close_id and sell_price > 0:
                    try:
                        result = tracker.close_holding(close_id, sell_date, sell_price)
                        st.success(f"Closed {result['ticker']}: P&L ${result['realized_pnl']:+,.2f} ({result['realized_pnl_pct']:+.1f}%)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter a valid sell price")

        else:
            st.info("No long-term holdings yet. Use the 'Add Position' tab to record your purchases.")

    # TAB 2: Add Position
    with tab2:
        st.subheader("Record a New Purchase")

        col1, col2 = st.columns(2)
        with col1:
            new_ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL").upper()
            new_shares = st.number_input("Shares", min_value=0.01, step=0.01)
            new_buy_date = st.date_input("Purchase Date", date.today())

        with col2:
            new_buy_price = st.number_input("Buy Price per Share", min_value=0.01, step=0.01)
            new_strategy = st.selectbox("Strategy", ["long_term", "active"])
            new_notes = st.text_input("Notes (optional)", placeholder="Reason for purchase")

        # Preview
        if new_ticker and new_shares and new_buy_price:
            cost = new_shares * new_buy_price
            st.markdown("---")
            st.markdown(f"**Preview:** Buy {new_shares} shares of **{new_ticker}** @ ${new_buy_price:.2f} = **${cost:,.2f}**")

        if st.button("Add Position", type="primary"):
            if new_ticker and new_shares > 0 and new_buy_price > 0:
                try:
                    holding_id = tracker.add_holding(
                        ticker=new_ticker,
                        buy_date=new_buy_date,
                        buy_price=new_buy_price,
                        shares=new_shares,
                        strategy_type=new_strategy,
                        notes=new_notes if new_notes else None
                    )
                    st.success(f"Added position #{holding_id}: {new_shares} shares of {new_ticker}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding position: {e}")
            else:
                st.warning("Please fill in all required fields")

    # TAB 3: Closed Positions
    with tab3:
        st.subheader("Closed Positions History")

        closed_df = tracker.get_closed_holdings("long_term")

        if not closed_df.empty:
            # Summary stats
            total_realized = closed_df["realized_pnl"].sum()
            total_closed_trades = len(closed_df)
            wins = len(closed_df[closed_df["realized_pnl"] > 0])
            win_rate = (wins / total_closed_trades * 100) if total_closed_trades > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Closed Trades", total_closed_trades)
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Total Realized", f"${total_realized:+,.2f}")
            with col4:
                avg_pnl = closed_df["realized_pnl_pct"].mean()
                st.metric("Avg Return", f"{avg_pnl:+.1f}%")

            # Closed positions table
            st.markdown("---")
            display_cols = ["ticker", "buy_date", "sell_date", "buy_price", "sell_price",
                          "shares", "realized_pnl", "realized_pnl_pct"]
            available = [c for c in display_cols if c in closed_df.columns]
            display_df = closed_df[available].copy()

            if "buy_price" in display_df.columns:
                display_df["buy_price"] = display_df["buy_price"].apply(lambda x: f"${x:.2f}")
            if "sell_price" in display_df.columns:
                display_df["sell_price"] = display_df["sell_price"].apply(lambda x: f"${x:.2f}" if x else "N/A")
            if "realized_pnl" in display_df.columns:
                display_df["realized_pnl"] = display_df["realized_pnl"].apply(lambda x: f"${x:+,.2f}" if x else "N/A")
            if "realized_pnl_pct" in display_df.columns:
                display_df["realized_pnl_pct"] = display_df["realized_pnl_pct"].apply(lambda x: f"{x:+.1f}%" if x else "N/A")

            display_df = display_df.rename(columns={
                "ticker": "Ticker",
                "buy_date": "Buy Date",
                "sell_date": "Sell Date",
                "buy_price": "Buy",
                "sell_price": "Sell",
                "shares": "Shares",
                "realized_pnl": "P&L",
                "realized_pnl_pct": "P&L %"
            })

            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No closed positions yet.")


def render_ai_stocks_page(services: dict):
    """Render AI Stocks page showing AI universe stocks with scores."""
    st.title("🤖 AI Stocks")
    st.markdown("AI universe stocks ranked by opportunity score. Pullbacks + oversold = higher scores.")

    # Market Mood Banner (Fear & Greed Index)
    try:
        from stockpulse.data.sentiment import SentimentStorage as _FGStorage
        _fg_storage = _FGStorage()
        _fg_data = _fg_storage.get_signals(["_MARKET"]).get("_MARKET", {}).get("fear_greed", {})
        _fg_score = _fg_data.get("data", {}).get("score", 0) if _fg_data else 0
        _fg_rating = _fg_data.get("data", {}).get("rating", "") if _fg_data else ""
        if _fg_score > 0:
            if _fg_score >= 75:
                _fg_color, _fg_emoji = "#22c55e", "🟢"
            elif _fg_score >= 55:
                _fg_color, _fg_emoji = "#fbbf24", "🟡"
            elif _fg_score >= 25:
                _fg_color, _fg_emoji = "#f97316", "🟠"
            else:
                _fg_color, _fg_emoji = "#ef4444", "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #1e293b, #0f172a); padding: 10px 20px; border-radius: 8px;
                        border: 1px solid #334155; margin-bottom: 15px; display: flex; align-items: center; justify-content: center; gap: 15px;">
                <span style="color: #94a3b8; font-size: 13px;">Market Mood</span>
                <span style="color: {_fg_color}; font-weight: bold; font-size: 18px;">{_fg_emoji} {_fg_score:.0f}</span>
                <span style="color: {_fg_color}; font-size: 13px;">{_fg_rating}</span>
                <span style="color: #475569; font-size: 11px;">CNN Fear & Greed Index</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass

    # Scoring methodology
    with st.expander("📊 Scoring Methodology (click to expand)"):
        st.markdown("""
        ### How AI Stocks Are Scored

        Each stock receives an **AI Score (0-100)** based on weighted factors:

        | Factor | Points | Description |
        |--------|--------|-------------|
        | **Base** | 50 | Starting point |
        | **30d Performance** | ±20 | Pullbacks score higher (-20%: +20, +30%: -15) |
        | **90d Performance** | ±15 | Medium-term trend |
        | **AI Category** | +10 | Infra: +10, Hyperscaler: +8, Software/Supply Chain: +7, Robotics/Biotech: +6, Space/Crypto: +5 |
        | **RSI (14)** | ±15 | Oversold <30: +15, Overbought >70: -5 |
        | **50-Day MA** | ±10 | Below MA = opportunity |
        | **Valuation** | ±10 | PEG <1: +10, P/E <20: +8 |

        **Score Interpretation:**
        - 🔥 **75+** = Strong Buy
        - ✅ **65-74** = Buy
        - ➡️ **55-64** = Hold
        - ⚠️ **<55** = Wait for pullback
        """)

    # Load AI stocks from database (no auto-scan)
    try:
        from stockpulse.scanner.ai_pulse import AIPulseScanner
        scanner = AIPulseScanner()

        # Try to get cached data first
        ai_stocks, scan_timestamp = scanner.get_cached_ai_stocks(max_age_hours=48)

        # Show data freshness and refresh option
        col_fresh, col_refresh = st.columns([3, 1])
        with col_fresh:
            if scan_timestamp:
                from datetime import datetime
                try:
                    scan_dt = datetime.fromisoformat(scan_timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
                    age = datetime.now() - scan_dt
                    hours_ago = age.total_seconds() / 3600
                    if hours_ago < 1:
                        freshness = f"{int(age.total_seconds() / 60)} minutes ago"
                    elif hours_ago < 24:
                        freshness = f"{hours_ago:.1f} hours ago"
                    else:
                        freshness = f"{hours_ago / 24:.1f} days ago"
                    st.caption(f"📅 Data from: {freshness}")
                except Exception:
                    st.caption(f"📅 Last scan: {scan_timestamp[:16]}")
            else:
                st.caption("📅 No cached data available")

        with col_refresh:
            if st.button("🔄 Refresh Data", key="refresh_ai_stocks"):
                with st.spinner("Scanning AI universe stocks..."):
                    ai_stocks = scanner.get_ai_stocks()
                    scanner.save_ai_stocks(ai_stocks)
                st.rerun()

        if not ai_stocks:
            st.warning("No AI stocks data. Run `stockpulse ai-scan` or click 'Refresh Data' to populate.")
            return

    except Exception as e:
        st.error(f"Error loading AI stocks: {e}")
        return

    # Load sentiment data
    sentiment_data = {}
    try:
        from stockpulse.data.sentiment import SentimentStorage
        storage = SentimentStorage()
        ai_tickers = [s.get("ticker") for s in ai_stocks if s.get("ticker")]
        sentiment_data = storage.get_todays_sentiment(ai_tickers)
    except Exception:
        pass

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("AI Stocks", len(ai_stocks))
    with col2:
        high_score = len([s for s in ai_stocks if s.get("ai_score", 0) >= 65])
        st.metric("High Score (65+)", high_score)
    with col3:
        avg_30d = sum(s.get("pct_30d", 0) for s in ai_stocks) / len(ai_stocks) if ai_stocks else 0
        st.metric("Avg 30d Return", f"{avg_30d:+.1f}%")
    with col4:
        best = ai_stocks[0] if ai_stocks else None
        st.metric("Top Pick", best.get("ticker", "N/A") if best else "N/A")

    # Sentiment Sources Overview
    with st.expander("📡 Sentiment Sources Overview", expanded=False):
        _source_cols = st.columns(5)
        _st_count = sum(1 for t, d in sentiment_data.items() if d.get("stocktwits", {}).get("total_messages", 0) > 0)
        _reddit_count = sum(1 for t, d in sentiment_data.items() if d.get("reddit", {}).get("total_messages", 0) > 0)
        _news_count = sum(1 for t, d in sentiment_data.items() if d.get("google_news", {}).get("total_messages", 0) > 0)
        with _source_cols[0]:
            st.metric("StockTwits", f"{_st_count} stocks")
        with _source_cols[1]:
            st.metric("Reddit", f"{_reddit_count} stocks")
        with _source_cols[2]:
            st.metric("Google News", f"{_news_count} stocks")
        with _source_cols[3]:
            # Load signals to check for analyst/insider data
            try:
                _sig_storage = SentimentStorage() if 'SentimentStorage' not in dir() else storage
                _signals = _sig_storage.get_signals([s.get("ticker") for s in ai_stocks[:20]])
                _analyst_count = sum(1 for t, d in _signals.items() if "analyst_rating" in d)
                st.metric("Analyst Ratings", f"{_analyst_count} stocks")
            except Exception:
                st.metric("Analyst Ratings", "—")
        with _source_cols[4]:
            try:
                _insider_count = sum(1 for t, d in _signals.items() if "insider_txn" in d)
                st.metric("Insider Data", f"{_insider_count} stocks")
            except Exception:
                st.metric("Insider Data", "—")

    # Top picks (Score 65+, pullback)
    st.markdown("---")
    st.subheader("🎯 Top AI Stock Picks")
    st.caption("High AI score (65+) + Recent pullback = potential opportunity")

    top_picks = [s for s in ai_stocks if s.get("ai_score", 0) >= 65 and s.get("pct_30d", 0) <= 0][:5]

    if top_picks:
        cols = st.columns(min(len(top_picks), 5))
        for idx, stock in enumerate(top_picks):
            with cols[idx]:
                ticker = stock.get("ticker", "N/A")
                score = stock.get("ai_score", 0)
                price = stock.get("current_price", 0)
                pct_30d = stock.get("pct_30d", 0)
                sent_display = format_sentiment(sentiment_data, ticker)

                st.markdown(f"""
                <div style="background: #f5f3ff; padding: 15px; border-radius: 8px; border: 2px solid #7c3aed; text-align: center;">
                    <h2 style="margin: 0; color: #6d28d9;">{ticker}</h2>
                    <p style="font-size: 24px; margin: 10px 0; color: #7c3aed;">{score:.0f}</p>
                    <p style="color: #6b7280; font-size: 14px;">${price:.2f}</p>
                    <p style="color: #22c55e; font-size: 13px;">{pct_30d:+.1f}% (30d)</p>
                    <p style="font-size: 12px;">{sent_display}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No strong picks right now. AI stocks may be extended - wait for a pullback.")

    # Category breakdown
    st.markdown("---")
    st.subheader("📁 Category Performance")

    # Group by category
    categories = {}
    for stock in ai_stocks:
        cat = stock.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(stock)

    cat_rows = []
    for cat, stocks in sorted(categories.items(), key=lambda x: -len(x[1])):
        if not stocks:
            continue
        avg_score = sum(s.get("ai_score", 0) for s in stocks) / len(stocks)
        avg_30d = sum(s.get("pct_30d", 0) for s in stocks) / len(stocks)
        top_pick = max(stocks, key=lambda s: s.get("ai_score", 0))

        # Calculate category sentiment
        cat_sent_scores = []
        for s in stocks:
            ticker = s.get("ticker", "")
            sent = sentiment_data.get(ticker, {})
            if sent.get("aggregate_score", 0) > 0:
                cat_sent_scores.append(sent.get("aggregate_score", 50))

        if cat_sent_scores:
            avg_sent = sum(cat_sent_scores) / len(cat_sent_scores)
            if avg_sent >= 60:
                sent_display = f"🟢 {avg_sent:.0f}"
            elif avg_sent <= 40:
                sent_display = f"🔴 {avg_sent:.0f}"
            else:
                sent_display = f"🟡 {avg_sent:.0f}"
        else:
            sent_display = "—"

        cat_rows.append({
            "Category": cat,
            "Stocks": len(stocks),
            "Avg AI Score": f"{avg_score:.0f}",
            "Sentiment": sent_display,
            "Avg 30d": f"{avg_30d:+.1f}%",
            "Top Pick": top_pick.get("ticker", "N/A")
        })

    if cat_rows:
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    # Reddit Buzz Section
    _reddit_buzz = []
    for ticker, sent in sentiment_data.items():
        reddit_data = sent.get("reddit", {})
        if reddit_data and reddit_data.get("total_messages", 0) > 0:
            _reddit_buzz.append({
                "Ticker": ticker,
                "Mentions": reddit_data.get("total_messages", 0),
                "Sentiment": f"{'🟢' if reddit_data.get('sentiment_label') == 'bullish' else '🔴' if reddit_data.get('sentiment_label') == 'bearish' else '🟡'} {reddit_data.get('sentiment_score', 50):.0f}",
                "Trending": "📈" if reddit_data.get("trending") else "",
                "Velocity": f"{reddit_data.get('message_velocity', 0):.1f}/hr",
            })

    if _reddit_buzz:
        st.markdown("---")
        st.subheader("🔥 Reddit Buzz")
        st.caption("Most-discussed AI stocks on Reddit today (r/wallstreetbets, r/stocks, r/investing)")
        _reddit_buzz.sort(key=lambda x: x["Mentions"], reverse=True)
        st.dataframe(pd.DataFrame(_reddit_buzz[:10]), use_container_width=True, hide_index=True)

    # Social Sentiment Summary section
    st.markdown("---")
    st.subheader("📊 Social Sentiment Summary")
    st.caption("Sentiment across all sources - sorted by score. AI analysis available for top movers.")

    # Get sentiment with details for display
    sentiment_with_details = []
    for ticker, sent in sentiment_data.items():
        if sent.get("aggregate_score", 0) > 0:
            st_data = sent.get("stocktwits", {})
            ai_data = sent.get("ai_analysis", {})
            sentiment_with_details.append({
                "ticker": ticker,
                "score": sent.get("aggregate_score", 50),
                "label": sent.get("aggregate_label", "neutral"),
                "bullish": st_data.get("bullish_count", 0),
                "bearish": st_data.get("bearish_count", 0),
                "total": st_data.get("total_messages", 0),
                "trending": st_data.get("trending", False),
                "ai_summary": ai_data.get("summary", "") if ai_data else "",
                "sample_messages": st_data.get("sample_messages", [])
            })

    if sentiment_with_details:
        # Sort by score
        sentiment_with_details.sort(key=lambda x: x["score"], reverse=True)

        # Show top bullish and bearish
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Most Bullish**")
            bullish = [s for s in sentiment_with_details if s["label"] == "bullish"][:5]
            if bullish:
                for s in bullish:
                    trending = " 📈" if s["trending"] else ""
                    st.markdown(f"**{s['ticker']}** - Score: {s['score']:.0f}{trending}")
                    st.caption(f"↑{s['bullish']} ↓{s['bearish']} ({s['total']} messages)")

                    # Show AI analysis and sample quotes
                    has_content = s["ai_summary"] or s.get("sample_messages")
                    if has_content:
                        with st.expander("Details"):
                            if s["ai_summary"]:
                                st.markdown("**AI Analysis:**")
                                st.write(s["ai_summary"][:500])
                            if s.get("sample_messages"):
                                st.markdown("**Sample Posts:**")
                                for msg in s["sample_messages"][:3]:
                                    sentiment_icon = "🟢" if msg.get("sentiment") == "Bullish" else "🔴" if msg.get("sentiment") == "Bearish" else "🟡"
                                    st.caption(f"{sentiment_icon} \"{msg.get('text', '')[:120]}...\"")
            else:
                st.caption("No bullish stocks found")

        with col2:
            st.markdown("**🔴 Most Bearish**")
            bearish = [s for s in sentiment_with_details if s["label"] == "bearish"][:5]
            if bearish:
                for s in bearish:
                    trending = " 📈" if s["trending"] else ""
                    st.markdown(f"**{s['ticker']}** - Score: {s['score']:.0f}{trending}")
                    st.caption(f"↑{s['bullish']} ↓{s['bearish']} ({s['total']} messages)")

                    # Show AI analysis and sample quotes
                    has_content = s["ai_summary"] or s.get("sample_messages")
                    if has_content:
                        with st.expander("Details"):
                            if s["ai_summary"]:
                                st.markdown("**AI Analysis:**")
                                st.write(s["ai_summary"][:500])
                            if s.get("sample_messages"):
                                st.markdown("**Sample Posts:**")
                                for msg in s["sample_messages"][:3]:
                                    sentiment_icon = "🟢" if msg.get("sentiment") == "Bullish" else "🔴" if msg.get("sentiment") == "Bearish" else "🟡"
                                    st.caption(f"{sentiment_icon} \"{msg.get('text', '')[:120]}...\"")
            else:
                st.caption("No bearish stocks found")

        # Full sentiment table
        with st.expander("📋 Full Sentiment Data (click to expand)"):
            sent_df = pd.DataFrame(sentiment_with_details)
            sent_df = sent_df[["ticker", "score", "label", "bullish", "bearish", "total", "trending"]]
            sent_df["score"] = sent_df["score"].round(0).astype(int)
            sent_df["label"] = sent_df["label"].apply(lambda x: f"{'🟢' if x == 'bullish' else '🔴' if x == 'bearish' else '🟡'} {x.upper()}")
            sent_df["trending"] = sent_df["trending"].apply(lambda x: "📈 Yes" if x else "")
            sent_df = sent_df.rename(columns={
                "ticker": "Ticker",
                "score": "Score",
                "label": "Sentiment",
                "bullish": "↑ Bullish",
                "bearish": "↓ Bearish",
                "total": "Total",
                "trending": "Trending"
            })
            st.dataframe(sent_df, use_container_width=True, hide_index=True)

        st.caption("**Score:** 0-100 (50=neutral) | **Trending:** High message velocity on StockTwits")
    else:
        st.info("No sentiment data available. Run `stockpulse sentiment-scan` to fetch social sentiment.")

    # All AI stocks table
    st.markdown("---")
    st.subheader("🤖 All AI Universe Stocks")

    # Add sentiment to stocks
    for stock in ai_stocks:
        stock["sentiment"] = format_sentiment(sentiment_data, stock.get("ticker", ""))

    display_df = pd.DataFrame(ai_stocks)

    # Select columns for display
    display_cols = ["ticker", "company_name", "category", "current_price", "ai_score", "sentiment", "pct_30d", "pct_90d"]
    available_cols = [c for c in display_cols if c in display_df.columns]
    display_df = display_df[available_cols].copy()

    # Format columns
    if "current_price" in display_df.columns:
        display_df["current_price"] = display_df["current_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    if "ai_score" in display_df.columns:
        display_df["ai_score"] = display_df["ai_score"].round(0).astype(int)
    if "pct_30d" in display_df.columns:
        display_df["pct_30d"] = display_df["pct_30d"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    if "pct_90d" in display_df.columns:
        display_df["pct_90d"] = display_df["pct_90d"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")

    display_df = display_df.rename(columns={
        "ticker": "Ticker",
        "company_name": "Company",
        "category": "Category",
        "current_price": "Price",
        "ai_score": "AI Score",
        "sentiment": "Sentiment",
        "pct_30d": "30d",
        "pct_90d": "90d"
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("**Score:** 75+ = Strong Buy | 65-74 = Buy | 55-64 = Hold | <55 = Wait")
    st.caption("**Sentiment:** 🟢 Bullish | 🔴 Bearish | 🟡 Neutral (from StockTwits)")

    # Detail view for selected ticker
    st.markdown("---")
    st.subheader("📊 Stock Details")

    ticker_list = [s.get("ticker") for s in ai_stocks if s.get("ticker")]
    selected_ticker = st.selectbox("Select ticker for details", ticker_list)

    if selected_ticker:
        stock_data = next((s for s in ai_stocks if s.get("ticker") == selected_ticker), None)
        if stock_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score = stock_data.get('ai_score', 0)
                score_label = "Strong Buy" if score >= 75 else "Buy" if score >= 65 else "Hold" if score >= 55 else "Wait"
                st.metric("AI Score", f"{score:.0f} ({score_label})")
            with col2:
                st.metric("Price", f"${stock_data.get('current_price', 0):.2f}")
            with col3:
                st.metric("30d Return", f"{stock_data.get('pct_30d', 0):+.1f}%")
            with col4:
                st.metric("Category", stock_data.get('category', 'Unknown'))

            # Score breakdown
            st.markdown("#### Score Breakdown")
            breakdown = stock_data.get("score_breakdown", {})
            if breakdown:
                _breakdown_col1, _breakdown_col2 = st.columns([1, 1])
                with _breakdown_col1:
                    breakdown_rows = []
                    for factor, data in breakdown.items():
                        if factor != "total":
                            breakdown_rows.append({
                                "Factor": data.get("label", factor),
                                "Points": f"{data.get('points', 0):+d}",
                                "Value": data.get("raw_value", "—")
                            })
                    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

                # Score radar chart
                with _breakdown_col2:
                    try:
                        from stockpulse.dashboard.charts import create_score_radar_chart
                        _radar_scores = {}
                        for factor, data in breakdown.items():
                            if factor != "total":
                                _radar_scores[data.get("label", factor)] = abs(data.get("points", 0))
                        if _radar_scores:
                            _radar_fig = create_score_radar_chart(_radar_scores, selected_ticker)
                            st.plotly_chart(_radar_fig, use_container_width=True)
                    except Exception:
                        pass

            # Sentiment Analysis section for selected stock
            st.markdown("#### Social Sentiment")
            ticker_sent = sentiment_data.get(selected_ticker, {})
            if ticker_sent.get("aggregate_score", 0) > 0:
                sent_score = ticker_sent.get("aggregate_score", 50)
                sent_label = ticker_sent.get("aggregate_label", "neutral")
                st_data = ticker_sent.get("stocktwits", {})
                reddit_data = ticker_sent.get("reddit", {})
                gnews_data = ticker_sent.get("google_news", {})
                ai_data = ticker_sent.get("ai_analysis", {})

                # Sentiment metrics
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1:
                    emoji = "🟢" if sent_label == "bullish" else ("🔴" if sent_label == "bearish" else "🟡")
                    st.metric("Sentiment Score", f"{emoji} {sent_score:.0f}")
                with scol2:
                    _total_bull = st_data.get("bullish_count", 0) + (reddit_data.get("bullish_count", 0) if reddit_data else 0)
                    st.metric("Bullish Messages", _total_bull)
                with scol3:
                    _total_bear = st_data.get("bearish_count", 0) + (reddit_data.get("bearish_count", 0) if reddit_data else 0)
                    st.metric("Bearish Messages", _total_bear)
                with scol4:
                    _is_trending = st_data.get("trending", False) or (reddit_data.get("trending", False) if reddit_data else False)
                    trending = "📈 Yes" if _is_trending else "No"
                    st.metric("Trending", trending)

                # Per-source breakdown
                with st.expander("📡 Source Breakdown"):
                    _src_rows = []
                    if st_data and st_data.get("total_messages", 0) > 0:
                        _src_rows.append({"Source": "StockTwits", "Score": f"{st_data.get('sentiment_score', 50):.0f}",
                                          "Bull": st_data.get("bullish_count", 0), "Bear": st_data.get("bearish_count", 0),
                                          "Total": st_data.get("total_messages", 0), "Trending": "📈" if st_data.get("trending") else ""})
                    if reddit_data and reddit_data.get("total_messages", 0) > 0:
                        _src_rows.append({"Source": "Reddit", "Score": f"{reddit_data.get('sentiment_score', 50):.0f}",
                                          "Bull": reddit_data.get("bullish_count", 0), "Bear": reddit_data.get("bearish_count", 0),
                                          "Total": reddit_data.get("total_messages", 0), "Trending": "📈" if reddit_data.get("trending") else ""})
                    if gnews_data and gnews_data.get("total_messages", 0) > 0:
                        _src_rows.append({"Source": "Google News", "Score": f"{gnews_data.get('sentiment_score', 50):.0f}",
                                          "Bull": gnews_data.get("bullish_count", 0), "Bear": gnews_data.get("bearish_count", 0),
                                          "Total": gnews_data.get("total_messages", 0), "Trending": ""})
                    # Load signals for this ticker
                    try:
                        _det_signals = storage.get_signals([selected_ticker]).get(selected_ticker, {})
                        _analyst = _det_signals.get("analyst_rating", {}).get("data", {})
                        if _analyst and _analyst.get("total_analysts", 0) > 0:
                            _src_rows.append({"Source": "Analyst Ratings", "Score": f"{_analyst.get('consensus_score', 50):.0f}",
                                              "Bull": _analyst.get("buy", 0) + _analyst.get("strong_buy", 0),
                                              "Bear": _analyst.get("sell", 0) + _analyst.get("strong_sell", 0),
                                              "Total": _analyst.get("total_analysts", 0), "Trending": ""})
                        _insider = _det_signals.get("insider_txn", {}).get("data", {})
                        if _insider and _insider.get("total_transactions", 0) > 0:
                            _src_rows.append({"Source": "Insider Activity", "Score": f"{_insider.get('insider_score', 50):.0f}",
                                              "Bull": _insider.get("buy_transactions", 0),
                                              "Bear": _insider.get("sell_transactions", 0),
                                              "Total": _insider.get("total_transactions", 0), "Trending": ""})
                    except Exception:
                        pass
                    if _src_rows:
                        st.dataframe(pd.DataFrame(_src_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No per-source breakdown available")

                # AI Analysis
                if ai_data and ai_data.get("summary"):
                    st.markdown("**AI Analysis:**")
                    st.info(ai_data.get("summary", ""))

                # Sample messages from all sources
                _all_samples = []
                for msg in st_data.get("sample_messages", [])[:3]:
                    _all_samples.append(("StockTwits", msg))
                if reddit_data:
                    for msg in reddit_data.get("sample_messages", [])[:3]:
                        _all_samples.append(("Reddit", msg))
                if gnews_data:
                    for msg in gnews_data.get("sample_messages", [])[:2]:
                        _all_samples.append(("News", msg))
                if _all_samples:
                    with st.expander("📝 Sample Messages"):
                        for source_name, msg in _all_samples:
                            _s_icon = "🟢" if msg.get("sentiment") in ("Bullish", "bullish") else "🔴" if msg.get("sentiment") in ("Bearish", "bearish") else "🟡"
                            _text = msg.get("text", str(msg))[:150] if isinstance(msg, dict) else str(msg)[:150]
                            st.caption(f"[{source_name}] {_s_icon} {_text}")
            else:
                st.caption("No sentiment data available for this stock. Run `stockpulse sentiment-scan` to fetch.")

            # Price chart
            st.markdown("#### Price History (3 Months)")
            try:
                price_data = services["ingestion"].get_daily_prices(
                    [selected_ticker],
                    start_date=date.today() - timedelta(days=90)
                )

                if not price_data.empty:
                    fig = create_price_chart(price_data, selected_ticker, show_volume=True)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load price chart: {e}")


def render_trillion_club_page(services: dict):
    """Render Trillion Club page showing mega-cap stocks and entry scores."""
    st.title("💎 Trillion+ Club")
    st.markdown("Tracking stocks that have hit $1T+ market cap in the last 30 days. Looking for optimal long-term entry points.")

    # Get trillion club data from database
    try:
        db = services.get("db") or get_db()
        trillion_df = db.fetchdf("""
            SELECT t.* FROM trillion_club t
            INNER JOIN (
                SELECT ticker, MAX(scan_date) as max_date
                FROM trillion_club
                GROUP BY ticker
            ) latest ON t.ticker = latest.ticker AND t.scan_date = latest.max_date
            ORDER BY t.entry_score DESC
        """)
    except Exception as e:
        st.error(f"Error loading trillion club data: {e}")
        trillion_df = pd.DataFrame()

    if trillion_df.empty:
        st.warning("No Trillion Club data yet. Run `stockpulse trillion-scan` to populate.")

        # Quick action button
        if st.button("🔍 Run Trillion Club Scan Now"):
            st.info("Run in terminal: `stockpulse trillion-scan`")
        return

    # Show data freshness
    col_fresh, col_refresh = st.columns([3, 1])
    with col_fresh:
        if "scan_date" in trillion_df.columns:
            latest_scan = trillion_df["scan_date"].max()
            from datetime import datetime
            try:
                scan_dt = datetime.fromisoformat(str(latest_scan))
                age = datetime.now() - scan_dt
                hours_ago = age.total_seconds() / 3600
                if hours_ago < 24:
                    freshness = f"{hours_ago:.1f} hours ago"
                else:
                    freshness = f"{hours_ago / 24:.1f} days ago"
                st.caption(f"📅 Data from: {freshness}")
            except Exception:
                st.caption(f"📅 Last scan: {latest_scan}")
    with col_refresh:
        if st.button("🔄 Refresh", key="refresh_trillion"):
            st.info("Run in terminal: `stockpulse trillion-scan`")

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Members", len(trillion_df))
    with col2:
        total_market_cap = trillion_df["market_cap"].sum() / 1e12 if "market_cap" in trillion_df.columns else 0
        st.metric("Total Market Cap", f"${total_market_cap:.1f}T")
    with col3:
        avg_score = trillion_df["entry_score"].mean() if "entry_score" in trillion_df.columns else 0
        st.metric("Avg Entry Score", f"{avg_score:.0f}")
    with col4:
        best_opp = trillion_df.iloc[0] if not trillion_df.empty else None
        st.metric("Best Entry", best_opp["ticker"] if best_opp is not None else "N/A")

    # Best entry opportunities highlight
    st.markdown("---")
    st.subheader("🎯 Best Entry Opportunities")
    st.caption("Stocks with Entry Score ≥ 70 (pullback + oversold conditions)")

    best_entries = trillion_df[trillion_df["entry_score"] >= 70] if "entry_score" in trillion_df.columns else pd.DataFrame()

    if not best_entries.empty:
        cols = st.columns(min(len(best_entries), 4))
        for idx, (_, entry) in enumerate(best_entries.head(4).iterrows()):
            with cols[idx]:
                ticker = entry.get("ticker", "N/A")
                score = entry.get("entry_score", 0)
                price = entry.get("current_price", 0)
                pct_from_high = entry.get("price_vs_30d_high_pct", 0)

                st.markdown(f"""
                <div style="background: #ecfdf5; padding: 15px; border-radius: 8px; border: 2px solid #22c55e; text-align: center;">
                    <h2 style="margin: 0; color: #15803d;">{ticker}</h2>
                    <p style="font-size: 24px; margin: 10px 0; color: #22c55e;">{score:.0f}</p>
                    <p style="color: #6b7280; font-size: 14px;">${price:.2f}</p>
                    <p style="color: #22c55e; font-size: 13px;">{pct_from_high:+.1f}% from high</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No strong entry opportunities right now (all scores < 70). Check back when stocks pull back.")

    # Category breakdown
    st.markdown("---")
    st.subheader("📁 Category Breakdown")

    if "category" in trillion_df.columns:
        category_counts = trillion_df.groupby("category").agg({
            "ticker": ["count", lambda x: ", ".join(x.tolist()[:5])]
        }).reset_index()
        category_counts.columns = ["Category", "Count", "Tickers"]
        st.dataframe(category_counts, use_container_width=True, hide_index=True)

    # Main trillion club table
    st.markdown("---")
    st.subheader("💎 All Trillion+ Club Members")

    # Add trend data
    from stockpulse.scanner.ai_pulse import AIPulseScanner
    scanner = AIPulseScanner()

    # Load sentiment data
    sentiment_data = {}
    try:
        from stockpulse.data.sentiment import SentimentStorage
        storage = SentimentStorage()
        tc_tickers = trillion_df["ticker"].tolist()
        sentiment_data = storage.get_todays_sentiment(tc_tickers)
    except Exception as e:
        pass  # Sentiment optional

    # Enrich with trends and sentiment
    members = trillion_df.to_dict('records')
    for member in members:
        trend = scanner.get_trend_data(member["ticker"])
        member["trend_symbol"] = trend["trend_symbol"]
        member["consecutive_days"] = trend["consecutive_days"]
        member["score_change_5d"] = trend["score_change_5d"]

        # Add sentiment
        member["sentiment"] = format_sentiment(sentiment_data, member["ticker"])

    trillion_df = pd.DataFrame(members)

    # Sentiment Summary section for Trillion Club
    if sentiment_data:
        st.markdown("---")
        st.subheader("📊 Social Sentiment - Trillion Club")
        st.caption("Sentiment from StockTwits for mega-cap stocks")

        # Build sentiment details
        tc_sentiment = []
        for ticker, sent in sentiment_data.items():
            if sent.get("aggregate_score", 0) > 0:
                st_data = sent.get("stocktwits", {})
                ai_data = sent.get("ai_analysis", {})
                tc_sentiment.append({
                    "ticker": ticker,
                    "score": sent.get("aggregate_score", 50),
                    "label": sent.get("aggregate_label", "neutral"),
                    "bullish": st_data.get("bullish_count", 0),
                    "bearish": st_data.get("bearish_count", 0),
                    "total": st_data.get("total_messages", 0),
                    "ai_summary": ai_data.get("summary", "") if ai_data else "",
                    "sample_messages": st_data.get("sample_messages", [])
                })

        if tc_sentiment:
            tc_sentiment.sort(key=lambda x: x["score"], reverse=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🟢 Most Bullish**")
                bullish = [s for s in tc_sentiment if s["label"] == "bullish"][:3]
                if bullish:
                    for s in bullish:
                        st.markdown(f"**{s['ticker']}** - Score: {s['score']:.0f}")
                        st.caption(f"↑{s['bullish']} ↓{s['bearish']} ({s['total']} messages)")

                        # Show AI analysis and sample quotes
                        has_content = s["ai_summary"] or s.get("sample_messages")
                        if has_content:
                            with st.expander("Details"):
                                if s["ai_summary"]:
                                    st.markdown("**AI Analysis:**")
                                    st.write(s["ai_summary"][:400])
                                if s.get("sample_messages"):
                                    st.markdown("**Sample Posts:**")
                                    for msg in s["sample_messages"][:3]:
                                        sentiment_icon = "🟢" if msg.get("sentiment") == "Bullish" else "🔴" if msg.get("sentiment") == "Bearish" else "🟡"
                                        st.caption(f"{sentiment_icon} \"{msg.get('text', '')[:120]}...\"")
                else:
                    st.caption("No bullish stocks")

            with col2:
                st.markdown("**🔴 Most Bearish**")
                bearish = [s for s in tc_sentiment if s["label"] == "bearish"][:3]
                if bearish:
                    for s in bearish:
                        st.markdown(f"**{s['ticker']}** - Score: {s['score']:.0f}")
                        st.caption(f"↑{s['bullish']} ↓{s['bearish']} ({s['total']} messages)")

                        # Show AI analysis and sample quotes
                        has_content = s["ai_summary"] or s.get("sample_messages")
                        if has_content:
                            with st.expander("Details"):
                                if s["ai_summary"]:
                                    st.markdown("**AI Analysis:**")
                                    st.write(s["ai_summary"][:400])
                                if s.get("sample_messages"):
                                    st.markdown("**Sample Posts:**")
                                    for msg in s["sample_messages"][:3]:
                                        sentiment_icon = "🟢" if msg.get("sentiment") == "Bullish" else "🔴" if msg.get("sentiment") == "Bearish" else "🟡"
                                        st.caption(f"{sentiment_icon} \"{msg.get('text', '')[:120]}...\"")
                else:
                    st.caption("No bearish stocks")

    # All Trillion Club table
    st.markdown("---")
    st.subheader("💎 All Trillion+ Club Members")

    # Format for display
    display_cols = ["ticker", "market_cap", "current_price", "price_vs_30d_high_pct", "entry_score", "sentiment", "category", "trend_symbol", "consecutive_days"]
    available_cols = [c for c in display_cols if c in trillion_df.columns]
    display_df = trillion_df[available_cols].copy()

    # Format columns
    if "market_cap" in display_df.columns:
        display_df["market_cap"] = display_df["market_cap"].apply(lambda x: f"${x/1e9:.0f}B" if pd.notna(x) else "")
    if "current_price" in display_df.columns:
        display_df["current_price"] = display_df["current_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    if "price_vs_30d_high_pct" in display_df.columns:
        display_df["price_vs_30d_high_pct"] = display_df["price_vs_30d_high_pct"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    if "entry_score" in display_df.columns:
        display_df["entry_score"] = display_df["entry_score"].round(0).astype(int)

    # Create trend column
    def format_trend(row):
        trend = row.get('trend_symbol', '➡️')
        days = row.get('consecutive_days', 0)
        return f"{trend} {days}d"

    display_df['trend'] = display_df.apply(format_trend, axis=1)

    # Select final columns
    final_cols = ["ticker", "market_cap", "current_price", "price_vs_30d_high_pct", "entry_score", "sentiment", "category", "trend"]
    final_cols = [c for c in final_cols if c in display_df.columns]
    display_df = display_df[final_cols]

    display_df = display_df.rename(columns={
        "ticker": "Ticker",
        "market_cap": "Market Cap",
        "current_price": "Price",
        "price_vs_30d_high_pct": "vs 30d High",
        "entry_score": "Entry Score",
        "sentiment": "Sentiment",
        "category": "Category",
        "trend": "Trend"
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Entry score legend
    st.caption("**Entry Score:** 75+ = Strong Entry (pullback + oversold) | 65-74 = Good Entry | 55-64 = Neutral | <55 = Extended (wait for pullback)")
    st.caption("**Trend:** 📈 Score improving | 📉 Score declining | ➡️ Stable | 🆕 New to list • **Xd** = days tracked")

    # Detail view for selected ticker
    st.markdown("---")
    st.subheader("📊 Stock Details")

    selected_ticker = st.selectbox("Select ticker for details", trillion_df["ticker"].unique())

    if selected_ticker:
        ticker_data = trillion_df[trillion_df["ticker"] == selected_ticker].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score = ticker_data.get('entry_score', 0)
            score_label = "Strong" if score >= 75 else "Good" if score >= 65 else "Neutral" if score >= 55 else "Extended"
            st.metric("Entry Score", f"{score:.0f} ({score_label})")
        with col2:
            st.metric("Price", f"${ticker_data.get('current_price', 0):.2f}")
        with col3:
            st.metric("vs 30d High", f"{ticker_data.get('price_vs_30d_high_pct', 0):+.1f}%")
        with col4:
            st.metric("Category", ticker_data.get('category', 'Unknown'))

        # Price chart
        st.markdown("#### Price History (3 Months)")
        try:
            price_data = services["ingestion"].get_daily_prices(
                [selected_ticker],
                start_date=date.today() - timedelta(days=90)
            )

            if not price_data.empty:
                fig = create_price_chart(price_data, selected_ticker, show_volume=True)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load price chart: {e}")


def render_ai_theses_page(services: dict):
    """Render AI Investment Theses page."""
    st.title("🧠 AI Investment Theses")
    st.markdown("Track and research investment theses using AI-powered analysis.")

    # Get theses from database
    try:
        db = services.get("db") or get_db()
        theses_df = db.fetchdf("""
            SELECT * FROM ai_theses ORDER BY updated_at DESC
        """)
    except Exception as e:
        st.error(f"Error loading theses: {e}")
        theses_df = pd.DataFrame()

    if theses_df.empty:
        st.warning("No investment theses tracked yet. Run `stockpulse ai-scan` to initialize default theses.")
        return

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Theses", len(theses_df[theses_df["status"] == "active"]) if "status" in theses_df.columns else len(theses_df))
    with col2:
        bullish = len(theses_df[theses_df["recommendation"] == "bullish"]) if "recommendation" in theses_df.columns else 0
        st.metric("Bullish", bullish)
    with col3:
        bearish = len(theses_df[theses_df["recommendation"] == "bearish"]) if "recommendation" in theses_df.columns else 0
        st.metric("Bearish", bearish)
    with col4:
        avg_conf = theses_df["confidence"].mean() if "confidence" in theses_df.columns else 0
        st.metric("Avg Confidence", f"{avg_conf:.0f}%")

    # Thesis cards
    st.markdown("---")
    st.subheader("📋 Investment Theses")

    for _, thesis in theses_df.iterrows():
        name = thesis.get("thesis_name", "Unnamed")
        description = thesis.get("description", "")
        tickers = thesis.get("tickers", "").split(",") if thesis.get("tickers") else []
        recommendation = thesis.get("recommendation", "neutral")
        confidence = thesis.get("confidence", 50)
        last_research = thesis.get("last_research", "")
        updated_at = thesis.get("updated_at", "")

        # Recommendation styling
        rec_colors = {
            "bullish": ("#22c55e", "📈"),
            "bearish": ("#ef4444", "📉"),
            "neutral": ("#6b7280", "➡️")
        }
        rec_color, rec_icon = rec_colors.get(recommendation, ("#6b7280", "➡️"))

        with st.expander(f"{rec_icon} {name} — {recommendation.upper()} ({confidence:.0f}%)", expanded=True):
            st.markdown(f"**Description:** {description}")
            st.markdown(f"**Tickers:** {', '.join(tickers)}")

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div style="background: {rec_color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {rec_color}; text-align: center;">
                    <h3 style="margin: 0; color: {rec_color};">{recommendation.upper()}</h3>
                    <p style="margin: 5px 0 0 0; font-size: 24px;">{confidence:.0f}%</p>
                    <p style="color: #6b7280; font-size: 12px;">confidence</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if last_research:
                    st.markdown("**Latest Research:**")
                    st.info(last_research[:500] + "..." if len(last_research) > 500 else last_research)
                else:
                    st.caption("No research analysis yet. Run `stockpulse ai-scan` to generate.")

            # Council Perspectives (if available)
            thesis_id = thesis.get("id")
            if thesis_id:
                try:
                    # Get council verdict
                    _verdict_df = db.fetchdf("""
                        SELECT * FROM council_verdicts
                        WHERE thesis_id = ? ORDER BY research_date DESC LIMIT 1
                    """, (thesis_id,))

                    _persp_df = db.fetchdf("""
                        SELECT * FROM council_perspectives
                        WHERE thesis_id = ? ORDER BY research_date DESC LIMIT 6
                    """, (thesis_id,))

                    if not _verdict_df.empty:
                        _v = _verdict_df.iloc[0]
                        _agreement = _v.get("agreement_score", 0)
                        _dissent = _v.get("dissenting_views", "")
                        st.markdown(f"**Council Consensus:** {_v.get('consensus', 'N/A')} "
                                    f"(Agreement: {_agreement:.0f}%)")
                        if _dissent:
                            st.caption(f"Dissenting views: {_dissent[:200]}")

                    if not _persp_df.empty:
                        with st.expander(f"🗣️ Council Perspectives ({len(_persp_df)} agents)"):
                            _p_cols = st.columns(min(len(_persp_df), 3))
                            for _p_idx, (_, _p_row) in enumerate(_persp_df.iterrows()):
                                with _p_cols[_p_idx % 3]:
                                    _p_rec = _p_row.get("recommendation", "neutral")
                                    _p_icon = "📈" if _p_rec == "bullish" else "📉" if _p_rec == "bearish" else "➡️"
                                    _p_conf = _p_row.get("confidence", 0)
                                    _p_name = str(_p_row.get("perspective", "Agent")).title()
                                    st.markdown(f"**{_p_icon} {_p_name}** — {_p_rec.upper()} ({_p_conf:.0f}%)")
                                    _p_text = str(_p_row.get("analysis", ""))[:200]
                                    st.caption(_p_text + "..." if len(str(_p_row.get("analysis", ""))) > 200 else _p_text)
                except Exception:
                    pass

            if updated_at:
                st.caption(f"Last updated: {updated_at}")

    # Research history
    st.markdown("---")
    st.subheader("📜 Research History")

    try:
        history_df = db.fetchdf("""
            SELECT
                t.thesis_name,
                r.research_date,
                r.recommendation,
                r.confidence,
                r.analysis
            FROM thesis_research r
            JOIN ai_theses t ON r.thesis_id = t.id
            ORDER BY r.research_date DESC
            LIMIT 20
        """)

        if not history_df.empty:
            # Format for display
            display_df = history_df.copy()
            if "confidence" in display_df.columns:
                display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
            if "analysis" in display_df.columns:
                display_df["analysis"] = display_df["analysis"].apply(lambda x: x[:100] + "..." if x and len(x) > 100 else x)

            display_df = display_df.rename(columns={
                "thesis_name": "Thesis",
                "research_date": "Date",
                "recommendation": "Recommendation",
                "confidence": "Confidence",
                "analysis": "Summary"
            })

            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No research history yet. Research is generated when running `stockpulse ai-scan`.")
    except Exception as e:
        st.warning(f"Could not load research history: {e}")

    # Add new thesis
    st.markdown("---")
    st.subheader("➕ Add New Thesis")

    with st.form("add_thesis"):
        thesis_name = st.text_input("Thesis Name", placeholder="e.g., Tesla Robot Thesis")
        thesis_description = st.text_area("Description", placeholder="Describe the investment thesis...")
        thesis_tickers = st.text_input("Related Tickers (comma-separated)", placeholder="e.g., TSLA, NVDA, ISRG")

        if st.form_submit_button("Add Thesis"):
            if thesis_name and thesis_description:
                try:
                    from stockpulse.scanner.ai_pulse import AIPulseScanner
                    scanner = AIPulseScanner()
                    tickers_list = [t.strip() for t in thesis_tickers.split(",") if t.strip()]
                    scanner.add_thesis(thesis_name, thesis_description, tickers_list)
                    st.success(f"Added thesis: {thesis_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding thesis: {e}")
            else:
                st.warning("Please provide both name and description")


def render_settings_page(services: dict):
    """Render Settings page."""
    st.title("⚙️ Settings")

    config = get_config()

    st.markdown("---")
    st.subheader("Current Configuration")

    tab1, tab2, tab3, tab4 = st.tabs(["Strategies", "Alerts", "Risk Management", "System"])

    with tab1:
        st.json(config.get("strategies", {}))

    with tab2:
        st.json(config.get("alerts", {}))

    with tab3:
        st.json(config.get("risk_management", {}))

    with tab4:
        st.json({
            "scanning": config.get("scanning", {}),
            "trading": config.get("trading", {}),
            "database": config.get("database", {})
        })

    st.markdown("---")
    st.subheader("Stock Universe")

    universe_df = services["universe"].get_universe_df()

    if not universe_df.empty:
        st.metric("Active Stocks", len(universe_df[universe_df["is_active"] == True]))

        with st.expander("View Universe"):
            st.dataframe(universe_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("System Status")

    staleness = services["ingestion"].check_data_staleness()

    col1, col2 = st.columns(2)

    with col1:
        if staleness.get("last_daily"):
            st.success(f"Last daily ingestion: {staleness['last_daily']}")
        else:
            st.warning("No daily data ingested yet")

    with col2:
        if staleness.get("last_intraday"):
            st.success(f"Last intraday ingestion: {staleness['last_intraday']}")
        else:
            st.warning("No intraday data ingested yet")

    if staleness.get("is_stale"):
        st.error("Data is stale! Check the scheduler.")


if __name__ == "__main__":
    main()
