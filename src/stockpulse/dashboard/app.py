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
    print(formatted)
    _debug_log.append({"time": timestamp, "level": level, "message": message})


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
        ["Live Signals", "Paper Portfolio", "Long-Term Holdings", "Performance", "Backtests", "Long-Term Watchlist", "Universe", "Settings", "Debug"],
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

    # Data status
    try:
        staleness = services["ingestion"].check_data_staleness()
        if staleness.get("last_daily"):
            st.sidebar.success("📊 Data: Loaded")
        else:
            st.sidebar.warning("📊 Data: No data yet")
    except:
        st.sidebar.warning("📊 Data: Unknown")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
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
    st.title("📡 Live Signals")

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

    # Signals table
    st.markdown("---")
    st.subheader("Active Signals")

    if not filtered.empty:
        # Only select columns that exist
        desired_cols = ["ticker", "strategy", "direction", "confidence",
                       "entry_price", "target_price", "stop_price", "created_at", "notes"]
        available_cols = [c for c in desired_cols if c in filtered.columns]
        display_df = filtered[available_cols].copy()

        if "confidence" in display_df.columns:
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0f}%")
        if "entry_price" in display_df.columns:
            display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
        if "target_price" in display_df.columns:
            display_df["target_price"] = display_df["target_price"].apply(lambda x: f"${x:.2f}")
        if "stop_price" in display_df.columns:
            display_df["stop_price"] = display_df["stop_price"].apply(lambda x: f"${x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No active signals matching your filters.")

        # Show helpful message if no data at all
        if signals_df.empty:
            st.markdown("""
            **No signals yet?** This is expected on first launch. To generate signals:

            1. **Load stock universe**: Click "Initialize Data" below
            2. **Wait for data ingestion**: This fetches price history for 100 stocks
            3. **Generate signals**: Strategies will analyze the data

            Signals are generated automatically when the scheduler runs, or you can trigger manually.
            """)

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
                                services["ingestion"].ingest_daily_data(tickers)
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
                    display_data.append({
                        "Status": p["status"],
                        "Ticker": p["ticker"],
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
                display_cols = ["status", "ticker", "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason"]
                available_cols = [c for c in display_cols if c in strategy_closed.columns]
                display_df = strategy_closed[available_cols].copy()

                if "entry_price" in display_df.columns:
                    display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
                if "exit_price" in display_df.columns:
                    display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:.2f}" if x else "N/A")
                if "pnl" in display_df.columns:
                    display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:+,.2f}" if x else "N/A")
                if "pnl_pct" in display_df.columns:
                    display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%" if x else "N/A")

                display_df = display_df.rename(columns={"status": "W/L", "entry_price": "Entry", "exit_price": "Exit", "pnl": "P&L", "pnl_pct": "P&L %", "exit_reason": "Reason"})
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
    st.title("📊 Performance Analytics")

    performance = services["positions"].get_performance_summary()

    # Equity Curve Section
    st.markdown("---")
    st.subheader("Portfolio Equity Curve")

    # Get current prices for mark-to-market (use LIVE prices)
    try:
        # Get list of open position tickers
        positions_df = services["positions"].get_open_positions(is_paper=True)
        tickers = positions_df["ticker"].tolist() if not positions_df.empty else []

        # Fetch live prices
        ingestion = services.get("ingestion")
        if ingestion and tickers:
            current_prices = ingestion.fetch_current_prices(tickers)
        else:
            current_prices = {}

        # Fallback to daily prices if needed
        if not current_prices:
            db = services.get("db") or get_db()
            prices_df = db.fetchdf("""
                SELECT ticker, close FROM prices_daily
                WHERE date = (SELECT MAX(date) FROM prices_daily)
            """)
            current_prices = dict(zip(prices_df["ticker"], prices_df["close"])) if not prices_df.empty else {}
    except Exception:
        current_prices = {}

    # Get equity curve with open positions marked to market
    equity_df = services["positions"].get_equity_curve_with_open_positions(current_prices, is_paper=True)

    if len(equity_df) > 1:
        fig = create_equity_curve(equity_df, title="Paper Portfolio Equity Curve", show_drawdown=True)
        st.plotly_chart(fig, use_container_width=True)

        # Show key equity metrics
        if not equity_df.empty:
            latest_equity = equity_df.iloc[-1]["equity"]
            initial_equity = equity_df.iloc[0]["equity"]
            total_return = ((latest_equity / initial_equity) - 1) * 100 if initial_equity > 0 else 0
            max_dd = equity_df["drawdown"].min() * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Equity", f"${latest_equity:,.2f}")
            with col2:
                st.metric("Total Return", f"{total_return:+.2f}%")
            with col3:
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
    else:
        st.info("No equity data yet. Start trading to see your equity curve!")

    st.markdown("---")
    st.subheader("Performance Summary")

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

    st.markdown("---")
    st.subheader("Strategy Performance")

    strategy_perf = services["positions"].get_strategy_performance()

    if not strategy_perf.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig = create_performance_chart(strategy_perf, "total_pnl")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_win_rate_chart(strategy_perf)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(strategy_perf, use_container_width=True, hide_index=True)
    else:
        st.info("No strategy performance data available yet.")

    st.markdown("---")
    st.subheader("P&L Distribution")

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
        watchlist_df = pd.DataFrame(opps)

        # Better display with trend, company name, sector, and price info
        display_cols = [
            "trend_symbol", "ticker", "company_name", "sector", "composite_score",
            "current_price", "week52_low", "week52_high", "price_vs_52w_low_pct"
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
            "trend_symbol": "Trend",
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "composite_score": "Score",
            "current_price": "Price",
            "week52_low": "52W Low",
            "week52_high": "52W High",
            "price_vs_52w_low_pct": "vs Low"
        }
        display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Trend legend
        st.caption("**Trend:** 📈 Strengthening | 📉 Weakening | ➡️ Stable | 🆕 New")

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
