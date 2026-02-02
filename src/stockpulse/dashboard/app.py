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

# Custom CSS - WCAG 2.1 AA compliant color contrast (4.5:1 minimum)
st.markdown("""
<style>
    /* ========================================
       FORCE LIGHT THEME - Override Streamlit
       ======================================== */

    /* Main app background */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #ffffff !important;
    }

    /* Main content area */
    .main .block-container {
        background-color: #ffffff !important;
        padding-top: 2rem !important;
    }

    /* ========================================
       SIDEBAR - Light gray background
       ======================================== */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #f0f2f6 !important;
    }

    /* Sidebar text - dark on light */
    [data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #1f2937 !important;
    }

    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] .stRadio label span {
        color: #1f2937 !important;
    }

    /* ========================================
       MAIN CONTENT TEXT - Dark on white
       ======================================== */

    /* All text defaults */
    .main p, .main span, .main label, .main div {
        color: #1f2937 !important;
    }

    /* Headers - very dark */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
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
       METRICS - High contrast
       ======================================== */
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* Positive delta - green */
    [data-testid="stMetricDelta"][data-testid-delta="positive"] {
        color: #059669 !important;
    }

    /* Negative delta - red */
    [data-testid="stMetricDelta"][data-testid-delta="negative"] {
        color: #dc2626 !important;
    }

    /* ========================================
       DATA TABLES
       ======================================== */
    .stDataFrame {
        border-radius: 8px !important;
    }

    /* Table headers */
    .stDataFrame thead th {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* Table cells */
    .stDataFrame tbody td {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }

    /* Alternating rows */
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: #f9fafb !important;
    }

    /* ========================================
       FORM ELEMENTS
       ======================================== */

    /* Select boxes */
    .stSelectbox label, .stMultiSelect label {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }

    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    .stSelectbox [data-baseweb="select"] * {
        color: #1f2937 !important;
    }

    /* Text inputs */
    .stTextInput label, .stNumberInput label {
        color: #1f2937 !important;
    }

    .stTextInput input, .stNumberInput input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Sliders */
    .stSlider label {
        color: #1f2937 !important;
    }

    /* ========================================
       BUTTONS
       ======================================== */
    .stButton button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border: none !important;
    }

    .stButton button:hover {
        background-color: #2563eb !important;
    }

    /* ========================================
       ALERTS / INFO BOXES
       ======================================== */
    .stAlert {
        border-radius: 8px !important;
    }

    /* Info box */
    [data-testid="stAlert"][data-baseweb="notification"] {
        background-color: #eff6ff !important;
        color: #1e40af !important;
    }

    /* Success box */
    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background-color: #ecfdf5 !important;
        color: #065f46 !important;
    }

    /* Warning box */
    .stWarning, [data-baseweb="notification"][kind="warning"] {
        background-color: #fffbeb !important;
        color: #92400e !important;
    }

    /* Error box */
    .stError, [data-baseweb="notification"][kind="negative"] {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
    }

    /* ========================================
       TABS
       ======================================== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f3f4f6 !important;
        border-radius: 8px !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #4b5563 !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }

    /* ========================================
       EXPANDERS
       ======================================== */
    .streamlit-expanderHeader {
        color: #1f2937 !important;
        font-weight: 500 !important;
        background-color: #f9fafb !important;
    }

    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }

    /* ========================================
       CODE BLOCKS
       ======================================== */
    code {
        background-color: #f3f4f6 !important;
        color: #1f2937 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    pre {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
    }

    /* ========================================
       CAPTIONS & SMALL TEXT
       ======================================== */
    .stCaption, small, .st-emotion-cache-1gulkj5 {
        color: #6b7280 !important;
    }

    /* ========================================
       PLOTLY CHARTS - ensure readable
       ======================================== */
    .js-plotly-plot .plotly .modebar {
        background-color: transparent !important;
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

    # Database
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
        ["Live Signals", "Paper Portfolio", "Performance", "Backtests", "Long-Term Watchlist", "Settings", "Debug"],
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
    elif page == "Performance":
        render_performance_page(services)
    elif page == "Backtests":
        render_backtests_page(services)
    elif page == "Long-Term Watchlist":
        render_watchlist_page(services)
    elif page == "Settings":
        render_settings_page(services)
    elif page == "Debug":
        render_debug_page(services)


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

    positions_df = services["positions"].get_open_positions()

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Open Positions", len(positions_df))

    with col2:
        if not positions_df.empty:
            total_value = (positions_df["entry_price"] * positions_df["shares"]).sum()
            st.metric("Total Value", f"${total_value:,.2f}")
        else:
            st.metric("Total Value", "$0.00")

    with col3:
        performance = services["positions"].get_performance_summary()
        total_pnl = performance.get("total_pnl", 0)
        st.metric("Total P&L", f"${total_pnl:+,.2f}")

    with col4:
        win_rate = performance.get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")

    st.markdown("---")
    st.subheader("Open Positions")

    if not positions_df.empty:
        display_cols = ["ticker", "direction", "entry_price", "shares", "entry_date", "strategy", "status"]
        available_cols = [c for c in display_cols if c in positions_df.columns]
        display_df = positions_df[available_cols].copy()

        if "entry_price" in display_df.columns:
            display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
        if "shares" in display_df.columns:
            display_df["shares"] = display_df["shares"].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    st.markdown("---")
    st.subheader("Recent Closed Positions")

    closed_df = services["positions"].get_closed_positions(start_date=date.today() - timedelta(days=30))

    if not closed_df.empty:
        display_cols = ["ticker", "direction", "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason", "strategy"]
        available_cols = [c for c in display_cols if c in closed_df.columns]
        display_df = closed_df[available_cols].copy()

        if "entry_price" in display_df.columns:
            display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
        if "exit_price" in display_df.columns:
            display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:.2f}" if x else "N/A")
        if "pnl" in display_df.columns:
            display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:+,.2f}" if x else "N/A")
        if "pnl_pct" in display_df.columns:
            display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%" if x else "N/A")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No closed positions in the last 30 days.")

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

    # Get current prices for mark-to-market
    try:
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
    """Render Backtests page."""
    st.title("🔬 Backtests")

    results_df = services["backtester"].get_backtest_results()

    if not results_df.empty:
        st.markdown("---")
        st.subheader("Backtest Results")

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
        st.info("No backtest results available. Run backtests to see results here.")

    st.markdown("---")
    st.subheader("Run New Backtest")

    col1, col2 = st.columns(2)

    with col1:
        strategy_options = [
            "rsi_mean_reversion", "bollinger_squeeze", "macd_volume",
            "zscore_mean_reversion", "momentum_breakout"
        ]
        selected_strategy = st.selectbox("Strategy", strategy_options)

    with col2:
        start_date = st.date_input("Start Date", date(2023, 1, 1))
        end_date = st.date_input("End Date", date.today())

    if st.button("Run Backtest"):
        st.info("Backtest functionality requires historical data. Run `stockpulse init` first.")


def render_watchlist_page(services: dict):
    """Render Long-Term Watchlist page."""
    st.title("📋 Long-Term Watchlist")

    try:
        watchlist_df = services["db"].fetchdf("""
            SELECT * FROM long_term_watchlist
            WHERE scan_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY composite_score DESC
        """)
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        watchlist_df = pd.DataFrame()

    if not watchlist_df.empty:
        st.markdown("---")
        st.subheader("Top Long-Term Opportunities")

        display_cols = [
            "ticker", "composite_score", "valuation_score",
            "technical_score", "pe_percentile", "price_vs_52w_low_pct",
            "reasoning", "scan_date"
        ]
        available_cols = [c for c in display_cols if c in watchlist_df.columns]
        display_df = watchlist_df[available_cols].copy()

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        selected_ticker = st.selectbox("Select ticker for details", watchlist_df["ticker"].unique())

        if selected_ticker:
            ticker_data = watchlist_df[watchlist_df["ticker"] == selected_ticker].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Composite Score", f"{ticker_data.get('composite_score', 0):.0f}")
            with col2:
                st.metric("P/E Percentile", f"{ticker_data.get('pe_percentile', 0):.0f}%")
            with col3:
                st.metric("vs 52W Low", f"{ticker_data.get('price_vs_52w_low_pct', 0):.1f}%")

            st.markdown(f"**Reasoning:** {ticker_data.get('reasoning', 'N/A')}")

            price_data = services["ingestion"].get_daily_prices(
                [selected_ticker],
                start_date=date.today() - timedelta(days=365)
            )

            if not price_data.empty:
                fig = create_price_chart(price_data, selected_ticker, show_volume=True)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No long-term opportunities identified yet. The scanner runs daily after market close.")


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
