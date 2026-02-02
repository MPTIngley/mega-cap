"""
StockPulse Dashboard - Streamlit Application

A comprehensive trading dashboard with:
- Live Signals
- Paper Portfolio
- Performance Analytics
- Backtests
- Long-Term Watchlist
- Settings
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta

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

# Custom CSS for clean styling
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stMetric { background: #f8f9fa; padding: 15px; border-radius: 10px; }
    .metric-positive { color: #27ae60 !important; }
    .metric-negative { color: #e74c3c !important; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


logger = get_logger(__name__)


@st.cache_resource
def init_services():
    """Initialize services (cached)."""
    import os

    print("=" * 60)
    print("STOCKPULSE DASHBOARD STARTUP")
    print("=" * 60)

    # Find and load config
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    print(f"[CONFIG] Looking for config at: {config_path}")
    print(f"[CONFIG] Config exists: {config_path.exists()}")

    if not config_path.exists():
        # Try alternative paths
        alt_paths = [
            Path.cwd() / "config" / "config.yaml",
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
        ]
        for alt in alt_paths:
            print(f"[CONFIG] Trying alternative: {alt}")
            if alt.exists():
                config_path = alt
                break

    load_config(config_path)
    print(f"[CONFIG] Loaded successfully")

    # Check environment variables
    print("\n[ENV] Environment Variables:")
    env_vars = [
        "STOCKPULSE_EMAIL_SENDER",
        "STOCKPULSE_EMAIL_RECIPIENT",
        "STOCKPULSE_EMAIL_PASSWORD",
        "STOCKPULSE_EMAIL_RECIPIENTS_CC",
    ]
    for var in env_vars:
        val = os.environ.get(var, "")
        if var == "STOCKPULSE_EMAIL_PASSWORD":
            display = "****" if val else "(not set)"
        else:
            display = val if val else "(not set)"
        print(f"  {var}: {display}")

    # Initialize database
    print("\n[DB] Initializing database...")
    db = get_db()
    print(f"[DB] Database path: {db.db_path}")
    print(f"[DB] Database exists: {db.db_path.exists()}")

    # Check table counts
    try:
        counts = {
            "universe": db.fetchone("SELECT COUNT(*) FROM universe")[0],
            "prices_daily": db.fetchone("SELECT COUNT(*) FROM prices_daily")[0],
            "signals": db.fetchone("SELECT COUNT(*) FROM signals")[0],
            "positions_paper": db.fetchone("SELECT COUNT(*) FROM positions_paper")[0],
        }
        print(f"[DB] Table counts: {counts}")
    except Exception as e:
        print(f"[DB] Error checking tables: {e}")

    # Initialize services
    print("\n[SERVICES] Initializing services...")
    services = {
        "db": db,
        "universe": UniverseManager(),
        "ingestion": DataIngestion(),
        "signals": SignalGenerator(),
        "positions": PositionManager(),
        "backtester": Backtester()
    }

    # Check universe
    tickers = services["universe"].get_active_tickers()
    print(f"[UNIVERSE] Active tickers: {len(tickers)}")
    if tickers:
        print(f"[UNIVERSE] Sample: {tickers[:5]}")

    print("\n" + "=" * 60)
    print("STARTUP COMPLETE - Dashboard ready")
    print("=" * 60 + "\n")

    return services


def main():
    """Main dashboard application."""
    # Initialize services
    try:
        services = init_services()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Make sure the database and config are properly set up.")
        return

    # Sidebar
    st.sidebar.title("📊 StockPulse")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Live Signals", "Paper Portfolio", "Performance", "Backtests", "Long-Term Watchlist", "Settings"],
        label_visibility="collapsed"
    )

    # Last update time
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

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


def render_signals_page(services: dict):
    """Render Live Signals page."""
    st.title("📡 Live Signals")

    # Get open signals
    signals_df = services["signals"].get_open_signals()

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_confidence = st.slider("Min Confidence", 0, 100, 60)

    with col2:
        strategies = ["All"] + list(signals_df["strategy"].unique()) if not signals_df.empty else ["All"]
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
        # Format for display
        display_df = filtered[[
            "ticker", "strategy", "direction", "confidence",
            "entry_price", "target_price", "stop_price", "created_at", "notes"
        ]].copy()

        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0f}%")
        display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
        display_df["target_price"] = display_df["target_price"].apply(lambda x: f"${x:.2f}")
        display_df["stop_price"] = display_df["stop_price"].apply(lambda x: f"${x:.2f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "direction": st.column_config.TextColumn(
                    "Direction",
                    help="BUY or SELL"
                )
            }
        )
    else:
        st.info("No active signals matching your filters.")

    # Signal details
    if not filtered.empty:
        st.markdown("---")
        st.subheader("Signal Details")

        selected_ticker = st.selectbox(
            "Select ticker to view chart",
            filtered["ticker"].unique()
        )

        if selected_ticker:
            # Get price data
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

    # Get open positions
    positions_df = services["positions"].get_open_positions()

    # Summary metrics
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

    # Open positions table
    st.markdown("---")
    st.subheader("Open Positions")

    if not positions_df.empty:
        display_cols = [
            "ticker", "direction", "entry_price", "shares",
            "entry_date", "strategy", "status"
        ]
        available_cols = [c for c in display_cols if c in positions_df.columns]
        display_df = positions_df[available_cols].copy()

        if "entry_price" in display_df.columns:
            display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
        if "shares" in display_df.columns:
            display_df["shares"] = display_df["shares"].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    # Closed positions
    st.markdown("---")
    st.subheader("Recent Closed Positions")

    closed_df = services["positions"].get_closed_positions(
        start_date=date.today() - timedelta(days=30)
    )

    if not closed_df.empty:
        display_cols = [
            "ticker", "direction", "entry_price", "exit_price",
            "pnl", "pnl_pct", "exit_reason", "strategy"
        ]
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


def render_performance_page(services: dict):
    """Render Performance Analytics page."""
    st.title("📊 Performance Analytics")

    # Overall metrics
    performance = services["positions"].get_performance_summary()

    st.markdown("---")
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

    # Strategy performance
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

        # Strategy table
        st.dataframe(strategy_perf, use_container_width=True, hide_index=True)
    else:
        st.info("No strategy performance data available yet.")

    # P&L Distribution
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

    # Get backtest results
    results_df = services["backtester"].get_backtest_results()

    if not results_df.empty:
        st.markdown("---")
        st.subheader("Backtest Results")

        # Summary table
        display_cols = [
            "strategy", "total_return_pct", "annualized_return_pct",
            "sharpe_ratio", "max_drawdown_pct", "win_rate",
            "profit_factor", "total_trades", "run_date"
        ]
        available_cols = [c for c in display_cols if c in results_df.columns]
        display_df = results_df[available_cols].copy()

        # Format percentages
        pct_cols = ["total_return_pct", "annualized_return_pct", "max_drawdown_pct", "win_rate"]
        for col in pct_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")

        # Format ratios
        ratio_cols = ["sharpe_ratio", "profit_factor"]
        for col in ratio_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Comparison chart
        st.markdown("---")
        st.subheader("Strategy Comparison")

        fig = create_performance_chart(
            results_df[["strategy", "total_return_pct"]].rename(columns={"total_return_pct": "total_pnl"}),
            "total_pnl"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No backtest results available. Run backtests to see results here.")

    # Run new backtest
    st.markdown("---")
    st.subheader("Run New Backtest")

    col1, col2 = st.columns(2)

    with col1:
        strategy_options = [
            "rsi_mean_reversion",
            "bollinger_squeeze",
            "macd_volume",
            "zscore_mean_reversion",
            "momentum_breakout"
        ]
        selected_strategy = st.selectbox("Strategy", strategy_options)

    with col2:
        start_date = st.date_input("Start Date", date(2023, 1, 1))
        end_date = st.date_input("End Date", date.today())

    if st.button("Run Backtest"):
        st.info("Backtest functionality requires historical data. Make sure data is loaded first.")


def render_watchlist_page(services: dict):
    """Render Long-Term Watchlist page."""
    st.title("📋 Long-Term Watchlist")

    # Get watchlist
    watchlist_df = services["db"].fetchdf("""
        SELECT * FROM long_term_watchlist
        WHERE scan_date >= DATE('now', '-7 days')
        ORDER BY composite_score DESC
    """)

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

        # Details for selected ticker
        st.markdown("---")
        selected_ticker = st.selectbox(
            "Select ticker for details",
            watchlist_df["ticker"].unique()
        )

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

            # Price chart
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

    # Display current configuration
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

    # Universe
    st.markdown("---")
    st.subheader("Stock Universe")

    universe_df = services["universe"].get_universe_df()

    if not universe_df.empty:
        st.metric("Active Stocks", len(universe_df[universe_df["is_active"] == True]))

        with st.expander("View Universe"):
            st.dataframe(universe_df, use_container_width=True, hide_index=True)

    # System status
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
