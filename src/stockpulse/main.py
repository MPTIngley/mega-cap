"""
StockPulse Main Entry Point.

Usage:
    stockpulse run          # Run scheduler and scan
    stockpulse dashboard    # Launch Streamlit dashboard
    stockpulse backtest     # Run backtests
    stockpulse ingest       # Run data ingestion
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import date

# Auto-load .env file before anything else
def _load_env():
    """Automatically load .env file from project root."""
    # Find .env file - check current dir and parent dirs
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

from stockpulse.utils.config import load_config
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="StockPulse - Automated Stock Scanning & Trading Signal System"
    )

    parser.add_argument(
        "command",
        choices=["run", "dashboard", "backtest", "ingest", "scan", "init", "optimize", "reset", "test-email", "digest", "longterm-backtest", "longterm-scan", "longterm-backfill", "longterm-reset", "fundamentals-refresh", "pe-backfill", "add-holding", "close-holding", "holdings"],
        help="Command to execute"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name for backtest"
    )

    parser.add_argument(
        "--keep-market-data",
        action="store_true",
        default=True,
        help="Keep historical price data when resetting (default: True)"
    )

    parser.add_argument(
        "--clear-all",
        action="store_true",
        default=False,
        help="Clear ALL data including historical prices (use with caution)"
    )

    # Holdings tracking arguments
    parser.add_argument(
        "--ticker",
        type=str,
        help="Stock ticker symbol for holdings commands"
    )

    parser.add_argument(
        "--buy-date",
        type=str,
        help="Purchase date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--buy-price",
        type=float,
        help="Purchase price per share"
    )

    parser.add_argument(
        "--shares",
        type=float,
        help="Number of shares"
    )

    parser.add_argument(
        "--strategy-type",
        type=str,
        choices=["active", "long_term"],
        default="long_term",
        help="Strategy type: 'active' or 'long_term'"
    )

    parser.add_argument(
        "--holding-id",
        type=int,
        help="Holding ID for close-holding command"
    )

    parser.add_argument(
        "--sell-date",
        type=str,
        help="Sell date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--sell-price",
        type=float,
        help="Sell price per share"
    )

    parser.add_argument(
        "--notes",
        type=str,
        help="Optional notes for the holding"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script location
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    try:
        load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Execute command
    if args.command == "run":
        run_scheduler()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "backtest":
        run_backtest(args.strategy)
    elif args.command == "ingest":
        run_ingestion()
    elif args.command == "scan":
        run_scan()
    elif args.command == "init":
        run_init()
    elif args.command == "optimize":
        run_optimize()
    elif args.command == "reset":
        run_reset(keep_market_data=not args.clear_all)
    elif args.command == "test-email":
        run_test_email()
    elif args.command == "digest":
        run_digest()
    elif args.command == "longterm-backtest":
        run_longterm_backtest()
    elif args.command == "longterm-scan":
        run_longterm_scan()
    elif args.command == "longterm-backfill":
        run_longterm_backfill()
    elif args.command == "longterm-reset":
        run_longterm_reset()
    elif args.command == "fundamentals-refresh":
        run_fundamentals_refresh()
    elif args.command == "pe-backfill":
        run_pe_backfill()
    elif args.command == "add-holding":
        run_add_holding(args)
    elif args.command == "close-holding":
        run_close_holding(args)
    elif args.command == "holdings":
        run_show_holdings(args)


def run_digest():
    """Send daily portfolio digest email now."""
    from stockpulse.alerts.alert_manager import AlertManager

    print("\nSending daily portfolio digest...")
    alert_manager = AlertManager()
    success = alert_manager.send_daily_digest()

    if success:
        print("[SUCCESS] Daily digest sent!")
    else:
        print("[FAILED] Could not send digest. Check email configuration.")


def run_test_email():
    """Test email configuration and send a test email."""
    import os
    from stockpulse.alerts.email_sender import EmailSender

    print("\n" + "="*60)
    print("EMAIL CONFIGURATION TEST")
    print("="*60)

    # Check environment variables
    sender = os.environ.get("STOCKPULSE_EMAIL_SENDER", "")
    recipient = os.environ.get("STOCKPULSE_EMAIL_RECIPIENT", "")
    password = os.environ.get("STOCKPULSE_EMAIL_PASSWORD", "")
    cc = os.environ.get("STOCKPULSE_EMAIL_RECIPIENTS_CC", "")

    print(f"\nEnvironment Variables:")
    print(f"  STOCKPULSE_EMAIL_SENDER:    {'[SET]' if sender else '[MISSING]'} {sender if sender else ''}")
    print(f"  STOCKPULSE_EMAIL_RECIPIENT: {'[SET]' if recipient else '[MISSING]'} {recipient if recipient else ''}")
    print(f"  STOCKPULSE_EMAIL_PASSWORD:  {'[SET]' if password else '[MISSING]'} {'*' * len(password) if password else ''}")
    print(f"  STOCKPULSE_EMAIL_RECIPIENTS_CC: {cc if cc else '[not set]'}")

    # Check .env file location
    from pathlib import Path
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent.parent / ".env",
        Path.home() / "mega-cap" / ".env",
    ]
    print(f"\n.env file search paths:")
    for p in env_paths:
        exists = p.exists()
        print(f"  {p}: {'[EXISTS]' if exists else '[not found]'}")

    # Initialize email sender
    print(f"\nInitializing EmailSender...")
    email_sender = EmailSender()

    print(f"  is_configured: {email_sender.is_configured}")
    print(f"  smtp_server: {email_sender.smtp_server}")
    print(f"  smtp_port: {email_sender.smtp_port}")

    if not email_sender.is_configured:
        print("\n[ERROR] Email is NOT configured. Check your .env file.")
        print("Make sure the .env file is in the project root directory.")
        return

    # Try to send a test email
    print(f"\nSending test email...")
    from datetime import datetime

    subject = f"StockPulse Test Email - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    body_html = """
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h1 style="color: #2c3e50;">StockPulse Email Test</h1>
        <p>If you're reading this, your email configuration is working correctly!</p>
        <p style="color: #27ae60; font-weight: bold;">Configuration verified.</p>
        <hr>
        <p style="color: #666; font-size: 12px;">This is an automated test email from StockPulse.</p>
    </body>
    </html>
    """

    try:
        result = email_sender.send_email(subject, body_html)
        if result:
            print(f"\n[SUCCESS] Test email sent!")
            print(f"  To: {recipient}")
            if cc:
                print(f"  CC: {cc}")
            print(f"\nCheck your inbox (and spam folder).")
        else:
            print(f"\n[FAILED] send_email returned False")
    except Exception as e:
        print(f"\n[ERROR] Exception while sending: {e}")
        import traceback
        traceback.print_exc()


def run_scheduler():
    """Run the scheduler for continuous scanning."""
    from datetime import datetime
    import time
    import pytz
    from stockpulse.scheduler import StockPulseScheduler
    from stockpulse.strategies.signal_generator import SignalGenerator
    from stockpulse.strategies.position_manager import PositionManager
    from stockpulse.scanner.long_term_scanner import LongTermScanner
    from stockpulse.alerts.alert_manager import AlertManager
    from stockpulse.strategies.base import SignalDirection
    from stockpulse.utils.config import get_config

    logger.info("Starting StockPulse scheduler...")

    scheduler = StockPulseScheduler()
    signal_generator = SignalGenerator()
    position_manager = PositionManager()
    long_term_scanner = LongTermScanner()
    alert_manager = AlertManager()
    config = get_config()

    # Helper function to check market hours
    def is_market_open() -> tuple[bool, str]:
        """Check if market is currently open. Returns (is_open, status_message)."""
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)

        # Check weekday
        if now.weekday() >= 5:
            day_name = "Saturday" if now.weekday() == 5 else "Sunday"
            return False, f"CLOSED ({day_name} - market reopens Monday 9:30 AM ET)"

        # Get market hours from config
        scanning_config = config.get("scanning", {})
        market_open_str = scanning_config.get("market_open", "09:30")
        market_close_str = scanning_config.get("market_close", "16:00")

        market_open = datetime.strptime(market_open_str, "%H:%M").time()
        market_close = datetime.strptime(market_close_str, "%H:%M").time()

        current_time = now.time()

        if current_time < market_open:
            # Before market open
            return False, f"CLOSED (pre-market - opens at {market_open_str} ET)"
        elif current_time > market_close:
            # After market close
            return False, f"CLOSED (after-hours - reopens tomorrow 9:30 AM ET)"
        else:
            return True, f"OPEN ({market_open_str} - {market_close_str} ET)"

    def on_intraday_scan(tickers):
        """Callback for intraday scans - generates signals, opens positions, sends alerts."""
        from stockpulse.strategies.signal_insights import SignalInsights

        print("\n" + "=" * 60)
        print("  INTRADAY SCAN RUNNING")
        print("=" * 60)
        print(f"  Scanning {len(tickers)} tickers...")

        signals = signal_generator.generate_signals(tickers)
        signal_insights = SignalInsights()

        # Separate BUY and SELL signals
        buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]
        sell_signals = [s for s in signals if s.direction == SignalDirection.SELL]

        print(f"  Found: {len(buy_signals)} BUY signals, {len(sell_signals)} SELL signals")
        logger.info(f"Generated {len(buy_signals)} BUY signals, {len(sell_signals)} SELL signals")

        # Per-strategy breakdown with near-misses
        all_strategies = ["rsi_mean_reversion", "macd_volume", "zscore_mean_reversion",
                         "momentum_breakout", "week52_low_bounce", "sector_rotation"]
        strategy_signals = {s: [] for s in all_strategies}
        for sig in buy_signals:
            if sig.strategy in strategy_signals:
                strategy_signals[sig.strategy].append(sig)

        # Get near-misses for strategies without signals
        near_misses = signal_insights.get_near_misses(tickers, top_n=3)

        # Get strategy capacity status
        strategy_status = signal_insights.get_strategy_status(position_manager)

        print("\n  Per-Strategy Breakdown:")
        print("-" * 60)
        for strat in all_strategies:
            sigs = strategy_signals.get(strat, [])
            status = strategy_status.get(strat, {})
            capacity_info = f"[{status.get('current_exposure_pct', 0):.0f}%/{status.get('max_allowed_pct', 70):.0f}% used]"

            if sigs:
                # Sort by confidence and show top 3
                top_sigs = sorted(sigs, key=lambda s: s.confidence, reverse=True)[:3]
                print(f"  {strat}: {len(sigs)} signals {capacity_info}")
                for sig in top_sigs:
                    upside = ((sig.target_price - sig.entry_price) / sig.entry_price * 100) if sig.entry_price > 0 else 0
                    print(f"    ✓ {sig.ticker} @ ${sig.entry_price:.2f}: {sig.confidence:.0f}% conf, +{upside:.1f}% upside")
            else:
                # Show near-misses when no signals
                nm = near_misses.get(strat, [])
                if nm:
                    print(f"  {strat}: 0 signals (near-misses below) {capacity_info}")
                    for stock in nm:
                        price_str = f"${stock['price']:.2f}" if stock.get('price') else "N/A"
                        print(f"    ○ {stock['ticker']} @ {price_str}: {stock['indicator']} - {stock['distance']}")
                else:
                    print(f"  {strat}: 0 signals (no stocks near criteria) {capacity_info}")
        print("-" * 60)

        if not signals:
            print("  Result: No signals generated")
            print("=" * 60 + "\n")
            logger.info("No signals generated")
            return

        # Get current portfolio state
        open_positions_df = position_manager.get_open_positions()
        portfolio_tickers = set(open_positions_df["ticker"].tolist()) if not open_positions_df.empty else set()

        # Calculate current portfolio exposure
        current_exposure_pct = 0.0
        if not open_positions_df.empty:
            total_invested = (open_positions_df["entry_price"] * open_positions_df["shares"]).sum()
            current_exposure_pct = (total_invested / position_manager.initial_capital) * 100

        # Filter SELL signals to stocks we own
        actionable_sells = [s for s in sell_signals if s.ticker in portfolio_tickers]

        # Get constraints from config
        risk_config = config.get("risk_management", {})
        max_positions = config.get("portfolio", {}).get("max_positions", 40)
        max_exposure_pct = risk_config.get("max_portfolio_exposure_pct", 80.0)

        # Rank signals by expected return: confidence × upside potential
        def expected_return(signal):
            upside = (signal.target_price - signal.entry_price) / signal.entry_price if signal.entry_price > 0 else 0
            return signal.confidence * upside * 100

        ranked_buys = sorted(buy_signals, key=expected_return, reverse=True)

        # Track what happens to each signal
        opened_positions = []  # (signal, size_pct, dollar_amount, sizing_details)
        blocked_signals = []   # (signal, reason, detailed_reasons)

        running_exposure_pct = current_exposure_pct
        current_positions = len(portfolio_tickers)
        min_position_size_pct = risk_config.get("min_position_size_pct", 3.0)

        # === CIRCUIT BREAKER CHECK ===
        # Block new entries if market drops > 3% intraday
        from stockpulse.utils.market_health import check_circuit_breaker
        circuit_breaker_ok, cb_message = check_circuit_breaker()
        print(f"  {cb_message}")

        if not circuit_breaker_ok:
            # Circuit breaker triggered - block ALL new entries
            for signal in ranked_buys:
                blocked_signals.append((signal, "Circuit breaker active", [{"icon": "🚨", "reason": cb_message}]))
            ranked_buys = []  # Clear all buys
            logger.warning(f"Circuit breaker blocked {len(blocked_signals)} potential trades")

        for signal in ranked_buys:
            # Calculate position size first (needed for add-to-position check)
            size_pct, sizing_details = position_manager.calculate_position_size_pct(signal, return_details=True)

            # Check if already in portfolio - but allow adding if conditions met
            if signal.ticker in portfolio_tickers:
                can_add, add_reason = position_manager._can_add_to_position(signal.ticker, size_pct)
                if not can_add:
                    blocked_signals.append((signal, add_reason, [{"icon": "📌", "reason": add_reason}]))
                    continue
                # Can add to position - continue with other checks
            else:
                # New position - check position limit
                if current_positions >= max_positions:
                    blocked_signals.append((signal, f"Max positions ({max_positions}) reached", []))
                    continue

            # Check if this would exceed portfolio exposure limit
            if running_exposure_pct + size_pct > max_exposure_pct:
                # Try to fit a smaller position
                remaining_pct = max_exposure_pct - running_exposure_pct
                if remaining_pct >= min_position_size_pct:
                    size_pct = remaining_pct
                else:
                    blocked_signals.append((signal, f"Portfolio exposure limit ({max_exposure_pct}%) reached", []))
                    continue

            # Check if this would exceed strategy concentration limit
            # Note: get_strategy_remaining_capacity_pct queries DB which includes positions
            # opened earlier in this scan, so no need to track separately
            strategy = signal.strategy
            strategy_remaining = position_manager.get_strategy_remaining_capacity_pct(strategy)

            if size_pct > strategy_remaining:
                # Try to fit a smaller position
                if strategy_remaining >= min_position_size_pct:
                    size_pct = strategy_remaining
                    logger.info(f"Reduced {signal.ticker} to {size_pct:.1f}% to fit strategy limit")
                else:
                    current_strat_pct = position_manager.get_strategy_current_exposure_pct(strategy)
                    blocked_signals.append((
                        signal,
                        f"Strategy {strategy} limit reached ({position_manager.max_per_strategy_pct}%)",
                        [{"icon": "📊", "reason": f"Strategy at {current_strat_pct:.1f}% (max {position_manager.max_per_strategy_pct:.0f}%)"}]
                    ))
                    continue

            # Check cooldown (churn prevention and loss cooldown)
            cooldown_ok, cooldown_reason = position_manager._check_cooldown(signal.ticker)
            if not cooldown_ok:
                blocked_signals.append((
                    signal,
                    cooldown_reason,
                    [{"icon": "⏱️", "reason": cooldown_reason}]
                ))
                continue

            # Check loss limit (too many consecutive losses on this ticker)
            loss_ok, loss_reason = position_manager._check_loss_limit(signal.ticker)
            if not loss_ok:
                blocked_signals.append((
                    signal,
                    loss_reason,
                    [{"icon": "🚫", "reason": loss_reason}]
                ))
                continue

            # Check sector concentration
            sector_ok, sector_reason = position_manager._check_sector_concentration(signal.ticker)
            if not sector_ok:
                blocked_signals.append((
                    signal,
                    sector_reason,
                    [{"icon": "📊", "reason": sector_reason}]
                ))
                continue

            # Try to open the position with the (potentially reduced) size
            try:
                pos_id = position_manager.open_position_from_signal(signal, override_size_pct=size_pct)
                if pos_id:
                    dollar_amount = position_manager.initial_capital * (size_pct / 100)
                    opened_positions.append((signal, size_pct, dollar_amount, sizing_details))
                    running_exposure_pct += size_pct
                    current_positions += 1
                    portfolio_tickers.add(signal.ticker)
                    d = sizing_details
                    logger.info(f"Opened {signal.ticker} at {size_pct:.1f}%: {d['base_size_pct']}% × {d['strategy_weight']:.1f}strat × {d['confidence_mult']:.2f}conf")
                else:
                    # Position manager returned None - get detailed blocking reasons
                    detailed_reasons = signal_insights.get_blocking_reasons(signal.ticker, position_manager)
                    if detailed_reasons:
                        reason_summary = "; ".join([r["reason"] for r in detailed_reasons])
                        blocked_signals.append((signal, reason_summary, detailed_reasons))
                    else:
                        blocked_signals.append((signal, "Blocked by risk check", []))
            except Exception as e:
                blocked_signals.append((signal, f"Error: {str(e)[:50]}", []))
                logger.error(f"Failed to open position for {signal.ticker}: {e}")

        # Print scan results
        print("-" * 60)
        if opened_positions:
            print(f"  OPENED: {len(opened_positions)} positions")
            for signal, size_pct, dollar_amount, _ in opened_positions:
                print(f"    + {signal.ticker}: {size_pct:.1f}% (${dollar_amount:,.0f})")
            logger.info(f"Opened {len(opened_positions)} positions, total exposure: {running_exposure_pct:.1f}%")
        else:
            print("  OPENED: 0 positions")

        if blocked_signals:
            print(f"  BLOCKED: {len(blocked_signals)} signals")
            for item in blocked_signals[:5]:
                signal = item[0]
                reason = item[1]
                detailed = item[2] if len(item) > 2 else []
                print(f"    - {signal.ticker} ({signal.confidence:.0f}% conf): {reason}")
                # Show additional details if available
                for detail in detailed:
                    if detail.get("detail"):
                        print(f"        {detail.get('icon', '')} {detail.get('detail', '')}")
            if len(blocked_signals) > 5:
                print(f"    ... and {len(blocked_signals) - 5} more")

        if actionable_sells:
            print(f"  SELL SIGNALS: {len(actionable_sells)} (holdings to consider selling)")

        print(f"  Portfolio exposure: {running_exposure_pct:.1f}%")

        # Update existing positions (check stops/targets) - MUST use LIVE prices!
        from stockpulse.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        # Get tickers with open positions
        open_pos_tickers = list(portfolio_tickers) if portfolio_tickers else []
        if open_pos_tickers:
            # Fetch LIVE prices for position exit checks - NOT database prices!
            print(f"  Checking {len(open_pos_tickers)} open positions against live prices...")
            live_prices = ingestion.fetch_current_prices(open_pos_tickers)
            if live_prices:
                updates = position_manager.update_positions(live_prices)
                for update in updates:
                    alert_manager.process_position_exit(update)
                    print(f"  → Closed {update['ticker']}: {update['exit_reason']} @ ${update['exit_price']:.2f}")

        # Send consolidated email ONLY if actual trades occurred (opened or actionable sells)
        initial_capital = position_manager.initial_capital

        # Build per-strategy signal summary with action status
        strategy_signal_summary = {}
        for strat in all_strategies:
            strat_signals = strategy_signals.get(strat, [])
            strat_summary = []
            for sig in sorted(strat_signals, key=lambda s: s.confidence, reverse=True)[:10]:
                # Check if this signal was opened
                was_opened = any(op[0].ticker == sig.ticker for op in opened_positions)
                # Check if it was blocked and why
                blocked_entry = next((b for b in blocked_signals if b[0].ticker == sig.ticker), None)

                if was_opened:
                    status = "OPENED"
                    reason = ""
                elif blocked_entry:
                    status = "BLOCKED"
                    reason = blocked_entry[1]
                else:
                    status = "NOT_ACTED"
                    reason = "Unknown"

                strat_summary.append({
                    "ticker": sig.ticker,
                    "confidence": sig.confidence,
                    "entry_price": sig.entry_price,
                    "target_price": sig.target_price,
                    "status": status,
                    "reason": reason,
                })
            strategy_signal_summary[strat] = strat_summary

        # Only send email if we made actual trades
        if opened_positions or actionable_sells:
            email_sent = alert_manager.send_scan_results_email(
                opened_positions=opened_positions,
                blocked_signals=blocked_signals[:10],  # Top 10 blocked
                sell_signals=actionable_sells,
                portfolio_exposure_pct=running_exposure_pct,
                initial_capital=initial_capital,
                near_misses=near_misses,
                strategy_status=strategy_status,
                strategy_signal_summary=strategy_signal_summary
            )

            if email_sent:
                print("  Email: SENT")
            else:
                print("  Email: Not sent (quiet hours)")
        else:
            print("  Email: Not sent (no trades executed)")

        print("=" * 60 + "\n")

    def on_daily_scan(tickers):
        """Callback for daily scans."""
        # Same as intraday but runs once at end of day
        on_intraday_scan(tickers)

    def on_long_term_scan(tickers):
        """Callback for long-term scanner."""
        opportunities = long_term_scanner.run_scan(tickers)
        long_term_scanner.send_digest(opportunities)

    def on_daily_digest():
        """Callback for daily portfolio digest email."""
        alert_manager.send_daily_digest()

    # Timezone and job names for schedule display
    et = pytz.timezone("US/Eastern")
    job_names = {
        "intraday_open": "Opening scan (09:32)",
        "intraday_first": "First 15-min (09:45)",
        "intraday_scan": "15-min scans (10:00-15:45)",
        "intraday_close": "Closing scan (15:58)",
        "daily_scan": "Daily scan",
        "daily_digest": "Daily digest",
        "long_term_scan": "Long-term scan",
    }

    def print_schedule():
        """Print full schedule after scan completes."""
        next_runs = scheduler.get_next_run_times()
        now = datetime.now(et)
        market_open, market_status = is_market_open()

        print("\n" + "-" * 50)
        print(f"  Date: {now.strftime('%Y-%m-%d')}  Time: {now.strftime('%H:%M ET')}  Market: {'OPEN' if market_open else 'CLOSED'}")

        # Check if weekend
        if now.weekday() >= 5:
            day_name = "Saturday" if now.weekday() == 5 else "Sunday"
            print(f"\n  📅 {day_name} - Markets closed until Monday 9:30 AM ET")
            print("-" * 50 + "\n")
            return

        print("\n  Daily Schedule:")

        # Define the trading day order
        day_order = [
            "intraday_open",    # 09:32
            "intraday_first",   # 09:45
            "intraday_scan",    # 10:00-15:45
            "intraday_close",   # 15:58
            "daily_scan",       # 16:30
            "daily_digest",     # 17:00
            "long_term_scan",   # 17:30
        ]

        # Find the next job (soonest)
        next_job_id = None
        next_job_time = None
        for job_id, next_time in next_runs.items():
            if next_time:
                if next_job_time is None or next_time < next_job_time:
                    next_job_time = next_time
                    next_job_id = job_id

        # Print in trading day order
        for job_id in day_order:
            if job_id in next_runs and next_runs[job_id]:
                next_time = next_runs[job_id]
                name = job_names.get(job_id, job_id)
                time_str = next_time.strftime('%H:%M ET')
                marker = "→" if job_id == next_job_id else " "
                print(f"  {marker} {time_str}  {name}")

        print("-" * 50 + "\n")

    # Wrap callbacks to print schedule after completion
    def on_intraday_scan_wrapper(tickers):
        on_intraday_scan(tickers)
        print_schedule()

    def on_daily_scan_wrapper(tickers):
        on_daily_scan(tickers)
        print_schedule()

    def on_long_term_scan_wrapper(tickers):
        on_long_term_scan(tickers)
        print_schedule()

    def on_daily_digest_wrapper():
        on_daily_digest()
        print_schedule()

    scheduler.on_intraday_scan = on_intraday_scan_wrapper
    scheduler.on_daily_scan = on_daily_scan_wrapper
    scheduler.on_long_term_scan = on_long_term_scan_wrapper
    scheduler.on_daily_digest = on_daily_digest_wrapper

    scheduler.start()

    logger.info("Scheduler started. Press Ctrl+C to stop.")

    try:
        last_status = ""

        # Show schedule at startup
        print_schedule()

        while True:
            try:
                # Show countdown to next scan
                next_runs = scheduler.get_next_run_times()
                now = datetime.now(et)

                # Check market status
                market_open, market_status = is_market_open()

                # Find the next job to run
                next_job = None
                next_job_time = None
                for job_id, next_time in next_runs.items():
                    if next_time:
                        if next_job_time is None or next_time < next_job_time:
                            next_job_time = next_time
                            next_job = job_id

                # Calculate countdown
                if next_job_time:
                    delta = next_job_time - now
                    total_seconds = int(delta.total_seconds())
                    if total_seconds > 0:
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, _ = divmod(remainder, 60)

                        # Build progress bar (24 chars = 24 hours max display)
                        if hours < 24:
                            filled = max(0, 24 - hours)
                            bar = "█" * filled + "░" * (24 - filled)
                        else:
                            bar = "░" * 24

                        next_job_name = job_names.get(next_job, next_job)
                        next_time_str = next_job_time.strftime('%H:%M ET')

                        # Build compact status line (include seconds for clarity)
                        status = (
                            f"\r  ⏱ {now.strftime('%H:%M:%S ET')} | "
                            f"Market: {'OPEN' if market_open else 'CLOSED'} | "
                            f"Next: {next_job_name} @ {next_time_str} ({hours}h {minutes}m) "
                            f"[{bar}]  "
                        )

                        # Always print (seconds ensure it changes each cycle)
                        print(status, end="", flush=True)
                    else:
                        # Job is running or overdue, show that
                        next_job_name = job_names.get(next_job, next_job)
                        status = (
                            f"\r  ⏱ {now.strftime('%H:%M:%S ET')} | "
                            f"Market: {'OPEN' if market_open else 'CLOSED'} | "
                            f"Running: {next_job_name}...  "
                        )
                        print(status, end="", flush=True)

            except Exception as e:
                # Don't let display errors freeze the loop
                logger.debug(f"Display update error: {e}")

            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print()  # New line after countdown
        logger.info("Shutting down...")
        scheduler.stop()


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    logger.info("Launching Streamlit dashboard...")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true"
    ])


def run_backtest(strategy_name: str | None = None):
    """Run backtests."""
    from stockpulse.strategies.backtest import Backtester
    from stockpulse.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    from stockpulse.strategies.bollinger_squeeze import BollingerSqueezeStrategy
    from stockpulse.strategies.macd_volume import MACDVolumeStrategy
    from stockpulse.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy
    from stockpulse.strategies.momentum_breakout import MomentumBreakoutStrategy
    from stockpulse.data.universe import UniverseManager
    from stockpulse.utils.config import get_config

    config = get_config()
    backtester = Backtester()
    universe = UniverseManager()

    # Get tickers
    tickers = universe.get_active_tickers()
    if not tickers:
        logger.info("No tickers in universe, using defaults")
        from stockpulse.data.universe import TOP_US_STOCKS
        tickers = TOP_US_STOCKS[:50]

    # Define strategies
    strategy_classes = {
        "rsi_mean_reversion": RSIMeanReversionStrategy,
        "bollinger_squeeze": BollingerSqueezeStrategy,
        "macd_volume": MACDVolumeStrategy,
        "zscore_mean_reversion": ZScoreMeanReversionStrategy,
        "momentum_breakout": MomentumBreakoutStrategy,
    }

    strategies_config = config.get("strategies", {})

    if strategy_name:
        if strategy_name not in strategy_classes:
            logger.error(f"Unknown strategy: {strategy_name}")
            return

        strategies_to_test = {strategy_name: strategy_classes[strategy_name]}
    else:
        strategies_to_test = strategy_classes

    for name, cls in strategies_to_test.items():
        logger.info(f"Running backtest for {name}...")

        strategy_config = strategies_config.get(name, {})
        strategy = cls(strategy_config)

        result = backtester.run_backtest(strategy, tickers)

        logger.info(f"""
        {name} Backtest Results:
        ========================
        Total Return: {result.total_return_pct:.2f}%
        Annualized Return: {result.annualized_return_pct:.2f}%
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Max Drawdown: {result.max_drawdown_pct:.2f}%
        Win Rate: {result.win_rate:.2f}%
        Profit Factor: {result.profit_factor:.2f}
        Total Trades: {result.total_trades}
        """)


def run_ingestion():
    """Run data ingestion."""
    from stockpulse.data.universe import UniverseManager
    from stockpulse.data.ingestion import DataIngestion

    logger.info("Running data ingestion...")

    universe = UniverseManager()
    ingestion = DataIngestion()

    # Refresh universe
    tickers = universe.refresh_universe(force=True)
    logger.info(f"Universe has {len(tickers)} tickers")

    # Run ingestion
    results = ingestion.run_daily_ingestion(tickers)

    logger.info(f"Ingestion complete: {results}")


def run_scan():
    """Run a single scan with market snapshot."""
    from stockpulse.data.universe import UniverseManager
    from stockpulse.strategies.signal_generator import SignalGenerator
    from stockpulse.alerts.alert_manager import AlertManager
    from stockpulse.strategies.position_manager import PositionManager
    from stockpulse.data.ingestion import DataIngestion
    from stockpulse.strategies.base import SignalDirection
    from stockpulse.utils.config import get_config
    import pandas as pd
    import numpy as np

    logger.info("Running scan...")

    universe = UniverseManager()
    signal_generator = SignalGenerator()
    alert_manager = AlertManager()
    position_manager = PositionManager()
    ingestion = DataIngestion()
    config = get_config()

    tickers = universe.get_active_tickers()
    if not tickers:
        logger.warning("No tickers in universe")
        return

    # Fetch fresh intraday data before scanning
    print("  Fetching latest market data...")
    try:
        ingestion.run_intraday_ingestion(tickers)
        print("  ✓ Data refreshed")
    except Exception as e:
        print(f"  ⚠ Data refresh failed: {e}")
        logger.warning(f"Data ingestion failed, using cached data: {e}")

    # Get current portfolio positions
    open_positions_df = position_manager.get_open_positions()
    portfolio_tickers = set(open_positions_df["ticker"].tolist()) if not open_positions_df.empty else set()

    # Get allocation weights from config
    allocation_weights = config.get("strategy_allocation", {})
    base_position_pct = config.get("risk_management", {}).get("max_position_size_pct", 5.0)
    initial_capital = config.get("portfolio", {}).get("initial_capital", 100000.0)

    # Generate signals
    all_signals = signal_generator.generate_signals(tickers)

    # Separate BUY and SELL signals
    buy_signals = [s for s in all_signals if s.direction == SignalDirection.BUY]
    sell_signals = [s for s in all_signals if s.direction == SignalDirection.SELL]

    # Filter SELL signals to only stocks we own
    actionable_sells = [s for s in sell_signals if s.ticker in portfolio_tickers]
    informational_sells = [s for s in sell_signals if s.ticker not in portfolio_tickers]

    # Get signal insights for per-strategy breakdown
    from stockpulse.strategies.signal_insights import SignalInsights
    signal_insights = SignalInsights()

    # Group buy signals by strategy
    all_strategies = ["rsi_mean_reversion", "macd_volume", "zscore_mean_reversion",
                     "momentum_breakout", "week52_low_bounce", "sector_rotation"]
    strategy_signals = {s: [] for s in all_strategies}
    for sig in buy_signals:
        if sig.strategy in strategy_signals:
            strategy_signals[sig.strategy].append(sig)

    # Get near-misses and strategy capacity
    near_misses = signal_insights.get_near_misses(tickers, top_n=3)
    strategy_status = signal_insights.get_strategy_status(position_manager)

    # Import strategy descriptions
    from stockpulse.strategies.signal_insights import STRATEGY_DESCRIPTIONS

    print("\n" + "=" * 80)
    print("  STOCKPULSE SCAN RESULTS")
    print("=" * 80)

    # === BUY SIGNALS ===
    print(f"\n  📈 BUY SIGNALS: {len(buy_signals)}")
    print("-" * 80)

    if buy_signals:
        # Sort by confidence
        buy_signals.sort(key=lambda s: s.confidence, reverse=True)
        for signal in buy_signals:
            weight = allocation_weights.get(signal.strategy, 1.0)
            position_size = base_position_pct * weight
            dollar_amount = initial_capital * (position_size / 100)
            shares = int(dollar_amount / signal.entry_price)

            print(f"  {signal.ticker:5} | {signal.strategy:25} | Conf: {signal.confidence:.0f}%")
            print(f"         Entry: ${signal.entry_price:.2f} → Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}")
            print(f"         Allocation: {position_size:.1f}% (${dollar_amount:,.0f}) = ~{shares} shares")
            if signal.notes:
                print(f"         Why: {signal.notes}")
            print()
    else:
        print("  No buy signals\n")

    # === PER-STRATEGY BREAKDOWN (after buy signals for easier reference) ===
    print(f"\n  📊 PER-STRATEGY BREAKDOWN:")
    print("-" * 80)
    for strat in all_strategies:
        sigs = strategy_signals.get(strat, [])
        status = strategy_status.get(strat, {})
        capacity_info = f"[{status.get('current_exposure_pct', 0):.0f}%/{status.get('max_allowed_pct', 70):.0f}% used]"
        strat_desc = STRATEGY_DESCRIPTIONS.get(strat, {}).get("short", strat)

        # Print strategy header with description
        print(f"\n  {strat_desc}")
        print(f"    Capacity: {capacity_info}")

        if sigs:
            top_sigs = sorted(sigs, key=lambda s: s.confidence, reverse=True)[:3]
            print(f"    {len(sigs)} signal(s):")
            for sig in top_sigs:
                upside = ((sig.target_price - sig.entry_price) / sig.entry_price * 100) if sig.entry_price > 0 else 0
                print(f"      ✓ {sig.ticker}: {sig.confidence:.0f}% confidence, +{upside:.1f}% potential upside")
        else:
            nm = near_misses.get(strat, [])
            if nm:
                print(f"    0 signals, but {len(nm)} stock(s) close to triggering:")
                for stock in nm:
                    print(f"      ○ {stock['ticker']}: {stock['indicator']} ({stock['distance']})")
            else:
                print(f"    0 signals (no stocks near criteria)")
    print("\n" + "-" * 80)

    # === ACTIONABLE SELL SIGNALS (stocks we own) ===
    print(f"\n  📉 SELL SIGNALS (Portfolio): {len(actionable_sells)}")
    print("-" * 80)

    if actionable_sells:
        actionable_sells.sort(key=lambda s: s.confidence, reverse=True)
        for signal in actionable_sells:
            print(f"  {signal.ticker:5} | {signal.strategy:25} | Conf: {signal.confidence:.0f}%")
            print(f"         Exit: ${signal.entry_price:.2f} → Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}")
            if signal.notes:
                print(f"         Why: {signal.notes}")
            print()
    else:
        print("  No sell signals for current holdings\n")

    # === INFORMATIONAL SELL SIGNALS (stocks we don't own) ===
    if informational_sells:
        print(f"\n  ℹ️  SELL SIGNALS (Not in Portfolio): {len(informational_sells)}")
        print("-" * 80)
        # Just show tickers, not full details
        sell_tickers = [s.ticker for s in informational_sells[:10]]
        print(f"  {', '.join(sell_tickers)}{'...' if len(informational_sells) > 10 else ''}")
        print("  (These are overbought stocks - avoid buying)\n")

    # === PORTFOLIO STATUS ===
    print("\n  💼 PORTFOLIO STATUS")
    print("-" * 80)
    if not open_positions_df.empty:
        # Fetch LIVE prices for portfolio positions
        portfolio_tickers_list = open_positions_df["ticker"].tolist()
        live_portfolio_prices = ingestion.fetch_current_prices(portfolio_tickers_list)

        print(f"  Open positions: {len(open_positions_df)}")
        total_value = 0
        total_cost = 0
        for _, pos in open_positions_df.iterrows():
            ticker = pos['ticker']
            entry_price = pos.get('entry_price', 0)
            shares = pos.get('shares', 0)
            cost_basis = entry_price * shares

            # Get live price and calculate P&L
            current_price = live_portfolio_prices.get(ticker, entry_price)
            current_value = current_price * shares
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_dollar = current_value - cost_basis

            total_value += current_value
            total_cost += cost_basis

            # Color indicator for P&L
            pnl_indicator = "📈" if pnl_pct >= 0 else "📉"
            print(f"    {ticker:5} | Entry: ${entry_price:.2f} → Now: ${current_price:.2f} | {pnl_indicator} {pnl_pct:+.1f}% (${pnl_dollar:+,.0f})")

        # Show portfolio totals
        total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        total_pnl_dollar = total_value - total_cost
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  Portfolio: ${total_value:,.0f} | Cost: ${total_cost:,.0f} | P&L: {total_pnl_pct:+.1f}% (${total_pnl_dollar:+,.0f})")
    else:
        print("  No open positions (paper portfolio empty)")
    print()

    # === MARKET SNAPSHOT ===
    print("-" * 80)
    print("  MARKET SNAPSHOT - Stocks Near Threshold (Live Prices)")
    print("-" * 80)

    try:
        from datetime import timedelta, datetime
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            et_tz = pytz.timezone("America/New_York")

        end_date = date.today()
        start_date = end_date - timedelta(days=60)

        # Get historical data for indicator calculation
        prices_df = ingestion.get_daily_prices(tickers[:30], start_date, end_date)

        # Fetch LIVE current prices
        live_prices = ingestion.fetch_current_prices(tickers[:30])
        price_fetch_time = datetime.now(et_tz).strftime("%Y-%m-%d %H:%M:%S ET")

        if not prices_df.empty:
            near_triggers = []

            for ticker in prices_df["ticker"].unique():
                ticker_data = prices_df[prices_df["ticker"] == ticker].copy()
                if len(ticker_data) < 20:
                    continue

                # Ensure date column is consistent type for sorting
                ticker_data["date"] = pd.to_datetime(ticker_data["date"])
                ticker_data = ticker_data.sort_values("date")

                # Update last row with live price for accurate indicator calculation
                if ticker in live_prices and live_prices[ticker] > 0:
                    close = live_prices[ticker]
                    ticker_data.iloc[-1, ticker_data.columns.get_loc("close")] = close
                else:
                    close = ticker_data.iloc[-1]["close"]

                # Calculate RSI with live-updated data
                delta = ticker_data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                # Calculate Z-score with live-updated data
                mean_20 = ticker_data["close"].rolling(20).mean().iloc[-1]
                std_20 = ticker_data["close"].rolling(20).std().iloc[-1]
                zscore = (close - mean_20) / std_20 if std_20 > 0 else 0

                status = []
                if current_rsi < 35:
                    status.append(f"RSI={current_rsi:.1f}")
                if zscore < -1.5:
                    status.append(f"Z={zscore:.2f}")
                if current_rsi > 65:
                    status.append(f"RSI={current_rsi:.1f}(OB)")

                if status:
                    near_triggers.append((ticker, close, " | ".join(status)))

            if near_triggers:
                print(f"  (prices updated at {price_fetch_time})")
                for ticker, price, status in near_triggers[:10]:
                    print(f"  {ticker:5} @ ${price:>8.2f} | {status}")
            else:
                print("  No stocks near threshold triggers")

    except Exception as e:
        logger.debug(f"Could not generate market snapshot: {e}")
        print("  Could not generate market snapshot")

    print("\n" + "=" * 80)
    print(f"  Scan complete. {len(buy_signals)} buys, {len(actionable_sells)} actionable sells.")
    print("=" * 80 + "\n")

    # === AUTO-OPEN POSITIONS FOR TOP BUY SIGNALS ===
    # Calculate current portfolio exposure
    current_exposure_pct = 0.0
    if not open_positions_df.empty:
        total_invested = (open_positions_df["entry_price"] * open_positions_df["shares"]).sum()
        current_exposure_pct = (total_invested / initial_capital) * 100

    risk_config = config.get("risk_management", {})
    max_positions = config.get("portfolio", {}).get("max_positions", 40)
    max_exposure_pct = risk_config.get("max_portfolio_exposure_pct", 80.0)

    # Rank by expected return
    def expected_return(signal):
        upside = (signal.target_price - signal.entry_price) / signal.entry_price if signal.entry_price > 0 else 0
        return signal.confidence * upside * 100

    ranked_buys = sorted(buy_signals, key=expected_return, reverse=True)

    opened_positions = []
    blocked_signals = []
    running_exposure_pct = current_exposure_pct
    current_positions = len(portfolio_tickers)
    min_position_size_pct = risk_config.get("min_position_size_pct", 3.0)

    # === CIRCUIT BREAKER CHECK ===
    from stockpulse.utils.market_health import check_circuit_breaker
    circuit_breaker_ok, cb_message = check_circuit_breaker()

    print(f"\n  🎯 OPENING POSITIONS (max {max_positions}, max {max_exposure_pct}% exposure)")
    print("-" * 80)
    print(f"  {cb_message}")

    if not circuit_breaker_ok:
        # Circuit breaker triggered - block ALL new entries
        for signal in ranked_buys:
            blocked_signals.append((signal, "Circuit breaker active", [{"icon": "🚨", "reason": cb_message}]))
        ranked_buys = []  # Clear all buys
        logger.warning(f"Circuit breaker blocked {len(blocked_signals)} potential trades")
        print(f"  🚨 Circuit breaker active - no new positions will be opened")

    for signal in ranked_buys:
        # Calculate position size first (needed for add-to-position check)
        size_pct, sizing_details = position_manager.calculate_position_size_pct(signal, return_details=True)

        # Check if already in portfolio - but allow adding if conditions met
        if signal.ticker in portfolio_tickers:
            can_add, add_reason = position_manager._can_add_to_position(signal.ticker, size_pct)
            if not can_add:
                blocked_signals.append((signal, add_reason, [{"icon": "📌", "reason": add_reason}]))
                print(f"  📌 {signal.ticker}: {add_reason}")
                continue
            print(f"  📈 {signal.ticker}: Adding to existing position")
        else:
            # New position - check position limit
            if current_positions >= max_positions:
                blocked_signals.append((signal, f"Max positions ({max_positions}) reached", []))
                continue

        if running_exposure_pct + size_pct > max_exposure_pct:
            remaining_pct = max_exposure_pct - running_exposure_pct
            if remaining_pct >= min_position_size_pct:
                size_pct = remaining_pct
            else:
                blocked_signals.append((signal, f"Portfolio exposure limit ({max_exposure_pct}%) reached", []))
                continue

        # Check if this would exceed strategy concentration limit
        # Note: get_strategy_remaining_capacity_pct queries DB which includes positions
        # opened earlier in this scan, so no need to track separately
        strategy = signal.strategy
        strategy_remaining = position_manager.get_strategy_remaining_capacity_pct(strategy)

        if size_pct > strategy_remaining:
            # Try to fit a smaller position
            if strategy_remaining >= min_position_size_pct:
                size_pct = strategy_remaining
                print(f"  📉 {signal.ticker}: reduced to {size_pct:.1f}% to fit strategy limit")
            else:
                current_strat_pct = position_manager.get_strategy_current_exposure_pct(strategy)
                blocked_signals.append((
                    signal,
                    f"Strategy {strategy} limit reached ({position_manager.max_per_strategy_pct}%)",
                    [{"icon": "📊", "reason": f"Strategy at {current_strat_pct:.1f}%"}]
                ))
                continue

        # Check cooldown (churn prevention and loss cooldown)
        cooldown_ok, cooldown_reason = position_manager._check_cooldown(signal.ticker)
        if not cooldown_ok:
            blocked_signals.append((
                signal,
                cooldown_reason,
                [{"icon": "⏱️", "reason": cooldown_reason}]
            ))
            print(f"  ⏱️ {signal.ticker}: {cooldown_reason}")
            continue

        # Check loss limit (too many consecutive losses on this ticker)
        loss_ok, loss_reason = position_manager._check_loss_limit(signal.ticker)
        if not loss_ok:
            blocked_signals.append((
                signal,
                loss_reason,
                [{"icon": "🚫", "reason": loss_reason}]
            ))
            print(f"  🚫 {signal.ticker}: {loss_reason}")
            continue

        # Check sector concentration
        sector_ok, sector_reason = position_manager._check_sector_concentration(signal.ticker)
        if not sector_ok:
            blocked_signals.append((
                signal,
                sector_reason,
                [{"icon": "📊", "reason": sector_reason}]
            ))
            print(f"  📊 {signal.ticker}: {sector_reason}")
            continue

        try:
            pos_id = position_manager.open_position_from_signal(signal, override_size_pct=size_pct)
            if pos_id:
                dollar_amount = initial_capital * (size_pct / 100)
                opened_positions.append((signal, size_pct, dollar_amount, sizing_details))
                running_exposure_pct += size_pct
                current_positions += 1
                portfolio_tickers.add(signal.ticker)
                # Show sizing calculation: base × strategy × conf_mult = raw → final
                d = sizing_details
                cap_note = " [CAPPED]" if d["was_capped"] else ""
                print(f"  ✅ {signal.ticker}: {size_pct:.1f}% (${dollar_amount:,.0f}) - {signal.strategy}")
                print(f"      Sizing: {d['base_size_pct']}% × {d['strategy_weight']:.1f}strat × {d['confidence_mult']:.2f}conf = {d['raw_size_pct']:.1f}% → {d['final_size_pct']:.1f}%{cap_note}")
            else:
                blocked_signals.append((signal, "Blocked by risk check", []))
                print(f"  ❌ {signal.ticker}: blocked by risk limits")
        except Exception as e:
            blocked_signals.append((signal, f"Error: {str(e)[:30]}", []))
            print(f"  ❌ {signal.ticker}: error - {e}")

    print(f"\n  Summary: {len(opened_positions)} opened, {len(blocked_signals)} blocked, {running_exposure_pct:.1f}% exposure")

    # Build per-strategy signal summary with action status
    strategy_signal_summary = {}
    for strat in all_strategies:
        strat_signals = strategy_signals.get(strat, [])
        strat_summary = []
        for sig in sorted(strat_signals, key=lambda s: s.confidence, reverse=True)[:10]:
            was_opened = any(op[0].ticker == sig.ticker for op in opened_positions)
            blocked_entry = next((b for b in blocked_signals if b[0].ticker == sig.ticker), None)

            if was_opened:
                status = "OPENED"
                reason = ""
            elif blocked_entry:
                status = "BLOCKED"
                reason = blocked_entry[1]
            else:
                status = "NOT_ACTED"
                reason = "Unknown"

            strat_summary.append({
                "ticker": sig.ticker,
                "confidence": sig.confidence,
                "entry_price": sig.entry_price,
                "target_price": sig.target_price,
                "status": status,
                "reason": reason,
            })
        strategy_signal_summary[strat] = strat_summary

    # === SEND RESULTS EMAIL ===
    if opened_positions or blocked_signals or actionable_sells:
        alert_manager.send_scan_results_email(
            opened_positions=opened_positions,
            blocked_signals=blocked_signals[:10],
            sell_signals=actionable_sells,
            portfolio_exposure_pct=running_exposure_pct,
            initial_capital=initial_capital,
            near_misses=near_misses,
            strategy_status=strategy_status,
            strategy_signal_summary=strategy_signal_summary
        )

    # Signal dashboard to refresh by updating last_scan_completed timestamp
    from stockpulse.data.database import get_db
    from datetime import datetime
    db = get_db()
    db.execute("""
        INSERT INTO system_state (key, value, updated_at)
        VALUES ('last_scan_completed', ?, current_timestamp)
        ON CONFLICT (key) DO UPDATE SET
            value = excluded.value,
            updated_at = current_timestamp
    """, (datetime.now().isoformat(),))
    logger.debug("Scan completion signal written to database")


def run_init():
    """Initialize the database and fetch initial data with full historical preload."""
    from stockpulse.data.database import get_db
    from stockpulse.data.universe import UniverseManager
    from stockpulse.data.ingestion import DataIngestion
    from datetime import timedelta

    logger.info("Initializing StockPulse with full data preload...")

    # Initialize database (creates tables)
    db = get_db()
    logger.info("Database initialized")

    # Refresh universe
    universe = UniverseManager()
    tickers = universe.refresh_universe(force=True)
    logger.info(f"Universe populated with {len(tickers)} tickers")

    # Fetch historical data - full 2 years for backtesting
    ingestion = DataIngestion()

    logger.info("Fetching 2 years of historical daily price data...")
    start_date = date.today() - timedelta(days=730)
    daily_df = ingestion.fetch_daily_prices(tickers, start_date=start_date, progress=True)
    daily_count = ingestion.store_daily_prices(daily_df)
    logger.info(f"Stored {daily_count} daily price records")

    logger.info("Fetching 5 days of intraday data...")
    intraday_df = ingestion.fetch_intraday_prices(tickers, period="5d")
    intraday_count = ingestion.store_intraday_prices(intraday_df)
    logger.info(f"Stored {intraday_count} intraday price records")

    logger.info("Fetching fundamentals...")
    fundamentals_df = ingestion.fetch_fundamentals(tickers[:50])  # Top 50 for fundamentals
    fundamentals_count = ingestion.store_fundamentals(fundamentals_df)
    logger.info(f"Stored {fundamentals_count} fundamental records")

    logger.info(f"""
    ==========================================
    StockPulse Initialization Complete!
    ==========================================
    Universe: {len(tickers)} stocks
    Daily prices: {daily_count} records (2 years)
    Intraday prices: {intraday_count} records (5 days)
    Fundamentals: {fundamentals_count} records

    Ready for trading! Run one of:
      - stockpulse run        # Start scheduler
      - stockpulse dashboard  # Launch dashboard
      - stockpulse scan       # Run single scan
      - stockpulse backtest   # Run backtests
    ==========================================
    """)


def run_optimize():
    """Run hyperparameter optimization for all strategies."""
    from stockpulse.strategies.optimizer import StrategyOptimizer, STRATEGY_CLASSES
    from stockpulse.data.universe import UniverseManager
    from datetime import timedelta
    import time

    logger.info("Starting hyperparameter optimization...")

    print("\n" + "=" * 70)
    print("  STOCKPULSE STRATEGY OPTIMIZER")
    print("  Maximizing returns with max 25% drawdown constraint")
    print("=" * 70)

    # Get tickers
    universe = UniverseManager()
    tickers = universe.get_active_tickers()

    if not tickers:
        print("\n  ⚠ No tickers in universe. Run 'stockpulse init' first.")
        logger.error("No tickers in universe. Run 'stockpulse init' first.")
        return

    # Use subset for faster optimization
    tickers = tickers[:30]
    print(f"\n  Using {len(tickers)} tickers for optimization")
    print(f"  Tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

    # Date range - use 18 months of data
    end_date = date.today()
    start_date = end_date - timedelta(days=540)
    print(f"  Date range: {start_date} to {end_date}")

    # Check if we have price data
    from stockpulse.data.database import get_db
    db = get_db()
    try:
        price_count = db.fetchone("SELECT COUNT(*) FROM prices_daily")
        price_count = price_count[0] if price_count else 0
        print(f"  Price records in database: {price_count:,}")
        if price_count < 1000:
            print("\n  ⚠ WARNING: Not enough price data for reliable optimization.")
            print("  Run 'stockpulse init' to fetch 2 years of historical data.")
            return
    except Exception as e:
        print(f"\n  ⚠ Database error: {e}")
        print("  Run 'stockpulse init' to initialize the database.")
        return

    # Initialize optimizer
    optimizer = StrategyOptimizer(max_drawdown_pct=25.0)

    all_results = {}
    total_strategies = len(STRATEGY_CLASSES)

    for i, strategy_name in enumerate(STRATEGY_CLASSES.keys(), 1):
        print(f"\n  [{i}/{total_strategies}] Optimizing {strategy_name}...")
        print("-" * 70)

        start_time = time.time()

        try:
            def progress_callback(current, total, result):
                bar_width = 40
                filled = int(bar_width * current / total)
                bar = "█" * filled + "░" * (bar_width - filled)
                ret = result.get("total_return_pct", 0)
                dd = result.get("max_drawdown_pct", 0)
                print(f"\r  [{bar}] {current}/{total} | Return: {ret:+.1f}% | DD: {dd:.1f}%", end="", flush=True)

            result = optimizer.optimize(
                strategy_name=strategy_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                objective="sharpe",
                max_iterations=50,
                progress_callback=progress_callback
            )

            all_results[strategy_name] = result

            elapsed = time.time() - start_time
            print()  # New line after progress bar

            # Show results
            constraint_status = "✓" if result.constraint_satisfied else "✗"
            print(f"\n  {constraint_status} {strategy_name}")
            print(f"    Final Value: ${result.final_value:,.0f} (from $100k)")
            print(f"    Return: {result.best_return:+.2f}% ± {result.return_std_pct:.1f}%")
            print(f"    Sharpe: {result.best_sharpe:.2f}")
            print(f"    Max DD: {result.best_drawdown:.2f}%")
            print(f"    Time: {elapsed:.1f}s")

            # Show key params
            key_params = {k: v for k, v in result.best_params.items()
                         if k not in ["enabled", "_optimized"]}
            print(f"    Params: {key_params}")

        except Exception as e:
            import traceback
            logger.error(f"Failed to optimize {strategy_name}: {e}")
            print(f"\n  ✗ {strategy_name} - FAILED: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 70)

    print("\n  Strategy                     Final Value     Return ± Std    Sharpe    Max DD   Constraint")
    print("  " + "-" * 94)

    for name, result in all_results.items():
        status = "✓ Met" if result.constraint_satisfied else "✗ Exceeded"
        print(f"  {name:28} ${result.final_value:>10,.0f}  {result.best_return:+6.1f}% ± {result.return_std_pct:4.1f}%  {result.best_sharpe:6.2f}  {result.best_drawdown:6.1f}%  {status}")

    # Save to config
    print("\n" + "-" * 70)
    print("  Saving optimized parameters to config/config.yaml...")

    try:
        optimizer.save_optimized_params(all_results)
        print("  ✓ Parameters saved successfully!")

        # Also save detailed results to JSON for future analysis
        json_file = optimizer.save_detailed_results(all_results)
        print(f"  ✓ Detailed results saved to: {json_file}")

        print("\n  To use new parameters:")
        print("    1. Restart stockpulse run/dashboard")
        print("    2. Commit and push: git add config/config.yaml && git commit -m 'Optimized params' && git push")
    except Exception as e:
        logger.error(f"Failed to save params: {e}")
        print(f"  ✗ Failed to save: {e}")

    print("\n" + "=" * 70 + "\n")


def run_reset(keep_market_data: bool = True):
    """Reset trading data while keeping historical prices."""
    from stockpulse.data.database import reset_trading_data

    print("\n" + "=" * 60)
    print("  STOCKPULSE DATA RESET")
    print("=" * 60)

    if keep_market_data:
        print("\n  Mode: Reset trading data ONLY")
        print("  ✓ Will KEEP: prices_daily, prices_intraday, fundamentals, universe")
        print("  ✗ Will DELETE: signals, positions, alerts, backtest_results")
    else:
        print("\n  Mode: FULL RESET (including historical data)")
        print("  ✗ Will DELETE: ALL data including historical prices")
        print("  ⚠ You will need to run 'stockpulse init' to fetch data again")

    print("\n  Resetting...")

    deleted = reset_trading_data(keep_market_data=keep_market_data)

    print("\n  Deleted records:")
    for table, count in deleted.items():
        print(f"    {table}: {count}")

    total = sum(deleted.values())
    print(f"\n  Total: {total} records deleted")

    if keep_market_data:
        print("\n  ✓ Historical market data preserved!")
        print("  Ready to run: stockpulse optimize")
    else:
        print("\n  ✓ All data cleared!")
        print("  Run: stockpulse init")

    print("\n" + "=" * 60 + "\n")


def run_longterm_backtest():
    """Run long-term scanner backtesting with 3-year hold strategy."""
    from stockpulse.scanner.long_term_backtest import run_long_term_backtest
    run_long_term_backtest()


def run_longterm_scan():
    """Run long-term scanner and display results."""
    from stockpulse.scanner.long_term_scanner import LongTermScanner
    from stockpulse.data.universe import UniverseManager

    print("\n" + "=" * 70)
    print("  LONG-TERM INVESTMENT SCANNER")
    print("  Scoring: Valuation, Technical, Quality, Insider, FCF, Earnings, Peers")
    print("=" * 70)

    universe = UniverseManager()
    tickers = universe.get_active_tickers()

    if not tickers:
        print("\n  No tickers in universe. Run 'stockpulse init' first.")
        return

    scanner = LongTermScanner()

    # Show database stats
    stats = scanner.get_watchlist_stats()
    print(f"\n  Historical Data: {stats['total_records']} records, "
          f"{stats['unique_dates']} unique dates, {stats['unique_tickers']} tickers")
    if stats['total_records'] == 0:
        print("  ⚠️  No historical data! Run 'stockpulse longterm-backfill' to build history")
    elif stats.get('records_last_7_days', 0) < 3:
        print(f"  ⚠️  Only {stats.get('records_last_7_days', 0)} records in last 7 days")
    else:
        print(f"  Date range: {stats.get('earliest_date')} to {stats.get('latest_date')}")

    print(f"\n  Scanning {len(tickers)} stocks...")
    opportunities = scanner.run_scan(tickers)

    if not opportunities:
        print("\n  No opportunities found meeting minimum score threshold.")
        print("=" * 70 + "\n")
        return

    print(f"\n  Found {len(opportunities)} opportunities (score >= 60)")
    print("\n" + "-" * 70)
    print("  TOP LONG-TERM OPPORTUNITIES")
    print("-" * 70)

    # Display top opportunities
    for i, opp in enumerate(opportunities[:15], 1):
        ticker = opp["ticker"]
        score = opp["composite_score"]
        reasoning = opp["reasoning"]

        print(f"\n  {i:2}. {ticker:5} | Score: {score:.0f}")

        # Show component scores
        components = []
        if opp.get("insider_score", 50) >= 65:
            components.append(f"Insider: {opp['insider_score']:.0f}")
        if opp.get("fcf_yield_score", 50) >= 65:
            components.append(f"FCF: {opp['fcf_yield_score']:.0f}")
        if opp.get("earnings_momentum_score", 50) >= 65:
            components.append(f"Earnings: {opp['earnings_momentum_score']:.0f}")
        if opp.get("valuation_score", 50) >= 65:
            components.append(f"Value: {opp['valuation_score']:.0f}")
        if opp.get("technical_score", 50) >= 65:
            components.append(f"Tech: {opp['technical_score']:.0f}")

        if components:
            print(f"      Highlights: {', '.join(components)}")

        print(f"      {reasoning}")

    # Summary table
    print("\n" + "-" * 70)
    print("  SCORE BREAKDOWN (Top 10)")
    print("-" * 70)
    print("  Ticker  Total  Value  Tech  Insdr  FCF   Earns  Peers  Div   Qual")
    print("  " + "-" * 68)

    for opp in opportunities[:10]:
        print(f"  {opp['ticker']:6} "
              f" {opp['composite_score']:5.0f} "
              f" {opp.get('valuation_score', 50):5.0f} "
              f" {opp.get('technical_score', 50):4.0f} "
              f" {opp.get('insider_score', 50):5.0f} "
              f" {opp.get('fcf_yield_score', 50):4.0f} "
              f" {opp.get('earnings_momentum_score', 50):5.0f} "
              f" {opp.get('peer_valuation_score', 50):5.0f} "
              f" {opp.get('dividend_score', 50):4.0f} "
              f" {opp.get('quality_score', 50):4.0f}")

    print("\n" + "=" * 70)
    print(f"  Scan complete. {len(opportunities)} opportunities saved to database.")
    print("=" * 70 + "\n")

    # Send digest if configured
    scanner_config = scanner.config.get("long_term_scanner", {})
    if scanner_config.get("send_digest", True):
        print("  Sending opportunity digest email...")
        scanner.send_digest(opportunities[:10])


def run_longterm_backfill():
    """Backfill 6 weeks of long-term scanner history."""
    from stockpulse.scanner.long_term_scanner import LongTermScanner

    print("\n" + "=" * 70)
    print("  LONG-TERM SCANNER BACKFILL")
    print("  Building 6 weeks of historical scan data")
    print("=" * 70 + "\n")

    scanner = LongTermScanner()

    print("  This may take a few minutes...")
    print("  Progress will be logged below.\n")

    records = scanner.backfill_history(days=42)

    print("\n" + "=" * 70)
    print(f"  Backfill complete: {records} historical records created")
    print("  You can now run 'stockpulse longterm-scan' to see trends")
    print("=" * 70 + "\n")


def run_longterm_reset():
    """Clear the long-term watchlist table for fresh data."""
    from stockpulse.data.database import get_db

    print("\n" + "=" * 70)
    print("  LONG-TERM WATCHLIST RESET")
    print("=" * 70)

    db = get_db()

    # Count existing records
    result = db.fetchone("SELECT COUNT(*) FROM long_term_watchlist")
    count = result[0] if result else 0

    print(f"\n  Current records: {count}")
    print("  This will delete all long-term watchlist data.")
    print("  Run 'stockpulse longterm-scan' after to repopulate with fresh data.\n")

    confirm = input("  Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        db.execute("DELETE FROM long_term_watchlist")
        print(f"\n  ✅ Deleted {count} records from long_term_watchlist")
        print("  Run 'stockpulse longterm-scan' to populate fresh data")
    else:
        print("\n  ❌ Cancelled")

    print("=" * 70 + "\n")


def run_fundamentals_refresh():
    """Refresh fundamentals data for all tickers in universe."""
    from stockpulse.data.ingestion import DataIngestion
    from stockpulse.data.universe import UniverseManager

    print("\n" + "=" * 70)
    print("  FUNDAMENTALS REFRESH")
    print("=" * 70)

    universe = UniverseManager()
    ingestion = DataIngestion()

    tickers = universe.get_active_tickers()
    print(f"\n  Fetching fundamentals for {len(tickers)} tickers...")
    print("  This may take a few minutes due to API rate limits.\n")

    fundamentals_df = ingestion.fetch_fundamentals(tickers)

    if not fundamentals_df.empty:
        count = ingestion.store_fundamentals(fundamentals_df)
        print(f"\n  ✅ Stored fundamentals for {count} tickers")
        print("  P/E percentile will improve as historical data accumulates.")
    else:
        print("\n  ❌ No fundamentals data retrieved")

    print("=" * 70 + "\n")


def run_pe_backfill():
    """Backfill historical P/E ratios from FMP and/or calculated values."""
    import os
    from stockpulse.data.universe import UniverseManager
    from stockpulse.data.fmp_data import FMPDataSource, backfill_calculated_pe

    print("\n" + "=" * 70)
    print("  P/E RATIO HISTORICAL BACKFILL")
    print("=" * 70)

    universe = UniverseManager()
    tickers = universe.get_active_tickers()

    print(f"\n  Tickers to process: {len(tickers)}")

    # Check if FMP API key is available
    fmp = FMPDataSource()
    fmp_available = fmp.is_available()

    if fmp_available:
        print("\n  📊 Method 1: Financial Modeling Prep (FMP) API")
        print("  Fetching 5 years of historical P/E ratios...")
        print("  (250 requests/day free tier)\n")

        fmp_records = fmp.backfill_historical_pe(tickers, years=5)
        print(f"\n  ✅ FMP: {fmp_records} historical P/E records stored")
    else:
        print("\n  ⚠️  FMP_API_KEY not set - skipping FMP backfill")
        print("  Get free key at: https://site.financialmodelingprep.com/developer/docs")

    print("\n  📊 Method 2: Calculate P/E from Price / EPS")
    print("  Using historical prices and current trailing EPS...")
    print("  (Note: yfinance API calls may be slow - be patient)\n")

    calc_records = backfill_calculated_pe(tickers, days_back=365, verbose=True)
    print(f"\n  ✅ Calculated: {calc_records} P/E records stored")

    # Show summary
    from stockpulse.data.database import get_db
    db = get_db()
    total = db.fetchone("SELECT COUNT(*) FROM fundamentals WHERE pe_ratio IS NOT NULL")
    unique_dates = db.fetchone("SELECT COUNT(DISTINCT date) FROM fundamentals WHERE pe_ratio IS NOT NULL")

    print("\n" + "-" * 50)
    print(f"  Total P/E records in database: {total[0] if total else 0}")
    print(f"  Unique dates with P/E data: {unique_dates[0] if unique_dates else 0}")
    print("-" * 50)
    print("\n  P/E percentile calculations will now be more accurate!")
    print("=" * 70 + "\n")


def run_add_holding(args):
    """Add a new holding to the tracker."""
    from stockpulse.tracker.holdings_tracker import HoldingsTracker
    from datetime import date as date_type

    tracker = HoldingsTracker()

    # Validate required arguments
    if not args.ticker:
        print("Error: --ticker is required")
        print("Usage: stockpulse add-holding --ticker AAPL --buy-date 2024-01-15 --buy-price 185.50 --shares 10")
        return

    if not args.buy_price:
        print("Error: --buy-price is required")
        return

    if not args.shares:
        print("Error: --shares is required")
        return

    # Use today's date if not specified
    buy_date = args.buy_date if args.buy_date else date_type.today().isoformat()

    holding_id = tracker.add_holding(
        ticker=args.ticker,
        buy_date=buy_date,
        buy_price=args.buy_price,
        shares=args.shares,
        strategy_type=args.strategy_type,
        notes=args.notes
    )

    cost = args.buy_price * args.shares
    print(f"\n  Added holding #{holding_id}")
    print(f"  {args.ticker}: {args.shares} shares @ ${args.buy_price:.2f} = ${cost:,.2f}")
    print(f"  Strategy: {args.strategy_type}")
    print(f"  Date: {buy_date}")
    if args.notes:
        print(f"  Notes: {args.notes}")
    print()


def run_close_holding(args):
    """Close an existing holding."""
    from stockpulse.tracker.holdings_tracker import HoldingsTracker
    from datetime import date as date_type

    tracker = HoldingsTracker()

    if not args.holding_id:
        # Show open holdings to help user
        print("\nOpen holdings:")
        holdings_df = tracker.get_open_holdings()
        if holdings_df.empty:
            print("  No open holdings")
        else:
            for _, row in holdings_df.iterrows():
                print(f"  #{row['id']}: {row['ticker']} - {row['shares']} shares @ ${row['buy_price']:.2f} ({row['strategy_type']})")
        print("\nUsage: stockpulse close-holding --holding-id 1 --sell-date 2024-06-15 --sell-price 195.00")
        return

    if not args.sell_price:
        print("Error: --sell-price is required")
        return

    # Use today's date if not specified
    sell_date = args.sell_date if args.sell_date else date_type.today().isoformat()

    try:
        result = tracker.close_holding(
            holding_id=args.holding_id,
            sell_date=sell_date,
            sell_price=args.sell_price
        )

        print(f"\n  Closed holding #{args.holding_id}")
        print(f"  {result['ticker']}: {result['shares']:.2f} shares")
        print(f"  Buy: ${result['buy_price']:.2f} -> Sell: ${result['sell_price']:.2f}")
        print(f"  P&L: ${result['realized_pnl']:+,.2f} ({result['realized_pnl_pct']:+.1f}%)")
        print()

    except ValueError as e:
        print(f"\nError: {e}")


def run_show_holdings(args):
    """Show current holdings."""
    from stockpulse.tracker.holdings_tracker import print_holdings_summary

    strategy = args.strategy_type if hasattr(args, 'strategy_type') and args.strategy_type != "long_term" else None

    # Show both by default, or filtered by strategy
    if strategy:
        print_holdings_summary(strategy)
    else:
        print_holdings_summary("long_term")
        print_holdings_summary("active")


if __name__ == "__main__":
    main()
