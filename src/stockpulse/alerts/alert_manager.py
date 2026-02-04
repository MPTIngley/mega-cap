"""Alert Manager - coordinates when and how to send alerts."""

from datetime import datetime, time
from typing import Any

import pandas as pd
import pytz

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.database import get_db
from stockpulse.strategies.base import Signal
from stockpulse.strategies.position_manager import PositionManager

from .email_sender import EmailSender

logger = get_logger(__name__)


class AlertManager:
    """
    Manages alert triggers and delivery.

    Handles:
    - Signal alerts (when new high-confidence signals fire)
    - Position exit alerts
    - Daily digest emails
    - Error alerts
    - Quiet hours enforcement
    """

    def __init__(self):
        """Initialize alert manager."""
        self.db = get_db()
        self.config = get_config()
        self.alerts_config = self.config.get("alerts", {})
        self.email_sender = EmailSender()
        self.position_manager = PositionManager()

        self.timezone = pytz.timezone(
            self.config.get("scanning", {}).get("timezone", "US/Eastern")
        )

        # Alert thresholds
        self.min_confidence = self.alerts_config.get("min_confidence_for_email", 70)
        self.send_on_signal = self.alerts_config.get("send_on_new_signal", True)
        self.send_on_exit = self.alerts_config.get("send_on_position_exit", True)
        self.send_digest = self.alerts_config.get("send_daily_digest", True)
        self.send_errors = self.alerts_config.get("send_error_alerts", True)

        # Quiet hours
        quiet_start = self.alerts_config.get("quiet_hours_start", "22:00")
        quiet_end = self.alerts_config.get("quiet_hours_end", "07:00")
        self.quiet_start = datetime.strptime(quiet_start, "%H:%M").time()
        self.quiet_end = datetime.strptime(quiet_end, "%H:%M").time()

    def _is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours."""
        now = datetime.now(self.timezone).time()

        # Handle overnight quiet hours (e.g., 22:00 to 07:00)
        if self.quiet_start > self.quiet_end:
            return now >= self.quiet_start or now <= self.quiet_end
        else:
            return self.quiet_start <= now <= self.quiet_end

    def _log_alert(
        self,
        alert_type: str,
        signal_id: int | None,
        subject: str,
        body: str,
        success: bool,
        error: str | None = None
    ) -> None:
        """Log alert to database."""
        try:
            recipient = self.email_sender.recipient
            self.db.execute("""
                INSERT INTO alerts_log (signal_id, alert_type, recipient, subject, body, sent_successfully, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, alert_type, recipient, subject, body, success, error))
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    def process_new_signal(self, signal: Signal) -> bool:
        """
        Process a new signal and send alert if appropriate.

        Args:
            signal: The new trading signal

        Returns:
            True if alert was sent
        """
        if not self.send_on_signal:
            return False

        if signal.confidence < self.min_confidence:
            logger.debug(f"Signal confidence {signal.confidence} below threshold {self.min_confidence}")
            return False

        if self._is_quiet_hours():
            logger.info(f"Quiet hours - queueing alert for {signal.ticker}")
            # Could implement queue here for later delivery
            return False

        signal_data = signal.to_dict()
        success = self.email_sender.send_signal_alert(signal_data)

        self._log_alert(
            "new_signal",
            None,  # signal_id would come from DB
            f"{signal.direction.value} {signal.ticker}",
            str(signal_data),
            success
        )

        return success

    def process_signals(self, signals: list[Signal]) -> int:
        """
        Process multiple signals and send alerts.
        DEPRECATED: Use send_consolidated_scan_alert instead.

        Args:
            signals: List of new signals

        Returns:
            Number of alerts sent
        """
        # Now sends consolidated email instead of individual ones
        if signals:
            success = self.send_consolidated_scan_alert(
                buy_signals=[s for s in signals if s.direction.value == "BUY"],
                sell_signals=[s for s in signals if s.direction.value == "SELL"],
                portfolio_tickers=set(),
                allocation_weights={},
                base_position_pct=5.0,
                initial_capital=100000.0
            )
            return 1 if success else 0
        return 0

    def send_consolidated_scan_alert(
        self,
        buy_signals: list[Signal],
        sell_signals: list[Signal],
        portfolio_tickers: set,
        allocation_weights: dict,
        base_position_pct: float,
        initial_capital: float
    ) -> bool:
        """
        Send a single consolidated email with all signals from a scan.

        Args:
            buy_signals: List of BUY signals
            sell_signals: List of actionable SELL signals (stocks in portfolio)
            portfolio_tickers: Set of tickers currently in portfolio
            allocation_weights: Strategy allocation weights from config
            base_position_pct: Base position size percentage
            initial_capital: Portfolio initial capital

        Returns:
            True if email sent successfully
        """
        if not self.send_on_signal:
            return False

        if self._is_quiet_hours():
            logger.info("Quiet hours - skipping scan alert")
            return False

        # Check if we have signals worth alerting
        high_conf_buys = [s for s in buy_signals if s.confidence >= self.min_confidence]
        high_conf_sells = [s for s in sell_signals if s.confidence >= self.min_confidence]

        if not high_conf_buys and not high_conf_sells:
            logger.info("No high-confidence signals to alert on")
            return False

        # Check if signals changed from last scan
        if not self._signals_changed(high_conf_buys, high_conf_sells):
            logger.info("Signals unchanged from last scan - skipping email")
            return False

        # Build consolidated email
        today = datetime.now().strftime('%Y-%m-%d %H:%M')
        subject = f"📊 StockPulse Scan: {len(high_conf_buys)} Buys, {len(high_conf_sells)} Sells"

        # Build BUY signals table
        buy_rows = ""
        for signal in sorted(high_conf_buys, key=lambda s: s.confidence, reverse=True)[:10]:
            weight = allocation_weights.get(signal.strategy, 1.0)
            position_pct = base_position_pct * weight
            dollar_amount = initial_capital * (position_pct / 100)
            shares = int(dollar_amount / signal.entry_price) if signal.entry_price > 0 else 0

            buy_rows += f"""
            <tr>
                <td><strong style="color: #27ae60;">{signal.ticker}</strong></td>
                <td>{signal.strategy}</td>
                <td>{signal.confidence:.0f}%</td>
                <td>${signal.entry_price:.2f}</td>
                <td>${signal.target_price:.2f}</td>
                <td>${signal.stop_price:.2f}</td>
                <td>{position_pct:.1f}% (${dollar_amount:,.0f})</td>
            </tr>
            """

        # Build SELL signals table
        sell_rows = ""
        for signal in sorted(high_conf_sells, key=lambda s: s.confidence, reverse=True)[:10]:
            sell_rows += f"""
            <tr>
                <td><strong style="color: #e74c3c;">{signal.ticker}</strong></td>
                <td>{signal.strategy}</td>
                <td>{signal.confidence:.0f}%</td>
                <td>${signal.entry_price:.2f}</td>
                <td>${signal.target_price:.2f}</td>
                <td>${signal.stop_price:.2f}</td>
                <td>IN PORTFOLIO</td>
            </tr>
            """

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #1a1a2e; color: #eee; }}
                .header {{ background: #16213e; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #16213e; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; color: #eee; }}
                th {{ background: #0f3460; }}
                .section-title {{ color: #fff; margin-top: 25px; padding: 10px; border-left: 4px solid #27ae60; }}
                .sell-title {{ border-left-color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 StockPulse Scan Results</h1>
                <p>{today}</p>
            </div>
            <div class="content">
                <h2 class="section-title">📈 BUY Signals ({len(high_conf_buys)})</h2>
                {"<table><tr><th>Ticker</th><th>Strategy</th><th>Confidence</th><th>Entry</th><th>Target</th><th>Stop</th><th>Allocation</th></tr>" + buy_rows + "</table>" if buy_rows else "<p>No high-confidence buy signals</p>"}

                <h2 class="section-title sell-title">📉 SELL Signals ({len(high_conf_sells)})</h2>
                {"<table><tr><th>Ticker</th><th>Strategy</th><th>Confidence</th><th>Exit</th><th>Target</th><th>Stop</th><th>Status</th></tr>" + sell_rows + "</table>" if sell_rows else "<p>No actionable sell signals for current holdings</p>"}

                <p style="color: #888; font-size: 12px; margin-top: 30px;">
                    Portfolio: {len(portfolio_tickers)} positions | Initial Capital: ${initial_capital:,.0f}<br>
                    <strong>Disclaimer:</strong> This is not financial advice. Paper trading only.
                </p>
            </div>
        </body>
        </html>
        """

        success = self.email_sender.send_email(subject, body_html)

        self._log_alert(
            "consolidated_scan",
            None,
            subject,
            f"Buys: {len(high_conf_buys)}, Sells: {len(high_conf_sells)}",
            success
        )

        # Store current signals for change detection
        self._store_last_signals(high_conf_buys, high_conf_sells)

        return success

    def _signals_changed(self, buy_signals: list[Signal], sell_signals: list[Signal]) -> bool:
        """Check if signals changed from last scan."""
        try:
            # Get last scan's signals from DB
            last_scan = self.db.fetchdf("""
                SELECT body FROM alerts_log
                WHERE alert_type = 'consolidated_scan'
                ORDER BY created_at DESC
                LIMIT 1
            """)

            if last_scan.empty:
                return True  # First scan, always send

            last_body = last_scan.iloc[0]["body"]
            current_tickers = set([s.ticker for s in buy_signals] + [s.ticker for s in sell_signals])
            current_str = ",".join(sorted(current_tickers))

            # Simple check: if tickers changed, send email
            return current_str not in last_body

        except Exception as e:
            logger.debug(f"Error checking signal changes: {e}")
            return True  # On error, send anyway

    def _store_last_signals(self, buy_signals: list[Signal], sell_signals: list[Signal]) -> None:
        """Store signal tickers for change detection."""
        # Already stored in _log_alert via the body field
        pass

    def send_scan_results_email(
        self,
        opened_positions: list[tuple],  # (signal, size_pct, dollar_amount, sizing_details) or (signal, size_pct, dollar_amount)
        blocked_signals: list[tuple],   # (signal, reason, detailed_reasons) or (signal, reason)
        sell_signals: list[Signal],
        portfolio_exposure_pct: float,
        initial_capital: float,
        near_misses: dict[str, list[dict]] | None = None,
        strategy_status: dict[str, dict] | None = None,
        strategy_signal_summary: dict[str, list[dict]] | None = None
    ) -> bool:
        """
        Send email showing actual scan results - what was opened and what was blocked.

        Args:
            opened_positions: List of (signal, size_pct, dollar_amount, sizing_details) for opened positions
                             sizing_details is optional for backwards compat
            blocked_signals: List of (signal, reason, detailed_reasons) for blocked signals
                            detailed_reasons is optional for backwards compat
            sell_signals: Actionable sell signals
            portfolio_exposure_pct: Current portfolio exposure percentage
            initial_capital: Initial capital amount
            near_misses: Optional dict of strategy -> list of near-miss stocks
            strategy_status: Optional dict of strategy -> status info
            strategy_signal_summary: Optional dict of strategy -> list of signal dicts with status
        """
        if self._is_quiet_hours():
            logger.info("Quiet hours - skipping scan alert")
            return False

        # Skip if nothing happened
        if not opened_positions and not blocked_signals and not sell_signals:
            logger.info("No scan activity to report")
            return False

        today = datetime.now().strftime('%Y-%m-%d %H:%M')
        subject = f"📊 StockPulse: Opened {len(opened_positions)} positions"
        if blocked_signals:
            subject += f", {len(blocked_signals)} blocked"

        # Build opened positions table
        opened_rows = ""
        total_allocated = 0.0
        for pos_data in opened_positions:
            # Handle both old format (3 elements) and new format (4 elements with sizing_details)
            if len(pos_data) >= 4:
                signal, size_pct, dollar_amount, sizing_details = pos_data[:4]
            else:
                signal, size_pct, dollar_amount = pos_data[:3]
                sizing_details = None

            total_allocated += size_pct
            upside_pct = ((signal.target_price - signal.entry_price) / signal.entry_price * 100) if signal.entry_price > 0 else 0

            # Build sizing formula string if we have details
            if sizing_details:
                d = sizing_details
                cap_note = " CAPPED" if d.get("was_capped", False) else ""
                sizing_str = f"{d['base_size_pct']}%×{d['strategy_weight']:.1f}strat×{d['confidence_mult']:.2f}conf={d['raw_size_pct']:.1f}%{cap_note}"
            else:
                sizing_str = ""

            opened_rows += f"""
            <tr>
                <td><strong style="color: #22c55e;">{signal.ticker}</strong></td>
                <td>{signal.strategy}</td>
                <td>{signal.confidence:.0f}%</td>
                <td>${signal.entry_price:.2f}</td>
                <td>${signal.target_price:.2f} (+{upside_pct:.1f}%)</td>
                <td>${signal.stop_price:.2f}</td>
                <td><strong>{size_pct:.1f}%</strong> (${dollar_amount:,.0f})<br/><small style="color: #94a3b8;">{sizing_str}</small></td>
            </tr>
            """

        # Build blocked signals table
        blocked_rows = ""
        for item in blocked_signals:
            signal = item[0]
            reason = item[1]
            detailed_reasons = item[2] if len(item) > 2 else []
            size_pct = self.position_manager.calculate_position_size_pct(signal)

            # Show detailed reasons if available
            reason_html = reason
            if detailed_reasons:
                detail_parts = [f"{r.get('icon', '')} {r.get('detail', r.get('reason', ''))}" for r in detailed_reasons if r.get('detail') or r.get('reason')]
                if detail_parts:
                    reason_html = f"{reason}<br/><small style='color: #94a3b8;'>{'; '.join(detail_parts)}</small>"

            blocked_rows += f"""
            <tr style="opacity: 0.7;">
                <td>{signal.ticker}</td>
                <td>{signal.strategy}</td>
                <td>{signal.confidence:.0f}%</td>
                <td>{size_pct:.1f}%</td>
                <td style="color: #f59e0b;">{reason_html}</td>
            </tr>
            """

        # Build sell signals table
        sell_rows = ""
        for signal in sorted(sell_signals, key=lambda s: s.confidence, reverse=True)[:5]:
            sell_rows += f"""
            <tr>
                <td><strong style="color: #ef4444;">{signal.ticker}</strong></td>
                <td>{signal.strategy}</td>
                <td>{signal.confidence:.0f}%</td>
                <td>${signal.entry_price:.2f}</td>
            </tr>
            """

        # Build strategy insights section with per-strategy signal breakdown
        from stockpulse.strategies.signal_insights import STRATEGY_DESCRIPTIONS
        strategy_insights_html = ""
        all_strategies = ["rsi_mean_reversion", "macd_volume", "zscore_mean_reversion",
                        "momentum_breakout", "week52_low_bounce", "sector_rotation"]

        if strategy_signal_summary:
            # Build per-strategy signal tables (top 3 for scan email)
            strategy_tables_html = ""
            for strat in all_strategies:
                signals_for_strat = strategy_signal_summary.get(strat, [])
                status = strategy_status.get(strat, {}) if strategy_status else {}
                exposure = status.get("current_exposure_pct", 0)
                max_pct = status.get("max_allowed_pct", 70)
                strat_desc = STRATEGY_DESCRIPTIONS.get(strat, {}).get("short", strat)

                if signals_for_strat:
                    signal_rows = ""
                    for sig in signals_for_strat[:3]:  # Top 3 for scan email
                        ticker = sig.get("ticker", "N/A")
                        conf = sig.get("confidence", 0)
                        entry = sig.get("entry_price", 0)
                        target = sig.get("target_price", 0)
                        sig_status = sig.get("status", "UNKNOWN")
                        reason = sig.get("reason", "")

                        upside = ((target - entry) / entry * 100) if entry > 0 else 0

                        # Color and icon based on status - bright colors for dark background
                        if sig_status == "OPENED":
                            status_html = "<span style='color: #4ade80;'>✅ OPENED</span>"
                        elif sig_status == "BLOCKED":
                            reason_short = reason[:40] + "..." if len(reason) > 40 else reason
                            status_html = f"<span style='color: #fbbf24;'>⏸ {reason_short}</span>"
                        else:
                            status_html = "<span style='color: #cbd5e1;'>—</span>"

                        signal_rows += f"""
                        <tr>
                            <td>{ticker}</td>
                            <td>{conf:.0f}%</td>
                            <td>${entry:.2f}</td>
                            <td>+{upside:.1f}%</td>
                            <td>{status_html}</td>
                        </tr>
                        """

                    strategy_tables_html += f"""
                    <div style="margin-bottom: 20px;">
                        <h3 style="color: #e2e8f0; font-size: 14px; margin: 10px 0 5px 0;">
                            {strat_desc}
                        </h3>
                        <p style="color: #94a3b8; font-size: 11px; margin: 0 0 8px 0;">
                            Capacity: {exposure:.0f}%/{max_pct:.0f}% used | Top {len(signals_for_strat[:3])} of {len(signals_for_strat)} signals:
                        </p>
                        <table style="font-size: 13px;">
                            <tr><th>Ticker</th><th>Conf</th><th>Entry</th><th>Upside</th><th>Status</th></tr>
                            {signal_rows}
                        </table>
                    </div>
                    """
                else:
                    # Show near-misses for strategies with no signals
                    nm = near_misses.get(strat, []) if near_misses else []
                    if nm:
                        nm_rows = ""
                        for n in nm[:3]:
                            nm_rows += f"<li>{n['ticker']}: {n['indicator']} ({n['distance']})</li>"
                        strategy_tables_html += f"""
                        <div style="margin-bottom: 15px;">
                            <h3 style="color: #e2e8f0; font-size: 14px; margin: 10px 0 5px 0;">
                                {strat_desc}
                            </h3>
                            <p style="color: #94a3b8; font-size: 11px; margin: 0 0 5px 0;">
                                Capacity: {exposure:.0f}%/{max_pct:.0f}% used | No signals, but close:
                            </p>
                            <ul style="color: #cbd5e1; font-size: 12px; margin: 5px 0; padding-left: 20px;">
                                {nm_rows}
                            </ul>
                        </div>
                        """
                    else:
                        strategy_tables_html += f"""
                        <div style="margin-bottom: 10px;">
                            <h3 style="color: #94a3b8; font-size: 14px; margin: 10px 0 5px 0;">
                                {strat_desc}
                            </h3>
                            <p style="color: #cbd5e1; font-size: 11px; margin: 0;">
                                Capacity: {exposure:.0f}%/{max_pct:.0f}% used | No signals
                            </p>
                        </div>
                        """

            strategy_insights_html = f"""
            <h2 class='section-title' style='border-left-color: #6366f1;'>📊 Per-Strategy Signal Breakdown</h2>
            {strategy_tables_html}
            """

        elif near_misses or strategy_status:
            # Fallback to simpler status view if no signal summary
            strat_rows = ""
            for strat in all_strategies:
                status = strategy_status.get(strat, {}) if strategy_status else {}
                nm = near_misses.get(strat, []) if near_misses else []

                exposure = status.get("current_exposure_pct", 0)
                max_pct = status.get("max_allowed_pct", 70)
                can_open = status.get("can_open_more", True)

                # Near-miss info
                nm_info = ""
                if nm:
                    nm_tickers = ", ".join([f"{n['ticker']} ({n['indicator']})" for n in nm[:2]])
                    nm_info = f"<br/><small style='color: #94a3b8;'>Near: {nm_tickers}</small>"

                status_icon = "✓" if can_open else "⏸"
                status_color = "#4ade80" if can_open else "#f59e0b"

                strat_rows += f"""
                <tr>
                    <td>{strat}</td>
                    <td>{exposure:.1f}%/{max_pct:.0f}%</td>
                    <td style="color: {status_color}">{status_icon}{nm_info}</td>
                </tr>
                """

            strategy_insights_html = f"""
            <h2 class='section-title' style='border-left-color: #6366f1;'>📊 Strategy Status</h2>
            <table>
                <tr><th>Strategy</th><th>Exposure</th><th>Status / Near-Misses</th></tr>
                {strat_rows}
            </table>
            """

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #0f172a; color: #e2e8f0; }}
                .header {{ background: #1e293b; color: white; padding: 20px; text-align: center; border-bottom: 3px solid #3b82f6; }}
                .content {{ padding: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #1e293b; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; color: #e2e8f0; }}
                th {{ background: #334155; font-weight: 600; }}
                .section-title {{ color: #f1f5f9; margin-top: 25px; padding: 10px; border-left: 4px solid #22c55e; background: #1e293b; }}
                .blocked-title {{ border-left-color: #f59e0b; }}
                .sell-title {{ border-left-color: #ef4444; }}
                .summary {{ background: #1e293b; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .stat {{ display: inline-block; margin-right: 30px; }}
                .stat-value {{ font-size: 1.5em; font-weight: bold; color: #3b82f6; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 StockPulse Scan Results</h1>
                <p>{today}</p>
            </div>
            <div class="content">
                <div class="summary">
                    <span class="stat"><span class="stat-value">{len(opened_positions)}</span> Positions Opened</span>
                    <span class="stat"><span class="stat-value">{total_allocated:.1f}%</span> Allocated</span>
                    <span class="stat"><span class="stat-value">{portfolio_exposure_pct:.1f}%</span> Total Exposure</span>
                    <span class="stat"><span class="stat-value">${initial_capital:,.0f}</span> Capital</span>
                </div>

                {"<h2 class='section-title'>✅ Positions Opened (" + str(len(opened_positions)) + ")</h2><table><tr><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Entry</th><th>Target</th><th>Stop</th><th>Allocation</th></tr>" + opened_rows + "</table>" if opened_rows else "<p style='color: #94a3b8;'>No positions opened this scan.</p>"}

                {"<h2 class='section-title blocked-title'>⏸️ Signals Blocked (" + str(len(blocked_signals)) + ")</h2><table><tr><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Would Be</th><th>Reason</th></tr>" + blocked_rows + "</table>" if blocked_rows else ""}

                {"<h2 class='section-title sell-title'>📉 Sell Signals (" + str(len(sell_signals)) + ")</h2><table><tr><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Price</th></tr>" + sell_rows + "</table>" if sell_rows else ""}

                {strategy_insights_html}

                <div style="margin-top: 40px; padding: 20px; background: #1e293b; border-radius: 8px; border-top: 2px solid #3b82f6;">
                    <h2 style="color: #94a3b8; font-size: 16px; margin-top: 0;">📚 Strategy Guide</h2>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #60a5fa; margin: 10px 0 5px 0; font-size: 13px;">RSI Mean Reversion</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 0; line-height: 1.5;">
                            RSI (Relative Strength Index) measures how "oversold" or "overbought" a stock is on a scale of 0-100.
                            When RSI drops below 30, the stock has fallen sharply and is considered oversold - historically, these stocks tend to bounce back.
                            <br/><strong>Settings:</strong> Buy when RSI &lt; 30, Sell when RSI &gt; 70, using 14-day lookback.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #60a5fa; margin: 10px 0 5px 0; font-size: 13px;">MACD Volume</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 0; line-height: 1.5;">
                            MACD (Moving Average Convergence Divergence) tracks momentum by comparing short-term vs long-term price trends.
                            When the fast trend crosses above the slow trend with strong volume, it signals the stock is gaining momentum.
                            <br/><strong>Settings:</strong> Buy on MACD crossover with 1.5x average volume. Uses 12/26 day EMAs and 9-day signal line.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #60a5fa; margin: 10px 0 5px 0; font-size: 13px;">Z-Score Mean Reversion</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 0; line-height: 1.5;">
                            Z-score measures how far a stock's price is from its recent average, in standard deviations.
                            A Z-score of -2.0 means the price is unusually low - like a rubber band stretched too far, it tends to snap back.
                            <br/><strong>Settings:</strong> Buy when Z-score &lt; -2.0, Sell when Z-score &gt; 1.0, using 20-day lookback.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #60a5fa; margin: 10px 0 5px 0; font-size: 13px;">Momentum Breakout</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 0; line-height: 1.5;">
                            This strategy catches stocks "breaking out" to new highs. When a stock breaks above its recent 20-day high with volume,
                            it often signals the start of an uptrend - like a stock breaking free from a ceiling.
                            <br/><strong>Settings:</strong> Buy on new 20-day high with 1.2x average volume. Target +8% gain.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #60a5fa; margin: 10px 0 5px 0; font-size: 13px;">52-Week Low Bounce</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 0; line-height: 1.5;">
                            Stocks near their yearly low can be bargains if fundamentally sound. This buys quality S&amp;P 500 stocks
                            near their 52-week low, betting on a rebound - like buying a brand-name product at clearance prices.
                            <br/><strong>Settings:</strong> Buy when within 10% of 52-week low.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #60a5fa; margin: 10px 0 5px 0; font-size: 13px;">Sector Rotation</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 0; line-height: 1.5;">
                            Different market sectors take turns leading. This identifies stocks outperforming the overall market,
                            betting on continued momentum - like backing the winning horse mid-race.
                            <br/><strong>Settings:</strong> Buy when relative strength &gt; 1.1 (10% better than market), 20-day lookback.
                        </p>
                    </div>

                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #475569;">
                        <p style="color: #94a3b8; font-size: 11px; margin: 0;">
                            <strong>Position Sizing:</strong> Base 5% × Strategy Weight × Confidence Multiplier, capped at 15% per position, 80% max portfolio exposure.<br/>
                            <strong>Disclaimer:</strong> Paper trading simulation only. This is not financial advice. Past performance does not guarantee future results.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        success = self.email_sender.send_email(subject, body_html)

        self._log_alert(
            "scan_results",
            None,
            subject,
            f"Opened: {len(opened_positions)}, Blocked: {len(blocked_signals)}, Exposure: {portfolio_exposure_pct:.1f}%",
            success
        )

        return success

    def process_position_exit(self, position_data: dict[str, Any]) -> bool:
        """
        Process a position exit and send alert.

        Args:
            position_data: Position closure data

        Returns:
            True if alert was sent
        """
        if not self.send_on_exit:
            return False

        if self._is_quiet_hours():
            return False

        success = self.email_sender.send_position_exit_alert(position_data)

        self._log_alert(
            "exit",
            position_data.get("signal_id"),
            f"Exit {position_data.get('ticker')}",
            str(position_data),
            success
        )

        return success

    def send_daily_digest(self) -> bool:
        """
        Send daily digest email with full portfolio status and strategy insights.

        Returns:
            True if sent successfully
        """
        if not self.send_digest:
            return False

        # Get today's signals
        signals_df = self.db.fetchdf("""
            SELECT * FROM signals
            WHERE DATE(created_at) = DATE('now')
            ORDER BY confidence DESC
        """)
        signals = signals_df.to_dict("records") if not signals_df.empty else []

        # Get open positions with current prices
        positions_df = self.position_manager.get_open_positions()
        positions = []
        if not positions_df.empty:
            # Get current prices for unrealized P&L
            tickers = positions_df["ticker"].tolist()
            current_prices = self._get_current_prices(tickers)

            for _, pos in positions_df.iterrows():
                ticker = pos["ticker"]
                entry_price = pos["entry_price"]
                shares = pos["shares"]
                current_price = current_prices.get(ticker, entry_price)

                if pos["direction"] == "BUY":
                    unrealized_pnl = (current_price - entry_price) * shares
                else:
                    unrealized_pnl = (entry_price - current_price) * shares

                unrealized_pct = (unrealized_pnl / (entry_price * shares)) * 100 if entry_price * shares > 0 else 0

                positions.append({
                    **pos.to_dict(),
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pct": unrealized_pct,
                    "position_value": current_price * shares
                })

        # Get today's trades (opened and closed)
        todays_opened = self.db.fetchdf("""
            SELECT * FROM positions_paper
            WHERE DATE(entry_date) = DATE('now')
            ORDER BY entry_date DESC
        """)

        todays_closed = self.db.fetchdf("""
            SELECT * FROM positions_paper
            WHERE DATE(exit_date) = DATE('now') AND status = 'closed'
            ORDER BY exit_date DESC
        """)

        # Get recent closed positions (last 7 days)
        recent_closed = self.db.fetchdf("""
            SELECT * FROM positions_paper
            WHERE status = 'closed'
            AND exit_date >= DATE('now', '-7 days')
            ORDER BY exit_date DESC
            LIMIT 10
        """)

        # Get performance summary
        performance = self.position_manager.get_performance_summary()

        # Calculate portfolio totals
        total_unrealized = sum(p["unrealized_pnl"] for p in positions)
        total_realized = performance.get("total_pnl", 0)
        initial_capital = self.position_manager.initial_capital
        portfolio_value = initial_capital + total_realized + total_unrealized

        portfolio_summary = {
            "initial_capital": initial_capital,
            "total_realized_pnl": total_realized,
            "total_unrealized_pnl": total_unrealized,
            "portfolio_value": portfolio_value,
            "total_return_pct": ((portfolio_value - initial_capital) / initial_capital) * 100,
            "positions_count": len(positions),
            "todays_opened": len(todays_opened) if not todays_opened.empty else 0,
            "todays_closed": len(todays_closed) if not todays_closed.empty else 0,
        }

        # Get strategy insights for digest
        strategy_insights = self._get_strategy_insights_for_digest(signals_df, positions_df)

        success = self.email_sender.send_daily_digest(
            signals=signals,
            positions=positions,
            performance=performance,
            portfolio_summary=portfolio_summary,
            todays_opened=todays_opened.to_dict("records") if not todays_opened.empty else [],
            todays_closed=todays_closed.to_dict("records") if not todays_closed.empty else [],
            recent_closed=recent_closed.to_dict("records") if not recent_closed.empty else [],
            strategy_insights=strategy_insights
        )

        self._log_alert(
            "digest",
            None,
            "Daily Digest",
            f"Portfolio: ${portfolio_value:,.2f}, Positions: {len(positions)}",
            success
        )

        return success

    def _get_strategy_insights_for_digest(
        self,
        signals_df: pd.DataFrame,
        positions_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Get strategy-level insights for the daily digest.

        Returns:
            Dict with strategy breakdowns, capacity info, blocked tickers, and per-strategy signals
        """
        from stockpulse.strategies.signal_insights import SignalInsights

        all_strategies = [
            "rsi_mean_reversion", "macd_volume", "zscore_mean_reversion",
            "momentum_breakout", "week52_low_bounce", "sector_rotation"
        ]

        insights = {
            "by_strategy": {},
            "blocked_tickers": [],
            "capacity_warnings": [],
        }

        try:
            signal_insights = SignalInsights()
            strategy_status = signal_insights.get_strategy_status(self.position_manager)

            # Get today's opened positions to determine which signals were acted on
            todays_opened = self.db.fetchdf("""
                SELECT ticker, strategy FROM positions_paper
                WHERE DATE(entry_date) = DATE('now')
            """)
            opened_tickers = set(todays_opened["ticker"].tolist()) if not todays_opened.empty else set()

            for strategy in all_strategies:
                status = strategy_status.get(strategy, {})
                current_exp = status.get("current_exposure_pct", 0)
                max_allowed = status.get("max_allowed_pct", 70)
                position_count = status.get("position_count", 0)
                can_open = status.get("can_open_more", True)

                # Get today's signals for this strategy with full details
                signal_details = []
                if not signals_df.empty and "strategy" in signals_df.columns:
                    strategy_signals = signals_df[signals_df["strategy"] == strategy].head(10)
                    signal_count = len(signals_df[signals_df["strategy"] == strategy])

                    for _, sig in strategy_signals.iterrows():
                        ticker = sig.get("ticker", "N/A")
                        was_opened = ticker in opened_tickers

                        # Determine why it wasn't acted on if not opened
                        if was_opened:
                            action_status = "OPENED"
                            action_reason = ""
                        else:
                            # Check blocking reasons
                            action_status = "NOT_ACTED"
                            blocking_reasons = signal_insights.get_blocking_reasons(ticker, self.position_manager)
                            if blocking_reasons:
                                action_status = "BLOCKED"
                                action_reason = blocking_reasons[0].get("reason", "Risk limit")
                            elif not can_open:
                                action_status = "BLOCKED"
                                action_reason = f"Strategy at capacity ({current_exp:.0f}%/{max_allowed:.0f}%)"
                            else:
                                action_reason = "Did not meet final criteria"

                        signal_details.append({
                            "ticker": ticker,
                            "confidence": sig.get("confidence", 0),
                            "entry_price": sig.get("entry_price", 0),
                            "target_price": sig.get("target_price", 0),
                            "status": action_status,
                            "reason": action_reason,
                        })
                else:
                    signal_count = 0

                insights["by_strategy"][strategy] = {
                    "signal_count": signal_count,
                    "signal_details": signal_details,  # Full details for top 10
                    "position_count": position_count,
                    "exposure_pct": current_exp,
                    "max_pct": max_allowed,
                    "utilization_pct": (current_exp / max_allowed * 100) if max_allowed > 0 else 0,
                    "can_open_more": can_open,
                }

                # Track capacity warnings
                if not can_open:
                    insights["capacity_warnings"].append({
                        "strategy": strategy,
                        "exposure_pct": current_exp,
                        "max_pct": max_allowed,
                    })

            # Get blocked tickers
            blocked = self.position_manager.get_blocked_tickers()
            insights["blocked_tickers"] = blocked[:10]  # Top 10 blocked

        except Exception as e:
            logger.warning(f"Error getting strategy insights: {e}")

        return insights

    def _get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get current prices for a list of tickers."""
        from stockpulse.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        prices = {}
        try:
            for ticker in tickers:
                df = ingestion.get_daily_prices([ticker], days=1)
                if not df.empty:
                    prices[ticker] = df[df["ticker"] == ticker]["close"].iloc[-1]
        except Exception as e:
            logger.warning(f"Error fetching current prices: {e}")
        return prices

    def send_long_term_digest(self, opportunities: list[dict]) -> bool:
        """
        Send daily long-term opportunities digest with trend analysis.

        Args:
            opportunities: List of long-term investment opportunities

        Returns:
            True if sent successfully
        """
        if not opportunities:
            return False

        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"📈 StockPulse Long-Term Opportunities - {today}"

        # Identify strong buys (score 68+, improving or stable, 3+ days)
        strong_buys = [
            opp for opp in opportunities
            if opp.get('composite_score', 0) >= 68
            and opp.get('change_vs_5d_avg', 0) >= -1
            and opp.get('consecutive_days', 0) >= 3
        ]

        # Build strong buys section
        strong_buys_html = ""
        if strong_buys:
            strong_rows = ""
            for opp in strong_buys[:5]:
                ticker = opp.get('ticker', 'N/A')
                score = opp.get('composite_score', 0)
                days = opp.get('consecutive_days', 0)
                trend = opp.get('trend_symbol', '➡️')
                change_5d = opp.get('change_vs_5d_avg', 0)
                reasoning = opp.get('reasoning', '')

                trend_text = f"Score improving (+{change_5d:.1f} vs 5d avg)" if change_5d > 0 else f"Score stable ({change_5d:+.1f} vs 5d avg)"

                strong_rows += f"""
                <tr style="background: #ecfdf5;">
                    <td><strong style="color: #059669; font-size: 18px;">{ticker}</strong></td>
                    <td style="text-align: center; font-size: 18px;"><strong>{score:.0f}</strong></td>
                    <td style="text-align: center;">{trend} {days}d</td>
                </tr>
                <tr style="background: #ecfdf5;">
                    <td colspan="3" style="padding: 5px 15px 15px; color: #047857; font-size: 13px; border-bottom: 2px solid #059669;">
                        <strong>Why:</strong> {reasoning}<br/>
                        <strong>Trend:</strong> {trend_text}, on list for {days} consecutive days
                    </td>
                </tr>
                """

            strong_buys_html = f"""
            <div style="background: #ecfdf5; border: 2px solid #059669; border-radius: 8px; padding: 15px; margin-bottom: 25px;">
                <h2 style="color: #047857; margin: 0 0 15px 0; font-size: 20px;">🔥 Strong Buy Candidates ({len(strong_buys)})</h2>
                <p style="color: #047857; font-size: 13px; margin: 0 0 15px 0;">
                    High score (68+) + Improving/stable trend + 3+ consecutive days on list
                </p>
                <table style="width: 100%;">
                    <tr>
                        <th style="text-align: left; color: #047857;">Ticker</th>
                        <th style="text-align: center; color: #047857;">Score</th>
                        <th style="text-align: center; color: #047857;">Trend</th>
                    </tr>
                    {strong_rows}
                </table>
            </div>
            """

        # Build main opportunities table with trend data
        rows = ""
        for opp in opportunities[:15]:
            ticker = opp.get('ticker', 'N/A')
            score = opp.get('composite_score', 0)
            reasoning = opp.get('reasoning', '')
            trend = opp.get('trend_symbol', '➡️')
            days = opp.get('consecutive_days', 0)
            change_5d = opp.get('change_vs_5d_avg', 0)
            is_new = opp.get('is_new', False)

            # Format trend info
            if is_new:
                trend_info = "🆕 New"
            else:
                sign = "+" if change_5d >= 0 else ""
                trend_info = f"{trend} {days}d ({sign}{change_5d:.1f})"

            rows += f"""
            <tr>
                <td><strong style="color: #059669; font-size: 16px;">{ticker}</strong></td>
                <td style="text-align: center;"><strong>{score:.0f}</strong></td>
                <td style="text-align: center;">{trend_info}</td>
                <td>{opp.get('price_vs_52w_low_pct', 0):.1f}%</td>
            </tr>
            <tr>
                <td colspan="4" style="padding: 5px 10px 15px 10px; color: #4b5563; font-size: 13px; border-bottom: 2px solid #e5e7eb;">
                    {reasoning}
                </td>
            </tr>
            """

        # Build score breakdown table (top 10)
        breakdown_rows = ""
        for opp in opportunities[:10]:
            trend = opp.get('trend_symbol', '➡️')
            days = opp.get('consecutive_days', 0)
            breakdown_rows += f"""
            <tr>
                <td><strong>{opp.get('ticker', 'N/A')}</strong></td>
                <td style="text-align: center;">{opp.get('composite_score', 0):.0f}</td>
                <td style="text-align: center;">{trend}{days}d</td>
                <td style="text-align: center;">{opp.get('valuation_score', 0):.0f}</td>
                <td style="text-align: center;">{opp.get('technical_score', 0):.0f}</td>
                <td style="text-align: center;">{opp.get('fcf_score', 0):.0f}</td>
                <td style="text-align: center;">{opp.get('earnings_score', 0):.0f}</td>
                <td style="text-align: center;">{opp.get('peer_score', 0):.0f}</td>
                <td style="text-align: center;">{opp.get('quality_score', 0):.0f}</td>
            </tr>
            """

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #f9fafb; }}
                .header {{ background: linear-gradient(135deg, #059669 0%, #047857 100%); color: white; padding: 25px; text-align: center; border-radius: 8px 8px 0 0; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 25px; background: white; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
                th {{ background: #f3f4f6; color: #374151; font-size: 12px; text-transform: uppercase; }}
                .section {{ margin-top: 30px; }}
                .section h2 {{ color: #059669; font-size: 18px; border-bottom: 2px solid #059669; padding-bottom: 8px; }}
                .breakdown-table th {{ font-size: 11px; text-align: center; padding: 8px 4px; }}
                .breakdown-table td {{ font-size: 12px; }}
                .weight-table {{ background: #f3f4f6; border-radius: 8px; padding: 15px; margin: 20px 0; }}
                .weight-table td {{ border: none; padding: 5px 15px; }}
                .legend {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px 15px; margin: 15px 0; font-size: 13px; }}
                .footer {{ background: #f3f4f6; padding: 20px; border-radius: 0 0 8px 8px; font-size: 12px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Long-Term Investment Opportunities</h1>
                <p>Daily Scan Results - {today}</p>
            </div>
            <div class="content">
                <p style="color: #4b5563;">
                    {len(opportunities)} stocks identified. Ranked by composite score with trend analysis.
                </p>

                {strong_buys_html}

                <div class="section">
                    <h2>All Opportunities (Top 15)</h2>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th style="text-align: center;">Score</th>
                            <th style="text-align: center;">Trend</th>
                            <th>vs 52W Low</th>
                        </tr>
                        {rows}
                    </table>
                </div>

                <div class="section">
                    <h2>Score Breakdown (Top 10)</h2>
                    <table class="breakdown-table">
                        <tr>
                            <th>Ticker</th>
                            <th>Score</th>
                            <th>Trend</th>
                            <th>Value</th>
                            <th>Tech</th>
                            <th>FCF</th>
                            <th>Earns</th>
                            <th>Peers</th>
                            <th>Qual</th>
                        </tr>
                        {breakdown_rows}
                    </table>
                </div>

                <div class="section">
                    <h2>Scoring Methodology</h2>
                    <p style="color: #4b5563; font-size: 13px;">
                        Each stock is scored from 0-100 across multiple dimensions. The composite score is a weighted average:
                    </p>
                    <table class="weight-table" style="font-size: 13px;">
                        <tr>
                            <td><strong>Valuation (25%)</strong></td>
                            <td>P/E ratio vs history & sector, PEG ratio, Price/Sales</td>
                        </tr>
                        <tr>
                            <td><strong>Technical (15%)</strong></td>
                            <td>RSI, distance from 52-week low, moving average position</td>
                        </tr>
                        <tr>
                            <td><strong>Insider Activity (10%)</strong></td>
                            <td>Recent insider buying/selling patterns</td>
                        </tr>
                        <tr>
                            <td><strong>Free Cash Flow (15%)</strong></td>
                            <td>FCF yield, FCF growth, FCF margin</td>
                        </tr>
                        <tr>
                            <td><strong>Earnings (15%)</strong></td>
                            <td>EPS beat rate, earnings growth, consistency</td>
                        </tr>
                        <tr>
                            <td><strong>Peer Comparison (10%)</strong></td>
                            <td>Relative valuation vs sector peers</td>
                        </tr>
                        <tr>
                            <td><strong>Dividend (5%)</strong></td>
                            <td>Yield, payout ratio, growth history</td>
                        </tr>
                        <tr>
                            <td><strong>Quality (5%)</strong></td>
                            <td>ROE, debt levels, profit margins</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div class="footer">
                <div style="margin-bottom: 15px;">
                    <strong>Trend Legend:</strong><br/>
                    📈 = Score improving vs 5-day avg | 📉 = Score declining | ➡️ = Stable | 🆕 = New to list<br/>
                    <strong>Xd</strong> = X consecutive days on the opportunities list
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>🔥 Strong Buy Criteria:</strong> Score 68+ AND improving/stable trend AND 3+ consecutive days
                </div>
                <strong>Disclaimer:</strong> This is not financial advice. These are automated screening results
                for research purposes only. Always do your own due diligence before investing.
                <br/><br/>
                StockPulse Long-Term Scanner | Runs daily at 5:30pm ET
            </div>
        </body>
        </html>
        """

        success = self.email_sender.send_email(subject, body_html)

        self._log_alert(
            "long_term_digest",
            None,
            subject,
            f"Opportunities: {len(opportunities)}",
            success
        )

        return success

    def send_error_alert(self, error_type: str, error_message: str) -> bool:
        """
        Send system error alert.

        Args:
            error_type: Type of error
            error_message: Error details

        Returns:
            True if sent successfully
        """
        if not self.send_errors:
            return False

        # Don't spam errors during quiet hours
        if self._is_quiet_hours():
            logger.warning(f"Error during quiet hours (not sending): {error_type}")
            return False

        success = self.email_sender.send_error_alert(error_type, error_message)

        self._log_alert(
            "error",
            None,
            error_type,
            error_message,
            success
        )

        return success

    def get_alert_history(self, days: int = 7) -> list[dict]:
        """Get recent alert history."""
        df = self.db.fetchdf("""
            SELECT * FROM alerts_log
            WHERE created_at > datetime('now', ?)
            ORDER BY created_at DESC
        """, (f'-{days} days',))
        return df.to_dict("records") if not df.empty else []

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        stats = self.db.fetchdf("""
            SELECT
                alert_type,
                COUNT(*) as total,
                SUM(CASE WHEN sent_successfully THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN NOT sent_successfully THEN 1 ELSE 0 END) as failed
            FROM alerts_log
            WHERE created_at > datetime('now', '-30 days')
            GROUP BY alert_type
        """)

        return {
            "by_type": stats.to_dict("records") if not stats.empty else [],
            "total_30d": stats["total"].sum() if not stats.empty else 0,
            "success_rate": (
                stats["successful"].sum() / stats["total"].sum() * 100
                if not stats.empty and stats["total"].sum() > 0 else 0
            )
        }
