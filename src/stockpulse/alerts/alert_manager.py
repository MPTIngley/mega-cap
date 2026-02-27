"""Alert Manager - coordinates when and how to send alerts."""

from datetime import datetime
from typing import Any

import pandas as pd
import pytz

from stockpulse.data.database import get_db
from stockpulse.strategies.base import Signal
from stockpulse.strategies.position_manager import PositionManager
from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger

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

        self.timezone = pytz.timezone(self.config.get("scanning", {}).get("timezone", "US/Eastern"))

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
        error: str | None = None,
    ) -> None:
        """Log alert to database."""
        try:
            recipient = self.email_sender.recipient
            self.db.execute(
                """
                INSERT INTO alerts_log (signal_id, alert_type, recipient, subject, body, sent_successfully, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (signal_id, alert_type, recipient, subject, body, success, error),
            )
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
            logger.debug(
                f"Signal confidence {signal.confidence} below threshold {self.min_confidence}"
            )
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
            success,
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
                initial_capital=100000.0,
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
        initial_capital: float,
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
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
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
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #f9fafb; color: #1f2937; }}
                .header {{ background: #1e40af; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #ffffff; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; color: #1f2937; }}
                th {{ background: #f3f4f6; color: #374151; }}
                .section-title {{ color: #1f2937; margin-top: 25px; padding: 10px; border-left: 4px solid #16a34a; }}
                .sell-title {{ border-left-color: #dc2626; }}
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

                <p style="color: #6b7280; font-size: 12px; margin-top: 30px;">
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
            success,
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
            current_tickers = set(
                [s.ticker for s in buy_signals] + [s.ticker for s in sell_signals]
            )
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
        opened_positions: list[
            tuple
        ],  # (signal, size_pct, dollar_amount, sizing_details) or (signal, size_pct, dollar_amount)
        blocked_signals: list[tuple],  # (signal, reason, detailed_reasons) or (signal, reason)
        sell_signals: list[Signal],
        portfolio_exposure_pct: float,
        initial_capital: float,
        near_misses: dict[str, list[dict]] | None = None,
        strategy_status: dict[str, dict] | None = None,
        strategy_signal_summary: dict[str, list[dict]] | None = None,
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

        today = datetime.now().strftime("%Y-%m-%d %H:%M")
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
            upside_pct = (
                ((signal.target_price - signal.entry_price) / signal.entry_price * 100)
                if signal.entry_price > 0
                else 0
            )

            # Build sizing formula string if we have details
            if sizing_details:
                d = sizing_details
                cap_note = " CAPPED" if d.get("was_capped", False) else ""
                sizing_str = f"{d['base_size_pct']}%×{d['strategy_weight']:.1f}strat×{d['confidence_mult']:.2f}conf={d['raw_size_pct']:.1f}%{cap_note}"
            else:
                sizing_str = ""

            opened_rows += f"""
            <tr>
                <td><strong style="color: #16a34a;">{signal.ticker}</strong></td>
                <td>{signal.strategy}</td>
                <td>{signal.confidence:.0f}%</td>
                <td>${signal.entry_price:.2f}</td>
                <td>${signal.target_price:.2f} (+{upside_pct:.1f}%)</td>
                <td>${signal.stop_price:.2f}</td>
                <td><strong>{size_pct:.1f}%</strong> (${dollar_amount:,.0f})<br/><small style="color: #6b7280;">{sizing_str}</small></td>
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
                detail_parts = [
                    f"{r.get('icon', '')} {r.get('detail', r.get('reason', ''))}"
                    for r in detailed_reasons
                    if r.get("detail") or r.get("reason")
                ]
                if detail_parts:
                    reason_html = f"{reason}<br/><small style='color: #6b7280;'>{'; '.join(detail_parts)}</small>"

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
        all_strategies = [
            "rsi_mean_reversion",
            "macd_volume",
            "zscore_mean_reversion",
            "momentum_breakout",
            "week52_low_bounce",
            "sector_rotation",
        ]

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

                        # Color and icon based on status
                        if sig_status == "OPENED":
                            status_html = "<span style='color: #16a34a;'>✅ OPENED</span>"
                        elif sig_status == "BLOCKED":
                            reason_short = reason[:40] + "..." if len(reason) > 40 else reason
                            status_html = f"<span style='color: #d97706;'>⏸ {reason_short}</span>"
                        else:
                            status_html = "<span style='color: #9ca3af;'>—</span>"

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
                        <h3 style="color: #1f2937; font-size: 14px; margin: 10px 0 5px 0;">
                            {strat_desc}
                        </h3>
                        <p style="color: #6b7280; font-size: 11px; margin: 0 0 8px 0;">
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
                            <h3 style="color: #1f2937; font-size: 14px; margin: 10px 0 5px 0;">
                                {strat_desc}
                            </h3>
                            <p style="color: #6b7280; font-size: 11px; margin: 0 0 5px 0;">
                                Capacity: {exposure:.0f}%/{max_pct:.0f}% used | No signals, but close:
                            </p>
                            <ul style="color: #374151; font-size: 12px; margin: 5px 0; padding-left: 20px;">
                                {nm_rows}
                            </ul>
                        </div>
                        """
                    else:
                        strategy_tables_html += f"""
                        <div style="margin-bottom: 10px;">
                            <h3 style="color: #6b7280; font-size: 14px; margin: 10px 0 5px 0;">
                                {strat_desc}
                            </h3>
                            <p style="color: #6b7280; font-size: 11px; margin: 0;">
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
                    nm_info = f"<br/><small style='color: #6b7280;'>Near: {nm_tickers}</small>"

                status_icon = "✓" if can_open else "⏸"
                status_color = "#16a34a" if can_open else "#d97706"

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
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #f9fafb; color: #1f2937; }}
                .header {{ background: #1e40af; color: white; padding: 20px; text-align: center; border-bottom: 3px solid #3b82f6; }}
                .content {{ padding: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #ffffff; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; color: #1f2937; }}
                th {{ background: #f3f4f6; font-weight: 600; color: #374151; }}
                .section-title {{ color: #1f2937; margin-top: 25px; padding: 10px; border-left: 4px solid #22c55e; background: #f0fdf4; }}
                .blocked-title {{ border-left-color: #f59e0b; background: #fefce8; }}
                .sell-title {{ border-left-color: #ef4444; background: #fef2f2; }}
                .summary {{ background: #eff6ff; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #bfdbfe; }}
                .stat {{ display: inline-block; margin-right: 30px; }}
                .stat-value {{ font-size: 1.5em; font-weight: bold; color: #1d4ed8; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 StockPulse Scan Results</h1>
                <p>{today}</p>
            </div>
            <div class="content">
                <div class="summary">
                    <span class="stat"><span class="stat-value">{len(opened_positions)}</span> Opened</span>
                    <span class="stat"><span class="stat-value">{len(blocked_signals)}</span> Blocked</span>
                    <span class="stat"><span class="stat-value">{portfolio_exposure_pct:.0f}%</span> Exposure</span>
                    <span class="stat"><span class="stat-value">{80.0 - portfolio_exposure_pct:.0f}%</span> Remaining</span>
                    <span class="stat"><span class="stat-value">${initial_capital * (1 - portfolio_exposure_pct / 100):,.0f}</span> Cash</span>
                </div>

                {"<h2 class='section-title'>✅ Positions Opened (" + str(len(opened_positions)) + ")</h2><table><tr><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Entry</th><th>Target</th><th>Stop</th><th>Allocation</th></tr>" + opened_rows + "</table>" if opened_rows else "<p style='color: #6b7280;'>No positions opened this scan.</p>"}

                {"<h2 class='section-title blocked-title'>⏸️ Signals Blocked (" + str(len(blocked_signals)) + ")</h2><table><tr><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Would Be</th><th>Reason</th></tr>" + blocked_rows + "</table>" if blocked_rows else ""}

                {"<h2 class='section-title sell-title'>📉 Sell Signals (" + str(len(sell_signals)) + ")</h2><table><tr><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Price</th></tr>" + sell_rows + "</table>" if sell_rows else ""}

                {strategy_insights_html}

                <div style="margin-top: 40px; padding: 20px; background: #f3f4f6; border-radius: 8px; border-top: 2px solid #3b82f6;">
                    <h2 style="color: #374151; font-size: 16px; margin-top: 0;">📚 Strategy Guide</h2>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #1d4ed8; margin: 10px 0 5px 0; font-size: 13px;">RSI Mean Reversion</h4>
                        <p style="color: #4b5563; font-size: 12px; margin: 0; line-height: 1.5;">
                            RSI (Relative Strength Index) measures how "oversold" or "overbought" a stock is on a scale of 0-100.
                            When RSI drops below 30, the stock has fallen sharply and is considered oversold - historically, these stocks tend to bounce back.
                            <br/><strong>Settings:</strong> Buy when RSI &lt; 30, Sell when RSI &gt; 70, using 14-day lookback.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #1d4ed8; margin: 10px 0 5px 0; font-size: 13px;">MACD Volume</h4>
                        <p style="color: #4b5563; font-size: 12px; margin: 0; line-height: 1.5;">
                            MACD (Moving Average Convergence Divergence) tracks momentum by comparing short-term vs long-term price trends.
                            When the fast trend crosses above the slow trend with strong volume, it signals the stock is gaining momentum.
                            <br/><strong>Settings:</strong> Buy on MACD crossover with 1.5x average volume. Uses 12/26 day EMAs and 9-day signal line.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #1d4ed8; margin: 10px 0 5px 0; font-size: 13px;">Z-Score Mean Reversion</h4>
                        <p style="color: #4b5563; font-size: 12px; margin: 0; line-height: 1.5;">
                            Z-score measures how far a stock's price is from its recent average, in standard deviations.
                            A Z-score of -2.0 means the price is unusually low - like a rubber band stretched too far, it tends to snap back.
                            <br/><strong>Settings:</strong> Buy when Z-score &lt; -2.0, Sell when Z-score &gt; 1.0, using 20-day lookback.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #1d4ed8; margin: 10px 0 5px 0; font-size: 13px;">Momentum Breakout</h4>
                        <p style="color: #4b5563; font-size: 12px; margin: 0; line-height: 1.5;">
                            This strategy catches stocks "breaking out" to new highs. When a stock breaks above its recent 20-day high with volume,
                            it often signals the start of an uptrend - like a stock breaking free from a ceiling.
                            <br/><strong>Settings:</strong> Buy on new 20-day high with 1.2x average volume. Target +8% gain.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #1d4ed8; margin: 10px 0 5px 0; font-size: 13px;">52-Week Low Bounce</h4>
                        <p style="color: #4b5563; font-size: 12px; margin: 0; line-height: 1.5;">
                            Stocks near their yearly low can be bargains if fundamentally sound. This buys quality S&amp;P 500 stocks
                            near their 52-week low, betting on a rebound - like buying a brand-name product at clearance prices.
                            <br/><strong>Settings:</strong> Buy when within 10% of 52-week low.
                        </p>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #1d4ed8; margin: 10px 0 5px 0; font-size: 13px;">Sector Rotation</h4>
                        <p style="color: #4b5563; font-size: 12px; margin: 0; line-height: 1.5;">
                            Different market sectors take turns leading. This identifies stocks outperforming the overall market,
                            betting on continued momentum - like backing the winning horse mid-race.
                            <br/><strong>Settings:</strong> Buy when relative strength &gt; 1.1 (10% better than market), 20-day lookback.
                        </p>
                    </div>

                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #d1d5db;">
                        <p style="color: #6b7280; font-size: 11px; margin: 0;">
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
            success,
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
            success,
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

                unrealized_pct = (
                    (unrealized_pnl / (entry_price * shares)) * 100
                    if entry_price * shares > 0
                    else 0
                )

                positions.append(
                    {
                        **pos.to_dict(),
                        "current_price": current_price,
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pct": unrealized_pct,
                        "position_value": current_price * shares,
                    }
                )

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
            strategy_insights=strategy_insights,
        )

        self._log_alert(
            "digest",
            None,
            "Daily Digest",
            f"Portfolio: ${portfolio_value:,.2f}, Positions: {len(positions)}",
            success,
        )

        return success

    def _get_strategy_insights_for_digest(
        self, signals_df: pd.DataFrame, positions_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Get strategy-level insights for the daily digest.

        Returns:
            Dict with strategy breakdowns, capacity info, blocked tickers, and per-strategy signals
        """
        from stockpulse.strategies.signal_insights import SignalInsights

        all_strategies = [
            "rsi_mean_reversion",
            "macd_volume",
            "zscore_mean_reversion",
            "momentum_breakout",
            "week52_low_bounce",
            "sector_rotation",
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
            opened_tickers = (
                set(todays_opened["ticker"].tolist()) if not todays_opened.empty else set()
            )

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
                            blocking_reasons = signal_insights.get_blocking_reasons(
                                ticker, self.position_manager
                            )
                            if blocking_reasons:
                                action_status = "BLOCKED"
                                action_reason = blocking_reasons[0].get("reason", "Risk limit")
                            elif not can_open:
                                action_status = "BLOCKED"
                                action_reason = (
                                    f"Strategy at capacity ({current_exp:.0f}%/{max_allowed:.0f}%)"
                                )
                            else:
                                action_reason = "Did not meet final criteria"

                        signal_details.append(
                            {
                                "ticker": ticker,
                                "confidence": sig.get("confidence", 0),
                                "entry_price": sig.get("entry_price", 0),
                                "target_price": sig.get("target_price", 0),
                                "status": action_status,
                                "reason": action_reason,
                            }
                        )
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
                    insights["capacity_warnings"].append(
                        {
                            "strategy": strategy,
                            "exposure_pct": current_exp,
                            "max_pct": max_allowed,
                        }
                    )

            # Get blocked tickers
            blocked = self.position_manager.get_blocked_tickers()
            insights["blocked_tickers"] = blocked[:10]  # Top 10 blocked

        except Exception as e:
            logger.warning(f"Error getting strategy insights: {e}")

        return insights

    def _get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get current LIVE prices for a list of tickers."""
        from stockpulse.data.ingestion import DataIngestion

        ingestion = DataIngestion()
        try:
            # Use fetch_current_prices for LIVE data, not stale database prices
            return ingestion.fetch_current_prices(tickers)
        except Exception as e:
            logger.warning(f"Error fetching current prices: {e}")
            return {}

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

        today = datetime.now().strftime("%Y-%m-%d")
        subject = f"📈 StockPulse Long-Term Opportunities - {today}"

        # Load sentiment data for long-term tickers
        sentiment_data = {}
        signals_data = {}
        try:
            from stockpulse.data.sentiment import SentimentStorage

            storage = SentimentStorage()
            lt_tickers = [o.get("ticker") for o in opportunities if o.get("ticker")]
            sentiment_data = storage.get_todays_sentiment(lt_tickers)
            signals_data = storage.get_signals(lt_tickers)
        except Exception as e:
            logger.debug(f"Sentiment data load skipped for Long-Term: {e}")

        # Identify strong buys (score 68+, improving or stable, 3+ days)
        strong_buys = [
            opp
            for opp in opportunities
            if opp.get("composite_score", 0) >= 68
            and opp.get("change_vs_5d_avg", 0) >= -1
            and opp.get("consecutive_days", 0) >= 3
        ]

        # Build strong buys section
        strong_buys_html = ""
        if strong_buys:
            strong_rows = ""
            for opp in strong_buys[:5]:
                ticker = opp.get("ticker", "N/A")
                company = opp.get("company_name", ticker)
                sector = opp.get("sector", "Unknown")
                score = opp.get("composite_score", 0)
                price = opp.get("current_price", 0)
                days = opp.get("consecutive_days", 0)
                trend = opp.get("trend_symbol", "➡️")
                change_5d = opp.get("change_vs_5d_avg", 0)
                reasoning = opp.get("reasoning", "")

                trend_text = (
                    f"Score improving (+{change_5d:.1f} vs 5d avg)"
                    if change_5d > 0
                    else f"Score stable ({change_5d:+.1f} vs 5d avg)"
                )

                # Get sentiment info with diagnostics
                sent = sentiment_data.get(ticker, {})
                sent_score = sent.get("aggregate_score", 0)
                sent_label = sent.get("aggregate_label", "")
                sent_emoji = (
                    "🟢" if sent_label == "bullish" else ("🔴" if sent_label == "bearish" else "🟡")
                )

                # Get StockTwits breakdown
                st = sent.get("stocktwits", {})
                st_bullish = st.get("bullish_count", 0)
                st_bearish = st.get("bearish_count", 0)
                st_neutral = st.get("neutral_count", 0)
                st_total = st.get("total_messages", 0)
                sample_msgs = st.get("sample_messages", [])

                # Build sentiment breakdown text
                sent_breakdown = ""
                if st_total > 0:
                    sent_breakdown = (
                        f" (🟢{st_bullish} 🔴{st_bearish} ⚪{st_neutral} of {st_total} posts)"
                    )

                # Get analyst/insider signals
                signals = signals_data.get(ticker, {})
                analyst = signals.get("analyst_rating", {}).get("data", {})
                insider = signals.get("insider_txn", {}).get("data", {})
                analyst_consensus = (
                    analyst.get("consensus", "").replace("_", " ").title() if analyst else ""
                )
                insider_sentiment = insider.get("insider_sentiment", "").title() if insider else ""

                # Build sample posts HTML (up to 2 samples)
                sample_html = ""
                if sample_msgs:
                    sample_items = []
                    for msg in sample_msgs[:2]:
                        msg_text = msg.get("text", "")[:80]
                        msg_sent = msg.get("sentiment", "")
                        msg_emoji = (
                            "🟢"
                            if msg_sent == "Bullish"
                            else ("🔴" if msg_sent == "Bearish" else "⚪")
                        )
                        if msg_text:
                            sample_items.append(f'{msg_emoji} "{msg_text}..."')
                    if sample_items:
                        sample_html = f"<br/><span style='color: #6b7280; font-size: 11px; font-style: italic;'>Sample posts: {' | '.join(sample_items)}</span>"

                strong_rows += f"""
                <tr style="background: #ecfdf5;">
                    <td>
                        <strong style="color: #059669; font-size: 18px;">{ticker}</strong>
                        <br/><span style="color: #047857; font-size: 12px;">{company}</span>
                        <br/><span style="color: #6b7280; font-size: 11px;">{sector}</span>
                    </td>
                    <td style="text-align: center; font-size: 16px;">${price:.2f}</td>
                    <td style="text-align: center; font-size: 18px;"><strong>{score:.0f}</strong></td>
                    <td style="text-align: center;">{trend} {days}d</td>
                </tr>
                <tr style="background: #ecfdf5;">
                    <td colspan="4" style="padding: 5px 15px 15px; color: #047857; font-size: 13px; border-bottom: 2px solid #059669;">
                        <strong>Why:</strong> {reasoning}<br/>
                        <strong>Trend:</strong> {trend_text}, on list for {days} consecutive days<br/>
                        <strong>Sentiment:</strong> {sent_emoji} {sent_score:.0f}/100{sent_breakdown}{f" | Analysts: {analyst_consensus}" if analyst_consensus else ""}{f" | Insiders: {insider_sentiment}" if insider_sentiment else ""}{sample_html}
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
                        <th style="text-align: left; color: #047857;">Company</th>
                        <th style="text-align: center; color: #047857;">Price</th>
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
            ticker = opp.get("ticker", "N/A")
            company = opp.get("company_name", ticker)
            sector = opp.get("sector", "Unknown")
            score = opp.get("composite_score", 0)
            price = opp.get("current_price", 0)
            reasoning = opp.get("reasoning", "")
            trend = opp.get("trend_symbol", "➡️")
            days = opp.get("consecutive_days", 0)
            change_5d = opp.get("change_vs_5d_avg", 0)
            is_new = opp.get("is_new", False)

            # Format trend info
            if is_new:
                trend_info = "🆕 New"
            else:
                sign = "+" if change_5d >= 0 else ""
                trend_info = f"{trend} {days}d ({sign}{change_5d:.1f})"

            # Get sentiment and signals for this ticker
            sent = sentiment_data.get(ticker, {})
            sent_score = sent.get("aggregate_score", 0)
            sent_label = sent.get("aggregate_label", "")
            sent_emoji = (
                "🟢" if sent_label == "bullish" else ("🔴" if sent_label == "bearish" else "🟡")
            )

            # Get StockTwits breakdown
            st = sent.get("stocktwits", {})
            st_bullish = st.get("bullish_count", 0)
            st_bearish = st.get("bearish_count", 0)
            st_neutral = st.get("neutral_count", 0)
            st_total = st.get("total_messages", 0)

            signals = signals_data.get(ticker, {})
            analyst = signals.get("analyst_rating", {}).get("data", {})
            insider = signals.get("insider_txn", {}).get("data", {})
            analyst_consensus = (
                analyst.get("consensus", "").replace("_", " ").title() if analyst else ""
            )
            insider_sentiment = insider.get("insider_sentiment", "").title() if insider else ""

            # Build sentiment info with breakdown
            if sent_score and st_total > 0:
                sentiment_info = (
                    f"{sent_emoji} {sent_score:.0f} (🟢{st_bullish} 🔴{st_bearish} ⚪{st_neutral})"
                )
            elif sent_score:
                sentiment_info = f"{sent_emoji} {sent_score:.0f}"
            else:
                sentiment_info = "—"

            signals_info = []
            if analyst_consensus:
                signals_info.append(f"Analysts: {analyst_consensus}")
            if insider_sentiment:
                signals_info.append(f"Insiders: {insider_sentiment}")
            signals_text = " | ".join(signals_info) if signals_info else ""

            rows += f"""
            <tr>
                <td>
                    <strong style="color: #059669; font-size: 16px;">{ticker}</strong>
                    <span style="color: #6b7280; font-size: 12px;"> - {company}</span>
                    <br/><span style="color: #9ca3af; font-size: 11px;">{sector}</span>
                </td>
                <td style="text-align: center;">${price:.2f}</td>
                <td style="text-align: center;"><strong>{score:.0f}</strong></td>
                <td style="text-align: center;">{trend_info}</td>
                <td>{opp.get("price_vs_52w_low_pct", 0):.1f}%</td>
            </tr>
            <tr>
                <td colspan="5" style="padding: 5px 10px 15px 10px; color: #4b5563; font-size: 13px; border-bottom: 2px solid #e5e7eb;">
                    {reasoning}{f" | Sentiment: {sentiment_info}" if sent_score else ""}{f" | {signals_text}" if signals_text else ""}
                </td>
            </tr>
            """

        # Build score breakdown table (top 10)
        breakdown_rows = ""
        for opp in opportunities[:10]:
            trend = opp.get("trend_symbol", "➡️")
            days = opp.get("consecutive_days", 0)
            ticker = opp.get("ticker", "N/A")
            company = opp.get("company_name", ticker)
            price = opp.get("current_price", 0)
            # Truncate company name if too long
            company_short = company[:20] + "..." if len(company) > 20 else company
            breakdown_rows += f"""
            <tr>
                <td><strong>{ticker}</strong><br/><span style="color: #6b7280; font-size: 10px;">{company_short}</span></td>
                <td style="text-align: center;">${price:.2f}</td>
                <td style="text-align: center;">{opp.get("composite_score", 0):.0f}</td>
                <td style="text-align: center;">{trend}{days}d</td>
                <td style="text-align: center;">{opp.get("valuation_score", 0):.0f}</td>
                <td style="text-align: center;">{opp.get("technical_score", 0):.0f}</td>
                <td style="text-align: center;">{opp.get("fcf_score", 0):.0f}</td>
                <td style="text-align: center;">{opp.get("earnings_score", 0):.0f}</td>
                <td style="text-align: center;">{opp.get("peer_score", 0):.0f}</td>
                <td style="text-align: center;">{opp.get("quality_score", 0):.0f}</td>
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
                    {len(opportunities) if opportunities else 0} stocks identified. Ranked by composite score with trend analysis.
                </p>

                {strong_buys_html}

                <div class="section">
                    <h2>All Opportunities (Top 15)</h2>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th style="text-align: center;">Price</th>
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
                            <th>Price</th>
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
            "long_term_digest", None, subject, f"Opportunities: {len(opportunities)}", success
        )

        return success

    def send_trillion_club_digest(self, trillion_club: list[dict]) -> bool:
        """
        Send Trillion+ Club digest email with entry point tracking.

        Dedicated email for mega-cap stocks ($1T+) and their entry scores.

        Args:
            trillion_club: List of Trillion+ Club members with entry scores

        Returns:
            True if sent successfully
        """
        if not trillion_club:
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        subject = f"💎 Trillion+ Club Entry Opportunities - {today}"

        # Load sentiment data for trillion club tickers
        sentiment_data = {}
        signals_data = {}
        try:
            from stockpulse.data.sentiment import SentimentStorage

            storage = SentimentStorage()
            tc_tickers = [m.get("ticker") for m in trillion_club if m.get("ticker")]
            sentiment_data = storage.get_todays_sentiment(tc_tickers)
            signals_data = storage.get_signals(tc_tickers)
        except Exception as e:
            logger.debug(f"Sentiment data load skipped for Trillion+: {e}")

        # Identify strong entries (score 70+, improving or stable, 3+ days)
        strong_entries = [
            m
            for m in trillion_club
            if m.get("entry_score", 0) >= 70
            and m.get("score_change_5d", 0) >= -1
            and m.get("consecutive_days", 0) >= 3
        ]

        # Build strong entries section (like strong_buys in longterm)
        strong_entries_html = ""
        if strong_entries:
            strong_rows = ""
            for opp in strong_entries[:5]:
                ticker = opp.get("ticker", "N/A")
                company = opp.get("company_name", ticker)
                category = opp.get("category", "Unknown")
                score = opp.get("entry_score", 0)
                price = opp.get("current_price", 0)
                days = opp.get("consecutive_days", 0)
                trend = opp.get("trend_symbol", "➡️")
                change_5d = opp.get("score_change_5d", 0)
                pct_from_high = opp.get("price_vs_30d_high_pct", 0)
                market_cap_b = opp.get("market_cap_b", 0)

                trend_text = (
                    f"Score improving (+{change_5d:.1f} vs 5d avg)"
                    if change_5d > 0
                    else f"Score stable ({change_5d:+.1f} vs 5d avg)"
                )
                reasoning = f"${market_cap_b:.0f}B market cap, {pct_from_high:+.1f}% from 30d high"

                # Get sentiment info with diagnostics
                sent = sentiment_data.get(ticker, {})
                sent_score = sent.get("aggregate_score", 0)
                sent_label = sent.get("aggregate_label", "")
                sent_emoji = (
                    "🟢" if sent_label == "bullish" else ("🔴" if sent_label == "bearish" else "🟡")
                )

                # Get StockTwits breakdown
                st = sent.get("stocktwits", {})
                st_bullish = st.get("bullish_count", 0)
                st_bearish = st.get("bearish_count", 0)
                st_neutral = st.get("neutral_count", 0)
                st_total = st.get("total_messages", 0)
                sample_msgs = st.get("sample_messages", [])

                # Build sentiment breakdown text
                sent_breakdown = ""
                if st_total > 0:
                    sent_breakdown = (
                        f" (🟢{st_bullish} 🔴{st_bearish} ⚪{st_neutral} of {st_total} posts)"
                    )

                # Get analyst/insider signals
                signals = signals_data.get(ticker, {})
                analyst = signals.get("analyst_rating", {}).get("data", {})
                insider = signals.get("insider_txn", {}).get("data", {})
                analyst_consensus = (
                    analyst.get("consensus", "").replace("_", " ").title() if analyst else ""
                )
                insider_sentiment = insider.get("insider_sentiment", "").title() if insider else ""

                # Build sample posts HTML (up to 2 samples)
                sample_html = ""
                if sample_msgs:
                    sample_items = []
                    for msg in sample_msgs[:2]:
                        msg_text = msg.get("text", "")[:80]
                        msg_sent = msg.get("sentiment", "")
                        msg_emoji = (
                            "🟢"
                            if msg_sent == "Bullish"
                            else ("🔴" if msg_sent == "Bearish" else "⚪")
                        )
                        if msg_text:
                            sample_items.append(f'{msg_emoji} "{msg_text}..."')
                    if sample_items:
                        sample_html = f"<br/><span style='color: #6b7280; font-size: 11px; font-style: italic;'>Sample posts: {' | '.join(sample_items)}</span>"

                strong_rows += f"""
                <tr style="background: #eff6ff;">
                    <td>
                        <strong style="color: #1d4ed8; font-size: 18px;">{ticker}</strong>
                        <br/><span style="color: #1e40af; font-size: 12px;">{company}</span>
                        <br/><span style="color: #6b7280; font-size: 11px;">{category}</span>
                    </td>
                    <td style="text-align: center; font-size: 16px;">${price:.2f}</td>
                    <td style="text-align: center; font-size: 18px;"><strong>{score:.0f}</strong></td>
                    <td style="text-align: center;">{trend} {days}d</td>
                </tr>
                <tr style="background: #eff6ff;">
                    <td colspan="4" style="padding: 5px 15px 15px; color: #1e40af; font-size: 13px; border-bottom: 2px solid #3b82f6;">
                        <strong>Why:</strong> {reasoning}<br/>
                        <strong>Trend:</strong> {trend_text}, on list for {days} consecutive days<br/>
                        <strong>Sentiment:</strong> {sent_emoji} {sent_score:.0f}/100{sent_breakdown}{f" | Analysts: {analyst_consensus}" if analyst_consensus else ""}{f" | Insiders: {insider_sentiment}" if insider_sentiment else ""}{sample_html}
                    </td>
                </tr>
                """

            strong_entries_html = f"""
            <div style="background: #eff6ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 15px; margin-bottom: 25px;">
                <h2 style="color: #1d4ed8; margin: 0 0 15px 0; font-size: 20px;">🎯 Strong Entry Candidates ({len(strong_entries)})</h2>
                <p style="color: #1e40af; font-size: 13px; margin: 0 0 15px 0;">
                    High score (70+) + Improving/stable trend + 3+ consecutive days on list
                </p>
                <table style="width: 100%;">
                    <tr>
                        <th style="text-align: left; color: #1d4ed8;">Company</th>
                        <th style="text-align: center; color: #1d4ed8;">Price</th>
                        <th style="text-align: center; color: #1d4ed8;">Score</th>
                        <th style="text-align: center; color: #1d4ed8;">Trend</th>
                    </tr>
                    {strong_rows}
                </table>
            </div>
            """

        # Build main table rows with reasoning (like longterm)
        tc_rows = ""
        for member in trillion_club[:15]:
            ticker = member.get("ticker", "N/A")
            company = member.get("company_name", ticker)
            market_cap_b = member.get("market_cap_b", 0)
            entry_score = member.get("entry_score", 50)
            price = member.get("current_price", 0)
            pct_from_high = member.get("price_vs_30d_high_pct", 0)
            category = member.get("category", "Other")
            trend = member.get("trend_symbol", "➡️")
            days = member.get("consecutive_days", 0)
            score_change_5d = member.get("score_change_5d", 0)
            is_new = member.get("is_new", False)

            # Format trend info
            if is_new:
                trend_info = "🆕 New"
            else:
                sign = "+" if score_change_5d >= 0 else ""
                trend_info = f"{trend} {days}d ({sign}{score_change_5d:.1f})"

            # Build reasoning
            reasoning = (
                f"${market_cap_b:.0f}B market cap, {pct_from_high:+.1f}% from 30d high, {category}"
            )

            # Get sentiment and signals for this ticker
            sent = sentiment_data.get(ticker, {})
            sent_score = sent.get("aggregate_score", 0)
            sent_label = sent.get("aggregate_label", "")
            sent_emoji = (
                "🟢" if sent_label == "bullish" else ("🔴" if sent_label == "bearish" else "🟡")
            )

            # Get StockTwits breakdown
            st = sent.get("stocktwits", {})
            st_bullish = st.get("bullish_count", 0)
            st_bearish = st.get("bearish_count", 0)
            st_neutral = st.get("neutral_count", 0)
            st_total = st.get("total_messages", 0)

            signals = signals_data.get(ticker, {})
            analyst = signals.get("analyst_rating", {}).get("data", {})
            insider = signals.get("insider_txn", {}).get("data", {})
            analyst_consensus = (
                analyst.get("consensus", "").replace("_", " ").title() if analyst else ""
            )
            insider_sentiment = insider.get("insider_sentiment", "").title() if insider else ""

            # Build sentiment info with breakdown
            if sent_score and st_total > 0:
                sentiment_info = (
                    f"{sent_emoji} {sent_score:.0f} (🟢{st_bullish} 🔴{st_bearish} ⚪{st_neutral})"
                )
            elif sent_score:
                sentiment_info = f"{sent_emoji} {sent_score:.0f}"
            else:
                sentiment_info = "—"

            signals_info = []
            if analyst_consensus:
                signals_info.append(f"Analysts: {analyst_consensus}")
            if insider_sentiment:
                signals_info.append(f"Insiders: {insider_sentiment}")
            signals_text = " | ".join(signals_info) if signals_info else ""

            tc_rows += f"""
            <tr>
                <td>
                    <strong style="color: #3b82f6; font-size: 16px;">{ticker}</strong>
                    <span style="color: #6b7280; font-size: 12px;"> - {company}</span>
                    <br/><span style="color: #9ca3af; font-size: 11px;">{category}</span>
                </td>
                <td style="text-align: center;">${price:.2f}</td>
                <td style="text-align: center;"><strong>{entry_score:.0f}</strong></td>
                <td style="text-align: center;">{trend_info}</td>
                <td style="color: {"#22c55e" if pct_from_high <= -5 else "#6b7280"};">{pct_from_high:+.1f}%</td>
            </tr>
            <tr>
                <td colspan="5" style="padding: 5px 10px 15px 10px; color: #4b5563; font-size: 13px; border-bottom: 2px solid #e5e7eb;">
                    {reasoning}{f" | Sentiment: {sentiment_info}" if sent_score else ""}{f" | {signals_text}" if signals_text else ""}
                </td>
            </tr>
            """

        # Build score breakdown table (top 10, like longterm)
        breakdown_rows = ""
        for member in trillion_club[:10]:
            ticker = member.get("ticker", "N/A")
            company = member.get("company_name", ticker)
            price = member.get("current_price", 0)
            score = member.get("entry_score", 50)
            trend = member.get("trend_symbol", "➡️")
            days = member.get("consecutive_days", 0)
            breakdown = member.get("score_breakdown", {})

            # Extract individual factor scores
            dist_pts = breakdown.get("distance_from_high", {}).get("points", 0)
            rsi_pts = breakdown.get("rsi", {}).get("points", 0)
            ma_pts = breakdown.get("ma_50", {}).get("points", 0)
            pe_pts = breakdown.get("pe_ratio", {}).get("points", 0)
            eg_pts = breakdown.get("earnings_growth", {}).get("points", 0)
            mom_pts = breakdown.get("momentum_20d", {}).get("points", 0)

            company_short = company[:20] + "..." if len(company) > 20 else company
            breakdown_rows += f"""
            <tr>
                <td><strong>{ticker}</strong><br/><span style="color: #6b7280; font-size: 10px;">{company_short}</span></td>
                <td style="text-align: center;">${price:.2f}</td>
                <td style="text-align: center;">{score:.0f}</td>
                <td style="text-align: center;">{trend}{days}d</td>
                <td style="text-align: center;">{dist_pts:+d}</td>
                <td style="text-align: center;">{rsi_pts:+d}</td>
                <td style="text-align: center;">{ma_pts:+d}</td>
                <td style="text-align: center;">{pe_pts:+d}</td>
                <td style="text-align: center;">{eg_pts:+d}</td>
                <td style="text-align: center;">{mom_pts:+d}</td>
            </tr>
            """

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #f9fafb; }}
                .header {{ background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 25px; text-align: center; border-radius: 8px 8px 0 0; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 25px; background: white; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
                th {{ background: #f3f4f6; color: #374151; font-size: 12px; text-transform: uppercase; }}
                .section {{ margin-top: 30px; }}
                .section h2 {{ color: #3b82f6; font-size: 18px; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
                .breakdown-table th {{ font-size: 11px; text-align: center; padding: 8px 4px; }}
                .breakdown-table td {{ font-size: 12px; }}
                .weight-table {{ background: #f3f4f6; border-radius: 8px; padding: 15px; margin: 20px 0; }}
                .weight-table td {{ border: none; padding: 5px 15px; }}
                .footer {{ background: #f3f4f6; padding: 20px; border-radius: 0 0 8px 8px; font-size: 12px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trillion+ Club Entry Opportunities</h1>
                <p>Daily Scan Results - {today}</p>
            </div>
            <div class="content">
                <p style="color: #4b5563;">
                    {len(trillion_club)} mega-cap stocks tracked. Ranked by entry score with trend analysis.
                </p>

                {strong_entries_html}

                <div class="section">
                    <h2>All Members (Top 15)</h2>
                    <table>
                        <tr>
                            <th>Company</th>
                            <th style="text-align: center;">Price</th>
                            <th style="text-align: center;">Score</th>
                            <th style="text-align: center;">Trend</th>
                            <th>vs 30d High</th>
                        </tr>
                        {tc_rows}
                    </table>
                </div>

                <div class="section">
                    <h2>Score Breakdown (Top 10)</h2>
                    <table class="breakdown-table">
                        <tr>
                            <th>Ticker</th>
                            <th>Price</th>
                            <th>Score</th>
                            <th>Trend</th>
                            <th>Dist</th>
                            <th>RSI</th>
                            <th>MA50</th>
                            <th>P/E</th>
                            <th>EG</th>
                            <th>Mom</th>
                        </tr>
                        {breakdown_rows}
                    </table>
                </div>

                <div class="section">
                    <h2>Scoring Methodology</h2>
                    <p style="color: #4b5563; font-size: 13px;">
                        Each stock is scored from 0-100 across multiple dimensions. The entry score is a weighted sum:
                    </p>
                    <table class="weight-table" style="font-size: 13px;">
                        <tr>
                            <td><strong>Base Score (50)</strong></td>
                            <td>Starting point for all stocks</td>
                        </tr>
                        <tr>
                            <td><strong>Distance from High (Dist)</strong></td>
                            <td>Pullback from 30-day high: -15%: +20, -10%: +15, -5%: +10</td>
                        </tr>
                        <tr>
                            <td><strong>RSI (14-day)</strong></td>
                            <td>Oversold/overbought: &lt;30: +15, &lt;40: +10, &gt;70: -10</td>
                        </tr>
                        <tr>
                            <td><strong>50-Day MA (MA50)</strong></td>
                            <td>Position vs moving average: 5%+ below: +10, below: +5</td>
                        </tr>
                        <tr>
                            <td><strong>P/E Ratio</strong></td>
                            <td>Valuation: &lt;15: +15, &lt;20: +10, &gt;40: -10, &gt;60: -15</td>
                        </tr>
                        <tr>
                            <td><strong>Earnings Growth (EG)</strong></td>
                            <td>Forward &lt; Trailing P/E: +5</td>
                        </tr>
                        <tr>
                            <td><strong>20-Day Momentum (Mom)</strong></td>
                            <td>Recent movement: -15%+: +10, -10% to 0%: +5, +20%+: -10</td>
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
                    <strong>🎯 Strong Entry Criteria:</strong> Score 70+ AND improving/stable trend AND 3+ consecutive days
                </div>
                <strong>Disclaimer:</strong> This is not financial advice. These are automated screening results
                for research purposes only. Always do your own due diligence before investing.
                <br/><br/>
                StockPulse Trillion+ Club Scanner | Runs daily at 5:30pm ET
            </div>
        </body>
        </html>
        """

        success = self.email_sender.send_email(subject, body_html)

        self._log_alert(
            "trillion_club_digest",
            None,
            subject,
            f"Trillion Club Members: {len(trillion_club)}",
            success,
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

        self._log_alert("error", None, error_type, error_message, success)

        return success

    def get_alert_history(self, days: int = 7) -> list[dict]:
        """Get recent alert history."""
        df = self.db.fetchdf(
            """
            SELECT * FROM alerts_log
            WHERE created_at > datetime('now', ?)
            ORDER BY created_at DESC
        """,
            (f"-{days} days",),
        )
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
                if not stats.empty and stats["total"].sum() > 0
                else 0
            ),
        }

    def send_ai_pulse_digest(
        self, scan_results: dict[str, Any], long_term_opportunities: list[dict] | None = None
    ) -> bool:
        """
        Send daily AI Pulse digest email.

        This is the consolidated daily email that includes:
        - AI Universe stock scanning with opportunity scores
        - Category breakdown (Hyperscalers, AI Infra, AI Software, etc.)
        - AI thesis research updates with real price data
        - Top picks with detailed score breakdowns
        - Market pulse summary

        Args:
            scan_results: Results from AIPulseScanner.run_scan()
            long_term_opportunities: Optional long-term opportunities to include

        Returns:
            True if sent successfully
        """
        today = datetime.now().strftime("%Y-%m-%d")

        ai_stocks = scan_results.get("ai_stocks", [])
        categories = scan_results.get("categories", {})
        category_scores = scan_results.get("category_scores", {})
        theses = scan_results.get("theses", [])
        market_pulse = scan_results.get("market_pulse", "")
        top_picks = scan_results.get("top_picks", [])
        worst_performers = scan_results.get("worst_performers", [])

        subject = f"🤖 AI Pulse Investment Opportunities - {today}"

        # Load sentiment data early (needed for strong buys section)
        sentiment_data = {}  # ticker -> sentiment dict
        category_sentiment = {}  # category -> list of sentiment scores
        try:
            from stockpulse.data.sentiment import SentimentStorage

            storage = SentimentStorage()
            ai_tickers = [s.get("ticker") for s in ai_stocks[:80] if s.get("ticker")]
            sentiment_data = storage.get_todays_sentiment(ai_tickers)

            # Build category sentiment aggregation
            for stock in ai_stocks:
                ticker = stock.get("ticker", "")
                cat = stock.get("category", "Other")
                if ticker in sentiment_data:
                    score = sentiment_data[ticker].get("aggregate_score", 50)
                    if cat not in category_sentiment:
                        category_sentiment[cat] = []
                    category_sentiment[cat].append(score)
        except Exception as e:
            logger.debug(f"Sentiment data load skipped: {e}")

        # === BUILD EMAIL HTML ===

        # Identify strong buys (score 55+, pullback, 3+ days)
        strong_buys = [
            stock
            for stock in top_picks[:10]
            if stock.get("ai_score", 0) >= 55 and stock.get("pct_30d", 0) <= 0
        ]

        # Build strong buys section (like longterm format)
        strong_buys_html = ""
        if strong_buys:
            strong_rows = ""
            for stock in strong_buys[:5]:
                ticker = stock.get("ticker", "N/A")
                company = stock.get("company_name", ticker)
                category = stock.get("category", "Unknown")
                score = stock.get("ai_score", 0)
                price = stock.get("current_price", 0)
                pct_30d = stock.get("pct_30d", 0)
                pct_90d = stock.get("pct_90d", 0)

                # Get sentiment for this ticker
                sent = sentiment_data.get(ticker, {})
                sent_score = sent.get("aggregate_score", 0)
                sent_label = sent.get("aggregate_label", "")
                if sent_score > 0:
                    if sent_label == "bullish":
                        sent_display = f'<span style="color: #22c55e;">🟢 {sent_score:.0f}</span>'
                    elif sent_label == "bearish":
                        sent_display = f'<span style="color: #ef4444;">🔴 {sent_score:.0f}</span>'
                    else:
                        sent_display = f'<span style="color: #eab308;">🟡 {sent_score:.0f}</span>'
                else:
                    sent_display = '<span style="color: #9ca3af;">-</span>'

                reasoning = f"{category} - 30d: {pct_30d:+.1f}%, 90d: {pct_90d:+.1f}%"

                strong_rows += f"""
                <tr style="background: #f5f3ff;">
                    <td>
                        <strong style="color: #6d28d9; font-size: 18px;">{ticker}</strong>
                        <br/><span style="color: #7c3aed; font-size: 12px;">{company}</span>
                        <br/><span style="color: #6b7280; font-size: 11px;">{category}</span>
                    </td>
                    <td style="text-align: center; font-size: 16px;">${price:.2f}</td>
                    <td style="text-align: center; font-size: 18px;"><strong>{score:.0f}</strong></td>
                    <td style="text-align: center;">{sent_display}</td>
                    <td style="text-align: center; color: {"#22c55e" if pct_30d <= -5 else "#6b7280"};">{pct_30d:+.1f}%</td>
                </tr>
                <tr style="background: #f5f3ff;">
                    <td colspan="5" style="padding: 5px 15px 15px; color: #6d28d9; font-size: 13px; border-bottom: 2px solid #7c3aed;">
                        <strong>Why:</strong> {reasoning}
                    </td>
                </tr>
                """

            strong_buys_html = f"""
            <div style="background: #f5f3ff; border: 2px solid #7c3aed; border-radius: 8px; padding: 15px; margin-bottom: 25px;">
                <h2 style="color: #6d28d9; margin: 0 0 15px 0; font-size: 20px;">🎯 Top AI Stock Picks ({len(strong_buys)})</h2>
                <p style="color: #7c3aed; font-size: 13px; margin: 0 0 15px 0;">
                    High AI score (65+) + Recent pullback = potential opportunity
                </p>
                <table style="width: 100%;">
                    <tr>
                        <th style="text-align: left; color: #6d28d9;">Company</th>
                        <th style="text-align: center; color: #6d28d9;">Price</th>
                        <th style="text-align: center; color: #6d28d9;">AI Score</th>
                        <th style="text-align: center; color: #6d28d9;">Sentiment</th>
                        <th style="text-align: center; color: #6d28d9;">30d</th>
                    </tr>
                    {strong_rows}
                </table>
            </div>
            """

        # Worst performers section (opportunities or warnings)
        worst_html = ""
        if worst_performers:
            worst_rows = ""
            for stock in worst_performers[:5]:
                ticker = stock.get("ticker", "")
                pct_30d = stock.get("pct_30d", 0)
                pct_90d = stock.get("pct_90d", 0)
                ai_score = stock.get("ai_score", 0)
                category = stock.get("category", "")

                worst_rows += f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td style="text-align: center; color: #ef4444; font-weight: bold;">{pct_30d:+.1f}%</td>
                    <td style="text-align: center; color: {"#ef4444" if pct_90d < 0 else "#22c55e"};">{pct_90d:+.1f}%</td>
                    <td style="text-align: center;">{ai_score:.0f}</td>
                    <td style="font-size: 12px;">{category}</td>
                </tr>
                """

            worst_html = f"""
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h3 style="color: #991b1b; margin: 0 0 10px 0; font-size: 16px;">📉 Biggest Pullbacks (30d)</h3>
                <p style="font-size: 12px; color: #7f1d1d; margin: 0 0 10px 0;">
                    These stocks have pulled back the most - could be opportunities or warnings.
                </p>
                <table style="width: 100%;">
                    <tr>
                        <th>Ticker</th>
                        <th style="text-align: center;">30d</th>
                        <th style="text-align: center;">90d</th>
                        <th style="text-align: center;">AI Score</th>
                        <th>Category</th>
                    </tr>
                    {worst_rows}
                </table>
            </div>
            """

        # Category scores section
        category_html = ""
        if category_scores:
            cat_rows = ""
            for cat, data in sorted(
                category_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True
            ):
                if data["count"] == 0:
                    continue
                avg_score = data["avg_score"]
                avg_30d = data["avg_30d"]
                count = data["count"]
                top_pick = data.get("top_pick", "-")

                # Get category sentiment
                cat_sent_scores = category_sentiment.get(cat, [])
                if cat_sent_scores:
                    avg_sent = sum(cat_sent_scores) / len(cat_sent_scores)
                    if avg_sent >= 60:
                        sent_emoji = "🟢"
                        sent_color = "#22c55e"
                    elif avg_sent <= 40:
                        sent_emoji = "🔴"
                        sent_color = "#ef4444"
                    else:
                        sent_emoji = "🟡"
                        sent_color = "#eab308"
                    sent_display = f"{sent_emoji} {avg_sent:.0f}"
                else:
                    sent_display = "-"
                    sent_color = "#9ca3af"

                # Calculate category health score
                # High avg_score + negative 30d = opportunity (WAIT)
                # High avg_score + positive 30d = extended (HOLD)
                # Low avg_score + negative 30d = avoid (SELL)
                if avg_30d <= -10:
                    if avg_score >= 50:
                        health = "OPPORTUNITY"
                        health_color = "#22c55e"
                    else:
                        health = "WEAKNESS"
                        health_color = "#ef4444"
                elif avg_30d <= -5:
                    health = "PULLBACK"
                    health_color = "#eab308"
                elif avg_30d >= 10:
                    health = "EXTENDED"
                    health_color = "#f97316"
                else:
                    health = "NEUTRAL"
                    health_color = "#6b7280"

                score_color = (
                    "#22c55e" if avg_score >= 55 else ("#eab308" if avg_score >= 40 else "#f97316")
                )
                perf_color = "#22c55e" if avg_30d > 0 else "#ef4444"

                cat_rows += f"""
                <tr>
                    <td><strong>{cat}</strong></td>
                    <td style="text-align: center;">{count}</td>
                    <td style="text-align: center; color: {score_color}; font-weight: bold;">{avg_score:.0f}</td>
                    <td style="text-align: center; color: {sent_color};">{sent_display}</td>
                    <td style="text-align: center; color: {perf_color}; font-weight: bold;">{avg_30d:+.1f}%</td>
                    <td style="text-align: center; color: {health_color}; font-weight: bold; font-size: 11px;">{health}</td>
                    <td style="font-size: 12px;">{top_pick}</td>
                </tr>
                """

            category_html = f"""
            <div class="section">
                <h2 style="color: #0891b2; border-bottom: 2px solid #0891b2;">📁 AI Categories Performance</h2>
                <p style="font-size: 12px; color: #6b7280; margin-bottom: 10px;">
                    Category health based on average AI score, social sentiment, and 30-day performance.
                    <strong style="color: #22c55e;">OPPORTUNITY</strong> = High score + pullback (potential entry) |
                    <strong style="color: #ef4444;">WEAKNESS</strong> = Low score + selloff (avoid) |
                    <strong style="color: #f97316;">EXTENDED</strong> = Running hot (wait for pullback)
                </p>
                <table>
                    <tr>
                        <th>Category</th>
                        <th style="text-align: center;">Stocks</th>
                        <th style="text-align: center;">AI Score</th>
                        <th style="text-align: center;">Sentiment</th>
                        <th style="text-align: center;">Avg 30d</th>
                        <th style="text-align: center;">Health</th>
                        <th>Top Pick</th>
                    </tr>
                    {cat_rows}
                </table>
            </div>
            """

        # All AI stocks table (top 15, with reasoning like longterm)
        stocks_rows = ""
        for stock in ai_stocks[:15]:
            ticker = stock.get("ticker", "N/A")
            company = stock.get("company_name", ticker)
            ai_score = stock.get("ai_score", 0)
            price = stock.get("current_price", 0)
            pct_30d = stock.get("pct_30d", 0)
            pct_90d = stock.get("pct_90d", 0)
            category = stock.get("category", "Other")

            # Get sentiment for this ticker
            sent = sentiment_data.get(ticker, {})
            sent_score = sent.get("aggregate_score", 0)
            sent_label = sent.get("aggregate_label", "")
            st_data = sent.get("stocktwits", {})
            sent_trending = st_data.get("trending", False)

            # Sentiment display
            if sent_score > 0:
                if sent_label == "bullish":
                    sent_emoji = "🟢"
                    sent_color = "#22c55e"
                elif sent_label == "bearish":
                    sent_emoji = "🔴"
                    sent_color = "#ef4444"
                else:
                    sent_emoji = "🟡"
                    sent_color = "#eab308"
                sent_display = f"{sent_emoji} {sent_score:.0f}"
                trending_badge = " 📈" if sent_trending else ""
            else:
                sent_display = "-"
                sent_color = "#9ca3af"
                trending_badge = ""

            reasoning = f"{category} - 30d: {pct_30d:+.1f}%, 90d: {pct_90d:+.1f}%"

            stocks_rows += f"""
            <tr>
                <td>
                    <strong style="color: #7c3aed; font-size: 16px;">{ticker}</strong>
                    <span style="color: #6b7280; font-size: 12px;"> - {company}</span>
                    <br/><span style="color: #9ca3af; font-size: 11px;">{category}</span>
                </td>
                <td style="text-align: center;">${price:.2f}</td>
                <td style="text-align: center;"><strong>{ai_score:.0f}</strong></td>
                <td style="text-align: center; color: {sent_color};">{sent_display}{trending_badge}</td>
                <td style="text-align: center; color: {"#22c55e" if pct_30d < 0 else "#6b7280"};">{pct_30d:+.1f}%</td>
            </tr>
            <tr>
                <td colspan="5" style="padding: 5px 10px 15px 10px; color: #4b5563; font-size: 13px; border-bottom: 2px solid #e5e7eb;">
                    {reasoning}
                </td>
            </tr>
            """

        # Build score breakdown table (top 10, like longterm)
        breakdown_rows = ""
        for stock in ai_stocks[:10]:
            ticker = stock.get("ticker", "N/A")
            company = stock.get("company_name", ticker)
            price = stock.get("current_price", 0)
            score = stock.get("ai_score", 50)
            pct_30d = stock.get("pct_30d", 0)
            pct_90d = stock.get("pct_90d", 0)
            category = stock.get("category", "Other")
            breakdown = stock.get("score_breakdown", {})

            # Extract individual factor scores
            perf_30d_pts = breakdown.get("perf_30d", {}).get("points", 0)
            perf_90d_pts = breakdown.get("perf_90d", {}).get("points", 0)
            cat_pts = breakdown.get("ai_category", {}).get("points", 0)
            rsi_pts = breakdown.get("rsi", {}).get("points", 0)
            ma_pts = breakdown.get("ma_50", {}).get("points", 0)
            val_pts = breakdown.get("valuation", {}).get("points", 0)

            company_short = company[:20] + "..." if len(company) > 20 else company
            breakdown_rows += f"""
            <tr>
                <td><strong>{ticker}</strong><br/><span style="color: #6b7280; font-size: 10px;">{company_short}</span></td>
                <td style="text-align: center;">${price:.2f}</td>
                <td style="text-align: center;">{score:.0f}</td>
                <td style="text-align: center;">{perf_30d_pts:+d}</td>
                <td style="text-align: center;">{perf_90d_pts:+d}</td>
                <td style="text-align: center;">{cat_pts:+d}</td>
                <td style="text-align: center;">{rsi_pts:+d}</td>
                <td style="text-align: center;">{ma_pts:+d}</td>
                <td style="text-align: center;">{val_pts:+d}</td>
            </tr>
            """

        # Thesis research section
        thesis_html = ""
        if theses:
            thesis_rows = ""
            for thesis in theses[:5]:
                name = thesis.get("thesis_name", "")
                recommendation = thesis.get("recommendation", "neutral")
                confidence = thesis.get("confidence", 50)
                analysis = thesis.get("analysis", "")[:400]
                thesis_tickers = thesis.get("tickers", [])
                tickers_str = ", ".join(thesis_tickers)

                # Get sentiment for thesis tickers
                thesis_sent_scores = []
                for t in thesis_tickers:
                    if t in sentiment_data:
                        thesis_sent_scores.append(sentiment_data[t].get("aggregate_score", 50))

                if thesis_sent_scores:
                    avg_thesis_sent = sum(thesis_sent_scores) / len(thesis_sent_scores)
                    if avg_thesis_sent >= 60:
                        sent_badge = f'<span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px;">🟢 Sentiment: {avg_thesis_sent:.0f}</span>'
                    elif avg_thesis_sent <= 40:
                        sent_badge = f'<span style="background: #fef2f2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px;">🔴 Sentiment: {avg_thesis_sent:.0f}</span>'
                    else:
                        sent_badge = f'<span style="background: #fef9c3; color: #854d0e; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px;">🟡 Sentiment: {avg_thesis_sent:.0f}</span>'
                else:
                    sent_badge = ""

                rec_color = {"bullish": "#22c55e", "bearish": "#ef4444", "neutral": "#6b7280"}.get(
                    recommendation, "#6b7280"
                )

                # Council verdict badge + perspective summaries
                council_html = ""
                perspectives = thesis.get("perspectives", [])
                consensus = thesis.get("consensus", {})
                verdict = thesis.get("verdict", "")

                if perspectives and verdict:
                    vote = consensus.get("vote_breakdown", {})
                    total = sum(vote.values())
                    majority = vote.get(recommendation, 0)
                    agreement = consensus.get("agreement_score", 0)

                    if agreement >= 1.0:
                        badge_bg, badge_color = "#dcfce7", "#166534"
                    elif agreement >= 0.75:
                        badge_bg, badge_color = "#fef9c3", "#854d0e"
                    else:
                        badge_bg, badge_color = "#fef2f2", "#991b1b"

                    council_badge = (
                        f'<span style="background: {badge_bg}; color: {badge_color}; '
                        f"padding: 2px 8px; border-radius: 4px; font-size: 11px; "
                        f'font-weight: bold; margin-left: 8px;">'
                        f"COUNCIL: {majority}/{total} {recommendation.upper()}</span>"
                    )

                    persp_lines = ""
                    for p in perspectives:
                        p_rec = p.get("recommendation", "neutral")
                        p_color = {
                            "bullish": "#22c55e",
                            "bearish": "#ef4444",
                            "neutral": "#6b7280",
                        }.get(p_rec, "#6b7280")
                        p_name = p.get("name", p.get("perspective", ""))
                        p_focus = p.get("focus", "")
                        p_conf = p.get("confidence", 0)

                        # Extract first 2-3 sentences for a fuller summary
                        raw = p.get("analysis", "")
                        sentences = raw.split(".")
                        p_summary = ".".join(sentences[:3]).strip()
                        if p_summary and not p_summary.endswith("."):
                            p_summary += "."
                        if len(p_summary) > 300:
                            p_summary = p_summary[:297] + "..."

                        p_rec_label = p_rec.upper()
                        persp_lines += (
                            f'<div style="margin: 8px 0; padding: 8px 10px; background: white; '
                            f'border-radius: 4px; border-left: 3px solid {p_color};">'
                            f'<div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-bottom: 3px;">'
                            f'{p_name} <span style="color: {p_color}; font-size: 11px;">{p_rec_label} ({p_conf:.0f}%)</span>'
                            f'<span style="color: #94a3b8; font-size: 10px; font-weight: normal; margin-left: 6px;">{p_focus}</span></div>'
                            f'<div style="font-size: 12px; color: #475569; line-height: 1.4;">{p_summary}</div>'
                            f"</div>"
                        )

                    dissent = thesis.get("dissent", "")
                    dissent_html = ""
                    if dissent:
                        dissent_html = (
                            f'<div style="margin-top: 8px; padding: 8px 10px; background: #fef2f2; '
                            f'border-left: 3px solid #ef4444; border-radius: 4px;">'
                            f'<div style="font-size: 11px; font-weight: bold; color: #991b1b; margin-bottom: 3px;">Key Disagreement</div>'
                            f'<div style="font-size: 12px; color: #7f1d1d; line-height: 1.4;">{dissent}</div>'
                            f"</div>"
                        )

                    council_html = f"""
                    <div style="margin: 10px 0 6px 0;">
                        {council_badge}
                    </div>
                    <div style="margin: 8px 0; padding: 8px; background: #f1f5f9; border-radius: 6px;">
                        {persp_lines}
                        {dissent_html}
                    </div>
                    """
                else:
                    council_html = f'<p style="margin: 0 0 10px 0; font-size: 13px; color: #334155;">{analysis}...</p>'

                thesis_rows += f"""
                <div style="margin-bottom: 15px; padding: 15px; background: #f8fafc; border-radius: 8px; border-left: 4px solid {rec_color};">
                    <h4 style="margin: 0 0 8px 0; color: #1e293b;">{name}{sent_badge}</h4>
                    <p style="margin: 0 0 8px 0; font-size: 12px; color: #64748b;">Tickers: {tickers_str}</p>
                    {council_html}
                    <div style="font-size: 12px;">
                        <span style="color: {rec_color}; font-weight: bold; text-transform: uppercase;">{recommendation}</span>
                        <span style="color: #6b7280;"> | Confidence: {confidence:.0f}%</span>
                    </div>
                </div>
                """

            thesis_html = f"""
            <div class="section">
                <h2 style="color: #6366f1; border-bottom: 2px solid #6366f1;">🧠 AI Investment Theses</h2>
                <p style="font-size: 12px; color: #6b7280; margin-bottom: 15px;">
                    Research based on actual price performance data. Theses are re-evaluated when stocks show significant moves.
                </p>
                {thesis_rows}
            </div>
            """
        else:
            # Show fallback message when no theses are available
            thesis_html = """
            <div class="section">
                <h2 style="color: #6366f1; border-bottom: 2px solid #6366f1;">🧠 AI Investment Theses</h2>
                <p style="font-size: 13px; color: #6b7280; padding: 15px; background: #f8fafc; border-radius: 8px;">
                    No active theses found. Run <code style="background: #e5e7eb; padding: 2px 6px; border-radius: 3px;">stockpulse ai-backfill</code> to initialize default AI investment theses.
                    Once configured, theses will be researched using Claude API for deeper analysis.
                </p>
            </div>
            """

        # Market pulse section
        pulse_html = ""
        if market_pulse:
            pulse_html = f"""
            <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 20px 0;">
                <h3 style="color: #92400e; margin: 0 0 10px 0; font-size: 16px;">📡 AI Market Pulse</h3>
                <div style="color: #78350f; font-size: 13px; white-space: pre-wrap;">{market_pulse}</div>
            </div>
            """

        # Social sentiment section (Phase 7 - isolated module)
        # sentiment_data already loaded at top of function
        sentiment_html = ""
        try:
            from stockpulse.data.sentiment import get_sentiment_summary_for_email

            ai_tickers = [s.get("ticker") for s in ai_stocks[:80] if s.get("ticker")]
            sentiment_html = get_sentiment_summary_for_email(ai_tickers, max_display=10)
        except Exception as e:
            logger.debug(f"Sentiment section skipped: {e}")

        # Market context header (Fear & Greed)
        market_context_html = ""
        try:
            from stockpulse.data.sentiment import SentimentStorage as _MCStorage
            _mc_storage = _MCStorage()
            _fg = _mc_storage.get_signals(["_MARKET"]).get("_MARKET", {}).get("fear_greed", {})
            _fg_score = _fg.get("data", {}).get("score", 0) if _fg else 0
            _fg_rating = _fg.get("data", {}).get("rating", "") if _fg else ""
            if _fg_score > 0:
                if _fg_score >= 75:
                    _fg_color = "#22c55e"
                elif _fg_score >= 55:
                    _fg_color = "#eab308"
                elif _fg_score >= 25:
                    _fg_color = "#f97316"
                else:
                    _fg_color = "#ef4444"
                market_context_html = f"""
                <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px;
                            padding: 10px 20px; margin-bottom: 20px; text-align: center;">
                    <span style="color: #64748b; font-size: 13px;">Market Mood: </span>
                    <span style="color: {_fg_color}; font-weight: bold; font-size: 18px;">{_fg_score:.0f}</span>
                    <span style="color: {_fg_color}; font-size: 13px;"> {_fg_rating}</span>
                    <span style="color: #94a3b8; font-size: 11px;"> (CNN Fear & Greed)</span>
                </div>
                """
        except Exception:
            pass

        # Sentiment reversal alerts
        reversal_html = ""
        try:
            from stockpulse.data.sentiment import SentimentStorage as _RevStorage
            _rev_storage = _RevStorage()
            _reversals = []
            for t in ai_tickers[:30]:
                _today_sent = sentiment_data.get(t, {}).get("aggregate_score", 50)
                if _today_sent == 50:
                    continue
                # Check 7-day history
                _hist = _rev_storage.get_hourly_trend(t, hours=168)
                if len(_hist) >= 3:
                    _avg_7d = sum(h.get("sentiment_score", 50) for h in _hist) / len(_hist)
                    _delta = _today_sent - _avg_7d
                    if abs(_delta) >= 20:
                        _direction = "bearish" if _delta < 0 else "bullish"
                        _reversals.append({
                            "ticker": t, "today": _today_sent,
                            "avg_7d": _avg_7d, "delta": _delta,
                            "direction": _direction,
                        })
            if _reversals:
                _reversals.sort(key=lambda x: abs(x["delta"]), reverse=True)
                _rev_rows = ""
                for rv in _reversals[:5]:
                    _rv_color = "#ef4444" if rv["direction"] == "bearish" else "#22c55e"
                    _rv_emoji = "📉" if rv["direction"] == "bearish" else "📈"
                    _rev_rows += f"""
                    <tr>
                        <td><strong>{rv['ticker']}</strong></td>
                        <td style="text-align: center;">{rv['avg_7d']:.0f}</td>
                        <td style="text-align: center; color: {_rv_color}; font-weight: bold;">{rv['today']:.0f}</td>
                        <td style="text-align: center; color: {_rv_color};">{_rv_emoji} {rv['delta']:+.0f}</td>
                    </tr>
                    """
                reversal_html = f"""
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <h3 style="color: #991b1b; margin: 0 0 10px 0; font-size: 16px;">⚠️ Sentiment Reversals</h3>
                    <p style="font-size: 12px; color: #7f1d1d; margin: 0 0 10px 0;">
                        Stocks with 20+ point sentiment shift vs 7-day average
                    </p>
                    <table style="width: 100%;">
                        <tr><th>Ticker</th><th style="text-align: center;">7d Avg</th><th style="text-align: center;">Now</th><th style="text-align: center;">Change</th></tr>
                        {_rev_rows}
                    </table>
                </div>
                """
        except Exception:
            pass

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #f9fafb; }}
                .header {{ background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%); color: white; padding: 25px; text-align: center; border-radius: 8px 8px 0 0; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 25px; background: white; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
                th {{ background: #f3f4f6; color: #374151; font-size: 12px; text-transform: uppercase; }}
                .section {{ margin-top: 30px; }}
                .section h2 {{ color: #7c3aed; font-size: 18px; border-bottom: 2px solid #7c3aed; padding-bottom: 8px; }}
                .breakdown-table th {{ font-size: 11px; text-align: center; padding: 8px 4px; }}
                .breakdown-table td {{ font-size: 12px; }}
                .weight-table {{ background: #f3f4f6; border-radius: 8px; padding: 15px; margin: 20px 0; }}
                .weight-table td {{ border: none; padding: 5px 15px; }}
                .footer {{ background: #f3f4f6; padding: 20px; border-radius: 0 0 8px 8px; font-size: 12px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Pulse Investment Opportunities</h1>
                <p>Daily Scan Results - {today}</p>
            </div>
            <div class="content">
                {market_context_html}
                <p style="color: #4b5563;">
                    {len(ai_stocks)} AI stocks tracked. Ranked by AI score with trend analysis.
                </p>

                {strong_buys_html}

                <div class="section">
                    <h2>All Opportunities (Top 15)</h2>
                    <table>
                        <tr>
                            <th>Company</th>
                            <th style="text-align: center;">Price</th>
                            <th style="text-align: center;">AI Score</th>
                            <th style="text-align: center;">Sentiment</th>
                            <th style="text-align: center;">30d</th>
                        </tr>
                        {stocks_rows}
                    </table>
                </div>

                <div class="section">
                    <h2>Score Breakdown (Top 10)</h2>
                    <table class="breakdown-table">
                        <tr>
                            <th>Ticker</th>
                            <th>Price</th>
                            <th>Score</th>
                            <th>30d</th>
                            <th>90d</th>
                            <th>Cat</th>
                            <th>RSI</th>
                            <th>MA</th>
                            <th>Val</th>
                        </tr>
                        {breakdown_rows}
                    </table>
                </div>

                {category_html}

                {thesis_html}

                {sentiment_html}

                {reversal_html}

                <div class="section">
                    <h2>Scoring Methodology</h2>
                    <p style="color: #4b5563; font-size: 13px;">
                        Each stock is scored from 0-100 across multiple dimensions. The AI score is a weighted sum:
                    </p>
                    <table class="weight-table" style="font-size: 13px;">
                        <tr>
                            <td><strong>Base Score (20)</strong></td>
                            <td>Starting point for all stocks</td>
                        </tr>
                        <tr>
                            <td><strong>30-Day Performance (30d)</strong></td>
                            <td>Continuous: pullback % &times; 0.5 (max +15). Extensions penalized.</td>
                        </tr>
                        <tr>
                            <td><strong>90-Day Performance (90d)</strong></td>
                            <td>Continuous: pullback % &times; 0.25 (max +10). Extensions penalized.</td>
                        </tr>
                        <tr>
                            <td><strong>AI Category (Cat)</strong></td>
                            <td>Infra: +15, Hyperscaler: +12, Software: +10, Robotics: +8</td>
                        </tr>
                        <tr>
                            <td><strong>RSI (14-day)</strong></td>
                            <td>Continuous: distance from 50 &times; 0.4 (max +12 oversold, -10 overbought)</td>
                        </tr>
                        <tr>
                            <td><strong>50-Day MA</strong></td>
                            <td>Continuous: distance below MA &times; 0.6 (max +8). Above MA penalized.</td>
                        </tr>
                        <tr>
                            <td><strong>Valuation (Val)</strong></td>
                            <td>PEG&lt;0.5: +12, PEG&lt;1: +8, P/E&lt;15: +8, P/E&gt;80: -10</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div class="footer">
                <div style="margin-bottom: 15px;">
                    <strong>Score Interpretation:</strong><br/>
                    80+: Strong Buy | 65-79: Buy | 50-64: Hold | &lt;50: Wait
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>🎯 Top Pick Criteria:</strong> AI Score 65+ AND recent pullback (30d &lt; 0)
                </div>
                <strong>Disclaimer:</strong> This is not financial advice. These are automated screening results
                for research purposes only. Always do your own due diligence before investing.
                <br/><br/>
                StockPulse AI Pulse Scanner | Runs daily at 5:30pm ET
            </div>
        </body>
        </html>
        """

        success = self.email_sender.send_email(subject, body_html)

        self._log_alert(
            "ai_pulse_digest",
            None,
            subject,
            f"AI Stocks: {len(ai_stocks)}, Theses: {len(theses)}",
            success,
        )

        return success
