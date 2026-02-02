"""Email sending functionality for StockPulse."""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Any

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger

logger = get_logger(__name__)


class EmailSender:
    """
    Email sender for trading alerts and digests.

    Supports:
    - HTML formatted emails
    - Gmail SMTP (with app password)
    - Multiple recipients (TO + CC)
    """

    def __init__(self):
        """Initialize email sender."""
        self.config = get_config()
        self.email_config = self.config.get("email", {})

        self.smtp_server = self.email_config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = self.email_config.get("smtp_port", 587)
        self.sender = self.email_config.get("sender") or os.environ.get("STOCKPULSE_EMAIL_SENDER", "")
        self.recipient = self.email_config.get("recipient") or os.environ.get("STOCKPULSE_EMAIL_RECIPIENT", "")
        self.password = os.environ.get("STOCKPULSE_EMAIL_PASSWORD", "")

        # Additional CC recipients (comma-separated)
        cc_env = os.environ.get("STOCKPULSE_EMAIL_RECIPIENTS_CC", "")
        self.cc_recipients = [r.strip() for r in cc_env.split(",") if r.strip()]

        self._configured = bool(self.sender and self.recipient and self.password)

        # Log configuration status
        if self._configured:
            logger.info(f"Email configured: sender={self.sender}, recipient={self.recipient}, cc={self.cc_recipients}")
        else:
            missing = []
            if not self.sender:
                missing.append("STOCKPULSE_EMAIL_SENDER")
            if not self.recipient:
                missing.append("STOCKPULSE_EMAIL_RECIPIENT")
            if not self.password:
                missing.append("STOCKPULSE_EMAIL_PASSWORD")
            logger.warning(f"Email not configured. Missing: {', '.join(missing)}")

    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return self._configured

    def send_email(
        self,
        subject: str,
        body_html: str,
        body_text: str | None = None,
        recipient: str | None = None
    ) -> bool:
        """
        Send an email.

        Args:
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body (optional, will be derived from HTML)
            recipient: Override recipient (optional)

        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            logger.warning("Email not configured, skipping send")
            return False

        to_addr = recipient or self.recipient

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender
            msg["To"] = to_addr

            # Add CC recipients if configured
            all_recipients = [to_addr]
            if self.cc_recipients:
                msg["Cc"] = ", ".join(self.cc_recipients)
                all_recipients.extend(self.cc_recipients)

            logger.debug(f"Sending email to: {all_recipients}")

            # Plain text version
            if body_text is None:
                # Simple HTML to text conversion
                body_text = body_html.replace("<br>", "\n").replace("</p>", "\n\n")
                import re
                body_text = re.sub(r"<[^>]+>", "", body_text)

            part1 = MIMEText(body_text, "plain")
            part2 = MIMEText(body_html, "html")

            msg.attach(part1)
            msg.attach(part2)

            # Send to all recipients (TO + CC)
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, all_recipients, msg.as_string())

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_signal_alert(self, signal_data: dict[str, Any]) -> bool:
        """
        Send alert for a new trading signal.

        Args:
            signal_data: Signal information dictionary

        Returns:
            True if sent successfully
        """
        ticker = signal_data.get("ticker", "UNKNOWN")
        direction = signal_data.get("direction", "BUY")
        strategy = signal_data.get("strategy", "unknown")
        confidence = signal_data.get("confidence", 0)
        entry_price = signal_data.get("entry_price", 0)
        target_price = signal_data.get("target_price", 0)
        stop_price = signal_data.get("stop_price", 0)
        notes = signal_data.get("notes", "")

        # Calculate risk/reward
        if direction == "BUY":
            risk = entry_price - stop_price
            reward = target_price - entry_price
        else:
            risk = stop_price - entry_price
            reward = entry_price - target_price

        risk_reward = reward / risk if risk > 0 else 0

        subject = f"🚨 StockPulse Signal: {direction} {ticker} (Confidence: {confidence:.0f}%)"

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }}
                .header {{ background: {'#28a745' if direction == 'BUY' else '#dc3545'}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                .metric-value {{ font-size: 18px; font-weight: bold; }}
                .notes {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{direction} {ticker}</h1>
                <p>Strategy: {strategy}</p>
            </div>
            <div class="content">
                <div class="metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.0f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Risk/Reward</div>
                    <div class="metric-value">{risk_reward:.2f}</div>
                </div>

                <table>
                    <tr>
                        <th>Entry Price</th>
                        <td>${entry_price:.2f}</td>
                    </tr>
                    <tr>
                        <th>Target Price</th>
                        <td>${target_price:.2f} ({((target_price - entry_price) / entry_price * 100):+.1f}%)</td>
                    </tr>
                    <tr>
                        <th>Stop Price</th>
                        <td>${stop_price:.2f} ({((stop_price - entry_price) / entry_price * 100):+.1f}%)</td>
                    </tr>
                </table>

                {f'<div class="notes"><strong>Notes:</strong> {notes}</div>' if notes else ''}

                <p style="color: #666; font-size: 12px; margin-top: 20px;">
                    Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET
                </p>
            </div>
        </body>
        </html>
        """

        return self.send_email(subject, body_html)

    def send_position_exit_alert(self, position_data: dict[str, Any]) -> bool:
        """
        Send alert when a position is closed.

        Args:
            position_data: Position closure information

        Returns:
            True if sent successfully
        """
        ticker = position_data.get("ticker", "UNKNOWN")
        direction = position_data.get("direction", "BUY")
        entry_price = position_data.get("entry_price", 0)
        exit_price = position_data.get("exit_price", 0)
        pnl = position_data.get("net_pnl", 0)
        pnl_pct = position_data.get("pnl_pct", 0)
        exit_reason = position_data.get("exit_reason", "unknown")
        strategy = position_data.get("strategy", "unknown")

        is_win = pnl > 0
        emoji = "✅" if is_win else "❌"

        subject = f"{emoji} StockPulse Exit: {ticker} {'+' if pnl > 0 else ''}{pnl_pct:.1f}% ({exit_reason})"

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }}
                .header {{ background: {'#28a745' if is_win else '#dc3545'}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .pnl {{ font-size: 32px; font-weight: bold; color: {'#28a745' if is_win else '#dc3545'}; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Position Closed: {ticker}</h1>
                <p>{direction} via {strategy}</p>
            </div>
            <div class="content">
                <p class="pnl">{'+' if pnl > 0 else ''}{pnl_pct:.2f}% (${pnl:+.2f})</p>

                <table>
                    <tr>
                        <th>Exit Reason</th>
                        <td>{exit_reason.upper()}</td>
                    </tr>
                    <tr>
                        <th>Entry Price</th>
                        <td>${entry_price:.2f}</td>
                    </tr>
                    <tr>
                        <th>Exit Price</th>
                        <td>${exit_price:.2f}</td>
                    </tr>
                </table>

                <p style="color: #666; font-size: 12px; margin-top: 20px;">
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET
                </p>
            </div>
        </body>
        </html>
        """

        return self.send_email(subject, body_html)

    def send_daily_digest(
        self,
        signals: list[dict],
        positions: list[dict],
        performance: dict[str, Any]
    ) -> bool:
        """
        Send daily digest email.

        Args:
            signals: Today's signals
            positions: Open positions
            performance: Performance summary

        Returns:
            True if sent successfully
        """
        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"📊 StockPulse Daily Digest - {today}"

        # Build signals section
        signals_html = ""
        if signals:
            signals_rows = ""
            for s in signals[:10]:  # Limit to 10
                signals_rows += f"""
                <tr>
                    <td>{s.get('ticker', 'N/A')}</td>
                    <td style="color: {'green' if s.get('direction') == 'BUY' else 'red'}">{s.get('direction', 'N/A')}</td>
                    <td>{s.get('confidence', 0):.0f}%</td>
                    <td>{s.get('strategy', 'N/A')}</td>
                </tr>
                """
            signals_html = f"""
            <h2>Today's Signals ({len(signals)})</h2>
            <table>
                <tr><th>Ticker</th><th>Direction</th><th>Confidence</th><th>Strategy</th></tr>
                {signals_rows}
            </table>
            """
        else:
            signals_html = "<p>No new signals today.</p>"

        # Build positions section
        positions_html = ""
        if positions:
            positions_rows = ""
            for p in positions[:10]:
                positions_rows += f"""
                <tr>
                    <td>{p.get('ticker', 'N/A')}</td>
                    <td>{p.get('direction', 'N/A')}</td>
                    <td>${p.get('entry_price', 0):.2f}</td>
                    <td>{p.get('strategy', 'N/A')}</td>
                </tr>
                """
            positions_html = f"""
            <h2>Open Positions ({len(positions)})</h2>
            <table>
                <tr><th>Ticker</th><th>Direction</th><th>Entry</th><th>Strategy</th></tr>
                {positions_rows}
            </table>
            """
        else:
            positions_html = "<p>No open positions.</p>"

        # Performance section
        total_pnl = performance.get("total_pnl", 0)
        win_rate = performance.get("win_rate", 0)
        total_trades = performance.get("total_trades", 0)

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f8f9fa; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>StockPulse Daily Digest</h1>
                <p>{today}</p>
            </div>
            <div class="content">
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" style="color: {'green' if total_pnl >= 0 else 'red'}">${total_pnl:+,.2f}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{win_rate:.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{total_trades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                </div>

                {signals_html}
                {positions_html}

                <p style="color: #666; font-size: 12px; margin-top: 30px; text-align: center;">
                    StockPulse - Automated Trading Signals<br>
                    This is not financial advice. Trade at your own risk.
                </p>
            </div>
        </body>
        </html>
        """

        return self.send_email(subject, body_html)

    def send_error_alert(self, error_type: str, error_message: str) -> bool:
        """
        Send system error alert.

        Args:
            error_type: Type of error
            error_message: Error details

        Returns:
            True if sent successfully
        """
        subject = f"⚠️ StockPulse Error: {error_type}"

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }}
                .header {{ background: #e74c3c; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .error {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>System Error</h1>
                <p>{error_type}</p>
            </div>
            <div class="content">
                <div class="error">
                    <pre>{error_message}</pre>
                </div>
                <p style="color: #666; font-size: 12px; margin-top: 20px;">
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET
                </p>
            </div>
        </body>
        </html>
        """

        return self.send_email(subject, body_html)
