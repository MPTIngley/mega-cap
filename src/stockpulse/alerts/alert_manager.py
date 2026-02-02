"""Alert Manager - coordinates when and how to send alerts."""

from datetime import datetime, time
from typing import Any

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

        Args:
            signals: List of new signals

        Returns:
            Number of alerts sent
        """
        sent_count = 0

        for signal in signals:
            if self.process_new_signal(signal):
                sent_count += 1

        return sent_count

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
        Send daily digest email.

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

        # Get open positions
        positions_df = self.position_manager.get_open_positions()
        positions = positions_df.to_dict("records") if not positions_df.empty else []

        # Get performance summary
        performance = self.position_manager.get_performance_summary()

        success = self.email_sender.send_daily_digest(signals, positions, performance)

        self._log_alert(
            "digest",
            None,
            "Daily Digest",
            f"Signals: {len(signals)}, Positions: {len(positions)}",
            success
        )

        return success

    def send_long_term_digest(self, opportunities: list[dict]) -> bool:
        """
        Send weekly long-term opportunities digest.

        Args:
            opportunities: List of long-term investment opportunities

        Returns:
            True if sent successfully
        """
        if not opportunities:
            return False

        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"📈 StockPulse Long-Term Opportunities - {today}"

        # Build opportunities table
        rows = ""
        for opp in opportunities[:15]:
            rows += f"""
            <tr>
                <td><strong>{opp.get('ticker', 'N/A')}</strong></td>
                <td>{opp.get('composite_score', 0):.0f}</td>
                <td>{opp.get('pe_percentile', 0):.0f}%</td>
                <td>{opp.get('price_vs_52w_low_pct', 0):.1f}%</td>
                <td style="font-size: 12px;">{opp.get('reasoning', '')[:50]}...</td>
            </tr>
            """

        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }}
                .header {{ background: #27ae60; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Long-Term Investment Opportunities</h1>
                <p>Weekly Scan Results</p>
            </div>
            <div class="content">
                <p>The following stocks have been identified as potential long-term investment opportunities based on valuation, technical, and quality metrics:</p>

                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Score</th>
                        <th>P/E Percentile</th>
                        <th>vs 52W Low</th>
                        <th>Notes</th>
                    </tr>
                    {rows}
                </table>

                <p style="color: #666; font-size: 12px; margin-top: 30px;">
                    <strong>Disclaimer:</strong> This is not financial advice. These are automated screening results
                    for research purposes only. Always do your own due diligence before investing.
                </p>
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
        df = self.db.fetchdf(f"""
            SELECT * FROM alerts_log
            WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '{days} days'
            ORDER BY created_at DESC
        """)
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
            WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
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
