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
        Send daily digest email with full portfolio status.

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

        success = self.email_sender.send_daily_digest(
            signals=signals,
            positions=positions,
            performance=performance,
            portfolio_summary=portfolio_summary,
            todays_opened=todays_opened.to_dict("records") if not todays_opened.empty else [],
            todays_closed=todays_closed.to_dict("records") if not todays_closed.empty else [],
            recent_closed=recent_closed.to_dict("records") if not recent_closed.empty else []
        )

        self._log_alert(
            "digest",
            None,
            "Daily Digest",
            f"Portfolio: ${portfolio_value:,.2f}, Positions: {len(positions)}",
            success
        )

        return success

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
