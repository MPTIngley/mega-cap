"""Scheduler for StockPulse - runs data ingestion and strategy scans."""

from datetime import datetime, time
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from stockpulse.utils.config import get_config
from stockpulse.utils.logging import get_logger
from stockpulse.data.universe import UniverseManager
from stockpulse.data.ingestion import DataIngestion
# Note: SQLite WAL mode handles concurrency - no need to release locks between jobs

logger = get_logger(__name__)


class StockPulseScheduler:
    """Scheduler for running periodic tasks."""

    def __init__(self):
        """Initialize scheduler."""
        self.config = get_config()
        self.scanning_config = self.config["scanning"]
        self.timezone = pytz.timezone(self.scanning_config["timezone"])

        self.scheduler = BackgroundScheduler(timezone=self.timezone)
        self.universe_manager = UniverseManager()
        self.data_ingestion = DataIngestion()

        # Callbacks for signal generation (set by main app)
        self.on_intraday_scan: Callable | None = None
        self.on_daily_scan: Callable | None = None
        self.on_long_term_scan: Callable | None = None
        self.on_daily_digest: Callable | None = None
        self.on_trillion_scan: Callable | None = None
        self.on_sentiment_scan: Callable | None = None
        self.on_hourly_sentiment_scan: Callable | None = None
        self.on_ai_pulse_scan: Callable | None = None

    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours."""
        now = datetime.now(self.timezone)

        # Check if weekday (Mon-Fri)
        if now.weekday() >= 5:
            return False

        market_open = datetime.strptime(
            self.scanning_config["market_open"], "%H:%M"
        ).time()
        market_close = datetime.strptime(
            self.scanning_config["market_close"], "%H:%M"
        ).time()

        return market_open <= now.time() <= market_close

    def _run_intraday_job(self) -> None:
        """Run intraday data ingestion and scanning."""
        if not self._is_market_hours():
            logger.debug("Skipping intraday job - outside market hours")
            return

        logger.info("Running intraday job")

        try:
            tickers = self.universe_manager.get_active_tickers()
            if not tickers:
                logger.warning("No active tickers in universe")
                return

            # Ingest latest intraday data
            self.data_ingestion.run_intraday_ingestion(tickers)

            # Run strategy scan if callback is set
            if self.on_intraday_scan:
                self.on_intraday_scan(tickers)

            logger.info("Intraday job completed")

        except Exception as e:
            logger.error(f"Error in intraday job: {e}", exc_info=True)

    def _run_daily_job(self) -> None:
        """Run daily data ingestion and scanning."""
        logger.info("Running daily job")

        try:
            # Refresh universe weekly
            tickers = self.universe_manager.refresh_universe()
            if not tickers:
                logger.warning("No tickers in universe")
                return

            # Run full daily ingestion
            self.data_ingestion.run_daily_ingestion(tickers)

            # Run daily strategy scan if callback is set
            if self.on_daily_scan:
                self.on_daily_scan(tickers)

            logger.info("Daily job completed")

        except Exception as e:
            logger.error(f"Error in daily job: {e}", exc_info=True)

    def _run_long_term_scan_job(self) -> None:
        """Run long-term investment scanner."""
        logger.info("Running long-term scan job")

        try:
            tickers = self.universe_manager.get_active_tickers()
            if not tickers:
                return

            if self.on_long_term_scan:
                self.on_long_term_scan(tickers)

            logger.info("Long-term scan job completed")

        except Exception as e:
            logger.error(f"Error in long-term scan job: {e}", exc_info=True)

    def _run_daily_digest_job(self) -> None:
        """Send daily portfolio digest email."""
        logger.info("Running daily digest job")

        try:
            if self.on_daily_digest:
                self.on_daily_digest()
            logger.info("Daily digest job completed")

        except Exception as e:
            logger.error(f"Error in daily digest job: {e}", exc_info=True)

    def _run_trillion_scan_job(self) -> None:
        """Run Trillion+ Club scanner."""
        logger.info("Running Trillion+ Club scan job")

        try:
            if self.on_trillion_scan:
                self.on_trillion_scan()
            logger.info("Trillion+ Club scan job completed")

        except Exception as e:
            logger.error(f"Error in Trillion+ Club scan job: {e}", exc_info=True)

    def _run_sentiment_scan_job(self) -> None:
        """Run daily sentiment scan for AI universe stocks."""
        logger.info("Running daily sentiment scan job")

        try:
            if self.on_sentiment_scan:
                self.on_sentiment_scan()
            logger.info("Daily sentiment scan job completed")

        except Exception as e:
            logger.error(f"Error in sentiment scan job: {e}", exc_info=True)

    def _run_hourly_sentiment_job(self) -> None:
        """Run hourly sentiment scan for top 20 AI stocks."""
        logger.info("Running hourly sentiment scan (top 20 AI stocks)")

        try:
            if self.on_hourly_sentiment_scan:
                self.on_hourly_sentiment_scan()
            logger.info("Hourly sentiment scan completed")

        except Exception as e:
            logger.error(f"Error in hourly sentiment scan job: {e}", exc_info=True)

    def _run_ai_pulse_scan_job(self) -> None:
        """Run AI Thesis scanner (AI-powered investment thesis research)."""
        logger.info("Running AI Thesis scan job")

        try:
            if self.on_ai_pulse_scan:
                self.on_ai_pulse_scan()
            logger.info("AI Thesis scan job completed")

        except Exception as e:
            logger.error(f"Error in AI Thesis scan job: {e}", exc_info=True)

    def start(self) -> None:
        """Start the scheduler with all jobs."""
        # Parse market hours from config
        market_open = self.scanning_config.get("market_open", "09:30")
        market_close = self.scanning_config.get("market_close", "16:00")
        open_hour, open_min = map(int, market_open.split(":"))
        close_hour, close_min = map(int, market_close.split(":"))

        # Opening scan - 2 minutes after market open (e.g., 9:32 AM)
        self.scheduler.add_job(
            self._run_intraday_job,
            CronTrigger(
                hour=open_hour,
                minute=open_min + 2,
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="intraday_open",
            name="Opening scan (market open +2 min)",
            replace_existing=True,
            misfire_grace_time=600,  # Allow up to 10 min late
            coalesce=True,           # Combine missed runs
            max_instances=1          # No overlapping
        )

        # First 15-min scan at 09:45 (after open scan at 09:32)
        self.scheduler.add_job(
            self._run_intraday_job,
            CronTrigger(
                hour=open_hour,
                minute=45,
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="intraday_first",
            name="First 15-min scan (09:45)",
            replace_existing=True,
            misfire_grace_time=600,
            coalesce=True,
            max_instances=1
        )

        # Regular intraday scans - every 15 minutes at :00, :15, :30, :45
        # Start at 10:00, run through 15:45
        self.scheduler.add_job(
            self._run_intraday_job,
            CronTrigger(
                hour=f"{open_hour + 1}-{close_hour - 1}",
                minute="0,15,30,45",
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="intraday_scan",
            name="15-min scans (10:00-15:45)",
            replace_existing=True,
            misfire_grace_time=600,
            coalesce=True,
            max_instances=1
        )

        # Closing scan - 2 minutes before market close (e.g., 3:58 PM)
        self.scheduler.add_job(
            self._run_intraday_job,
            CronTrigger(
                hour=close_hour - 1 if close_min < 2 else close_hour,
                minute=close_min - 2 if close_min >= 2 else 60 + close_min - 2,
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="intraday_close",
            name="Closing scan (market close -2 min)",
            replace_existing=True,
            misfire_grace_time=600,
            coalesce=True,
            max_instances=1
        )

        # Daily job - 30 minutes after market close
        self.scheduler.add_job(
            self._run_daily_job,
            CronTrigger(
                hour=16,
                minute=30,
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="daily_scan",
            name="Daily data ingestion and scan",
            replace_existing=True,
            misfire_grace_time=7200,  # 2 hours - allow late start
            coalesce=True,
            max_instances=1
        )

        # Hourly sentiment scan - Every hour during market hours (9:30-16:00 ET)
        # Only scans top 20 AI stocks, NO Haiku (save costs)
        sentiment_config = self.config.get("sentiment", {})
        if sentiment_config.get("hourly_enabled", True):
            self.scheduler.add_job(
                self._run_hourly_sentiment_job,
                CronTrigger(
                    hour="10-15",  # 10:00-15:00 (covers 10am-4pm market hours)
                    minute=30,  # Run at :30 to avoid market open/close
                    day_of_week="mon-fri",
                    timezone=self.timezone
                ),
                id="sentiment_hourly",
                name="Hourly sentiment for top 20 AI stocks",
                replace_existing=True,
                misfire_grace_time=300,
                coalesce=True,
                max_instances=1
            )

        # Daily sentiment scan - 17:00 ET on weekdays (before AI Pulse to cache data)
        # Full scan of 80 tickers + analyst ratings + insider data
        if sentiment_config.get("enabled", True):
            run_time = sentiment_config.get("run_time", "17:00").split(":")
            self.scheduler.add_job(
                self._run_sentiment_scan_job,
                CronTrigger(
                    hour=int(run_time[0]),
                    minute=int(run_time[1]),
                    day_of_week="mon-fri",
                    timezone=self.timezone
                ),
                id="sentiment_scan",
                name="Daily sentiment scanner (full)",
                replace_existing=True,
                misfire_grace_time=7200,  # 2 hours - allow late start
                coalesce=True,
                max_instances=1
            )

        # Long-term scanner - 17:15 ET on weekdays (after sentiment completes)
        lt_config = self.config.get("long_term_scanner", {})
        if lt_config.get("enabled", True):
            run_time = lt_config.get("run_time", "17:15").split(":")
            self.scheduler.add_job(
                self._run_long_term_scan_job,
                CronTrigger(
                    hour=int(run_time[0]),
                    minute=int(run_time[1]),
                    day_of_week="mon-fri",
                    timezone=self.timezone
                ),
                id="long_term_scan",
                name="Long-term investment scanner",
                replace_existing=True,
                misfire_grace_time=7200,  # 2 hours - allow late start
                coalesce=True,
                max_instances=1
            )

        # Trillion+ Club scanner - 17:20 ET on weekdays (5 min after long-term)
        trillion_config = self.config.get("trillion_club", {})
        if trillion_config.get("enabled", True):
            run_time = trillion_config.get("run_time", "17:20").split(":")
            self.scheduler.add_job(
                self._run_trillion_scan_job,
                CronTrigger(
                    hour=int(run_time[0]),
                    minute=int(run_time[1]),
                    day_of_week="mon-fri",
                    timezone=self.timezone
                ),
                id="trillion_club_scan",
                name="Trillion+ Club mega-cap scanner",
                replace_existing=True,
                misfire_grace_time=7200,  # 2 hours - allow late start
                coalesce=True,
                max_instances=1
            )

        # AI Thesis scanner - 17:30 ET on weekdays (uses cached sentiment data)
        ai_config = self.config.get("ai_pulse", {})
        if ai_config.get("enabled", True):
            run_time = ai_config.get("run_time", "17:30").split(":")
            self.scheduler.add_job(
                self._run_ai_pulse_scan_job,
                CronTrigger(
                    hour=int(run_time[0]),
                    minute=int(run_time[1]),
                    day_of_week="mon-fri",
                    timezone=self.timezone
                ),
                id="ai_thesis_scan",
                name="AI Thesis research scanner",
                replace_existing=True,
                misfire_grace_time=7200,  # 2 hours - allow late start
                coalesce=True,
                max_instances=1
            )

        # Daily digest email - configured time (default 17:00 ET)
        alerts_config = self.config.get("alerts", {})
        if alerts_config.get("send_daily_digest", True):
            digest_time = alerts_config.get("daily_digest_time", "17:00").split(":")
            self.scheduler.add_job(
                self._run_daily_digest_job,
                CronTrigger(
                    hour=int(digest_time[0]),
                    minute=int(digest_time[1]),
                    day_of_week="mon-fri",
                    timezone=self.timezone
                ),
                id="daily_digest",
                name="Daily portfolio digest email",
                replace_existing=True,
                misfire_grace_time=7200,  # 2 hours - allow late start
                coalesce=True,
                max_instances=1
            )

        self.scheduler.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    def run_now(self, job_type: str = "daily") -> None:
        """Manually trigger a job."""
        if job_type == "intraday":
            self._run_intraday_job()
        elif job_type == "daily":
            self._run_daily_job()
        elif job_type == "long_term":
            self._run_long_term_scan_job()
        elif job_type == "trillion":
            self._run_trillion_scan_job()
        elif job_type == "sentiment":
            self._run_sentiment_scan_job()
        elif job_type == "ai_pulse":
            self._run_ai_pulse_scan_job()
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    def get_next_run_times(self) -> dict[str, datetime | None]:
        """Get next scheduled run times for all jobs."""
        result = {}
        for job in self.scheduler.get_jobs():
            result[job.id] = job.next_run_time
        return result
