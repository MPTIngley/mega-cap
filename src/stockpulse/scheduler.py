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
from stockpulse.data.database import release_db_locks

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
        finally:
            # Release database locks to allow dashboard to connect
            release_db_locks()

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
        finally:
            release_db_locks()

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
        finally:
            release_db_locks()

    def _run_daily_digest_job(self) -> None:
        """Send daily portfolio digest email."""
        logger.info("Running daily digest job")

        try:
            if self.on_daily_digest:
                self.on_daily_digest()
            logger.info("Daily digest job completed")

        except Exception as e:
            logger.error(f"Error in daily digest job: {e}", exc_info=True)

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
            replace_existing=True
        )

        # Regular intraday scans - every 15 minutes at :00, :15, :30, :45
        # Only during market hours (handled by _is_market_hours check in the job)
        self.scheduler.add_job(
            self._run_intraday_job,
            CronTrigger(
                hour=f"{open_hour}-{close_hour - 1}",
                minute="0,15,30,45",
                day_of_week="mon-fri",
                timezone=self.timezone
            ),
            id="intraday_scan",
            name="Intraday scan (every 15 min)",
            replace_existing=True
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
            replace_existing=True
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
            replace_existing=True
        )

        # Long-term scanner - 17:30 ET on weekdays
        lt_config = self.config.get("long_term_scanner", {})
        if lt_config.get("enabled", True):
            run_time = lt_config.get("run_time", "17:30").split(":")
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
                replace_existing=True
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
                replace_existing=True
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
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    def get_next_run_times(self) -> dict[str, datetime | None]:
        """Get next scheduled run times for all jobs."""
        result = {}
        for job in self.scheduler.get_jobs():
            result[job.id] = job.next_run_time
        return result
