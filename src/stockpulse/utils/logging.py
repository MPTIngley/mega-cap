"""Logging configuration for StockPulse."""

import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET_TZ = pytz.timezone("America/New_York")


class EasternTimeFormatter(logging.Formatter):
    """Custom formatter that uses Eastern Time for timestamps."""

    def formatTime(self, record, datefmt=None):
        """Override to use Eastern Time."""
        ct = datetime.fromtimestamp(record.created, tz=ET_TZ)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S ET")
        return s


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Format with Eastern Time
        formatter = EasternTimeFormatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S ET"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        log_dir = Path("logs")
        if log_dir.exists() or _try_create_log_dir(log_dir):
            file_handler = logging.FileHandler(log_dir / "stockpulse.log")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _try_create_log_dir(log_dir: Path) -> bool:
    """Try to create log directory."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False
