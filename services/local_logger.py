"""
Local Logger — append-only CSV alert log.

Every confirmed SOS alert is written to ``logs/alerts.csv`` regardless
of whether Supabase is available.
"""

import csv
import logging
import os
from typing import Dict

import config

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "timestamp",
    "alert_type",
    "severity",
    "total_people",
    "male_count",
    "female_count",
    "unknown_count",
    "confidence",
    "status",
]


class LocalLogger:
    """Writes alert rows to a local CSV file."""

    def __init__(self):
        os.makedirs(os.path.dirname(config.ALERTS_CSV), exist_ok=True)

        # Write header if the file doesn't exist yet
        if not os.path.exists(config.ALERTS_CSV):
            try:
                with open(config.ALERTS_CSV, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(_CSV_COLUMNS)
                logger.info("Created alert log: %s", config.ALERTS_CSV)
            except Exception as exc:
                logger.error("Cannot create alert CSV: %s", exc)

    def log_alert(self, alert: Dict):
        """Append one alert row."""
        try:
            with open(config.ALERTS_CSV, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([alert.get(col, "") for col in _CSV_COLUMNS])
        except Exception as exc:
            logger.error("Failed to write alert to CSV: %s", exc)
