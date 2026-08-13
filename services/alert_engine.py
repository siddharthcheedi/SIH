"""
Alert Engine — coordinates alert generation, sound, and storage.

When the SOS detector transitions to CRITICAL, the engine:
  1. Plays a local warning sound (non-blocking).
  2. Logs the event to the local CSV.
  3. Stores the event in Supabase.
  4. Keeps a rolling list of recent alerts for the UI overlay.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List

from services.local_logger import LocalLogger
from services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)


class AlertEngine:
    """Generates, logs, and announces SOS alerts."""

    def __init__(
        self,
        supabase: SupabaseService,
        csv_logger: LocalLogger,
        session_id: str,
    ):
        self.supabase = supabase
        self.csv_logger = csv_logger
        self.session_id = session_id
        self.recent_alerts: List[Dict] = []
        self.total_alerts: int = 0

    # -----------------------------------------------------------------
    def trigger_alert(self, counts: Dict[str, int], confidence: float = 0.95):
        """
        Fire a single SOS alert.

        Parameters
        ----------
        counts : dict
            ``{"total": …, "male": …, "female": …, "unknown": …}``
        confidence : float
            Overall gesture confidence (0–1).
        """
        now = datetime.now(timezone.utc)

        alert: Dict = {
            "timestamp": now.isoformat(),
            "alert_type": "SOS",
            "severity": "CRITICAL",
            "total_people": counts.get("total", 0),
            "male_count": counts.get("male", 0),
            "female_count": counts.get("female", 0),
            "unknown_count": counts.get("unknown", 0),
            "confidence": round(confidence, 3),
            "status": "CONFIRMED",
        }

        logger.warning(
            "🚨 SOS ALERT — People: %d  M:%d  F:%d  U:%d",
            alert["total_people"],
            alert["male_count"],
            alert["female_count"],
            alert["unknown_count"],
        )

        # 1. Sound (non-blocking)
        self._play_sound()

        # 2. Local CSV
        self.csv_logger.log_alert(alert)

        # 3. Supabase
        self.supabase.store_alert(alert, self.session_id)

        # 4. Recent-alert list (for overlay)
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > 10:
            self.recent_alerts.pop(0)

        self.total_alerts += 1

    # -----------------------------------------------------------------
    @staticmethod
    def _play_sound():
        """Play a warning beep in a background thread (Windows-first)."""

        def _beep():
            try:
                import winsound  # Windows only

                winsound.Beep(1000, 300)
                winsound.Beep(1500, 300)
                winsound.Beep(2000, 300)
            except ImportError:
                # Fallback — terminal bell
                print("\a\a\a", flush=True)
            except Exception:
                print("\a", flush=True)

        threading.Thread(target=_beep, daemon=True).start()
