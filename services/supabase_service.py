"""
Supabase Service — structured event storage.

Stores only alert metadata and session info.  **Never** uploads raw
frames, video, or images.

If the Supabase connection is unavailable, the rest of the application
continues to function; alerts are logged locally via :mod:`services.local_logger`.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import config

logger = logging.getLogger(__name__)


class SupabaseService:
    """Light wrapper around the Supabase Python client."""

    def __init__(self):
        self.connected = False
        self.client = None

        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            logger.warning("Supabase credentials not configured — running OFFLINE")
            return

        try:
            from supabase import create_client

            self.client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            # Quick health-check (non-essential table read)
            self.client.table("sessions").select("id").limit(1).execute()
            self.connected = True
            logger.info("Supabase connected ✔")
        except Exception as exc:
            logger.warning("Supabase unavailable: %s — running OFFLINE", exc)

    # -----------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------
    def create_session(self) -> str:
        """Insert a new session row; return its UUID."""
        if not self.connected:
            local_id = str(uuid.uuid4())
            logger.info("Local session ID: %s", local_id)
            return local_id
        try:
            result = (
                self.client.table("sessions")
                .insert({"started_at": datetime.now(timezone.utc).isoformat()})
                .execute()
            )
            sid = result.data[0]["id"]
            logger.info("Supabase session created: %s", sid)
            return sid
        except Exception as exc:
            logger.warning("Failed to create Supabase session: %s", exc)
            return str(uuid.uuid4())

    def end_session(self, session_id: str, total_alerts: int):
        """Update the session with end time and alert count."""
        if not self.connected:
            return
        try:
            self.client.table("sessions").update(
                {
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "total_alerts": total_alerts,
                }
            ).eq("id", session_id).execute()
            logger.info("Session %s closed in Supabase", session_id)
        except Exception as exc:
            logger.warning("Failed to close Supabase session: %s", exc)

    # -----------------------------------------------------------------
    # Alert storage
    # -----------------------------------------------------------------
    def store_alert(self, alert: Dict, session_id: str):
        """Insert an alert row into Supabase."""
        if not self.connected:
            return
        try:
            data = {**alert, "session_id": session_id}
            self.client.table("alerts").insert(data).execute()
            logger.info("Alert stored in Supabase")
        except Exception as exc:
            logger.warning("Failed to store alert in Supabase: %s", exc)
