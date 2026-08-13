"""
Dashboard Overlay — professional OpenCV monitoring UI.

Layout (rendered every frame)::

    ┌──────────────────────────┬──────────────────┐
    │  WOMEN SAFETY ANALYTICS  │   STATUS PANEL   │
    │                          │                  │
    │     CAMERA FEED          │  SYSTEM   ONLINE │
    │                          │  CAMERA   ONLINE │
    │  [bounding boxes +       │  AI       ONLINE │
    │   gender labels]         │  SUPABASE ONLINE │
    │                          │                  │
    │  [pose skeleton]         │  PEOPLE: 3       │
    │                          │  MALE:   1       │
    │                          │  FEMALE: 2       │
    │                          │  UNKNOWN:0       │
    │                          │                  │
    │                          │  SOS: NORMAL     │
    │                          │  FPS: 18.5       │
    │                          │                  │
    │                          │  RECENT ALERTS   │
    │                          │  09:42 SOS CRIT  │
    └──────────────────────────┴──────────────────┘

When SOS is CRITICAL a large red overlay appears on the camera feed.
"""

import cv2
import numpy as np
from typing import Any, Dict, List
from datetime import datetime

import config
from vision.sos_detector import SOSState


# ─── Colour palette (BGR) ────────────────────────────────────────────
_BG_PANEL = (25, 25, 35)
_BG_HEADER = (35, 35, 50)
_WHITE = (255, 255, 255)
_GRAY = (160, 160, 160)
_CYAN = (230, 200, 0)
_GREEN = (0, 200, 80)
_YELLOW = (0, 210, 255)
_ORANGE = (0, 140, 255)
_RED = (60, 60, 255)
_BRIGHT_RED = (0, 0, 255)
_PINK = (180, 105, 255)
_BLUE_LIGHT = (255, 180, 0)

# Gender → box colour
_GENDER_COLOURS = {
    "Male": _BLUE_LIGHT,
    "Female": _PINK,
    "Unknown": _GRAY,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class DashboardOverlay:
    """Renders the full monitoring dashboard each frame."""

    def __init__(self):
        self.panel_w = config.PANEL_WIDTH

    # ─── Public entry point ──────────────────────────────────────────
    def render(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """
        Compose the dashboard canvas.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR webcam frame.
        data : dict
            Keys: detections, counts, sos_state, sos_progress,
            fps, supabase_online, recent_alerts.

        Returns
        -------
        np.ndarray  –  the composited dashboard image.
        """
        h, w = frame.shape[:2]
        canvas_w = w + self.panel_w
        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)

        # --- Draw detections on the camera feed ----------------------
        annotated = frame.copy()
        self._draw_detections(annotated, data.get("detections", []))

        # --- SOS warning overlay on camera feed ----------------------
        sos_state: SOSState = data.get("sos_state", SOSState.NORMAL)
        if sos_state == SOSState.CRITICAL:
            self._draw_sos_overlay(annotated)
        elif sos_state == SOSState.WARNING:
            self._draw_warning_bar(annotated, data.get("sos_progress", 0.0))

        # --- Place annotated camera feed on canvas -------------------
        canvas[:h, :w] = annotated

        # --- Draw right-side panel -----------------------------------
        panel_x = w
        canvas[:, panel_x:] = _BG_PANEL
        self._draw_panel(canvas, panel_x, h, data)

        return canvas

    # ─── Detection boxes ─────────────────────────────────────────────
    def _draw_detections(
        self, frame: np.ndarray, detections: list
    ):
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            colour = _GENDER_COLOURS.get(det.gender, _GRAY)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background
            label_parts = []
            if det.track_id is not None:
                label_parts.append(f"#{det.track_id}")
            label_parts.append(det.gender)
            if det.gender_confidence > 0:
                label_parts.append(f"{det.gender_confidence:.2f}")
            label = " ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 4),
                _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

            # Confidence under the box
            conf_txt = f"Conf: {det.confidence:.2f}"
            cv2.putText(
                frame, conf_txt, (x1, y2 + 16),
                _FONT, 0.4, colour, 1, cv2.LINE_AA,
            )

    # ─── SOS overlays on camera feed ─────────────────────────────────
    @staticmethod
    def _draw_sos_overlay(frame: np.ndarray):
        """Draw a large semi-transparent red alert over the camera feed."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Red tint on whole frame
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Central warning box
        bx1, by1 = w // 6, h // 3
        bx2, by2 = w - w // 6, h // 3 + h // 3
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), _BRIGHT_RED, 3)
        cv2.rectangle(frame, (bx1 + 3, by1 + 3), (bx2 - 3, by2 - 3), (0, 0, 0), -1)
        cv2.addWeighted(
            frame[by1 + 3 : by2 - 3, bx1 + 3 : bx2 - 3],
            0.3,
            np.full_like(frame[by1 + 3 : by2 - 3, bx1 + 3 : bx2 - 3], (0, 0, 60)),
            0.7,
            0,
            frame[by1 + 3 : by2 - 3, bx1 + 3 : bx2 - 3],
        )

        # Text
        line1 = "!! DISTRESS SIGNAL DETECTED !!"
        line2 = "SOS ALERT"
        (tw1, _), _ = cv2.getTextSize(line1, _FONT_BOLD, 0.9, 2)
        (tw2, _), _ = cv2.getTextSize(line2, _FONT_BOLD, 1.2, 2)
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        cv2.putText(
            frame, line1, (cx - tw1 // 2, cy - 15),
            _FONT_BOLD, 0.9, _WHITE, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line2, (cx - tw2 // 2, cy + 35),
            _FONT_BOLD, 1.2, _BRIGHT_RED, 2, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_warning_bar(frame: np.ndarray, progress: float):
        """Yellow progress bar during WARNING state."""
        h, w = frame.shape[:2]
        bar_h = 6
        bar_w = int(w * progress)
        cv2.rectangle(frame, (0, 0), (bar_w, bar_h), _YELLOW, -1)
        cv2.putText(
            frame, "HAND GESTURE DETECTED — KEEP SIGNALLING...",
            (10, 25), _FONT, 0.6, _YELLOW, 1, cv2.LINE_AA,
        )

    # ─── Right-side status panel ─────────────────────────────────────
    def _draw_panel(
        self,
        canvas: np.ndarray,
        px: int,
        h: int,
        data: Dict[str, Any],
    ):
        """Draw the status/info panel on the right side of the canvas."""
        pw = self.panel_w
        y = 0  # running y-offset

        # ── Header ───────────────────────────────────────────────────
        header_h = 50
        cv2.rectangle(canvas, (px, 0), (px + pw, header_h), _BG_HEADER, -1)
        cv2.putText(
            canvas, "WOMEN SAFETY", (px + 10, 22),
            _FONT_BOLD, 0.6, _CYAN, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "ANALYTICS", (px + 10, 42),
            _FONT_BOLD, 0.6, _CYAN, 1, cv2.LINE_AA,
        )
        y = header_h + 10

        # ── System status ────────────────────────────────────────────
        y = self._section(canvas, px, y, "SYSTEM STATUS")
        y = self._status_row(canvas, px, y, "SYSTEM", "ONLINE", _GREEN)
        y = self._status_row(canvas, px, y, "CAMERA", "ONLINE", _GREEN)
        y = self._status_row(canvas, px, y, "AI", "ONLINE", _GREEN)
        db_ok = data.get("supabase_online", False)
        y = self._status_row(
            canvas, px, y, "SUPABASE",
            "ONLINE" if db_ok else "OFFLINE",
            _GREEN if db_ok else _RED,
        )
        y += 8

        # ── People count ─────────────────────────────────────────────
        counts = data.get("counts", {})
        y = self._section(canvas, px, y, "PEOPLE COUNT")
        y = self._stat_row(canvas, px, y, "TOTAL", str(counts.get("total", 0)), _WHITE)
        y = self._stat_row(canvas, px, y, "MALE", str(counts.get("male", 0)), _BLUE_LIGHT)
        y = self._stat_row(canvas, px, y, "FEMALE", str(counts.get("female", 0)), _PINK)
        y = self._stat_row(canvas, px, y, "UNKNOWN", str(counts.get("unknown", 0)), _GRAY)
        y += 8

        # ── SOS status ───────────────────────────────────────────────
        sos: SOSState = data.get("sos_state", SOSState.NORMAL)
        sos_colour = {
            SOSState.NORMAL: _GREEN,
            SOSState.WARNING: _YELLOW,
            SOSState.CRITICAL: _BRIGHT_RED,
        }.get(sos, _GREEN)
        y = self._section(canvas, px, y, "SOS STATUS")

        # Status badge
        cv2.rectangle(canvas, (px + 10, y), (px + pw - 10, y + 28), sos_colour, -1)
        cv2.putText(
            canvas, sos.value, (px + 15, y + 20),
            _FONT_BOLD, 0.65, (0, 0, 0), 1, cv2.LINE_AA,
        )
        y += 36

        # ── FPS ──────────────────────────────────────────────────────
        fps = data.get("fps", 0.0)
        cv2.putText(
            canvas, f"FPS: {fps:.1f}", (px + 12, y + 14),
            _FONT, 0.5, _GRAY, 1, cv2.LINE_AA,
        )
        y += 28

        # ── Recent alerts ────────────────────────────────────────────
        y = self._section(canvas, px, y, "RECENT ALERTS")
        alerts: List[Dict] = data.get("recent_alerts", [])
        if not alerts:
            cv2.putText(
                canvas, "No alerts", (px + 12, y + 14),
                _FONT, 0.4, _GRAY, 1, cv2.LINE_AA,
            )
        else:
            for alert in reversed(alerts[-5:]):
                ts = alert.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts)
                    ts_str = dt.strftime("%H:%M:%S")
                except Exception:
                    ts_str = ts[:8]
                line = f"{ts_str}  {alert.get('alert_type','')}  {alert.get('severity','')}"
                cv2.putText(
                    canvas, line, (px + 12, y + 14),
                    _FONT, 0.38, _RED, 1, cv2.LINE_AA,
                )
                y += 18
                if y > h - 20:
                    break

    # ─── Helper drawers ──────────────────────────────────────────────
    @staticmethod
    def _section(canvas, px, y, title):
        """Draw a section header line and return the new y."""
        cv2.line(canvas, (px + 10, y), (px + config.PANEL_WIDTH - 10, y), (50, 50, 65), 1)
        y += 16
        cv2.putText(canvas, title, (px + 12, y), _FONT, 0.42, _CYAN, 1, cv2.LINE_AA)
        y += 18
        return y

    @staticmethod
    def _status_row(canvas, px, y, label, value, colour):
        cv2.putText(canvas, label, (px + 14, y + 12), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (px + 130, y + 12), _FONT, 0.42, colour, 1, cv2.LINE_AA)
        return y + 20

    @staticmethod
    def _stat_row(canvas, px, y, label, value, colour):
        cv2.putText(canvas, label, (px + 14, y + 14), _FONT, 0.48, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (px + 150, y + 14), _FONT_BOLD, 0.55, colour, 1, cv2.LINE_AA)
        return y + 22
