"""
Women Safety Analytics — MVP Entry Point
=========================================

Run:  ``python main.py``

Opens the laptop webcam and starts real-time monitoring with:
  • YOLO person detection
  • Apparent gender classification
  • MediaPipe pose-based SOS gesture detection
  • Local + Supabase alert logging

Press **Q** to quit.
"""

import logging
import sys
import time

import cv2
import numpy as np

import config
from services.alert_engine import AlertEngine
from services.local_logger import LocalLogger
from services.supabase_service import SupabaseService
from ui.overlay import DashboardOverlay
from vision.pipeline import SafetyPipeline

# ── Logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _warmup_pipeline(pipeline: SafetyPipeline):
    """
    Run one dummy inference to trigger any lazy downloads (e.g. YOLO's
    ``lap`` dependency) and JIT compilation BEFORE the camera starts.
    This prevents the webcam from timing out on the first frame.
    """
    logger.info("Warming up models (first inference) …")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        pipeline.process_frame(dummy)
    except Exception:
        pass  # Errors on a black frame are expected
    logger.info("Warm-up complete")


def main():
    print()
    print("=" * 56)
    print("   WOMEN SAFETY ANALYTICS — MVP")
    print("   Real-time monitoring prototype")
    print("=" * 56)
    print()

    # ── 1. Services ──────────────────────────────────────────────────
    logger.info("Connecting to Supabase …")
    supabase_svc = SupabaseService()

    logger.info("Initialising local logger …")
    csv_logger = LocalLogger()

    session_id = supabase_svc.create_session()

    # ── 2. Vision pipeline ───────────────────────────────────────────
    try:
        pipeline = SafetyPipeline()
    except Exception as exc:
        logger.critical("Pipeline init failed: %s", exc)
        logger.critical("Make sure models are downloaded: python scripts/download_models.py")
        sys.exit(1)

    # ── 2b. Warm up models before opening camera ─────────────────────
    _warmup_pipeline(pipeline)

    # ── 3. Alert engine ──────────────────────────────────────────────
    alert_engine = AlertEngine(supabase_svc, csv_logger, session_id)

    # ── 4. UI ────────────────────────────────────────────────────────
    overlay = DashboardOverlay()

    # ── 5. Open camera ───────────────────────────────────────────────
    logger.info("Opening camera index %d …", config.CAMERA_INDEX)

    # Try DirectShow backend first (most reliable on Windows),
    # then fall back to the default backend.
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("DirectShow backend failed, trying default …")
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        logger.critical(
            "Cannot open camera %d. Check CAMERA_INDEX or close other apps using the webcam.",
            config.CAMERA_INDEX,
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %d × %d", actual_w, actual_h)

    # Wait for camera to fully initialise (some webcams need this)
    time.sleep(1.0)

    # Wait for the first valid frame before entering the main loop
    logger.info("Waiting for first frame …")
    first_frame_ok = False
    for attempt in range(60):  # Up to ~3 seconds
        ret, _ = cap.read()
        if ret:
            first_frame_ok = True
            break
        time.sleep(0.05)

    if not first_frame_ok:
        logger.critical("Camera opened but cannot read frames. Try closing other apps or check drivers.")
        cap.release()
        sys.exit(1)

    logger.info("Camera streaming")

    # ── 6. Status banner ─────────────────────────────────────────────
    print()
    print("  SYSTEM:   ONLINE")
    print("  CAMERA:   ONLINE")
    print("  AI:       ONLINE")
    print(f"  SUPABASE: {'ONLINE' if supabase_svc.connected else 'OFFLINE'}")
    print()
    print("  Press Q to quit.")
    print()

    # ── 7. Main loop ─────────────────────────────────────────────────
    fps: float = 0.0
    fps_frame_count: int = 0
    fps_timer: float = time.time()
    consecutive_failures: int = 0
    MAX_CONSECUTIVE_FAILURES = 30  # ~1 second of failures before giving up

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Camera disconnected after %d failed reads", consecutive_failures)
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0  # Reset on successful read

            # -- Process --
            result = pipeline.process_frame(frame)

            # -- SOS alert trigger (ONLY when a female is explicitly detected) --
            female_count = result["counts"].get("female", 0)
            if result["sos_triggered"]:
                if female_count > 0:
                    alert_engine.trigger_alert(result["counts"], confidence=0.95)
                else:
                    logger.info(
                        "SOS gesture detected but no female classified "
                        "(male=%d, unknown=%d) — alert suppressed",
                        result["counts"].get("male", 0),
                        result["counts"].get("unknown", 0),
                    )

            # -- FPS --
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # -- Render dashboard --
            dashboard_data = {
                "detections": result["detections"],
                "counts": result["counts"],
                "sos_state": result["sos_state"],
                "sos_progress": result.get("sos_progress", 0.0),
                "fps": fps,
                "supabase_online": supabase_svc.connected,
                "recent_alerts": alert_engine.recent_alerts,
            }
            canvas = overlay.render(frame, dashboard_data)

            # -- Show --
            cv2.imshow(config.WINDOW_NAME, canvas)

            # -- Quit on Q --
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Quit requested")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # -- Cleanup --
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        supabase_svc.end_session(session_id, alert_engine.total_alerts)
        logger.info("Session ended — %d alerts generated", alert_engine.total_alerts)
        print()
        print("=" * 56)
        print("  Session complete.")
        print(f"  Total alerts: {alert_engine.total_alerts}")
        print(f"  Alerts CSV:   {config.ALERTS_CSV}")
        print("=" * 56)


if __name__ == "__main__":
    main()
