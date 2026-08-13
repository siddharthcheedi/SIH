"""
Pose Detection — MediaPipe Tasks API (Python 3.13+ compatible).

Uses ``mediapipe.tasks.python.vision.PoseLandmarker`` with the
lightweight ``pose_landmarker_lite.task`` model.

Returns a list of 33 NormalizedLandmark objects (same indices as the
legacy Solutions API) that the SOS detector consumes.
"""

import logging
import os
from typing import Optional, List

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None  # type: ignore


class PoseDetector:
    """
    Wrapper around ``mediapipe.tasks.python.vision.PoseLandmarker``.

    Public API
    ----------
    * :meth:`detect` — run pose estimation on a BGR frame.
    * :meth:`draw`   — draw skeleton overlay.
    * :meth:`close`  — release resources.
    """

    # Body-skeleton connections for drawing
    _CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),      # face left
        (0, 4), (4, 5), (5, 6), (6, 8),      # face right
        (11, 12),                              # shoulders
        (11, 13), (13, 15),                    # left arm
        (12, 14), (14, 16),                    # right arm
        (11, 23), (12, 24),                    # torso
        (23, 24),                              # hips
        (23, 25), (25, 27), (27, 29), (29, 31),  # left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # right leg
    ]

    def __init__(self):
        logger.info("Initialising MediaPipe Pose (Tasks API) …")
        self.available = False
        self._frame_timestamp_ms = 0

        if not MEDIAPIPE_AVAILABLE:
            logger.error("mediapipe is not installed.  pip install mediapipe")
            return

        model_path = config.POSE_MODEL_PATH
        if not os.path.exists(model_path):
            logger.error("Pose model not found: %s", model_path)
            logger.error("Run:  python scripts/download_models.py")
            return

        try:
            base_options = mp_tasks.BaseOptions(
                model_asset_path=model_path
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.available = True
            logger.info("  MediaPipe Pose ready  (Tasks API, lite model)")
        except Exception as exc:
            logger.error("MediaPipe Pose init failed: %s", exc)

    # -----------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Optional[List]:
        """
        Run pose estimation.

        Returns
        -------
        list of NormalizedLandmark (len=33) or ``None``
            Each landmark has ``.x``, ``.y``, ``.z``, ``.visibility``.
            Indices match the standard MediaPipe Pose landmark IDs
            (0=NOSE, 11=LEFT_SHOULDER, 15=LEFT_WRIST, etc.).
        """
        if not self.available:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self._frame_timestamp_ms += 33  # ~30 fps
            result = self.landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                return result.pose_landmarks[0]  # first person
            return None
        except Exception as exc:
            logger.warning("Pose detection error: %s", exc)
            return None

    # -----------------------------------------------------------------
    def draw(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose skeleton overlay on *frame*."""
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]

        # Draw connections
        for start_idx, end_idx in self._CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx].x * w),
                        int(landmarks[start_idx].y * h))
                pt2 = (int(landmarks[end_idx].x * w),
                        int(landmarks[end_idx].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmark dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        return frame

    # -----------------------------------------------------------------
    def close(self):
        """Release underlying MediaPipe resources."""
        if self.available:
            try:
                self.landmarker.close()
            except Exception:
                pass
