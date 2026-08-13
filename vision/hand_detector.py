"""
Hand Detection — MediaPipe Tasks API HandLandmarker.

Detects 21 hand landmarks per hand and determines whether the hand
is open (fingers extended) or closed (fist).

Used by the SOS detector to recognise an open/close pattern.
"""

import logging
import os
from typing import Optional, List, Tuple

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

# ── Finger landmark indices ──────────────────────────────────────────
# Each finger: [MCP (knuckle), PIP, DIP, TIP]
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_INDEX_MCP = 5
_MIDDLE_TIP = 12
_MIDDLE_MCP = 9
_RING_TIP = 16
_RING_MCP = 13
_PINKY_TIP = 20
_PINKY_MCP = 17


class HandDetector:
    """
    MediaPipe HandLandmarker wrapper.

    Detects hands and classifies them as open or closed based on
    finger extension.
    """

    def __init__(self):
        logger.info("Initialising MediaPipe Hand Landmarker …")
        self.available = False
        self._frame_timestamp_ms = 0

        if not MEDIAPIPE_AVAILABLE:
            logger.error("mediapipe is not installed.")
            return

        model_path = config.HAND_MODEL_PATH
        if not os.path.exists(model_path):
            logger.error("Hand model not found: %s", model_path)
            logger.error("Run:  python scripts/download_models.py")
            return

        try:
            base_options = mp_tasks.BaseOptions(
                model_asset_path=model_path
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self.available = True
            logger.info("  MediaPipe Hand Landmarker ready")
        except Exception as exc:
            logger.error("Hand Landmarker init failed: %s", exc)

    # -----------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[List]:
        """
        Detect hands in the frame.

        Returns
        -------
        list of landmark lists
            One list of 21 NormalizedLandmark per detected hand.
            Empty list if no hands found.
        """
        if not self.available:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self._frame_timestamp_ms += 33  # ~30 fps
            result = self.landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )

            if result.hand_landmarks:
                return result.hand_landmarks
            return []
        except Exception as exc:
            logger.warning("Hand detection error: %s", exc)
            return []

    # -----------------------------------------------------------------
    @staticmethod
    def is_hand_open(landmarks) -> bool:
        """
        Determine if a hand is open (fingers extended).

        A finger is considered extended if its tip is farther from the
        wrist than its MCP (knuckle) joint.  We check 4 fingers
        (excluding the thumb which bends differently).

        Returns True if at least 3 of 4 fingers are extended.
        """
        if len(landmarks) < 21:
            return False

        wrist = landmarks[_WRIST]
        extended = 0

        # Check index, middle, ring, pinky
        finger_pairs = [
            (_INDEX_TIP, _INDEX_MCP),
            (_MIDDLE_TIP, _MIDDLE_MCP),
            (_RING_TIP, _RING_MCP),
            (_PINKY_TIP, _PINKY_MCP),
        ]

        for tip_idx, mcp_idx in finger_pairs:
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]

            # Tip should be farther from wrist than MCP when extended
            # Using y-distance primarily (fingers point up/down)
            tip_dist = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
            mcp_dist = ((mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2) ** 0.5

            if tip_dist > mcp_dist * 1.1:  # 10% margin
                extended += 1

        return extended >= 3  # At least 3 of 4 fingers extended

    @staticmethod
    def is_hand_closed(landmarks) -> bool:
        """
        Determine if a hand is closed (fist).

        Returns True if at most 1 finger is extended.
        """
        if len(landmarks) < 21:
            return False

        wrist = landmarks[_WRIST]
        extended = 0

        finger_pairs = [
            (_INDEX_TIP, _INDEX_MCP),
            (_MIDDLE_TIP, _MIDDLE_MCP),
            (_RING_TIP, _RING_MCP),
            (_PINKY_TIP, _PINKY_MCP),
        ]

        for tip_idx, mcp_idx in finger_pairs:
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]

            tip_dist = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
            mcp_dist = ((mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2) ** 0.5

            if tip_dist > mcp_dist * 1.1:
                extended += 1

        return extended <= 1  # At most 1 finger extended

    # -----------------------------------------------------------------
    def close(self):
        """Release resources."""
        if self.available:
            try:
                self.landmarker.close()
            except Exception:
                pass
