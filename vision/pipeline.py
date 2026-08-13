"""
Vision Pipeline — orchestrates all CV components per frame.

Components wired together:
  PersonDetector  →  GenderClassifier  →  PersonTracker
  HandDetector    →  SOSDetector
  PoseDetector    (optional, for skeleton drawing)
"""

import logging
from typing import Any, Dict, List

import numpy as np

import config
from vision.person_detector import PersonDetector, PersonDetection
from vision.gender_classifier import GenderClassifier
from vision.tracker import PersonTracker
from vision.hand_detector import HandDetector
from vision.pose_detector import PoseDetector
from vision.sos_detector import SOSDetector, SOSState

logger = logging.getLogger(__name__)


class SafetyPipeline:
    """
    Central coordinator — call :meth:`process_frame` once per webcam
    frame to get detection results, gender counts, and SOS state.
    """

    def __init__(self):
        logger.info("═" * 50)
        logger.info("  Initialising Safety Pipeline")
        logger.info("═" * 50)

        self.person_detector = PersonDetector()
        self.gender_classifier = GenderClassifier()
        self.person_tracker = PersonTracker()
        self.hand_detector = HandDetector()
        self.pose_detector = PoseDetector()
        self.sos_detector = SOSDetector()

        self.frame_count: int = 0

        logger.info("═" * 50)
        logger.info("  Pipeline ready")
        logger.info("═" * 50)

    # -----------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run the full detection pipeline on one BGR frame.

        Returns
        -------
        dict with keys:
            detections    – list[PersonDetection]
            counts        – {"total", "male", "female", "unknown"}
            sos_state     – SOSState enum
            sos_triggered – bool (True on the exact CRITICAL transition)
            sos_progress  – float (0.0 – 1.0)
            hand_detected – bool
        """
        self.frame_count += 1
        self.person_tracker.tick()

        # 1. Person detection (with ByteTrack)
        detections: List[PersonDetection] = self.person_detector.detect(frame)

        # 2. Gender classification (skip some frames for perf)
        do_gender = (self.frame_count % config.GENDER_SKIP_FRAMES == 0)
        active_ids: List[int] = []

        for det in detections:
            # Assign a synthetic ID if tracking failed
            tid = det.track_id if det.track_id is not None else (
                self.frame_count * 10000 + len(active_ids)
            )
            active_ids.append(tid)

            if do_gender and self.gender_classifier.available:
                gender, conf = self.gender_classifier.classify(frame, det.bbox)
                self.person_tracker.update(tid, gender, conf)
            else:
                self.person_tracker.update(tid)

            # Fill in the smoothed gender on the detection object
            sg, sc = self.person_tracker.get_smoothed_gender(tid)
            det.gender = sg
            det.gender_confidence = sc

        # 3. Hand detection for SOS gesture
        hand_landmarks_list = self.hand_detector.detect(frame)

        # 4. SOS state machine (uses hand open/close pattern)
        sos_state = self.sos_detector.update(hand_landmarks_list)

        # 5. Aggregate counts
        counts = self.person_tracker.get_counts(active_ids)

        return {
            "detections": detections,
            "counts": counts,
            "sos_state": sos_state,
            "sos_triggered": self.sos_detector.just_triggered,
            "sos_progress": self.sos_detector.warning_progress,
            "hand_detected": len(hand_landmarks_list) > 0,
        }

    # -----------------------------------------------------------------
    def close(self):
        """Release resources held by sub-components."""
        self.pose_detector.close()
        self.hand_detector.close()
