"""
Person Detection — YOLO11n with built-in ByteTrack tracking.

Uses the Ultralytics pretrained COCO model to detect only the "person"
class (class 0).  ByteTrack tracking is enabled by default so that each
person gets a stable temporary ID across frames.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


class PersonDetection:
    """Data class for a single detected person."""

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        track_id: Optional[int] = None,
    ):
        self.bbox = bbox  # (x1, y1, x2, y2) — pixel coordinates
        self.confidence = confidence
        self.track_id = track_id
        # Gender fields are filled in later by the pipeline
        self.gender: str = "Unknown"
        self.gender_confidence: float = 0.0

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class PersonDetector:
    """
    YOLO-based person detector.

    Loads the lightweight YOLO11n model and filters for the "person" class
    only.  Supports optional ByteTrack tracking for stable IDs.
    """

    def __init__(self):
        logger.info("Loading YOLO model: %s", config.YOLO_MODEL)
        try:
            self.model = YOLO(config.YOLO_MODEL)
            self.available = True
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            self.available = False
            raise

    def detect(
        self, frame: np.ndarray, use_tracking: bool = True
    ) -> List[PersonDetection]:
        """
        Detect persons in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the webcam.
        use_tracking : bool
            If ``True``, use YOLO's built-in ByteTrack tracker so that
            each person gets a persistent ID.

        Returns
        -------
        list[PersonDetection]
        """
        detections: List[PersonDetection] = []
        if not self.available:
            return detections

        try:
            if use_tracking:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[config.PERSON_CLASS_ID],
                    conf=config.YOLO_CONFIDENCE,
                    verbose=False,
                )
            else:
                results = self.model(
                    frame,
                    classes=[config.PERSON_CLASS_ID],
                    conf=config.YOLO_CONFIDENCE,
                    verbose=False,
                )

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    conf = float(boxes.conf[i].cpu().numpy())

                    track_id: Optional[int] = None
                    if use_tracking and boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())

                    detections.append(
                        PersonDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            track_id=track_id,
                        )
                    )
        except Exception as e:
            logger.warning("Person detection error: %s", e)

        return detections
