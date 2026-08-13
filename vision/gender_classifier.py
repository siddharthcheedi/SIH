"""
Apparent Gender Classification — InsightFace (ArcFace).

Pipeline
--------
Person bounding box  →  InsightFace FaceAnalysis  →  Male / Female / Unknown

Uses the ``buffalo_sc`` model pack (small & fast) from InsightFace:
  • RetinaFace  — face detection  (much better than old SSD)
  • ArcFace     — face recognition + gender/age estimation

GPU acceleration is automatic when ``onnxruntime-gpu`` is installed.

IMPORTANT — this provides **Apparent Gender Classification** /
**AI-estimated Gender**.  It does NOT determine biological sex,
perform facial recognition, or identify individuals.
"""

import logging
import os
from typing import Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# InsightFace gender mapping
_GENDER_MAP = {0: "Female", 1: "Male"}


class GenderClassifier:
    """InsightFace-based apparent-gender classifier."""

    def __init__(self):
        logger.info("Initialising gender classifier (InsightFace) …")
        self.available = False
        self.app = None
        self._cached_frame_id = None
        self._cached_faces = []

        try:
            from insightface.app import FaceAnalysis

            # Detect available providers
            providers = self._get_providers()
            logger.info("  ONNX providers: %s", providers)

            # Use buffalo_l model pack with detection and genderage modules
            self.app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "genderage"],
                providers=providers,
            )
            # det_size=(640, 640) for full-frame webcam resolution
            self.app.prepare(ctx_id=0, det_size=(640, 640))

            self.available = True
            gpu_str = "GPU" if "CUDAExecutionProvider" in providers else "CPU"
            logger.info("  InsightFace ready  (buffalo_l genderage, %s)", gpu_str)

        except ImportError:
            logger.error("insightface package not found! Please run: pip install insightface onnxruntime")
        except Exception as exc:
            logger.error("InsightFace init failed: %s", exc)

    # -----------------------------------------------------------------
    @staticmethod
    def _get_providers():
        """Return the best available ONNX runtime providers."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                # Test if CUDA DLLs can actually be loaded
                try:
                    sess_opt = ort.SessionOptions()
                    sess_opt.log_severity_level = 3
                    ort.InferenceSession(b"\x08\x00", sess_options=sess_opt, providers=["CUDAExecutionProvider"])
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except Exception:
                    pass
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    # -----------------------------------------------------------------
    def _get_faces_for_frame(self, frame: np.ndarray):
        """Run InsightFace on full frame once and cache results for the frame."""
        frame_id = id(frame)
        if self._cached_frame_id != frame_id:
            self._cached_frame_id = frame_id
            try:
                self._cached_faces = self.app.get(frame)
            except Exception as exc:
                logger.warning("InsightFace face detection error: %s", exc)
                self._cached_faces = []
        return self._cached_faces

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def classify(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[str, float]:
        """
        Classify apparent gender for one detected person by matching full-frame faces.
        """
        if not self.available or self.app is None:
            return "Unknown", 0.0

        faces = self._get_faces_for_frame(frame)
        if not faces:
            return "Unknown", 0.0

        h, w = frame.shape[:2]
        px1, py1, px2, py2 = bbox

        # Find face whose bounding box overlaps or center is inside person bbox
        best_face = None
        best_score = 0.0

        for face in faces:
            fb = face.bbox  # [fx1, fy1, fx2, fy2]
            fcx = (fb[0] + fb[2]) / 2.0
            fcy = (fb[1] + fb[3]) / 2.0

            # Allow slight padding around person bbox (15% margin)
            margin_x = (px2 - px1) * 0.15
            margin_y = (py2 - py1) * 0.15

            if (px1 - margin_x) <= fcx <= (px2 + margin_x) and (py1 - margin_y) <= fcy <= (py2 + margin_y):
                score = float(face.det_score)
                if score > best_score:
                    best_score = score
                    best_face = face

        if best_face is None:
            return "Unknown", 0.0

        raw_gender = getattr(best_face, "gender", None)
        if raw_gender is None:
            return "Unknown", 0.0

        gender_code = int(raw_gender)  # 0=Female, 1=Male
        gender_label = _GENDER_MAP.get(gender_code, "Unknown")
        conf = float(getattr(best_face, "det_score", 0.8))

        return gender_label, conf
