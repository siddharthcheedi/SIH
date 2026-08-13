"""
Women Safety Analytics MVP — Configuration
==========================================
All configurable values in one place.
Loaded from environment variables (.env) with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# ============================================================
# Camera Configuration
# ============================================================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "720"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))

# ============================================================
# YOLO Person Detection
# ============================================================
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
PERSON_CLASS_ID = 0  # COCO class ID for "person"

# ============================================================
# Gender Classification — InsightFace (ArcFace)
# ============================================================
# InsightFace auto-downloads models to ~/.insightface/models/
# Uses buffalo_sc (small, fast) model pack
GENDER_CONFIDENCE_THRESHOLD = float(os.getenv("GENDER_CONFIDENCE", "0.5"))
GENDER_SMOOTHING_FRAMES = int(os.getenv("GENDER_SMOOTHING_FRAMES", "10"))

# ============================================================
# SOS Gesture Detection
# ============================================================
POSE_MODEL_PATH = str(MODELS_DIR / "pose_landmarker_lite.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")

SOS_CONFIRMATION_SECONDS = float(os.getenv("SOS_CONFIRMATION_SECONDS", "1.5"))
SOS_COOLDOWN_SECONDS = float(os.getenv("SOS_COOLDOWN_SECONDS", "15"))
SOS_MIN_VISIBILITY = float(os.getenv("SOS_MIN_VISIBILITY", "0.5"))

# Hand open/close gesture pattern
SOS_HAND_CYCLES_REQUIRED = int(os.getenv("SOS_HAND_CYCLES", "3"))  # open→close cycles needed
SOS_HAND_WINDOW_SECONDS = float(os.getenv("SOS_HAND_WINDOW", "5.0"))  # time window for cycles

# ============================================================
# Performance
# ============================================================
TARGET_FPS = int(os.getenv("TARGET_FPS", "120"))
GENDER_SKIP_FRAMES = int(os.getenv("GENDER_SKIP_FRAMES", "2"))  # Classify gender every N frames

# ============================================================
# Supabase
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ============================================================
# Local Logging
# ============================================================
ALERTS_CSV = str(LOGS_DIR / "alerts.csv")

# ============================================================
# UI / Display
# ============================================================
WINDOW_NAME = "Women Safety Analytics — MVP"
PANEL_WIDTH = 300  # Right-side dashboard panel width in pixels
