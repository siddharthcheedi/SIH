#!/usr/bin/env python3
"""
Test that every pretrained model can be loaded successfully.

Run:  ``python scripts/test_models.py``

Expected output (all-green)::

    [OK]  YOLO person detector loaded
    [OK]  Face detector loaded
    [OK]  Gender classifier loaded
    [OK]  MediaPipe Pose initialised
    [OK]  Supabase connection successful
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

os.chdir(PROJECT_DIR)

import config  # noqa: E402


def _ok(msg):
    print(f"  [OK]      {msg}")


def _warn(msg):
    print(f"  [WARNING] {msg}")


def _fail(msg):
    print(f"  [FAIL]    {msg}")


def test_yolo():
    try:
        from ultralytics import YOLO

        model = YOLO(config.YOLO_MODEL)
        _ok(f"YOLO person detector loaded  ({config.YOLO_MODEL})")
        return True
    except Exception as exc:
        _fail(f"YOLO load failed: {exc}")
        return False


def test_face_detector():
    import cv2

    pass  # Placeholder for future YOLO config tests


def test_insightface():
    """Test InsightFace face analysis + gender classifier."""
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        # Check for GPU
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            _ok("ONNX Runtime GPU available  (CUDAExecutionProvider)")
            use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            _warn("ONNX Runtime GPU not found — using CPU (install onnxruntime-gpu for GPU)")
            use_providers = ["CPUExecutionProvider"]

        # Load the model (auto-downloads on first run)
        app = FaceAnalysis(name="buffalo_sc", providers=use_providers)
        app.prepare(ctx_id=0, det_size=(320, 320))
        _ok("InsightFace loaded  (buffalo_sc — face detection + gender)")
        return True
    except ImportError as exc:
        _fail(f"InsightFace not installed: {exc}")
        _warn("Run: pip install insightface onnxruntime")
        return False
    except Exception as exc:
        _fail(f"InsightFace init failed: {exc}")
        return False


def test_mediapipe():
    # Check model file exists
    if not os.path.exists(config.POSE_MODEL_PATH):
        _fail(f"Pose model not found: {config.POSE_MODEL_PATH}")
        _warn("Run:  python scripts/download_models.py")
        return False

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_tasks.BaseOptions(
            model_asset_path=config.POSE_MODEL_PATH
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        landmarker.close()
        _ok("MediaPipe Pose initialised  (Tasks API, lite model)")
        return True
    except Exception as exc:
        _fail(f"MediaPipe init failed: {exc}")
        return False


def test_supabase():
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        _warn("Supabase not configured (no URL/KEY in .env)")
        _ok("Local inference can continue without Supabase")
        return True  # Not a fatal error

    try:
        from supabase import create_client

        client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        client.table("sessions").select("id").limit(1).execute()
        _ok("Supabase connection successful")
        return True
    except Exception as exc:
        _warn(f"Supabase unavailable: {exc}")
        _ok("Local inference can continue without Supabase")
        return True


def main():
    print()
    print("=" * 56)
    print("  Women Safety Analytics MVP — Model Tests")
    print("=" * 56)
    print()

    results = []
    for name, fn in [
        ("YOLO", test_yolo),
        ("InsightFace", test_insightface),
        ("MediaPipe", test_mediapipe),
        ("Supabase", test_supabase),
    ]:
        ok = fn()
        results.append((name, ok))
        print()

    print("─" * 56)
    all_ok = all(ok for _, ok in results)
    if all_ok:
        print("  All components ready!  Run:  python main.py")
    else:
        print("  ⚠ Some components failed. Fix the errors above.")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
