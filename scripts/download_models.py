#!/usr/bin/env python3
"""
Download all required pretrained models for the Women Safety Analytics MVP.

Run:  ``python scripts/download_models.py``

Models downloaded
-----------------
1. Face detector   — SSD ResNet-10 (from official OpenCV repos)
2. Gender classifier — Levi-Hassner CNN (from LearnOpenCV / OpenCV community)
3. YOLO11n          — downloaded automatically by the ``ultralytics`` package

MediaPipe models are bundled with the ``mediapipe`` pip package and do
not require a separate download step.
"""

import os
import sys
import urllib.request
import ssl

# Resolve paths relative to the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# ── Model definitions ────────────────────────────────────────────────
MODELS = [
    {
        "name": "Face Detector — Config (deploy.prototxt)",
        "filename": "deploy.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        ],
        "min_size": 1_000,
        "description": "SSD face-detector network definition (OpenCV official)",
    },
    {
        "name": "Face Detector — Weights (res10_300x300_ssd)",
        "filename": "res10_300x300_ssd_iter_140000.caffemodel",
        "urls": [
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        ],
        "min_size": 5_000_000,  # ~10.7 MB
        "description": "SSD ResNet-10 face-detector weights (OpenCV 3rd-party)",
    },
    {
        "name": "Gender Classifier — Config (gender_deploy.prototxt)",
        "filename": "gender_deploy.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
            "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_deploy.prototxt",
        ],
        "min_size": 500,
        "description": "Levi-Hassner gender CNN architecture definition",
    },
    {
        "name": "Gender Classifier — Weights (gender_net.caffemodel)",
        "filename": "gender_net.caffemodel",
        "urls": [
            # eveningglow repo — confirmed available (41.5 MB)
            "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel",
            # Fallback mirrors
            "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel",
        ],
        "min_size": 20_000_000,  # ~44 MB
        "description": "Levi-Hassner gender CNN weights",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────
def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r         {bar}  {pct:5.1f}%  ({downloaded:,} / {total_size:,})")
    else:
        sys.stdout.write(f"\r         Downloaded {downloaded:,} bytes …")
    sys.stdout.flush()


def _is_lfs_pointer(filepath: str) -> bool:
    """Check if a downloaded file is a Git LFS pointer instead of real data."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(44)
        return header.startswith(b"version https://git-lfs.github.com")
    except Exception:
        return False


def download_model(model_info: dict) -> bool:
    """Download a single model file, trying multiple URLs."""
    dest = os.path.join(MODELS_DIR, model_info["filename"])
    min_size = model_info["min_size"]

    # Already downloaded?
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        if not _is_lfs_pointer(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [OK]   {model_info['name']}  ({size_mb:.1f} MB)")
            return True

    # Try each URL in order
    # Create an SSL context that doesn't verify (for corporate proxies)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in model_info["urls"]:
        print(f"  [DL]   {model_info['name']}")
        print(f"         URL: {url}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
            print()  # newline after progress bar

            # Validate
            if not os.path.exists(dest):
                print(f"         ⚠ File not created")
                continue

            fsize = os.path.getsize(dest)
            if fsize < min_size:
                print(f"         ⚠ File too small ({fsize:,} bytes < {min_size:,})")
                os.remove(dest)
                continue

            if _is_lfs_pointer(dest):
                print(f"         ⚠ Got Git LFS pointer — trying next URL")
                os.remove(dest)
                continue

            size_mb = fsize / (1024 * 1024)
            print(f"  [OK]   Downloaded ({size_mb:.1f} MB)")
            return True

        except Exception as exc:
            print(f"\n         ⚠ Download failed: {exc}")
            if os.path.exists(dest):
                os.remove(dest)
            continue

    # All URLs failed
    print(f"  [FAIL] {model_info['name']}  — could not download")
    print(f"         Please download manually and place at:")
    print(f"         {dest}")
    return False


def download_yolo():
    """Trigger the YOLO model auto-download via ultralytics."""
    print()
    print("  [DL]   YOLO11n (via ultralytics auto-download) …")
    try:
        from ultralytics import YOLO

        # This will auto-download to the CWD / ultralytics cache
        _model = YOLO("yolo11n.pt")
        print("  [OK]   YOLO11n model ready")
        return True
    except Exception as exc:
        print(f"  [FAIL] YOLO auto-download failed: {exc}")
        print("         Make sure ultralytics is installed: pip install ultralytics")
        return False


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 56)
    print("  Women Safety Analytics MVP — Model Downloader")
    print("=" * 56)
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. OpenCV DNN models (face + gender)
    results = []
    for model in MODELS:
        ok = download_model(model)
        results.append((model["name"], ok))

    # 2. YOLO
    yolo_ok = download_yolo()
    results.append(("YOLO11n", yolo_ok))

    # 3. MediaPipe Pose Landmarker model
    pose_model = {
        "name": "MediaPipe Pose Landmarker (lite)",
        "filename": "pose_landmarker_lite.task",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        ],
        "min_size": 1_000_000,  # ~5 MB
        "description": "MediaPipe Tasks API pose estimation model",
    }
    pose_ok = download_model(pose_model)
    results.append((pose_model["name"], pose_ok))

    # 4. MediaPipe Hand Landmarker model
    hand_model = {
        "name": "MediaPipe Hand Landmarker",
        "filename": "hand_landmarker.task",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        ],
        "min_size": 1_000_000,  # ~5 MB
        "description": "MediaPipe Tasks API hand landmark detection model",
    }
    hand_ok = download_model(hand_model)
    results.append((hand_model["name"], hand_ok))

    print()
    try:
        import mediapipe  # noqa: F401
        print(f"  [OK]   MediaPipe package installed (v{mediapipe.__version__})")
        results.append(("MediaPipe Package", True))
    except ImportError:
        print("  [FAIL] mediapipe not installed. Run: pip install mediapipe")
        results.append(("MediaPipe Package", False))

    # Summary
    print()
    print("─" * 56)
    print("  SUMMARY")
    print("─" * 56)
    all_ok = True
    for name, ok in results:
        status = "✔" if ok else "✘"
        print(f"    {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All models ready! Run:  python main.py")
    else:
        print("  ⚠ Some models failed — check the errors above.")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
