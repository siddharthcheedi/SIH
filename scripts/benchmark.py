#!/usr/bin/env python3
"""
Benchmark — measure per-component inference latency.

Run:  ``python scripts/benchmark.py``

Creates a synthetic 640×480 test image and measures:
  • YOLO person detection
  • Face detection (OpenCV DNN)
  • Gender classification (OpenCV DNN)
  • MediaPipe Pose estimation
  • Total pipeline estimate

Results are approximate — they give a quick sanity-check on whether
the laptop can sustain real-time processing.
"""

import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

os.chdir(PROJECT_DIR)

import config  # noqa: E402

WARMUP_ITERS = 3
BENCH_ITERS = 20


def _bench(label, fn, iters=BENCH_ITERS):
    """Time *fn* over *iters* calls; print average ms."""
    # Warm-up
    for _ in range(WARMUP_ITERS):
        fn()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms
    print(f"  {label:<28s}  {elapsed:7.1f} ms")
    return elapsed


def main():
    print()
    print("=" * 56)
    print("  Women Safety Analytics MVP — Benchmark")
    print("=" * 56)
    print()

    # Synthetic test frame (dark image with a rectangle as a "person")
    frame = np.random.randint(40, 180, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 50), (440, 450), (200, 180, 160), -1)  # fake person

    total_ms = 0.0

    # ── YOLO ─────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO

        model = YOLO(config.YOLO_MODEL)
        ms = _bench("YOLO person detection", lambda: model(frame, classes=[0], conf=0.5, verbose=False))
        total_ms += ms
    except Exception as exc:
        print(f"  [SKIP] YOLO: {exc}")

    # ── Face detection ───────────────────────────────────────────────
    try:
        face_net = cv2.dnn.readNet(config.FACE_MODEL_PATH, config.FACE_PROTO_PATH)

        def _face():
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            face_net.setInput(blob)
            face_net.forward()

        ms = _bench("Face detection (SSD)", _face)
        total_ms += ms
    except Exception as exc:
        print(f"  [SKIP] Face detection: {exc}")

    # ── Gender classification ────────────────────────────────────────
    try:
        gender_net = cv2.dnn.readNet(config.GENDER_MODEL_PATH, config.GENDER_PROTO_PATH)
        face_crop = cv2.resize(frame[50:250, 250:400], (227, 227))

        def _gender():
            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227), (78.43, 87.77, 114.90)
            )
            gender_net.setInput(blob)
            gender_net.forward()

        ms = _bench("Gender classification (CNN)", _gender)
        total_ms += ms
    except Exception as exc:
        print(f"  [SKIP] Gender classification: {exc}")

    # ── MediaPipe Pose ───────────────────────────────────────────────
    try:
        import mediapipe as mp

        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        def _pose():
            pose.process(rgb)

        ms = _bench("MediaPipe Pose (complexity=0)", _pose)
        total_ms += ms
        pose.close()
    except Exception as exc:
        print(f"  [SKIP] MediaPipe Pose: {exc}")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("─" * 56)
    print(f"  Estimated total pipeline  {total_ms:7.1f} ms")
    if total_ms > 0:
        est_fps = 1000.0 / total_ms
        print(f"  Estimated FPS             {est_fps:7.1f}")
        if est_fps >= 15:
            print("  ✔ Should run comfortably in real-time")
        elif est_fps >= 8:
            print("  ⚠ Borderline — consider reducing resolution or GENDER_SKIP_FRAMES")
        else:
            print("  ✘ Likely too slow — try reducing FRAME_WIDTH/FRAME_HEIGHT")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
