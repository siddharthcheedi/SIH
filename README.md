# Women Safety Analytics — MVP

> **Real-time women safety monitoring prototype** using computer vision
> on a laptop webcam.  Detects persons, classifies apparent gender,
> counts male / female / unknown, and triggers SOS alerts when a
> distress gesture is detected.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [MVP Objectives](#mvp-objectives)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Hardware Requirements](#hardware-requirements)
6. [Installation](#installation)
7. [Pretrained Models](#pretrained-models)
8. [Supabase Setup](#supabase-setup)
9. [Environment Configuration](#environment-configuration)
10. [How to Run](#how-to-run)
11. [SOS Gesture Instructions](#sos-gesture-instructions)
12. [Demo Scenarios](#demo-scenarios)
13. [Troubleshooting](#troubleshooting)
14. [Privacy](#privacy)
15. [MVP Limitations](#mvp-limitations)
16. [Final Product Roadmap](#final-product-roadmap)

---

## Project Overview

This is the **Minimum Viable Product (MVP)** for the Women Safety
Analytics system.  It proves the core computer-vision concept on a
single laptop webcam — no servers, no cloud processing, no CCTV
network.

The OpenCV window **is** the entire user interface.

```
        LAPTOP WEBCAM
              ↓
      YOLO PERSON DETECTION
              ↓
     GENDER CLASSIFICATION
              ↓
      MALE / FEMALE COUNT
              ↓
    MEDIAPIPE POSE ESTIMATION
              ↓
     SOS GESTURE DETECTION
              ↓
     TEMPORAL CONFIRMATION
              ↓
       ALERT GENERATION
          ↙       ↘
    OpenCV Alert   Supabase
    + Sound         Log
```

---

## MVP Objectives

| # | Objective | Status |
|---|---|---|
| 1 | Real-time person detection | ✔ |
| 2 | Apparent gender classification | ✔ |
| 3 | Male / Female / Unknown counting | ✔ |
| 4 | SOS distress gesture detection | ✔ |
| 5 | Temporal confirmation (no false triggers) | ✔ |
| 6 | Local visual + audible alert | ✔ |
| 7 | Structured alert storage in Supabase | ✔ |
| 8 | Local CSV alert logging | ✔ |
| 9 | Session management | ✔ |
| 10 | Graceful offline mode (no Supabase) | ✔ |

---

## Architecture

```
women-safety-mvp/
│
├── main.py                  ← Entry point
├── config.py                ← All configuration
├── requirements.txt
├── .env.example
│
├── models/                  ← Pretrained model files
│   ├── MODELS.md            ← Detailed model documentation
│   └── README.md
│
├── scripts/
│   ├── download_models.py   ← One-click model setup
│   ├── test_models.py       ← Component health check
│   └── benchmark.py         ← CPU performance test
│
├── vision/
│   ├── person_detector.py   ← YOLO11n + ByteTrack
│   ├── gender_classifier.py ← OpenCV DNN face + gender
│   ├── tracker.py           ← Gender temporal smoothing
│   ├── pose_detector.py     ← MediaPipe Pose
│   ├── sos_detector.py      ← SOS state machine
│   └── pipeline.py          ← Orchestrator
│
├── services/
│   ├── alert_engine.py      ← Alert coordination + sound
│   ├── supabase_service.py  ← Supabase CRUD
│   └── local_logger.py      ← CSV logging
│
├── ui/
│   └── overlay.py           ← OpenCV dashboard
│
├── logs/
│   └── alerts.csv           ← Local alert log
│
└── supabase/
    └── schema.sql           ← Database tables
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Person Detection | YOLO11n (Ultralytics) |
| Person Tracking | ByteTrack (built into Ultralytics) |
| Face Detection | SSD ResNet-10 (OpenCV DNN) |
| Gender Classification | Levi-Hassner CNN (OpenCV DNN) |
| Pose Estimation | MediaPipe BlazePose (complexity=0) |
| Database | Supabase PostgreSQL |
| Local Logging | CSV |
| UI | OpenCV window with custom overlay |
| Audio Alert | `winsound` (Windows) / terminal bell |

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 | Windows 10/11 |
| Python | 3.10 | 3.11+ |
| CPU | Any modern CPU | Intel i5 / AMD Ryzen 5+ |
| RAM | 4 GB | 8 GB+ |
| Webcam | Built-in laptop camera | Any USB / built-in webcam |
| GPU | Not required | NVIDIA GPU (optional, auto-used) |
| Internet | For model download & Supabase | — |

---

## Installation

### 1. Clone / Copy the Project

```bash
cd SIH/women-safety-mvp
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
x```

> **Note:** This will install PyTorch (required by Ultralytics YOLO).
> First install may take a few minutes and ~2 GB of disk space.

### 4. Download Pretrained Models

```bash
python scripts/download_models.py
```

This downloads:
- Face detector (~10.7 MB)
- Gender classifier (~44 MB)
- YOLO11n (~5.4 MB, via Ultralytics auto-download)
- MediaPipe models (bundled with pip package)

### 5. Test Models

```bash
python scripts/test_models.py
```

Expected output:
```
  [OK]  YOLO person detector loaded
  [OK]  Face detector loaded
  [OK]  Gender classifier loaded
  [OK]  MediaPipe Pose initialised
  [OK]  Supabase connection successful  (or WARNING if not configured)
```

### 6. Benchmark (Optional)

```bash
python scripts/benchmark.py
```

---

## Pretrained Models

See [`models/MODELS.md`](models/MODELS.md) for complete documentation
of every model including:
- Source and official URLs
- License terms (including YOLO AGPL-3.0 / Enterprise)
- Input / output specifications
- Gender model evaluation (3 candidates compared)
- Known limitations

### Model Licenses Summary

| Model | License |
|---|---|
| YOLO11n | AGPL-3.0 or Ultralytics Enterprise |
| Face Detector (SSD) | Apache 2.0 / BSD-3 (OpenCV) |
| Gender CNN (Caffe) | Academic use (Levi & Hassner) |
| MediaPipe Pose | Apache 2.0 |

> **⚠ Verify** that your intended use (SIH / academic / commercial)
> complies with the applicable licence for each model.

---

## Supabase Setup

### 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign up / log in.
2. Click **New Project**.
3. Choose a name, set a database password, select a region.
4. Wait for the project to provision.

### 2. Create Database Tables

1. In your Supabase dashboard, go to **SQL Editor**.
2. Click **New Query**.
3. Copy the contents of [`supabase/schema.sql`](supabase/schema.sql)
   and paste it into the editor.
4. Click **Run**.

This creates:
- `sessions` — tracks each application run
- `alerts` — stores confirmed SOS events

### 3. Get Your Credentials

1. Go to **Settings → API** in your Supabase dashboard.
2. Copy the **Project URL** (e.g., `https://xxxx.supabase.co`).
3. Copy the **anon / public** API key.

### 4. Create `.env` File

```bash
cp .env.example .env
```

Edit `.env`:
```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
```

> **Without Supabase:** The MVP works perfectly without Supabase.
> Leave the `.env` fields empty and the system runs in offline mode.
> Alerts are still saved to `logs/alerts.csv`.

---

## Environment Configuration

All settings are in [`config.py`](config.py) with sensible defaults.
Override via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `FRAME_WIDTH` | `640` | Capture width |
| `FRAME_HEIGHT` | `480` | Capture height |
| `YOLO_MODEL` | `yolo11n.pt` | YOLO model file |
| `YOLO_CONFIDENCE` | `0.5` | Person detection threshold |
| `GENDER_CONFIDENCE` | `0.6` | Gender classification threshold |
| `GENDER_SMOOTHING_FRAMES` | `10` | Temporal smoothing window |
| `SOS_CONFIRMATION_SECONDS` | `1.5` | How long to hold gesture |
| `SOS_COOLDOWN_SECONDS` | `15` | Seconds between alerts |
| `GENDER_SKIP_FRAMES` | `3` | Run gender every N frames |
| `SUPABASE_URL` | (empty) | Supabase project URL |
| `SUPABASE_KEY` | (empty) | Supabase anon key |

---

## How to Run

```bash
python main.py
```

The application will:
1. Load configuration
2. Connect to Supabase (or go offline)
3. Load YOLO, gender classifier, MediaPipe
4. Open the webcam
5. Start real-time monitoring

**Press Q** to quit.

---

## SOS Gesture Instructions

The MVP recognises **one** distress gesture:

### 🙌 Both Hands Raised Above the Head

1. Face the camera.
2. Raise **both hands** clearly above your head.
3. Keep your **elbows above your shoulders**.
4. **Hold the gesture** for ~1.5 seconds.

**What happens:**
- Yellow progress bar appears (**WARNING** state)
- After 1.5s → SOS status turns **CRITICAL**
- Red overlay: *"!! DISTRESS SIGNAL DETECTED !!"*
- Warning beep plays
- Alert logged to Supabase + CSV
- 15-second cooldown prevents duplicate alerts

**Tips for demo:**
- Stand 1–2 metres from the camera
- Ensure good lighting
- Keep hands clearly visible
- Raise hands high — wrists above the nose

---

## Demo Scenarios

### Demo 1 — Person Detection
One person enters the camera view.
→ Bounding box appears with gender label and confidence.

### Demo 2 — Multiple People
Multiple people enter the scene.
→ Each person gets a bounding box. Counts update.

### Demo 3 — SOS Alert
A person raises both hands above head for 1.5s.
→ WARNING → CRITICAL. Red overlay + sound + logging.

### Demo 4 — Alert Cooldown
Continue holding the gesture after an alert.
→ Only one alert fires. No spam. Cooldown shows in status.

### Demo 5 — Supabase Offline
Run without Supabase configured.
→ Status shows OFFLINE. Everything else continues.
→ Alerts saved to `logs/alerts.csv`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot open camera 0" | Close other apps using the webcam. Try `CAMERA_INDEX=1` |
| "Missing model file" | Run `python scripts/download_models.py` |
| Model download fails | Check internet connection. See manual download links in `models/MODELS.md` |
| Low FPS (< 10) | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT`. Increase `GENDER_SKIP_FRAMES` |
| Gender always "Unknown" | Ensure face is visible. Check lighting. The model needs a clear face |
| SOS not triggering | Hold gesture steadily for 1.5s. Stand closer. Ensure good visibility |
| Supabase errors | Verify `.env` credentials. Check that schema.sql has been run |
| `mediapipe` import error | `pip install mediapipe` — version ≥ 0.10.0 |
| `winsound` not found | Only available on Windows. Other OS will use terminal bell |

---

## Privacy

This system is designed with privacy in mind:

- ✅ All video processing is **local** (on your laptop)
- ✅ **No raw video, frames, or images** are uploaded anywhere
- ✅ **No facial recognition** or identity recognition
- ✅ **No biometric data** is computed or stored
- ✅ Only **structured alert metadata** (counts, timestamps) goes to Supabase
- ✅ Supabase credentials are in environment variables (never hard-coded)
- ✅ The webcam feed **never** leaves the laptop

---

## MVP Limitations

This MVP does **NOT** implement:

- ❌ Lone woman at night detection
- ❌ Woman surrounded by men detection
- ❌ Hotspot detection
- ❌ Historical incident prediction
- ❌ Multi-camera CCTV
- ❌ IP cameras / RTSP streams
- ❌ Police dispatch or emergency calling
- ❌ Facial recognition or identity recognition
- ❌ Advanced behaviour prediction
- ❌ City-scale deployment
- ❌ Edge / GPU-optimised deployment

These features belong to the **final product** (see roadmap below).

---

## Final Product Roadmap

The MVP will evolve into a comprehensive safety analytics platform:

```
CCTV / IP CAMERA NETWORK
          ↓
    VIDEO INGESTION
          ↓
   YOLO PERSON DETECTION
          ↓
   MULTI-PERSON TRACKING
          ↓
  GENDER / CONTEXT ANALYTICS
          ↓
   POSE + GESTURE ANALYSIS
          ↓
    BEHAVIOUR ANALYSIS
          ↓
     CONTEXT ENGINE
          ↓
      RISK ENGINE
          ↓
    REAL-TIME ALERTS
          ↓
   HISTORICAL DATABASE
          ↓
   HOTSPOT ANALYTICS
          ↓
  AUTHORITY DASHBOARD
```

### Future Features

1. **Lone woman at night detection** — identify when a woman is alone
   in a poorly-lit area
2. **Woman surrounded by men** — detect potentially unsafe group
   dynamics
3. **More SOS gestures** — hand signals, body language patterns
4. **Behaviour / anomaly analysis** — detect unusual movement patterns
5. **Multiple CCTV streams** — process several cameras simultaneously
6. **Camera / location management** — map-based camera registry
7. **Historical alert analysis** — trends, patterns, heatmaps
8. **Risk scoring** — per-location, per-time risk levels
9. **Hotspot identification** — areas with frequent incidents
10. **Map-based visualisation** — geographic dashboard
11. **Authority dashboard** — web-based control centre for security teams
12. **Scalable GPU / edge deployment** — NVIDIA Jetson, cloud processing

---

## License

This project uses multiple pretrained models with different licences.
See [`models/MODELS.md`](models/MODELS.md) for full details.

The project code itself is provided for the Smart India Hackathon
(SIH) prototype.  Check the applicable licences before any commercial
deployment.
