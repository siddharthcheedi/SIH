# Model Documentation — Women Safety Analytics MVP

Every pretrained model used by this project is documented below with
its license, source, and known limitations.

> **Important — License Compliance**
> Before using any model in a commercial, competition, or government
> context, verify that your intended use complies with the applicable
> licence terms listed below.

---

## 1. Person Detection — YOLO11n

| Field | Details |
|---|---|
| **Model name** | YOLO11n (YOLOv11 Nano) |
| **Purpose** | Real-time person detection |
| **Task** | Object detection (COCO 80-class) — filtered to class 0 ("person") |
| **Source** | Ultralytics |
| **Official URL** | https://docs.ultralytics.com/models/yolo11/ |
| **Repository** | https://github.com/ultralytics/ultralytics |
| **Download method** | Automatic — `YOLO("yolo11n.pt")` downloads on first load |
| **Local path** | `yolo11n.pt` (project root, or Ultralytics cache) |
| **Input** | BGR image, any size (resized internally to 640×640) |
| **Output** | Bounding boxes, class IDs, confidence scores, optional track IDs |
| **Approx. size** | ~5.4 MB |
| **License** | **AGPL-3.0** (open-source) or **Ultralytics Enterprise** (commercial) |
| **CPU suitability** | Excellent — nano variant designed for edge / CPU deployment |
| **Known limitations** | May miss heavily occluded or very small persons; confidence drops at distance |
| **Alternative models** | YOLOv8n, YOLOv5n, YOLO11s (slightly larger / more accurate) |

### ⚠ YOLO11 License Notice

Ultralytics publishes YOLO11 under a **dual-licence** model:

1. **AGPL-3.0** — free for open-source projects whose entire source
   code is also released under a compatible open-source licence.
   Any modifications or derivative works must also be open-sourced.

2. **Ultralytics Enterprise Licence** — a paid licence for proprietary
   / commercial use that does not require open-sourcing your code.

If you are building this for **SIH (Smart India Hackathon)** or any
non-commercial academic prototype, AGPL-3.0 is typically acceptable.
For any commercial deployment, **consult the Ultralytics licensing
page**: https://www.ultralytics.com/license

---

## 2. Face Detection — SSD ResNet-10

| Field | Details |
|---|---|
| **Model name** | res10_300x300_ssd_iter_140000 |
| **Purpose** | Locate faces within person crops (pre-step for gender classification) |
| **Task** | Face detection |
| **Source** | OpenCV (official DNN samples) |
| **Official URL** | https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector |
| **Weights URL** | https://github.com/opencv/opencv_3rdparty (dnn_samples_face_detector_20170830 branch) |
| **Download method** | `python scripts/download_models.py` |
| **Local path** | `models/res10_300x300_ssd_iter_140000.caffemodel` + `models/deploy.prototxt` |
| **Input** | BGR image resized to 300×300, mean-subtracted |
| **Output** | Bounding boxes + confidence for detected faces |
| **Approx. size** | ~10.7 MB (weights) + ~28 KB (prototxt) |
| **License** | Released as part of OpenCV samples — **Apache 2.0 / BSD-3-Clause** |
| **CPU suitability** | Excellent — lightweight SSD architecture |
| **Known limitations** | May miss faces at extreme angles, heavy occlusion, or very low resolution |
| **Alternative models** | OpenCV Haar cascade, MTCNN, MediaPipe Face Detection |

> **Privacy note** — The face detector is used *only* to locate the
> facial region for gender classification.  It does **not** identify
> individuals.  No facial embeddings, encodings, or biometric data
> are computed or stored.

---

## 3. Gender Classification — Levi-Hassner CNN

| Field | Details |
|---|---|
| **Model name** | gender_net (Levi & Hassner, 2015) |
| **Purpose** | Apparent gender classification from face crops |
| **Task** | Binary image classification (Male / Female) |
| **Source** | Gil Levi & Tal Hassner — *"Age and Gender Classification using Convolutional Neural Networks"*, IEEE CVPR Workshops 2015. Weights hosted by LearnOpenCV (Satya Mallick / OpenCV.org) |
| **Paper** | https://talhassner.github.io/home/publication/2015_CVPR |
| **Download method** | `python scripts/download_models.py` |
| **Local path** | `models/gender_net.caffemodel` + `models/gender_deploy.prototxt` |
| **Input** | 227×227 BGR face crop, mean-subtracted |
| **Output** | Softmax probabilities for [Male, Female] |
| **Approx. size** | ~44 MB (weights) + ~1 KB (prototxt) |
| **License** | Originally released for academic use. Weights are widely redistributed under permissive terms by the OpenCV community. Verify with the original authors for commercial use. |
| **CPU suitability** | Excellent — simple AlexNet-variant, ~5 ms inference on modern CPU |
| **Known limitations** | Trained on limited demographics (2015); accuracy degrades on non-frontal faces, children, and underrepresented groups. Outputs "Apparent Gender" only. |
| **Alternative models** | See "Gender Model Evaluation" below |

### Gender Model Evaluation

Three candidate models were evaluated for this MVP:

| Model | Architecture | Input | Size | CPU Speed | Accuracy | License | Chosen? |
|---|---|---|---|---|---|---|---|
| **gender_net (Caffe)** | AlexNet-variant | 227×227 face | ~44 MB | ~5 ms | ~86% (Adience) | Academic / OpenCV community | **✔ Yes** |
| sgdkn/gender-classification (HuggingFace) | ViT-base | 224×224 image | ~330 MB (+PyTorch) | ~80 ms | 76.1% (self-reported) | Not clearly specified | ✘ |
| onnx-community/gender-classification-ONNX | ViT variant | 224×224 face | ~90 MB (+onnxruntime) | ~20 ms | ~94% (UTKFace) | Apache 2.0 | Backup option |

**Rationale for selecting gender_net (Caffe):**
- Zero additional Python dependencies (OpenCV DNN is already installed)
- Fastest CPU inference (~5 ms vs 80+ ms for ViT-based models)
- Removes the need for `transformers` package (~500 MB install)
- Well-proven in production OpenCV pipelines
- Good-enough accuracy for an MVP demo

**Known limitation:** The Caffe model was trained in 2015 on the
Adience dataset.  It may be less accurate on diverse demographics
than more modern models.  The ONNX community model is documented as
a higher-accuracy alternative if more precision is needed.

### Important Ethical Notes

- The system labels output as **"Apparent Gender Classification"** or
  **"AI-estimated Gender"**.
- It does **NOT** determine verified biological sex.
- It does **NOT** perform facial recognition or identity recognition.
- It does **NOT** store face images or biometric data.
- Low-confidence predictions are labelled **"Unknown"**.

---

## 4. Pose Estimation — MediaPipe Pose

| Field | Details |
|---|---|
| **Model name** | MediaPipe Pose (BlazePose) |
| **Purpose** | Body landmark estimation for SOS gesture detection |
| **Task** | 2D/3D pose estimation — 33 body landmarks |
| **Source** | Google MediaPipe |
| **Official URL** | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |
| **Repository** | https://github.com/google-ai-edge/mediapipe |
| **Download method** | Bundled with the `mediapipe` pip package — no separate download |
| **Local path** | Managed internally by MediaPipe |
| **Input** | RGB image (any size) |
| **Output** | 33 normalised body landmarks with visibility scores |
| **Approx. size** | ~3 MB (complexity=0, lite model, bundled in pip package) |
| **License** | **Apache 2.0** |
| **CPU suitability** | Excellent — BlazePose designed for mobile / edge |
| **Known limitations** | Single-person pose estimation (processes the most prominent person). May struggle with extreme occlusion. |
| **Alternative models** | MediaPipe Pose (complexity=1 or 2), OpenPose, MoveNet |

### API Note

This MVP uses the **MediaPipe Solutions API** (legacy):
```python
import mediapipe as mp
pose = mp.solutions.pose.Pose(model_complexity=0)
```

The newer `mp.tasks` API is also available but requires downloading
separate `.task` model files.  The Solutions API is simpler and
sufficient for the MVP.

---

## Model File Checklist

After running `python scripts/download_models.py`, you should have:

```
models/
  deploy.prototxt                            (~28 KB)
  res10_300x300_ssd_iter_140000.caffemodel   (~10.7 MB)
  gender_deploy.prototxt                     (~1 KB)
  gender_net.caffemodel                      (~44 MB)
```

Plus `yolo11n.pt` in the project root (or the Ultralytics cache).

MediaPipe models are bundled inside the `mediapipe` Python package.
