# 🎱 Railbird AI — Match Studio & 2D Physics Engine

AI-powered pool/billiards vision tracking and physics analysis platform with sub-pixel ball detection, multi-cushion bank prediction, ghost ball aiming geometry, and interactive 2D table radar.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tienanhnguyen996/billard/blob/main/train_pool_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)

---

## 🌟 Key Capabilities

1. **Multi-Camera Computer Vision**:
   - **Matchroom Broadcast Perspective** (`hoang_sao.mp4`): 720p 30fps European Open match between Duong Quoc Hoang and Pijus Labutis. Frame-accurate 9-ball tracking, spin classification (Center Stun, Draw, Follow), and break rack explosion dynamics.
   - **Angled Club CCTV** (`SAMPLE.mp4`): 1080p club security camera footage with perspective homography calibration and rail reflection modeling.
   - **Top-Down Overhead Camera** (`overhead_drill.mp4`): True overhead 1080p drill footage tracking cue ball corridors and target ball progression.

2. **2D Radar & Physics Engine**:
   - Homography projection mapping perspective video coordinates $(x, y)$ to 2D table coordinates $(rx, ry)$ ($1000 \times 500$ mm).
   - Ghost Ball tangent line calculation with $90^\circ$ stun deflection and physical cushion reflection angles.
   - Zero phantom mid-air trajectories.

3. **High-Performance Architecture**:
   - **Backend**: Multithreaded Python HTTP/REST server (`app_server.py`) with HTTP 206 Byte-Range partial content streaming for sub-frame video scrubbing.
   - **Frontend**: Clean React 19 + TypeScript + Vite architecture with synchronized video HUD canvas and SVG table radar.

---

## ☁️ Zero-Local-Load Model Training (Google Colab)

To protect local PC hardware and avoid thermal throttling or VRAM limitations (e.g. GTX 1650 4GB), model training runs entirely in Google Cloud:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tienanhnguyen996/billard/blob/main/train_pool_colab.ipynb)

- **Notebook**: [`train_pool_colab.ipynb`](train_pool_colab.ipynb)
- **Dataset**: `kevinstotz/pool-ball-detection` (Hugging Face)
- **Model**: YOLO26s / YOLO11s trained at high resolution (`imgsz=1280`)
- **Hardware**: Free Google Colab T4 / L4 GPU
- **Output**: Automatically downloads `best_pool_1280.pt` upon training completion.

---

## 🚀 Quickstart

### 1. Launch Backend Server
```bash
python app_server.py
```
Server starts listening on `http://localhost:8080/`.

### 2. Open Web Dashboard
Open your browser and navigate to:
```
http://localhost:8080/
```

### 3. Run QA Assertion Test Suite
```bash
python production_test_suite.py
```

---

## 📁 Repository Structure

```
├── app_server.py              # Multithreaded REST + video streaming backend
├── production_test_suite.py   # End-to-end QA assertion test suite
├── train_pool_colab.ipynb     # 1-Click Google Colab cloud training notebook
├── train_ball_classifier.py   # Stage-2 PyTorch CNN classifier definition
├── hoang_sao_session.json     # Broadcast session metadata & ball telemetry
├── session_data.json          # CCTV session metadata & ball telemetry
├── overhead_session.json      # Overhead top-cam session metadata
├── frontend/                  # React 19 + TypeScript + Vite source
└── pretrained_models/         # Pretrained PyTorch weights (.pt)
```
