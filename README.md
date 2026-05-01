# 🦾 JARVIS Social Proximity Analysis System
### *Iron Man Themed Social Distancing Detection — Jetson Nano*

---

## Overview
Real-time social distancing detection using **YOLOv8-Pose** on NVIDIA Jetson Nano.
Detects human keypoints, computes hip-center Euclidean distances between all detected persons,
and flags violations with a full Iron Man / JARVIS HUD.

---

## Project Structure
```
jarvis-social-distancing/
├── jarvis_detector.py   ← Main backend (YOLOv8 + Flask stream)
├── ironman_ui.html      ← Iron Man JARVIS browser dashboard
├── results/             ← Auto-created: violation_log.csv + videos
└── README.md
```

---

## Setup
```bash
pip3 install ultralytics flask opencv-python numpy
```

---

## Run
```bash
# Terminal 1 — start detector
python3 jarvis_detector.py --camera 0

# Then open browser
firefox ironman_ui.html
```

### Arguments
| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index |
| `--distance` | `150` | Violation pixel threshold |
| `--port` | `8080` | Flask server port |
| `--width` | `960` | Capture width |
| `--height` | `540` | Capture height |

---

## How It Works
1. **YOLOv8n-Pose** detects all persons and 17 COCO keypoints per person
2. **Hip center** = midpoint of left_hip + right_hip keypoints
3. **Euclidean distance** computed between all person pairs
4. If `distance < threshold` → **violation flagged**
5. Results streamed to browser via **MJPEG** at `localhost:8080/stream`
6. Live stats served via **REST API** at `localhost:8080/status`
7. All violations logged to `results/violation_log.csv`

---

## Social Distancing Logic
```python
# Hip midpoint per person
hip_center = (left_hip + right_hip) / 2

# Euclidean distance between two persons
distance = sqrt((x1-x2)² + (y1-y2)²)

# Flag violation
if distance < DISTANCE_THRESHOLD:
    → RED skeleton + alert banner
```

---

## JARVIS HUD Features
- Arc Reactor animated logo
- Live MJPEG camera feed with scanning line
- Personnel count + violation count
- Per-person TARGET cards (SAFE / BREACH)
- Keypoint confidence bars
- Threat level bar
- Flashing PROXIMITY BREACH alert banner
- Violation log with timestamps
- Adjustable threshold slider
- CSV violation logging

---

*"Sometimes you gotta run before you can walk." — Tony Stark*
