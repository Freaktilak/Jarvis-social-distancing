#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   S.H.I.E.L.D. SOCIAL PROXIMITY ANALYSIS SYSTEM            ║
║   Powered by JARVIS — YOLOv8-Pose + Flask Stream            ║
║   Stark Industries — Jetson AI Platform                      ║
╚══════════════════════════════════════════════════════════════╝

Author  : TILAKRAJ
Platform: Jetson Nano / Linux
Run     : python3 jarvis_detector.py --camera 0
UI      : Open ironman_ui.html in Firefox
"""

import cv2
import math
import time
import threading
import datetime
import argparse
import os
import json
import csv
from flask import Flask, Response, jsonify, send_file
from ultralytics import YOLO

# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="JARVIS Social Distancing Detector")
parser.add_argument("--camera",    type=int,   default=0,     help="Camera index")
parser.add_argument("--distance",  type=int,   default=200,   help="Violation threshold in pixels")
parser.add_argument("--port",      type=int,   default=8080,  help="Flask server port")
parser.add_argument("--width",     type=int,   default=960,   help="Capture width")
parser.add_argument("--height",    type=int,   default=540,   help="Capture height")
args = parser.parse_args()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DISTANCE_THRESHOLD = args.distance
ALERT_COOLDOWN     = 0.5   # seconds — shorter so violations register faster

# COCO 17 keypoint indices
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
LEFT_HIP  = 11
RIGHT_HIP = 12

# Skeleton connections (COCO pairs)
SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,5),(0,6)
]

# Iron Man colors (BGR)
C_CYAN  = (255, 210,   0)
C_RED   = ( 30,  30, 220)
C_GOLD  = ( 30, 165, 255)
C_WHITE = (255, 255, 255)
C_GREEN = ( 60, 220,  60)

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
state = {
    "frame_jpeg":  None,
    "persons":     [],
    "violations":  [],
    "anomalies":   [],
    "frame_count": 0,
    "fps":         0.0,
    "threshold":   DISTANCE_THRESHOLD,
}
state_lock = threading.Lock()

# ─── CSV LOGGER ───────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
LOG_FILE = "results/violation_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["timestamp", "frame", "pair", "distance_px", "persons_in_frame"])

def log_violation(frame_num, pair, dist, n_persons):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, frame_num, str(pair), round(dist), n_persons])

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_kp(kps, idx):
    """Return (x, y) if confidence > 0.3, else None."""
    if kps is None or idx >= len(kps):
        return None
    x, y, c = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (int(x), int(y)) if c > 0.3 else None


def hip_center(kps):
    lh = get_kp(kps, LEFT_HIP)
    rh = get_kp(kps, RIGHT_HIP)
    if lh and rh:
        return ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
    return lh or rh


def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def draw_skeleton(frame, kps, color):
    for a, b in SKELETON:
        pa = get_kp(kps, a)
        pb = get_kp(kps, b)
        if pa and pb:
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    for i in range(17):
        pt = get_kp(kps, i)
        if pt:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def draw_bracket_box(frame, x1, y1, x2, y2, color, label=""):
    L, t = 20, 2
    for ox, oy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (ox,oy), (ox+dx*L, oy),    color, t, cv2.LINE_AA)
        cv2.line(frame, (ox,oy), (ox, oy+dy*L),    color, t, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (x1+4, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_danger_line(frame, p1, p2, dist):
    cv2.line(frame, p1, p2, C_RED, 2, cv2.LINE_AA)
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    cv2.circle(frame, mid, 10, C_RED, -1)
    cv2.putText(frame, f"{int(dist)}px", (mid[0]+12, mid[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,230,255), 1, cv2.LINE_AA)


def alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.6):
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0: return
    import numpy as np
    overlay = __import__('numpy').full_like(sub, color)
    cv2.addWeighted(overlay, alpha, sub, 1-alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def draw_hud(frame, W, H, n_persons, n_violations, fps, frame_count, alert):
    import numpy as np
    now = datetime.datetime.now().strftime("%H:%M:%S")

    # Top-left panel
    sub = frame[0:115, 0:370]
    panel = np.zeros_like(sub); panel[:] = (10,8,2)
    cv2.addWeighted(panel, 0.75, sub, 0.25, 0, sub)
    frame[0:115, 0:370] = sub

    cv2.putText(frame, "S.H.I.E.L.D.  PROXIMITY  ANALYSIS",
                (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30,165,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERSONNEL DETECTED : {n_persons}",
                (8,46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"VIOLATIONS         : {n_violations}",
                (8,70), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                C_RED if n_violations > 0 else C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}    FRAME: {frame_count:06d}",
                (8,94), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1, cv2.LINE_AA)

    # Top-right panel
    xr = W-295
    sub2 = frame[0:82, xr:W]
    panel2 = np.zeros_like(sub2); panel2[:] = (10,8,2)
    cv2.addWeighted(panel2, 0.75, sub2, 0.25, 0, sub2)
    frame[0:82, xr:W] = sub2

    cv2.putText(frame, "JARVIS THREAT ENGINE",
                (xr+8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30,165,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TIME   : {now}",
                (xr+8,48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_CYAN, 1, cv2.LINE_AA)
    status = "BREACH DETECTED" if n_violations > 0 else "ALL CLEAR"
    cv2.putText(frame, f"STATUS : {status}",
                (xr+8,72), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                C_RED if n_violations > 0 else C_GREEN, 1, cv2.LINE_AA)

    # Bottom bar
    sub3 = frame[H-34:H, 0:W]
    panel3 = np.zeros_like(sub3); panel3[:] = (10,8,2)
    cv2.addWeighted(panel3, 0.75, sub3, 0.25, 0, sub3)
    frame[H-34:H, 0:W] = sub3
    cv2.putText(frame,
                f"THRESHOLD: {DISTANCE_THRESHOLD}px  |  YOLOv8-Pose  |  Stark Industries AI",
                (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1, cv2.LINE_AA)

    # Alert banner
    if alert and n_violations > 0:
        bx = W//2 - 220
        sub4 = frame[8:52, bx:bx+440]
        panel4 = np.zeros_like(sub4); panel4[:] = (0,0,140)
        cv2.addWeighted(panel4, 0.7, sub4, 0.3, 0, sub4)
        frame[8:52, bx:bx+440] = sub4
        cv2.putText(frame, "!  SOCIAL DISTANCE VIOLATION  !",
                    (bx+20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (0,230,255), 2, cv2.LINE_AA)

# ─── PROCESSING THREAD ────────────────────────────────────────────────────────

def processing_loop():
    global DISTANCE_THRESHOLD

    print("[JARVIS] Loading YOLOv8n-pose model...")
    model = YOLO("yolov8n-pose.pt")
    print("[JARVIS] Model ready.\n")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[JARVIS] Camera online — {W}x{H}")
    print(f"[JARVIS] Violation threshold: {DISTANCE_THRESHOLD}px")
    print(f"[JARVIS] Stream at http://localhost:{args.port}/stream")
    print(f"[JARVIS] Status at http://localhost:{args.port}/status\n")

    frame_count  = 0
    last_alert_t = 0.0
    alert_active = False
    t_start      = time.time()
    fps          = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        DISTANCE_THRESHOLD = state["threshold"]

        # ── YOLOv8 inference ─────────────────────────────────
        results = model(frame, verbose=False)

        all_kps   = []
        all_boxes = []

        for r in results:
            if r.keypoints is None:
                continue
            kps_data = r.keypoints.data.cpu().numpy()
            boxes    = r.boxes.xyxy.cpu().numpy() if r.boxes else []
            for i in range(len(kps_data)):
                all_kps.append(kps_data[i])
                all_boxes.append(boxes[i] if i < len(boxes) else None)

        # ── Hip centers ───────────────────────────────────────
        centers = [hip_center(kps) for kps in all_kps]

        # ── Violation detection ───────────────────────────────
        violators    = set()
        violation_list = []

        for i in range(len(all_kps)):
            for j in range(i+1, len(all_kps)):
                ci, cj = centers[i], centers[j]
                if ci and cj:
                    d = euclidean(ci, cj)
                    if d < DISTANCE_THRESHOLD:
                        violators.add(i)
                        violators.add(j)
                        violation_list.append([i, j, round(d, 1)])
                        draw_danger_line(frame, ci, cj, d)
                        log_violation(frame_count, (i,j), d, len(all_kps))

        n_violations = len(violators) // 2

        # ── Alert timing ──────────────────────────────────────
        now_t = time.time()
        if n_violations > 0 and (now_t - last_alert_t) > ALERT_COOLDOWN:
            last_alert_t = now_t
            alert_active = True
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[JARVIS ⚠] VIOLATION at {ts} — "
                  f"{n_violations} pair(s) within {DISTANCE_THRESHOLD}px")
        elif n_violations == 0:
            alert_active = False

        # ── Draw persons ──────────────────────────────────────
        for idx, kps in enumerate(all_kps):
            is_v  = idx in violators
            color = C_RED if is_v else C_CYAN
            draw_skeleton(frame, kps, color)

            box = all_boxes[idx]
            if box is not None:
                x1,y1,x2,y2 = map(int, box)
                label = f"ID:{idx} {'BREACH' if is_v else 'SAFE'}"
                draw_bracket_box(frame, x1,y1,x2,y2, color, label)

            c = centers[idx]
            if c:
                cv2.drawMarker(frame, c, C_GOLD,
                               cv2.MARKER_CROSS, 16, 2)

        # ── FPS ───────────────────────────────────────────────
        elapsed = time.time() - t_start
        if elapsed > 0:
            fps = frame_count / elapsed

        # ── HUD ───────────────────────────────────────────────
        draw_hud(frame, W, H,
                 len(all_kps), n_violations,
                 fps, frame_count, alert_active)

        # ── Encode JPEG ───────────────────────────────────────
        _, buffer = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # ── Build persons payload for UI ──────────────────────
        persons_payload = []
        for idx, kps in enumerate(all_kps):
            c = centers[idx]
            kp_list = []
            for i in range(17):
                pt = get_kp(kps, i)
                conf = float(kps[i][2]) if i < len(kps) else 0.0
                kp_list.append({
                    "name":       KP_NAMES[i],
                    "x":          pt[0] if pt else 0,
                    "y":          pt[1] if pt else 0,
                    "confidence": round(conf, 2)
                })
            persons_payload.append({
                "id":        idx,
                "hip_center": list(c) if c else None,
                "keypoints":  kp_list
            })

        # ── Update shared state ───────────────────────────────
        with state_lock:
            state["frame_jpeg"]  = buffer.tobytes()
            state["persons"]     = persons_payload
            state["violations"]  = violation_list
            state["frame_count"] = frame_count
            state["fps"]         = round(fps, 1)

    cap.release()

# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

def generate_frames():
    while True:
        with state_lock:
            frame = state["frame_jpeg"]
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)   # ~30fps cap

@app.route('/')
def index():
    return send_file('ironman_ui.html')

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with state_lock:
        return jsonify({
            "persons":     state["persons"],
            "violations":  state["violations"],
            "anomalies":   state["anomalies"],
            "frame_count": state["frame_count"],
            "fps":         state["fps"],
            "threshold":   state["threshold"],
        })

@app.route('/set_threshold/<int:val>')
def set_threshold(val):
    with state_lock:
        state["threshold"] = max(50, min(500, val))
    return jsonify({"threshold": state["threshold"]})

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║   JARVIS PROXIMITY ANALYSIS SYSTEM — ONLINE      ║")
    print(f"║   Camera    : {args.camera:<34}║")
    print(f"║   Threshold : {args.distance}px                             ║")
    print(f"║   Port      : {args.port:<34}║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Start processing in background thread
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()

    # Start Flask
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
