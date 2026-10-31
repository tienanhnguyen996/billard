"""
Railbird AI - Full-Stack Production Backend Server
Features:
- Multithreaded Python HTTP/REST Server (ThreadingHTTPServer)
- HTTP Byte-Range (206 Partial Content) support for smooth HTML5 <video> scrubbing
- REST API Endpoints:
    GET  /api/status        -> GPU device, CUDA status, loaded models, DB records, active jobs
    POST /api/detect_frame  -> Live GPU detection + neural classification + Optimal 3-Ball Runout Route
    POST /api/export_clip   -> On-demand FFmpeg/OpenCV shot slicing (16:9 or 9:16 vertical)
    POST /api/analyze_job   -> Spawns asynchronous video analysis thread
    GET  /api/job_status    -> Polls background analysis job status & progress
    GET  /api/db            -> Returns persistent DB (calibrations, bookmarks, drills)
    POST /api/db            -> Saves calibrations, bookmarks, or drills to railbird_db.json
- Static file serving for web dashboard (index.html, .js, .json, .mp4, .jpg)
"""

import os
import sys
import re
import json
import time
import base64
import math
import mimetypes
import threading
import subprocess
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import cv2
import numpy as np

import torch
import imageio_ffmpeg

# Import detection, classification, and geometry helpers
try:
    from neural_ball_classifier import classify_ball_crops_neural, CLASSES, DISPLAY_NAMES, BALL_COLORS_BGR
except ImportError:
    classify_ball_crops_neural = None
    CLASSES = []
    DISPLAY_NAMES = {}
    BALL_COLORS_BGR = {}

try:
    from test_hs_detector import (
        HS_TABLE_POLY, HS_POCKETS, RADAR_W, RADAR_H, project_hs_radar, detect_hs_balls
    )
except ImportError:
    HS_TABLE_POLY, HS_POCKETS, RADAR_W, RADAR_H = None, [], 1000, 500
    project_hs_radar, detect_hs_balls = None, None

try:
    from shot_rating_engine import compute_geometry, compute_difficulty_and_grade
except ImportError:
    compute_geometry, compute_difficulty_and_grade = None, None

BASE_DIR = os.path.abspath("c:/billard")
DB_PATH = os.path.join(BASE_DIR, "railbird_db.json")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Thread-safe in-memory background job tracking
JOBS_LOCK = threading.Lock()
JOBS = {}
DB_LOCK = threading.Lock()


# Initialize default database if not exists
def load_db():
    with DB_LOCK:
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        # Default starter DB
        default_db = {
            "version": "1.0.0",
            "calibrations": [
                {
                    "id": "matchroom_tv",
                    "table_name": "European Open 2023 - Matchroom Diamond 9ft",
                    "resolution": "1280x720",
                    "corners": [
                        [438, 212],
                        [838, 212],
                        [908, 596],
                        [368, 596]
                    ],
                    "created_at": "2026-10-31T00:00:00Z"
                },
                {
                    "id": "ban_11_12",
                    "table_name": "Billiard Club Ban 11-12 (Aplus 9ft)",
                    "resolution": "1920x1080",
                    "corners": [
                        [558, 416],
                        [1234, 318],
                        [1242, 672],
                        [442, 716]
                    ],
                    "created_at": "2026-10-31T00:00:00Z"
                }
            ],
            "bookmarks": [
                {
                    "id": "bm_1",
                    "session_id": "hoang_sao",
                    "shot_number": 11,
                    "title": "Hoang Sao 6-Green Power Draw",
                    "notes": "Full-table backspin draw pulling cue ball from top-right to bottom rail.",
                    "created_at": "2026-10-31T00:00:00Z"
                }
            ],
            "drills": [
                {
                    "id": "drill_1",
                    "title": "Extreme Backspin Draw Drill",
                    "target_ball": "6-Green",
                    "cue_position": [366.5, 333.1],
                    "object_position": [281.4, 267.4],
                    "target_pocket": "TR",
                    "required_grade": "S"
                },
                {
                    "id": "drill_2",
                    "title": "Corner Pocket Thin Cut Drill",
                    "target_ball": "1-Yellow",
                    "cue_position": [631.0, 450.0],
                    "object_position": [442.0, 716.0],
                    "target_pocket": "BL",
                    "required_grade": "A"
                }
            ]
        }
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(default_db, f, indent=2)
        return default_db


def save_db(data):
    with DB_LOCK:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# Optimal 3-Ball Runout Route Planner
BALL_ORDER = [
    "1-Yellow", "1_Yellow",
    "2-Blue", "2_Blue",
    "3-Red", "3_Red",
    "4-Pink", "4_Pink",
    "5-Orange", "5_Orange",
    "5-Purple", "5_Purple",
    "6-Green", "6_Green",
    "7-Brown", "7_Brown",
    "8-Ball", "8_Ball",
    "9-Stripe", "9_Stripe",
    "10-BlueStripe", "10_BlueStripe"
]

def get_ball_rank(label):
    lbl = label.strip()
    for i, name in enumerate(BALL_ORDER):
        if name.lower() in lbl.lower():
            return i
    # Extract leading number if present
    nums = re.findall(r"\d+", lbl)
    if nums:
        return int(nums[0])
    return 99


def compute_optimal_3ball_route(cue_ball, object_balls, pockets):
    """
    Computes optimal pocket, ghost ball, and ideal leave zones for the lowest 3 balls on the table.
    """
    if not cue_ball or not object_balls:
        return []

    # Sort object balls by ball order
    sorted_obs = sorted(object_balls, key=lambda b: get_ball_rank(b.get("label", "")))
    target_obs = sorted_obs[:3]

    route_steps = []
    current_cb_rx = float(cue_ball.get("rx", 500.0))
    current_cb_ry = float(cue_ball.get("ry", 250.0))

    for idx, ob in enumerate(target_obs):
        ob_rx = float(ob.get("rx", 500.0))
        ob_ry = float(ob.get("ry", 250.0))
        ob_label = ob.get("label", f"Ball-{idx+1}")

        # Find pocket offering most favorable cut angle (< 65 deg) and shortest distance
        best_pocket = None
        best_gb = None
        best_cut = 999.0
        best_margin = 0.0

        for pkt in pockets:
            p_rx = float(pkt.get("rx", 0.0))
            p_ry = float(pkt.get("ry", 0.0))

            dx_tgt = p_rx - ob_rx
            dy_tgt = p_ry - ob_ry
            dist_tgt = math.hypot(dx_tgt, dy_tgt)
            if dist_tgt < 1.0:
                continue
            ux_tgt, uy_tgt = dx_tgt / dist_tgt, dy_tgt / dist_tgt

            # Ghost Ball: OB - 22.5 * u_tgt
            gb_rx = ob_rx - 22.5 * ux_tgt
            gb_ry = ob_ry - 22.5 * uy_tgt

            # Cue Ball approach vector
            dx_app = gb_rx - current_cb_rx
            dy_app = gb_ry - current_cb_ry
            dist_app = math.hypot(dx_app, dy_app)
            if dist_app < 1.0:
                dist_app = 1.0
            ux_app, uy_app = dx_app / dist_app, dy_app / dist_app

            dot = max(-1.0, min(1.0, ux_app * ux_tgt + uy_app * uy_tgt))
            cut = math.degrees(math.acos(dot))
            if cut > 90.0:
                cut = 180.0 - cut

            # Check cut feasibility
            if cut < best_cut:
                best_cut = cut
                best_pocket = pkt
                best_gb = (gb_rx, gb_ry)
                cos_cut = max(0.1, math.cos(math.radians(cut)))
                best_margin = round(math.degrees(math.atan2(11.25, max(20.0, dist_tgt))) * cos_cut, 1)

        if not best_pocket:
            best_pocket = pockets[0]
            best_gb = (ob_rx - 22.5, ob_ry)
            best_cut = 30.0
            best_margin = 2.0

        # Calculate ideal Cue Ball leave zone for the next ball
        next_ob = target_obs[idx + 1] if idx + 1 < len(target_obs) else None
        if next_ob:
            next_rx = float(next_ob.get("rx", 500.0))
            next_ry = float(next_ob.get("ry", 250.0))
            # Center leave zone along midpoint towards next ball
            leave_rx = round(0.55 * best_gb[0] + 0.45 * next_rx, 1)
            leave_ry = round(0.55 * best_gb[1] + 0.45 * next_ry, 1)
            leave_desc = f"Natural roll zone towards {next_ob.get('label', 'next ball')}"
        else:
            leave_rx = round(best_gb[0] + 60.0 * (1 if best_gb[0] < 500 else -1), 1)
            leave_ry = round(best_gb[1] + 40.0 * (1 if best_gb[1] < 250 else -1), 1)
            leave_desc = "Center table resting position"

        difficulty = max(15, min(95, int(round(best_cut * 1.3 + math.hypot(ob_rx - current_cb_rx, ob_ry - current_cb_ry) * 0.05))))

        route_steps.append({
            "step": idx + 1,
            "ball": ob_label,
            "ball_pos": {"rx": round(ob_rx, 1), "ry": round(ob_ry, 1)},
            "pocket": best_pocket.get("name", best_pocket.get("id", "Pocket")),
            "pocket_id": best_pocket.get("id", "P"),
            "pocket_pos": {"rx": round(best_pocket.get("rx", 0.0), 1), "ry": round(best_pocket.get("ry", 0.0), 1)},
            "ghost_ball": {"rx": round(best_gb[0], 1), "ry": round(best_gb[1], 1)},
            "cut_angle_deg": round(best_cut, 1),
            "pocket_margin_deg": best_margin,
            "difficulty": difficulty,
            "leave_zone": {
                "rx": leave_rx,
                "ry": leave_ry,
                "radius": 35.0,
                "description": leave_desc
            }
        })

        # Advance simulated cue ball to leave zone
        current_cb_rx = leave_rx
        current_cb_ry = leave_ry

    return route_steps


class RailbirdProductionHandler(SimpleHTTPRequestHandler):
    """
    HTTP Request Handler with:
    1. HTTP Byte-Range (206 Partial Content) support for smooth <video> scrubbing
    2. JSON REST APIs under /api/*
    3. Static asset delivery
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_OPTIONS(self):
        """CORS preflight handling."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.end_headers()

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # REST API Routes
        if path.startswith("/api/"):
            self.handle_api_get(path, parse_qs(parsed.query))
            return

        # Serve static files with HTTP Range (206 Partial Content) support
        self.handle_static_with_range()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if not path.startswith("/api/"):
            self.send_error(404, "Unknown POST endpoint")
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body = {}

        self.handle_api_post(path, body)

    # -------------------------------------------------------------
    # Byte-Range File Streaming
    # -------------------------------------------------------------
    def handle_static_with_range(self):
        raw_path = urlparse(self.path).path.lstrip("/")
        if not raw_path:
            raw_path = "index.html"

        file_path = os.path.abspath(os.path.join(BASE_DIR, raw_path))

        # Security check: disallow paths outside BASE_DIR
        if not file_path.startswith(BASE_DIR) or not os.path.isfile(file_path):
            self.send_error(404, f"File not found: {raw_path}")
            return

        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            if file_path.endswith(".mp4"):
                mime_type = "video/mp4"
            elif file_path.endswith(".json"):
                mime_type = "application/json"
            elif file_path.endswith(".js"):
                mime_type = "application/javascript"
            else:
                mime_type = "application/octet-stream"

        range_header = self.headers.get("Range")
        if range_header:
            # Parse Range: bytes=start-end
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                if start >= file_size:
                    self.send_response(416, "Requested Range Not Satisfiable")
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                end = min(end, file_size - 1)
                content_len = end - start + 1

                self.send_response(206, "Partial Content")
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_len))
                self.send_header("Accept-Ranges", "bytes")
                self.send_cors_headers()
                self.end_headers()

                with open(file_path, "rb") as f:
                    f.seek(start)
                    bytes_remaining = content_len
                    chunk_size = 64 * 1024
                    while bytes_remaining > 0:
                        chunk = f.read(min(chunk_size, bytes_remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, ConnectionError):
                            return
                        bytes_remaining -= len(chunk)
                return

        # Full file delivery (200 OK)
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_cors_headers()
        self.end_headers()

        with open(file_path, "rb") as f:
            chunk_size = 64 * 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, ConnectionError):
                    return

    # -------------------------------------------------------------
    # API GET Handlers
    # -------------------------------------------------------------
    def handle_api_get(self, path, query_params):
        if path == "/api/status":
            cuda_avail = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

            detector_1280_path = os.path.join(BASE_DIR, "pretrained_models", "best_pool_1280.pt")
            detector_path = detector_1280_path if os.path.exists(detector_1280_path) else os.path.join(BASE_DIR, "pretrained_models", "railbird_ball_best.pt")
            cls_path = os.path.join(BASE_DIR, "pretrained_models", "railbird_ball_cls.pt")

            db = load_db()

            with JOBS_LOCK:
                active_jobs = len([j for j in JOBS.values() if j.get("status") == "running"])

            status_payload = {
                "success": True,
                "server": "Railbird AI Production Engine (Threading)",
                "cuda_available": cuda_avail,
                "device_name": device_name,
                "pytorch_version": torch.__version__,
                "models": {
                    "ball_detector": {
                        "name": os.path.basename(detector_path),
                        "exists": os.path.exists(detector_path),
                        "size_bytes": os.path.getsize(detector_path) if os.path.exists(detector_path) else 0,
                        "is_high_res_1280": os.path.exists(detector_1280_path)
                    },
                    "ball_classifier": {
                        "name": "railbird_ball_cls.pt",
                        "exists": os.path.exists(cls_path),
                        "size_bytes": os.path.getsize(cls_path) if os.path.exists(cls_path) else 0
                    }
                },
                "sessions_available": [
                    f for f in os.listdir(BASE_DIR) if f.endswith("_session.json") or f.endswith("_data.json")
                ],
                "db_summary": {
                    "calibrations": len(db.get("calibrations", [])),
                    "bookmarks": len(db.get("bookmarks", [])),
                    "drills": len(db.get("drills", []))
                },
                "active_jobs_count": active_jobs,
                "timestamp": time.time()
            }
            self.send_json_response(200, status_payload)
            return

        elif path == "/api/job_status":
            job_id = query_params.get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self.send_json_response(404, {"success": False, "error": f"Job '{job_id}' not found"})
                return
            self.send_json_response(200, {"success": True, "job": job})
            return

        elif path == "/api/db":
            db = load_db()
            self.send_json_response(200, {"success": True, "db": db})
            return

        elif path == "/api/download_status":
            try:
                from scripts.monitor_downloads import check_download
                status = check_download()
            except Exception as e:
                status = {"status": "error", "error": str(e)}
            self.send_json_response(200, {"success": True, "download": status})
            return

        self.send_json_response(404, {"success": False, "error": f"Unknown GET endpoint: {path}"})

    # -------------------------------------------------------------
    # API POST Handlers
    # -------------------------------------------------------------
    def handle_api_post(self, path, body):
        if path == "/api/detect_frame":
            t0 = time.time()
            img_b64 = body.get("image_base64", "")
            session_id = body.get("session_id", "hoang_sao")

            if not img_b64:
                self.send_json_response(400, {"success": False, "error": "Missing image_base64"})
                return

            # Strip data URI prefix if present
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                self.send_json_response(400, {"success": False, "error": f"Invalid image base64: {e}"})
                return

            if frame is None:
                self.send_json_response(400, {"success": False, "error": "Could not decode image"})
                return

            # Detect balls on frame
            balls = []
            try:
                if session_id == "hoang_sao" and detect_hs_balls is not None:
                    # Matchroom camera frame size is 1280x720
                    if frame.shape[:2] != (720, 1280):
                        frame_hs = cv2.resize(frame, (1280, 720))
                    else:
                        frame_hs = frame
                    balls = detect_hs_balls(frame_hs, t_sec=438.0)
                else:
                    # Universal detection using YOLO or blob
                    from analyze_any_video import parse_corners_string, detect_balls_in_frame, load_yolo_detector
                    db = load_db()
                    calib = next((c for c in db.get("calibrations", []) if c["id"] == session_id), db["calibrations"][0])
                    src_c = np.array(calib["corners"], dtype=np.float32)
                    dst_c = np.array([[0, 0], [RADAR_W, 0], [RADAR_W, RADAR_H], [0, RADAR_H]], dtype=np.float32)
                    H_mat = cv2.getPerspectiveTransform(src_c, dst_c)
                    model = load_yolo_detector()
                    balls = detect_balls_in_frame(frame, src_c, H_mat, model)
            except Exception as e:
                balls = []

            # Neural refinement with Stage-2 Classifier
            if classify_ball_crops_neural is not None and len(balls) > 0:
                centers = [(b["x"], b["y"]) for b in balls]
                neural_preds = classify_ball_crops_neural(frame, centers)
                for idx, nb in enumerate(neural_preds):
                    if idx < len(balls):
                        # Merge neural confidence and label
                        if nb["confidence"] > 0.40 and "Cue" not in balls[idx]["label"]:
                            balls[idx]["neural_label"] = nb["label"]
                            balls[idx]["neural_conf"] = nb["confidence"]

            # Compute Optimal 3-Ball Runout Route
            cue_ball = next((b for b in balls if "Cue" in b.get("label", "")), balls[0] if balls else None)
            object_balls = [b for b in balls if b != cue_ball]
            pockets = HS_POCKETS if session_id == "hoang_sao" else [
                {"id": "TL", "name": "Top-Left", "rx": 0, "ry": 0},
                {"id": "TM", "name": "Top-Middle", "rx": 500, "ry": 0},
                {"id": "TR", "name": "Top-Right", "rx": 1000, "ry": 0},
                {"id": "BL", "name": "Bottom-Left", "rx": 0, "ry": 500},
                {"id": "BM", "name": "Bottom-Middle", "rx": 500, "ry": 500},
                {"id": "BR", "name": "Bottom-Right", "rx": 1000, "ry": 500},
            ]

            best_route = compute_optimal_3ball_route(cue_ball, object_balls, pockets)
            latency_ms = round((time.time() - t0) * 1000.0, 1)

            res = {
                "success": True,
                "session_id": session_id,
                "ball_count": len(balls),
                "balls": balls,
                "best_route": best_route,
                "latency_ms": latency_ms
            }
            self.send_json_response(200, res)
            return

        elif path == "/api/export_clip":
            session_id = body.get("session_id", "hoang_sao")
            shot_num = int(body.get("shot_number", 11))
            fmt = body.get("format", "16x9")

            # Look up shot in hoang_sao_session.json or session_data.json
            session_file = "hoang_sao_session.json" if "hoang" in session_id else "session_data.json"
            session_path = os.path.join(BASE_DIR, session_file)
            if not os.path.exists(session_path):
                self.send_json_response(404, {"success": False, "error": f"Session file {session_file} not found"})
                return

            with open(session_path, "r") as f:
                s_data = json.load(f)

            shot = next((s for s in s_data.get("shots", []) if s.get("shot_number") == shot_num), None)
            if not shot:
                self.send_json_response(404, {"success": False, "error": f"Shot #{shot_num} not found in {session_id}"})
                return

            t0 = shot["start_time"]
            t1 = shot["end_time"]
            dur = t1 - t0

            out_filename = f"export_shot_{session_id}_{shot_num}_{fmt.replace(':', 'x')}.mp4"
            out_clip_path = os.path.join(CLIPS_DIR, out_filename)

            source_video = os.path.join(BASE_DIR, s_data.get("video_file", "hoang_sao.mp4"))
            if not os.path.exists(source_video):
                source_video = os.path.join(BASE_DIR, "hoang_sao.mp4")

            # Slice and render on demand
            try:
                if fmt == "9x16":
                    # Vertical 720x1280
                    vf_filter = "crop=720:640:280:65,pad=720:1280:0:0:black"
                    subprocess.run([
                        FFMPEG_EXE, "-y", "-ss", str(t0), "-t", str(dur),
                        "-i", source_video,
                        "-vf", vf_filter,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", out_clip_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                else:
                    # 16x9 standard
                    subprocess.run([
                        FFMPEG_EXE, "-y", "-ss", str(t0), "-t", str(dur),
                        "-i", source_video,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", out_clip_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                size_bytes = os.path.getsize(out_clip_path)
                self.send_json_response(200, {
                    "success": True,
                    "session_id": session_id,
                    "shot_number": shot_num,
                    "format": fmt,
                    "duration_sec": round(dur, 2),
                    "file_path": out_clip_path,
                    "download_url": f"/clips/{out_filename}",
                    "size_bytes": size_bytes
                })
            except Exception as e:
                self.send_json_response(500, {"success": False, "error": f"FFmpeg clip export failed: {e}"})
            return

        elif path == "/api/analyze_job":
            video_input = body.get("video_path_or_yt_url", body.get("video_path", "SAMPLE.mp4"))
            corners_str = body.get("corners", "558,416,1234,318,1242,672,442,716")
            max_sec = body.get("end_sec", None)
            out_json = body.get("out", "session_custom.json")

            job_id = f"job_{int(time.time())}"
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "running",
                    "progress": 5.0,
                    "video": video_input,
                    "corners": corners_str,
                    "start_time": time.time(),
                    "result": None,
                    "error": None
                }

            def worker():
                try:
                    from analyze_any_video import run_analysis
                    vid_full = os.path.abspath(os.path.join(BASE_DIR, video_input)) if not os.path.isabs(video_input) else video_input
                    out_full = os.path.abspath(os.path.join(BASE_DIR, out_json)) if not os.path.isabs(out_json) else out_json

                    payload = run_analysis(
                        video_path=vid_full,
                        corners_str=corners_str,
                        out_json_path=out_full,
                        max_seconds=float(max_sec) if max_sec else None,
                        stride=3
                    )
                    with JOBS_LOCK:
                        JOBS[job_id]["status"] = "completed"
                        JOBS[job_id]["progress"] = 100.0
                        JOBS[job_id]["result"] = {
                            "shots_count": len(payload.get("shots", [])),
                            "duration_sec": payload.get("duration_sec"),
                            "pot_pct": payload.get("summary", {}).get("pot_pct")
                        }
                except Exception as ex:
                    with JOBS_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id]["error"] = str(ex)

            th = threading.Thread(target=worker, daemon=True)
            th.start()

            self.send_json_response(200, {
                "success": True,
                "job_id": job_id,
                "status": "running",
                "message": "Analysis started in background thread"
            })
            return

        elif path == "/api/upload_video":
            video_input = body.get("video_path", body.get("filename", "SAMPLE.mp4"))
            b64_data = body.get("video_data_base64", "")
            preset = body.get("preset", "ban_11_12")
            corners_str = body.get("corners", "")
            max_sec = body.get("end_sec", None)
            table_name = body.get("title", f"Uploaded Session ({preset})")

            # Determine corners based on preset if not explicitly specified
            if not corners_str:
                if preset == "matchroom":
                    corners_str = "438,212,838,212,908,596,368,596"
                elif preset == "topcam":
                    corners_str = "0,0,1280,0,1280,720,0,720"
                else:
                    corners_str = "558,416,1234,318,1242,672,442,716"

            job_id = f"job_up_{int(time.time())}"
            target_video_rel = f"uploads/{os.path.basename(video_input)}"
            target_video_abs = os.path.join(BASE_DIR, target_video_rel)

            # If base64 payload provided, write to uploads folder
            if b64_data:
                try:
                    if "," in b64_data:
                        b64_data = b64_data.split(",", 1)[1]
                    file_bytes = base64.b64decode(b64_data)
                    with open(target_video_abs, "wb") as f:
                        f.write(file_bytes)
                except Exception as ex:
                    self.send_json_response(400, {"success": False, "error": f"Failed to save uploaded video: {ex}"})
                    return
            else:
                # Video file already on disk (e.g. SAMPLE.mp4, hoang_sao.mp4)
                if os.path.exists(os.path.join(BASE_DIR, video_input)):
                    target_video_abs = os.path.join(BASE_DIR, video_input)
                    target_video_rel = video_input
                elif os.path.exists(target_video_abs):
                    pass
                else:
                    target_video_abs = os.path.join(BASE_DIR, "SAMPLE.mp4")
                    target_video_rel = "SAMPLE.mp4"

            out_json_rel = f"uploads/session_{job_id}.json"
            out_json_abs = os.path.join(BASE_DIR, out_json_rel)

            with JOBS_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "processing",
                    "progress": 15.0,
                    "stage": "Video ingested. Initializing CV analysis...",
                    "video": target_video_rel,
                    "start_time": time.time(),
                    "result": None,
                    "error": None
                }

            def upload_worker():
                try:
                    with JOBS_LOCK:
                        JOBS[job_id]["progress"] = 30.0
                        JOBS[job_id]["stage"] = "Computing 4-corner homography & table radar..."

                    from analyze_any_video import run_analysis
                    payload = run_analysis(
                        video_path=target_video_abs,
                        corners_str=corners_str,
                        out_json_path=out_json_abs,
                        max_seconds=float(max_sec) if max_sec else None,
                        stride=3,
                        export_js=True,
                        table_name=table_name
                    )

                    with JOBS_LOCK:
                        JOBS[job_id]["status"] = "completed"
                        JOBS[job_id]["progress"] = 100.0
                        JOBS[job_id]["stage"] = "Analysis complete!"
                        JOBS[job_id]["result"] = {
                            "shots_count": len(payload.get("shots", [])),
                            "duration_sec": payload.get("duration_sec"),
                            "pot_pct": payload.get("summary", {}).get("pot_pct", 0),
                            "session_path": out_json_rel,
                            "video_path": target_video_rel,
                            "payload": payload
                        }
                except Exception as ex:
                    with JOBS_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id]["error"] = str(ex)

            th = threading.Thread(target=upload_worker, daemon=True)
            th.start()

            self.send_json_response(200, {
                "success": True,
                "job_id": job_id,
                "status": "processing",
                "video_url": f"/{target_video_rel}",
                "session_url": f"/{out_json_rel}"
            })
            return

        elif path == "/api/db":
            action = body.get("action", "")
            data_item = body.get("data", {})
            db = load_db()

            if action == "save_calibration":
                calibs = db.setdefault("calibrations", [])
                calib_id = data_item.get("id", f"calib_{int(time.time())}")
                data_item["id"] = calib_id
                data_item["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                # Update or append
                idx = next((i for i, c in enumerate(calibs) if c.get("id") == calib_id), -1)
                if idx >= 0:
                    calibs[idx] = data_item
                else:
                    calibs.append(data_item)
                save_db(db)
                self.send_json_response(200, {"success": True, "action": action, "item": data_item})
                return

            elif action == "save_bookmark":
                bms = db.setdefault("bookmarks", [])
                bm_id = data_item.get("id", f"bm_{int(time.time())}")
                data_item["id"] = bm_id
                data_item["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                bms.append(data_item)
                save_db(db)
                self.send_json_response(200, {"success": True, "action": action, "item": data_item})
                return

            elif action == "save_drill":
                drills = db.setdefault("drills", [])
                d_id = data_item.get("id", f"drill_{int(time.time())}")
                data_item["id"] = d_id
                drills.append(data_item)
                save_db(db)
                self.send_json_response(200, {"success": True, "action": action, "item": data_item})
                return

            self.send_json_response(400, {"success": False, "error": f"Unsupported db action: '{action}'"})
            return

        self.send_json_response(404, {"success": False, "error": f"Unknown POST endpoint: {path}"})

    def send_json_response(self, code, data):
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(json_bytes)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json_bytes)


def run_server(port=8080):
    load_db()
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, RailbirdProductionHandler)
    print("=" * 65)
    print("🎱 RAILBIRD AI - PRODUCTION HTTP & CV BACKEND SERVER")
    print("=" * 65)
    print(f"  Server listening on: http://localhost:{port}/")
    print(f"  Root Directory     : {BASE_DIR}")
    print(f"  Byte-Range Support : Enabled (HTTP 206 Partial Content)")
    print(f"  GPU Acceleration   : {'cuda:0 (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("  Endpoints:")
    print("    GET  /api/status")
    print("    POST /api/detect_frame")
    print("    POST /api/export_clip")
    print("    POST /api/analyze_job")
    print("    GET  /api/job_status?id=<job_id>")
    print("    GET  /api/db")
    print("    POST /api/db")
    print("=" * 65)
    while True:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")
            httpd.server_close()
            break
        except Exception as e:
            print(f"[WARN] Server recovered from exception: {e}")
            time.sleep(0.2)


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
    run_server(port_arg)
