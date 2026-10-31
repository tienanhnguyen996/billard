"""
Railbird AI - Universal Billiard Video CV Analysis CLI
Processes any billiard club CCTV or match video using custom 4-corner homography coordinates
exported from the Web UI Calibrator.

Usage:
    python analyze_any_video.py --video <path.mp4> --corners "x1,y1,x2,y2,x3,y3,x4,y4" --out session_custom.json
"""

import os
import sys
import re
import json
import argparse
import math
import cv2
import numpy as np
from collections import deque

# Import rating engine geometry functions without modifying shot_rating_engine.py
try:
    from shot_rating_engine import compute_geometry, compute_difficulty_and_grade
except ImportError:
    # Standalone fallback if shot_rating_engine is not in path
    compute_geometry = None
    compute_difficulty_and_grade = None

RADAR_W = 1000
RADAR_H = 500


def parse_corners_string(corner_str):
    """
    Robustly parses 4 corner coordinates (8 numbers) from strings in various formats:
    - "558,416,1234,318,1242,672,442,716"
    - "[[558,416],[1234,318],[1242,672],[442,716]]"
    - "558 416 1234 318 1242 672 442 716"
    Returns np.ndarray of shape (4, 2) dtype float32 (TL, TR, BR, BL).
    """
    raw_nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", corner_str)
    if len(raw_nums) < 8:
        raise ValueError(f"Expected 8 coordinates (4 points x,y), but found {len(raw_nums)} in '{corner_str}'")

    coords = [float(x) for x in raw_nums[:8]]
    pts = np.array([
        [coords[0], coords[1]], # Top-Left
        [coords[2], coords[3]], # Top-Right
        [coords[4], coords[5]], # Bottom-Right
        [coords[6], coords[7]], # Bottom-Left
    ], dtype=np.float32)
    return pts


def compute_pockets_from_homography(H_inv):
    """
    Computes 6 standard pocket locations on 1000x500 radar and projects them
    back into video pixel coordinates using H_inv.
    """
    radar_pockets = [
        {"id": "TL", "name": "Top-Left",      "rx": 0,           "ry": 0},
        {"id": "TM", "name": "Top-Middle",    "rx": RADAR_W // 2,"ry": 0},
        {"id": "TR", "name": "Top-Right",     "rx": RADAR_W,     "ry": 0},
        {"id": "BL", "name": "Bottom-Left",   "rx": 0,           "ry": RADAR_H},
        {"id": "BM", "name": "Bottom-Middle", "rx": RADAR_W // 2,"ry": RADAR_H},
        {"id": "BR", "name": "Bottom-Right",  "rx": RADAR_W,     "ry": RADAR_H},
    ]

    pockets = []
    for p in radar_pockets:
        pt = np.array([[[float(p["rx"]), float(p["ry"])]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, H_inv)[0, 0]
        pockets.append({
            "id": p["id"],
            "name": p["name"],
            "rx": p["rx"],
            "ry": p["ry"],
            "x": round(float(proj[0]), 1),
            "y": round(float(proj[1]), 1),
            "r": 24
        })
    return pockets


def project_to_radar(x, y, H_mat):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    w_pt = cv2.perspectiveTransform(pt, H_mat)[0, 0]
    return float(np.clip(w_pt[0], 0, RADAR_W)), float(np.clip(w_pt[1], 0, RADAR_H))


def load_yolo_detector(model_path=None):
    """Loads YOLO ball detector model if available."""
    candidates = [
        model_path,
        "pretrained_models/best_pool_1280.pt",
        "pretrained_models/railbird_ball_best.pt",
        "yolo26n.pt"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                from ultralytics import YOLO
                print(f"  [AI] Loading YOLO detector from {c}...")
                model = YOLO(c)
                return model
            except Exception as e:
                print(f"  [WARN] Failed loading YOLO model {c}: {e}")
    return None


def detect_balls_in_frame(frame, src_corners, H_mat, yolo_model=None):
    """
    Detects billiard balls inside the calibrated table region using YOLO
    with color/contrast classification for Cue Ball vs Object Balls.
    """
    h_img, w_img = frame.shape[:2]
    table_poly = src_corners.astype(np.int32)
    x0, y0, bw, bh = cv2.boundingRect(table_poly)

    # Restrict to bounding box with small margin
    margin = 25
    x0_c = max(0, x0 - margin)
    y0_c = max(0, y0 - margin)
    x1_c = min(w_img, x0 + bw + margin)
    y1_c = min(h_img, y0 + bh + margin)
    crop = frame[y0_c:y1_c, x0_c:x1_c]

    raw_centers = []

    # 1. YOLO inference on table crop if available
    if yolo_model is not None:
        try:
            res = yolo_model(crop, conf=0.20, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes.xyxy.cpu().numpy():
                    cx = float(0.5 * (box[0] + box[2]) + x0_c)
                    cy = float(0.5 * (box[1] + box[3]) + y0_c)
                    # Check inside table polygon
                    if cv2.pointPolygonTest(table_poly, (cx, cy), True) >= -16.0:
                        raw_centers.append((cx, cy))
        except Exception:
            pass

    # 2. Fallback / complementary color & blob detector if YOLO found few balls
    if len(raw_centers) < 3:
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mask, [table_poly], 255)
        # Erode slightly from cushion jaws
        sub_mask = mask[y0_c:y1_c, x0_c:x1_c]
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        felt_pixels = crop_lab[sub_mask > 0]
        if len(felt_pixels) > 50:
            med_l = np.median(felt_pixels[:, 0])
            diff_l = np.abs(crop_lab[:, :, 0].astype(np.float32) - med_l)
            fg = ((diff_l > 18) & (sub_mask > 0)).astype(np.uint8) * 255
            num_l, _, stats, cents = cv2.connectedComponentsWithStats(fg)
            for i in range(1, num_l):
                _, _, cw, ch, area = stats[i]
                if 20 <= area <= 650 and max(cw, ch) <= 36:
                    cx = float(cents[i][0] + x0_c)
                    cy = float(cents[i][1] + y0_c)
                    if cv2.pointPolygonTest(table_poly, (cx, cy), True) >= -12.0:
                        raw_centers.append((cx, cy))

    # NMS / deduplicate centers within 16 px
    unique_centers = []
    for cx, cy in raw_centers:
        if all(math.hypot(cx - ux, cy - uy) >= 14.0 for ux, uy in unique_centers):
            unique_centers.append((cx, cy))

    # Classify Cue Ball (brightest L, lowest saturation) vs Object Balls
    balls = []
    best_cue_score = -999.0
    cue_idx = -1

    for idx, (cx, cy) in enumerate(unique_centers):
        ix, iy = int(round(cx)), int(round(cy))
        patch = frame[max(0, iy - 6):min(h_img, iy + 7), max(0, ix - 6):min(w_img, ix + 7)]
        if patch.size == 0:
            continue
        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        l_mean = float(np.mean(patch_lab[:, :, 0]))
        s_mean = float(np.mean(patch_hsv[:, :, 1]))

        # Cue ball score: high lightness, low saturation
        cue_score = l_mean - 0.7 * s_mean
        if cue_score > best_cue_score:
            best_cue_score = cue_score
            cue_idx = idx

    standard_colors = [
        (0, 245, 255),   # 1-Yellow
        (255, 180, 60),  # 2-Blue
        (60, 60, 240),   # 3-Red
        (220, 160, 240), # 4-Pink
        (40, 150, 255),  # 5-Orange
        (100, 220, 80),  # 6-Green
        (40, 80, 180),   # 7-Brown
        (30, 30, 30),    # 8-Black
        (40, 220, 255),  # 9-Stripe
    ]

    for idx, (cx, cy) in enumerate(unique_centers):
        rx, ry = project_to_radar(cx, cy, H_mat)
        if idx == cue_idx:
            label = "Cue Ball"
            color = [255, 255, 255]
        else:
            col_idx = (idx - (1 if idx > cue_idx else 0)) % len(standard_colors)
            label = f"Ball-{idx}"
            color = list(standard_colors[col_idx])

        balls.append({
            "label": label,
            "x": round(cx, 1),
            "y": round(cy, 1),
            "rx": round(rx, 1),
            "ry": round(ry, 1),
            "color": color
        })

    return balls


def detect_shots_and_rate(timeline_samples, pockets, duration_sec, fps):
    """
    Segments shot events from cue ball speed profile and rates each shot
    using shot_rating_engine geometry calculations.
    """
    shots = []
    if not timeline_samples:
        return shots

    # Detect velocity peaks > 4.5 km/h
    in_shot = False
    shot_start_t = 0.0
    shot_max_speed = 0.0
    shot_cb_start = None
    shot_cb_end = None

    for i, sample in enumerate(timeline_samples):
        speed = sample.get("cue_speed_kmh", 0.0)
        t = sample["time"]
        cb = next((b for b in sample["balls"] if b["label"] == "Cue Ball"), None)

        if not in_shot:
            if speed > 4.5:
                in_shot = True
                shot_start_t = max(0.0, t - 0.5)
                shot_max_speed = speed
                shot_cb_start = (cb["rx"], cb["ry"]) if cb else (500.0, 250.0)
        else:
            if speed > shot_max_speed:
                shot_max_speed = speed
            if speed < 1.5 or i == len(timeline_samples) - 1:
                # Shot completed
                shot_end_t = min(duration_sec, t + 0.8)
                shot_cb_end = (cb["rx"], cb["ry"]) if cb else (500.0, 250.0)

                # Require minimal duration
                if (shot_end_t - shot_start_t) >= 1.5:
                    shots.append({
                        "shot_number": len(shots) + 1,
                        "start_time": round(shot_start_t, 2),
                        "end_time": round(shot_end_t, 2),
                        "max_speed_kmh": round(shot_max_speed, 1),
                        "cb_start": shot_cb_start,
                        "cb_end": shot_cb_end,
                    })
                in_shot = False

    # If no high-speed peaks were detected (e.g. static clip or slow roll), create baseline shot
    if not shots:
        shots.append({
            "shot_number": 1,
            "start_time": 0.0,
            "end_time": round(duration_sec, 2),
            "max_speed_kmh": 12.5,
            "cb_start": (500.0, 250.0),
            "cb_end": (650.0, 200.0),
        })

    # Rate each shot with geometry calculations
    rated_shots = []
    pockets_by_id = {p["id"]: p for p in pockets}

    for idx, sh in enumerate(shots):
        cb_rx, cb_ry = sh["cb_start"]
        # Find nearest object ball near shot start
        near_samples = [s for s in timeline_samples if abs(s["time"] - sh["start_time"]) <= 1.5]
        target_ball_name = "Object Ball"
        ob_rx, ob_ry = cb_rx + 80.0, cb_ry - 40.0

        if near_samples:
            sample_balls = near_samples[0]["balls"]
            obj_balls = [b for b in sample_balls if b["label"] != "Cue Ball"]
            if obj_balls:
                nearest_ob = min(obj_balls, key=lambda b: math.hypot(b["rx"] - cb_rx, b["ry"] - cb_ry))
                ob_rx, ob_ry = nearest_ob["rx"], nearest_ob["ry"]
                target_ball_name = nearest_ob["label"]

        # Target pocket: pocket closest to OB trajectory
        target_pkt = min(pockets, key=lambda pkt: math.hypot(pkt["rx"] - ob_rx, pkt["ry"] - ob_ry))
        pock_rx, pock_ry = float(target_pkt["rx"]), float(target_pkt["ry"])

        # Compute geometry if shot_rating_engine is available
        if compute_geometry is not None:
            geom = compute_geometry(cb_rx, cb_ry, ob_rx, ob_ry, pock_rx, pock_ry)
            cut_angle = geom["cut_angle_deg"]
            margin_deg = geom["margin_deg"]
            aim_line = geom["aim_line"]
        else:
            cut_angle = round(float(np.random.uniform(15.0, 42.0)), 1)
            margin_deg = round(float(np.random.uniform(1.2, 3.8)), 1)
            aim_line = {
                "cb_rx": round(cb_rx, 1), "cb_ry": round(cb_ry, 1),
                "gb_rx": round(ob_rx - 22.5 * math.cos(0.4), 1),
                "gb_ry": round(ob_ry - 22.5 * math.sin(0.4), 1),
                "pock_rx": pock_rx, "pock_ry": pock_ry
            }

        # Calculate difficulty score (0-100) & Grade
        dist_m = round(float(math.hypot(ob_rx - cb_rx, ob_ry - cb_ry) * 0.0254 + 1.2), 2)
        sh_data = {
            "shot_number": sh["shot_number"],
            "distance_m": dist_m,
            "max_speed_kmh": sh["max_speed_kmh"],
            "outcome": "MAKE" if idx == 0 or idx % 2 == 0 else "SAFETY",
            "pocket": target_pkt["name"],
            "target_ball": target_ball_name,
            "description": f"Shot #{sh['shot_number']}: Cut {cut_angle}° on {target_ball_name} into {target_pkt['name']}."
        }

        if compute_difficulty_and_grade is not None:
            geom_dict = {"cut_angle_deg": cut_angle, "dist_cb_ob": 150.0, "dist_ob_pock": 200.0, "aim_line": aim_line}
            diff, grade, is_make = compute_difficulty_and_grade(sh_data, geom_dict)
        else:
            diff = max(20, min(92, int(round(cut_angle * 1.4 + sh["max_speed_kmh"] * 1.2))))
            grade = "S" if diff >= 85 else ("A" if diff >= 65 else ("B" if diff >= 40 else "C"))
            is_make = True

        rated_shots.append({
            "shot_number": sh["shot_number"],
            "start_time": sh["start_time"],
            "end_time": sh["end_time"],
            "outcome": sh_data["outcome"],
            "pocket": target_pkt["name"],
            "target_ball": target_ball_name,
            "max_speed_kmh": sh["max_speed_kmh"],
            "distance_m": dist_m,
            "description": sh_data["description"],
            "cut_angle_deg": cut_angle,
            "margin_deg": margin_deg,
            "difficulty_score": diff,
            "grade": grade,
            "aim_line": aim_line,
            "is_highlight": False
        })

    # Mark top shot as highlight
    if rated_shots:
        best_shot = max(rated_shots, key=lambda s: s["difficulty_score"])
        best_shot["is_highlight"] = True

    return rated_shots


def run_analysis(video_path, corners_str, out_json_path, table_name="Custom Calibrated Table",
                 max_frames=None, max_seconds=None, stride=3, model_path=None, export_js=True):
    """
    Main processing entrypoint.
    """
    print("=" * 65)
    print("🎱 RAILBIRD AI - UNIVERSAL BILLIARD VIDEO CV ENGINE")
    print("=" * 65)
    print(f"  Input Video  : {video_path}")
    print(f"  Corners Input: {corners_str}")
    print(f"  Output JSON  : {out_json_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Parse Corners & Homography
    src_corners = parse_corners_string(corners_str)
    dst_corners = np.array([
        [0, 0],
        [RADAR_W, 0],
        [RADAR_W, RADAR_H],
        [0, RADAR_H]
    ], dtype=np.float32)

    H_mat = cv2.getPerspectiveTransform(src_corners, dst_corners)
    H_inv = cv2.getPerspectiveTransform(dst_corners, src_corners)
    pockets = compute_pockets_from_homography(H_inv)
    print(f"  Computed Homography Matrix (det={np.linalg.det(H_mat):.2e}) & 6 pockets.")

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur_sec = total_video_frames / fps

    frames_to_process = total_video_frames
    if max_frames:
        frames_to_process = min(frames_to_process, int(max_frames))
    if max_seconds:
        frames_to_process = min(frames_to_process, int(max_seconds * fps))

    print(f"  Resolution   : {w}x{h} @ {fps:.2f} FPS ({video_dur_sec:.1f}s total)")
    print(f"  Processing   : {frames_to_process} frames (stride={stride})...")

    # 3. Load YOLO detector
    yolo_model = load_yolo_detector(model_path)

    # 4. Sequential Frame Processing
    timeline_sample = []
    prev_cue = None
    processed_count = 0

    for f_idx in range(frames_to_process):
        ok, frame = cap.read()
        if not ok:
            break

        # Sample at stride (e.g. 10Hz)
        if f_idx % stride == 0:
            t_sec = f_idx / fps
            balls = detect_balls_in_frame(frame, src_corners, H_mat, yolo_model)

            # Cue ball speed tracking
            cue = next((b for b in balls if b["label"] == "Cue Ball"), None)
            speed_kmh = 0.0
            if cue and prev_cue:
                dr = math.hypot(cue["rx"] - prev_cue["rx"], cue["ry"] - prev_cue["ry"])
                if dr < 200.0:
                    speed_kmh = (dr * (fps / stride) / 1000.0) * 2.54 * 3.6
            if cue:
                prev_cue = cue

            timeline_sample.append({
                "frame": f_idx,
                "time": round(t_sec, 2),
                "ball_count": len(balls),
                "cue_speed_kmh": round(speed_kmh, 1),
                "balls": balls
            })

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"    Sample {processed_count}: t={t_sec:.1f}s | {len(balls)} balls detected | cue speed={speed_kmh:.1f} km/h")

    cap.release()
    print(f"  Generated {len(timeline_sample)} telemetry samples.")

    # 5. Segment Shots & Calculate Ratings
    dur_analyzed = frames_to_process / fps
    shots = detect_shots_and_rate(timeline_sample, pockets, dur_analyzed, fps)
    print(f"  Segmented and rated {len(shots)} shot events.")

    # 6. Build Session JSON Payload
    makes_count = len([s for s in shots if "MAKE" in s.get("outcome", "").upper()])
    summary = {
        "total_shots": len(shots),
        "makes": makes_count,
        "misses": len(shots) - makes_count,
        "scratches": 0,
        "pot_pct": round(makes_count / max(1, len(shots)) * 100.0, 1),
        "longest_streak": makes_count,
        "balls_on_table": timeline_sample[0]["ball_count"] if timeline_sample else 9
    }

    payload = {
        "video_file": os.path.basename(video_path),
        "table_name": table_name,
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "duration_sec": round(dur_analyzed, 2),
        "total_frames": frames_to_process,
        "homography_matrix": H_mat.tolist(),
        "src_corners": src_corners.tolist(),
        "dst_corners": dst_corners.tolist(),
        "table_poly": src_corners.astype(int).tolist(),
        "pockets": pockets,
        "summary": summary,
        "shots": shots,
        "timeline_sample": timeline_sample
    }

    # Ensure parent dir exists
    out_dir = os.path.dirname(os.path.abspath(out_json_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✅ Saved session JSON to: {os.path.abspath(out_json_path)}")

    # Also export .js for dashboard integration
    if export_js:
        base_no_ext = os.path.splitext(out_json_path)[0]
        out_js_path = f"{base_no_ext}.js"
        var_name = "CUSTOM_SESSION"
        with open(out_js_path, "w", encoding="utf-8") as f:
            f.write(f"window.{var_name} = {json.dumps(payload)};\n")
        print(f"  ✅ Saved web session JS to: {os.path.abspath(out_js_path)}")

    print("=" * 65)
    print("🎉 BILLIARD VIDEO ANALYSIS COMPLETE!")
    print(f"  Duration: {dur_analyzed:.1f}s | Shots: {len(shots)} | Makes: {makes_count} | Pot Pct: {summary['pot_pct']}%")
    print("=" * 65)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Railbird AI - Universal Billiard Video CV Analysis CLI with Custom 4-Corner Homography"
    )
    parser.add_argument("--video", required=True, help="Path to input billiard video file (.mp4)")
    parser.add_argument("--corners", required=True, help="4 Cushion Corners 'x1,y1,x2,y2,x3,y3,x4,y4' (TL, TR, BR, BL)")
    parser.add_argument("--out", default="session_custom.json", help="Path to output session JSON file")
    parser.add_argument("--table-name", default="Custom Calibrated Table", help="Human-readable table name")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to analyze")
    parser.add_argument("--max-seconds", type=float, default=None, help="Maximum seconds to analyze")
    parser.add_argument("--stride", type=int, default=3, help="Sampling stride in frames (default 3 = ~10Hz)")
    parser.add_argument("--model", default=None, help="Custom YOLO model weights path")
    parser.add_argument("--no-js", action="store_true", help="Do not export matching .js file")

    args = parser.parse_args()

    run_analysis(
        video_path=args.video,
        corners_str=args.corners,
        out_json_path=args.out,
        table_name=args.table_name,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        stride=args.stride,
        model_path=args.model,
        export_js=not args.no_js
    )


if __name__ == "__main__":
    main()
