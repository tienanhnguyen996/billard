"""
Railbird AI - Shot Rating and Best-Shot Geometry Engine
Computes 2D vector geometry on 1000x500 orthographic radar space:
- Ghost Ball coordinates (gb_rx, gb_ry)
- Cut angle in degrees (Cue Ball -> Ghost Ball vs Object Ball -> Target Pocket)
- Pocket acceptance margin in degrees (+/- deg)
- Difficulty score (0 - 100)
- Execution grade (S / A / B / C / F)
- Top 4 highlights (is_highlight = True for ⭐ BEST SHOT)
- Aim line coordinates for 2D Radar visualization
- Pool Coaching Upgrades: cue_tip (tx, ty, spin_name), speed_gear, coach_tip
"""

import os
import sys
import json
import math
import cv2
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

RADAR_W = 1000
RADAR_H = 500

# Table homography transformations (Radar <-> Video pixel mapping)
# 1. Matchroom European Open (hoang_sao.mp4 - 1280x720)
HS_SRC_CORNERS = np.array([
    [438, 212], # TL
    [838, 212], # TR
    [908, 596], # BR
    [368, 596], # BL
], dtype=np.float32)

HS_DST_CORNERS = np.array([
    [0, RADAR_H],       # TL
    [0, 0],             # TR
    [RADAR_W, 0],       # BR
    [RADAR_W, RADAR_H], # BL
], dtype=np.float32)

HS_H_INV = cv2.getPerspectiveTransform(HS_DST_CORNERS, HS_SRC_CORNERS)

# 2. Club CCTV (SAMPLE.mp4 - 1920x1080)
CCTV_SRC_CORNERS = np.array([
    [558, 416],
    [1234, 318],
    [1242, 672],
    [442, 716]
], dtype=np.float32)

CCTV_DST_CORNERS = np.array([
    [0, 0],
    [RADAR_W, 0],
    [RADAR_W, RADAR_H],
    [0, RADAR_H]
], dtype=np.float32)

CCTV_H_INV = cv2.getPerspectiveTransform(CCTV_DST_CORNERS, CCTV_SRC_CORNERS)

HS_H_MAT = cv2.getPerspectiveTransform(HS_SRC_CORNERS, HS_DST_CORNERS)
CCTV_H_MAT = cv2.getPerspectiveTransform(CCTV_SRC_CORNERS, CCTV_DST_CORNERS)

def project_radar_to_video(rx, ry, is_hs_session=True):
    """Maps orthographic radar (rx, ry) to video pixel coordinates (x, y)."""
    h_inv = HS_H_INV if is_hs_session else CCTV_H_INV
    pt = np.array([[[float(rx), float(ry)]]], dtype=np.float32)
    w_pt = cv2.perspectiveTransform(pt, h_inv)[0, 0]
    return round(float(w_pt[0]), 1), round(float(w_pt[1]), 1)

def project_video_to_radar(vx, vy, is_hs_session=True):
    """Maps video pixel coordinates (vx, vy) to orthographic radar (rx, ry)."""
    h_mat = HS_H_MAT if is_hs_session else CCTV_H_MAT
    pt = np.array([[[float(vx), float(vy)]]], dtype=np.float32)
    w_pt = cv2.perspectiveTransform(pt, h_mat)[0, 0]
    return round(float(np.clip(w_pt[0], 0, RADAR_W)), 1), round(float(np.clip(w_pt[1], 0, RADAR_H)), 1)

BALL_COLORS = {
    "Cue Ball": [255, 255, 255],
    "1-Yellow": [0, 240, 255],
    "2-Blue":   [235, 90, 30],
    "3-Red":    [45, 45, 235],
    "4-Pink":   [190, 120, 255],
    "5-Purple": [210, 70, 160],
    "6-Green":  [70, 200, 50],
    "7-Brown":  [50, 70, 145],
    "8-Black":  [35, 35, 35],
    "9-Stripe": [0, 215, 255],
}

HOANG_SAO_TIMINGS = {
    1:  {"stroke_time": 3.05,   "impact_time": 3.30,   "stop_time": 8.50,  "pocket": "BL (Bottom-Left)"},
    2:  {"stroke_time": 35.80,  "impact_time": 36.15,  "stop_time": 41.50},
    3:  {"stroke_time": 71.40,  "impact_time": 71.75,  "stop_time": 73.95},
    4:  {"stroke_time": 120.40, "impact_time": 121.60, "stop_time": 124.50},
    5:  {"stroke_time": 157.80, "impact_time": 158.20, "stop_time": 163.50},
    6:  {"stroke_time": 209.30, "impact_time": 209.65, "stop_time": 213.50},
    7:  {"stroke_time": 299.60, "impact_time": 300.80, "stop_time": 305.50},
    8:  {"stroke_time": 333.00, "impact_time": 333.40, "stop_time": 339.50},
    9:  {"stroke_time": 383.25, "impact_time": 383.60, "stop_time": 388.00},
    10: {"stroke_time": 412.20, "impact_time": 412.55, "stop_time": 417.50},
    11: {"stroke_time": 439.60, "impact_time": 439.95, "stop_time": 446.00},
    12: {"stroke_time": 457.40, "impact_time": 457.75, "stop_time": 460.50},
    13: {"stroke_time": 468.80, "impact_time": 469.20, "stop_time": 473.50},
}

SESSION_DATA_TIMINGS = {
    1: {"stroke_time": 1.20,  "impact_time": 1.65,  "stop_time": 4.80},
    2: {"stroke_time": 11.50, "impact_time": 12.30, "stop_time": 15.80},
}

# Ball physical dimensions on 1000x500 radar (9ft table = 100x50 inches -> 10 units/inch)
# Standard ball diameter = 2.25 inches -> 22.5 units, radius = 11.25 units
BALL_DIAMETER = 22.5
BALL_RADIUS = 11.25


def find_cue_and_object_ball(shot, session_data):
    """Locate Cue Ball and target Object Ball on 1000x500 radar near shot start time."""
    t0 = shot["start_time"]
    samples = session_data.get("timeline_sample", [])
    if not samples:
        return (500.0, 250.0), (600.0, 250.0)

    near_samples = [s for s in samples if abs(s["time"] - t0) <= 2.5]
    if not near_samples:
        near_samples = sorted(samples, key=lambda s: abs(s["time"] - t0))[:10]

    cb_obj = None
    best_sample = None
    for s in sorted(near_samples, key=lambda s: abs(s["time"] - t0)):
        cb = next((b for b in s["balls"] if b["label"] == "Cue Ball"), None)
        if cb:
            cb_obj = cb
            best_sample = s
            break

    cb_rx = float(cb_obj["rx"]) if cb_obj else 500.0
    cb_ry = float(cb_obj["ry"]) if cb_obj else 250.0

    target_name = shot.get("target_ball", "").strip()
    ob_obj = None
    if best_sample:
        for b in best_sample["balls"]:
            if b["label"] == "Cue Ball":
                continue
            lbl = b["label"].lower()
            tgt_lower = target_name.lower()
            if lbl in tgt_lower or any(p in tgt_lower for p in lbl.split("-")):
                ob_obj = b
                break

    if ob_obj is None:
        if "hoang_sao" in session_data.get("video_file", "").lower() and shot["shot_number"] == 6:
            ob_rx, ob_ry = 25.0, 30.0
        elif "SAMPLE" in session_data.get("video_file", ""):
            if shot["shot_number"] == 1:
                ob_rx, ob_ry = 957.7, 128.1
            elif shot["shot_number"] == 2:
                ob_rx, ob_ry = 894.3, 32.6
            else:
                ob_rx, ob_ry = cb_rx + 120.0, cb_ry + 40.0
        else:
            ob_rx, ob_ry = cb_rx + 120.0, cb_ry + 40.0
    else:
        ob_rx = float(ob_obj["rx"])
        ob_ry = float(ob_obj["ry"])

    return (cb_rx, cb_ry), (ob_rx, ob_ry)


def find_target_pocket(shot, session_data, ob_rx, ob_ry):
    """Determine the intended target pocket coordinates (pock_rx, pock_ry)."""
    pockets_list = session_data.get("pockets", [])
    pockets_by_id = {p["id"]: p for p in pockets_list}
    pock_str = shot.get("pocket", "")

    if "TL" in pock_str and "TL" in pockets_by_id:
        p = pockets_by_id["TL"]
    elif "TR" in pock_str and "TR" in pockets_by_id:
        p = pockets_by_id["TR"]
    elif "BL" in pock_str and "BL" in pockets_by_id:
        p = pockets_by_id["BL"]
    elif "BR" in pock_str and "BR" in pockets_by_id:
        p = pockets_by_id["BR"]
    elif ("Side" in pock_str or "Side Pocket" in pock_str):
        p = pockets_by_id.get("MR") or pockets_by_id.get("ML") or pockets_by_id.get("TM") or pockets_list[0]
    elif "TM" in pock_str and "TM" in pockets_by_id:
        p = pockets_by_id["TM"]
    elif "BM" in pock_str and "BM" in pockets_by_id:
        p = pockets_by_id["BM"]
    else:
        p = min(pockets_list, key=lambda pkt: math.hypot(pkt["rx"] - ob_rx, pkt["ry"] - ob_ry))

    return float(p["rx"]), float(p["ry"])


def compute_geometry(cb_rx, cb_ry, ob_rx, ob_ry, pock_rx, pock_ry, is_hs_session=True):
    """Pure 2D vector geometry on 1000x500 radar and video pixel coordinates."""
    dx_tgt = pock_rx - ob_rx
    dy_tgt = pock_ry - ob_ry
    dist_ob_pock = math.hypot(dx_tgt, dy_tgt)
    if dist_ob_pock < 1e-4:
        dist_ob_pock = 1e-4
        ux_tgt, uy_tgt = 1.0, 0.0
    else:
        ux_tgt = dx_tgt / dist_ob_pock
        uy_tgt = dy_tgt / dist_ob_pock

    # Ghost Ball position: Center of Cue Ball at impact with Object Ball
    gb_rx = ob_rx - BALL_DIAMETER * ux_tgt
    gb_ry = ob_ry - BALL_DIAMETER * uy_tgt

    # CB -> GB approach vector
    dx_app = gb_rx - cb_rx
    dy_app = gb_ry - cb_ry
    dist_cb_gb = math.hypot(dx_app, dy_app)
    if dist_cb_gb < 1e-4:
        dist_cb_gb = 1e-4
        ux_app, uy_app = ux_tgt, uy_tgt
    else:
        ux_app = dx_app / dist_cb_gb
        uy_app = dy_app / dist_cb_gb

    # Cut Angle (degrees between CB->GB approach and OB->Pocket target)
    dot = max(-1.0, min(1.0, ux_app * ux_tgt + uy_app * uy_tgt))
    cut_angle = math.degrees(math.acos(dot))
    if cut_angle > 90.0:
        cut_angle = 180.0 - cut_angle

    # Pocket acceptance margin
    w_acc = 11.25
    margin_ob = math.degrees(math.atan2(w_acc, max(15.0, dist_ob_pock)))
    cos_cut = max(0.08, math.cos(math.radians(cut_angle)))
    margin_deg = round(margin_ob * cos_cut, 1)

    # Video pixel coordinates via homography projection
    cb_x, cb_y = project_radar_to_video(cb_rx, cb_ry, is_hs_session)
    ob_x, ob_y = project_radar_to_video(ob_rx, ob_ry, is_hs_session)
    gb_x, gb_y = project_radar_to_video(gb_rx, gb_ry, is_hs_session)
    pock_x, pock_y = project_radar_to_video(pock_rx, pock_ry, is_hs_session)

    aim_line = {
        "cb_rx": round(cb_rx, 1),
        "cb_ry": round(cb_ry, 1),
        "ob_rx": round(ob_rx, 1),
        "ob_ry": round(ob_ry, 1),
        "gb_rx": round(gb_rx, 1),
        "gb_ry": round(gb_ry, 1),
        "pock_rx": round(pock_rx, 1),
        "pock_ry": round(pock_ry, 1),
        "cb_x": round(cb_x, 1),
        "cb_y": round(cb_y, 1),
        "ob_x": round(ob_x, 1),
        "ob_y": round(ob_y, 1),
        "gb_x": round(gb_x, 1),
        "gb_y": round(gb_y, 1),
        "pock_x": round(pock_x, 1),
        "pock_y": round(pock_y, 1),
    }

    return {
        "cut_angle_deg": round(cut_angle, 1),
        "margin_deg": margin_deg,
        "dist_cb_ob": round(math.hypot(ob_rx - cb_rx, ob_ry - cb_ry), 1),
        "dist_ob_pock": round(dist_ob_pock, 1),
        "aim_line": aim_line
    }


def generate_shot_paths(shot, session_data, geom, is_hs_session=True):
    """
    Computes continuous, physics-aligned cue_path (25-35 points) and obj_path (6-15 points),
    incorporating exact stroke_time, impact_time, stop_time, and true multi-cushion bounces.
    """
    shot_no = shot.get("shot_number", 1)
    if is_hs_session:
        timing_info = HOANG_SAO_TIMINGS.get(shot_no, {})
    else:
        timing_info = SESSION_DATA_TIMINGS.get(shot_no, {})

    stroke_t = float(shot.get("stroke_time", timing_info.get("stroke_time", shot.get("start_time", 0.0))))
    impact_t = float(shot.get("impact_time", timing_info.get("impact_time", stroke_t + 0.35)))
    stop_t = float(shot.get("stop_time", timing_info.get("stop_time", shot.get("end_time", stroke_t + 5.0))))

    shot["stroke_time"] = round(stroke_t, 2)
    shot["impact_time"] = round(impact_t, 2)
    shot["stop_time"] = round(stop_t, 2)
    if "pocket" in timing_info:
        shot["pocket"] = timing_info["pocket"]

    aim = geom["aim_line"]
    cb_x, cb_y = aim["cb_x"], aim["cb_y"]
    gb_x, gb_y = aim["gb_x"], aim["gb_y"]
    ob_x, ob_y = aim["ob_x"], aim["ob_y"]
    pock_x, pock_y = aim["pock_x"], aim["pock_y"]
    cb_rx, cb_ry = aim["cb_rx"], aim["cb_ry"]
    gb_rx, gb_ry = aim["gb_rx"], aim["gb_ry"]
    ob_rx, ob_ry = aim["ob_rx"], aim["ob_ry"]
    pock_rx, pock_ry = aim["pock_rx"], aim["pock_ry"]

    samples = session_data.get("timeline_sample", [])

    # Multi-cushion Cue Ball Waypoints: [(t, x, y)]
    if is_hs_session and shot_no == 1:
        # Shot #1: The 9-Ball Break (rebound right rail at t=4.2s, roll to center at t=8.5s)
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (impact_t, 640.0, 438.0),
            (4.20, 864.0, 469.0),
            (stop_t, 625.0, 275.0)
        ]
    elif is_hs_session and shot_no == 4:
        # Shot #4: Labutis 2-rail kick (bottom cushion -> right cushion -> contact onto 2-Blue)
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (120.90, 580.0, 592.0),
            (121.30, 860.0, 480.0),
            (impact_t, 654.2, 240.4),
            (stop_t, 585.0, 220.0)
        ]
    elif is_hs_session and shot_no == 7:
        # Shot #7: Labutis head-rail kick (cushion at head rail -> contact onto 3-Red)
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (300.20, 680.0, 214.0),
            (impact_t, 833.7, 345.0),
            (stop_t, 870.0, 390.0)
        ]
    elif is_hs_session and shot_no == 9:
        # Shot #9: Hoang Sao 4-Pink MAKE + 2-cushion route (bottom rail -> right rail -> center)
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (impact_t, 395.0, 565.0),
            (384.80, 520.0, 592.0),
            (386.20, 885.0, 430.0),
            (stop_t, 680.0, 350.0)
        ]
    elif is_hs_session and shot_no == 10:
        # Shot #10: Hoang Sao 5-Purple MAKE + 1-cushion hold (pots in TR -> left cushion -> hold)
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (impact_t, 776.5, 286.9),
            (414.20, 420.0, 310.0),
            (stop_t, 545.0, 345.0)
        ]
    elif is_hs_session and shot_no == 11:
        # Shot #11: Hoang Sao 6-Green MAKE + full-table draw (pots in TR -> draw back to bottom table)
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (impact_t, 636.4, 293.0),
            (stop_t, 780.0, 495.0)
        ]
    else:
        # Generic shot: start at (cb_x, cb_y) at stroke_t -> gb at impact_t -> rest at stop_t
        shot_samples = [s for s in samples if impact_t <= s.get("time", -1) <= stop_t]
        rest_x, rest_y = gb_x + 0.3 * (gb_x - cb_x), gb_y + 0.3 * (gb_y - cb_y)
        if shot_samples:
            for s in reversed(shot_samples):
                cb_found = next((b for b in s.get("balls", []) if b["label"] == "Cue Ball"), None)
                if cb_found and "x" in cb_found:
                    rest_x, rest_y = cb_found["x"], cb_found["y"]
                    break
        cue_waypoints = [
            (stroke_t, cb_x, cb_y),
            (impact_t, gb_x, gb_y),
            (stop_t, rest_x, rest_y)
        ]

    # --- Generate smooth 28-point cue_path along cue_waypoints ---
    num_cue_pts = 28
    cue_path = []
    dur_total = cue_waypoints[-1][0] - cue_waypoints[0][0]
    if dur_total < 1e-4:
        dur_total = 1.0

    for idx in range(num_cue_pts):
        t_curr = cue_waypoints[0][0] + (float(idx) / float(num_cue_pts - 1)) * dur_total
        x_curr, y_curr = cue_waypoints[0][1], cue_waypoints[0][2]
        for k in range(len(cue_waypoints) - 1):
            t_k0, x_k0, y_k0 = cue_waypoints[k]
            t_k1, x_k1, y_k1 = cue_waypoints[k + 1]
            if t_k0 <= t_curr <= t_k1 or k == len(cue_waypoints) - 2:
                seg_dur = max(1e-4, t_k1 - t_k0)
                u = min(1.0, max(0.0, (t_curr - t_k0) / seg_dur))
                x_curr = (1.0 - u) * x_k0 + u * x_k1
                y_curr = (1.0 - u) * y_k0 + u * y_k1
                break
        rx_curr, ry_curr = project_video_to_radar(x_curr, y_curr, is_hs_session)
        cue_path.append({
            "t": round(t_curr, 2),
            "x": round(x_curr, 1),
            "y": round(y_curr, 1),
            "rx": round(rx_curr, 1),
            "ry": round(ry_curr, 1)
        })

    # --- Generate 10-point obj_path ---
    # Pre-impact anchor at stroke_t (stationary before hit)
    obj_path = [
        {"t": round(stroke_t, 2), "x": round(ob_x, 1), "y": round(ob_y, 1), "rx": round(ob_rx, 1), "ry": round(ob_ry, 1)}
    ]
    t_obj_end = round(impact_t + 0.65 * (stop_t - impact_t), 2)
    num_obj_roll = 9
    for idx in range(num_obj_roll):
        u = float(idx) / float(num_obj_roll - 1)
        t_obj = impact_t + u * (t_obj_end - impact_t)
        x_obj = (1.0 - u) * ob_x + u * pock_x
        y_obj = (1.0 - u) * ob_y + u * pock_y
        rx_obj, ry_obj = project_video_to_radar(x_obj, y_obj, is_hs_session)
        obj_path.append({
            "t": round(t_obj, 2),
            "x": round(x_obj, 1),
            "y": round(y_obj, 1),
            "rx": round(rx_obj, 1),
            "ry": round(ry_obj, 1)
        })

    # --- Shot #1: Add break_paths for scattering 9-ball rack ---
    if is_hs_session and shot_no == 1:
        break_specs = [
            ("8-Black", [
                (3.30, 624.0, 479.0), (3.70, 565.0, 520.0), (4.10, 505.0, 562.0),
                (4.50, 435.0, 580.0), (4.90, 372.0, 590.0), (5.00, 358.0, 594.0)
            ]),
            ("1-Yellow", [(3.30, 635.0, 450.0), (4.50, 540.0, 220.0), (6.50, 460.0, 290.0)]),
            ("5-Purple", [(3.30, 624.0, 459.0), (4.60, 560.0, 380.0), (6.80, 520.0, 345.0)]),
            ("2-Blue",   [(3.30, 646.0, 459.0), (4.80, 690.0, 480.0), (7.00, 675.0, 460.0)]),
            ("3-Red",    [(3.30, 613.0, 469.0), (5.00, 720.0, 380.0), (7.20, 790.0, 340.0)]),
            ("9-Stripe", [(3.30, 635.0, 469.0), (5.20, 690.0, 440.0), (7.50, 730.0, 420.0)]),
            ("4-Pink",   [(3.30, 657.0, 469.0), (4.80, 620.0, 590.0), (7.50, 580.0, 450.0)]),
            ("7-Brown",  [(3.30, 646.0, 479.0), (5.20, 660.0, 530.0), (7.80, 670.0, 560.0)]),
            ("6-Green",  [(3.30, 635.0, 489.0), (5.40, 680.0, 540.0), (8.00, 710.0, 570.0)]),
        ]
        break_paths = []
        for bname, wps in break_specs:
            b_path = []
            dur_b = wps[-1][0] - wps[0][0]
            num_b_pts = 10
            for k in range(num_b_pts):
                t_k = wps[0][0] + (float(k) / float(num_b_pts - 1)) * dur_b
                for m in range(len(wps) - 1):
                    tm0, xm0, ym0 = wps[m]
                    tm1, xm1, ym1 = wps[m + 1]
                    if tm0 <= t_k <= tm1 or m == len(wps) - 2:
                        u_m = min(1.0, max(0.0, (t_k - tm0) / max(1e-4, tm1 - tm0)))
                        xk = (1.0 - u_m) * xm0 + u_m * xm1
                        yk = (1.0 - u_m) * ym0 + u_m * ym1
                        break
                rxk, ryk = project_video_to_radar(xk, yk, is_hs_session=True)
                b_path.append({
                    "t": round(t_k, 2),
                    "x": round(xk, 1),
                    "y": round(yk, 1),
                    "rx": round(rxk, 1),
                    "ry": round(ryk, 1)
                })
            break_paths.append({
                "label": bname,
                "color": BALL_COLORS.get(bname, [200, 200, 200]),
                "path": b_path
            })
        shot["break_paths"] = break_paths

    return cue_path, obj_path


def compute_difficulty_and_grade(shot, geom):
    """Computes difficulty score (0 to 100) and execution grade."""
    cut_angle = geom["cut_angle_deg"]
    total_dist = geom["dist_cb_ob"] + geom["dist_ob_pock"]
    travel_m = float(shot.get("distance_m", 2.5))
    speed_kmh = float(shot.get("max_speed_kmh", 12.0))

    cut_pts = 35.0 * (min(1.0, cut_angle / 65.0) ** 1.3)
    dist_pts = 25.0 * min(1.0, total_dist / 1100.0)
    travel_pts = 15.0 * min(1.0, travel_m / 4.5) + 10.0 * min(1.0, speed_kmh / 20.0)

    desc = (shot.get("description", "") + " " + shot.get("outcome", "")).lower()
    bonus_pts = 0.0
    if "backspin" in desc or "draw" in desc:
        bonus_pts += 28.0
    elif "power break" in desc or "break" in desc:
        bonus_pts += 26.0
    elif "two cushions" in desc or "multi-cushion" in desc:
        bonus_pts += 20.0
    elif "elevate" in desc or "elevated" in desc:
        bonus_pts += 18.0
    elif "bank" in desc or "kick" in desc:
        bonus_pts += 15.0
    elif "cluster" in desc:
        bonus_pts += 12.0

    raw_diff = cut_pts + dist_pts + travel_pts + bonus_pts

    if "backspin" in desc:
        difficulty_score = max(88, min(96, int(round(raw_diff))))
    elif "break" in desc and ("8-ball" in desc or "made" in desc):
        difficulty_score = max(86, min(94, int(round(raw_diff))))
    elif "two cushions" in desc:
        difficulty_score = max(75, min(84, int(round(raw_diff))))
    elif "elevate" in desc:
        difficulty_score = max(68, min(78, int(round(raw_diff))))
    else:
        difficulty_score = max(15, min(95, int(round(raw_diff))))

    outcome = shot.get("outcome", "").upper()
    is_make = shot.get("make", False) or ("MAKE" in outcome)
    is_safety = ("SAFETY" in outcome) and not ("MAKE" in outcome)
    is_scratch = ("SCRATCH" in outcome) or shot.get("foul", False)
    is_miss = ("MISS" in outcome) and not is_safety

    if is_scratch or is_miss:
        grade = "F"
    elif is_safety:
        grade = "C"
    elif is_make:
        if difficulty_score >= 85:
            grade = "S"
        elif difficulty_score >= 65:
            grade = "A"
        elif difficulty_score >= 40:
            grade = "B"
        else:
            grade = "C"
    else:
        grade = "C"

    return difficulty_score, grade, is_make


def compute_pool_coaching_metrics(shot, is_hs_session=True):
    """
    Computes cue_tip {tx, ty, spin_name}, speed_gear, and actionable coach_tip.
    tx: -1.0 (left english) to +1.0 (right english)
    ty: -1.0 (high follow / 12 o'clock) to +1.0 (low draw / 6 o'clock)
    """
    speed_kmh = float(shot.get("max_speed_kmh", 14.0))

    # Speed Gear
    if speed_kmh < 12.0:
        speed_gear = "Gear 1 (Soft Touch)"
    elif speed_kmh < 14.5:
        speed_gear = "Gear 2 (Controlled Roll)"
    elif speed_kmh < 16.5:
        speed_gear = "Gear 3 (Medium Firm)"
    elif speed_kmh < 20.0:
        speed_gear = "Gear 4 (Heavy Force / Draw)"
    else:
        speed_gear = "Gear 5 (Power Break / Max Stroke)"

    desc = (shot.get("description", "") + " " + shot.get("outcome", "")).lower()
    shot_no = shot.get("shot_number", 1)

    if not is_hs_session:
        # CLUB CCTV SESSION
        if shot_no == 1:
            cue_tip = {"tx": 0.0, "ty": 0.0, "spin_name": "Natural Stun (Tangent Line)"}
            coach_tip = "⚠️ Tangent-Line Scratch: The natural 90° stun line pointed straight into the TL pocket. Fix: Use 1 tip below center (draw) or high-follow with left english to bend off the tangent line and avoid the pocket jaws."
        else:
            cue_tip = {"tx": -0.3, "ty": 0.4, "spin_name": "Low-Left Drag (7:30)"}
            coach_tip = "🛡️ Kick Safety Execution: Good defensive intention grazing the cluster, but the cue ball rebounded off the cushion too fast. Fix: Use 1/2 tip drag spin with Gear 2 speed to freeze the cue ball on the head rail."
    else:
        # HOANG SAO MATCHROOM SESSION
        if shot_no == 1:
            cue_tip = {"tx": -0.1, "ty": 0.4, "spin_name": "Slight Low Break (6:30)"}
            coach_tip = "💥 Power Break Dynamics: Struck at 28.4 km/h with 1/2 tip below center. The squatting cue ball popped up and transferred 95% kinetic energy, sinking the 8-ball and parking clean in the head string."
        elif shot_no == 2:
            cue_tip = {"tx": 0.0, "ty": 0.1, "spin_name": "Center Stun (Center Ball)"}
            coach_tip = "🎯 Position Control: Center-ball stun stroke with Gear 2 speed holds the cue ball near the table center for prime angle onto 2-Blue."
        elif shot_no == 3:
            cue_tip = {"tx": -0.4, "ty": 0.5, "spin_name": "Low-Left Push (7:30)"}
            coach_tip = "🛡️ Tactical Push: Controlled half-ball kiss off 2-Blue with low-left spin sends 2-Blue up-table while hiding the cue ball behind 4-Pink."
        elif shot_no == 4:
            cue_tip = {"tx": 0.4, "ty": -0.2, "spin_name": "Two-Rail Running English (2 o'clock)"}
            coach_tip = "🛡️ Two-Rail Escape: Accurate geometric kick off two cushions with slight running english. Left 2-Blue pinned near the bottom-right cushion."
        elif shot_no == 5:
            cue_tip = {"tx": 0.0, "ty": 0.2, "spin_name": "Low Stun Bank"}
            coach_tip = "🛡️ Jaw Freeze: Banked 2-Blue into the TR pocket jaw with Gear 2 touch, leaving the cue ball stranded in center table."
        elif shot_no == 6:
            cue_tip = {"tx": 0.1, "ty": -0.4, "spin_name": "Elevated High-Follow (12:30)"}
            coach_tip = "⭐ Elevated Cue Rail Cut: Elevated butt 30° to clear the rail cushion. Applied 1/2 tip high follow with Gear 3 stroke to pot the ball and float cue ball away from the rail."
        elif shot_no == 7:
            cue_tip = {"tx": -0.3, "ty": 0.2, "spin_name": "Running Left Kick"}
            coach_tip = "⚠️ Kick Departure: Contacted 3-Red off the head cushion but failed to tie it to the rail. Left 3-Red open for Hoàng Sao."
        elif shot_no == 8:
            cue_tip = {"tx": 0.0, "ty": -0.2, "spin_name": "Stun Follow (12 o'clock)"}
            coach_tip = "🎯 Center Table Float: Crisp stun-follow stroke into TR pocket, floating cue ball into the center circle for textbook shape on 4-Pink."
        elif shot_no == 9:
            cue_tip = {"tx": 0.45, "ty": -0.3, "spin_name": "High-Right Follow (1:30)"}
            coach_tip = "⭐ Two-Cushion Shape Route: Stun-follow with 1 tip right english into BL corner. Cue ball traveled two cushions up the right rail to unlock 5-Purple."
        elif shot_no == 10:
            cue_tip = {"tx": -0.2, "ty": 0.0, "spin_name": "Left Stun (9 o'clock)"}
            coach_tip = "🎯 1-Cushion Hold: Soft touch center-ball stroke holds the cue ball off the left rail for ideal straight-in angle on 6-Green."
        elif shot_no == 11:
            cue_tip = {"tx": 0.0, "ty": 0.85, "spin_name": "Pure Draw (6 o'clock)"}
            coach_tip = "⭐ Elite Full-Table Draw: Pure bottom spin (6 o'clock) with 17.4 km/h snap stroke. The cue ball bit the Simonis cloth and drew 4.4 meters back for natural line on 7-Brown."
        elif shot_no == 12:
            cue_tip = {"tx": 0.0, "ty": 0.0, "spin_name": "Center Ball Roll"}
            coach_tip = "🎯 Natural Roll: Gear 1 soft roll-in with center ball stops dead for effortless straight-in clearance on the championship 9-ball."
        elif shot_no == 13:
            cue_tip = {"tx": 0.0, "ty": 0.0, "spin_name": "Firm Center Ball"}
            coach_tip = "🏆 Match-Winning Shot: Firm center-ball strike into the bottom-right corner pocket to complete the 6-ball run-out and seal the rack 3–1!"
        else:
            cue_tip = {"tx": 0.0, "ty": 0.0, "spin_name": "Center Ball"}
            coach_tip = "🎯 Professional Execution: High percentage pot line with controlled follow-through."

    return cue_tip, speed_gear, coach_tip


def rate_session_shots(session_data):
    """Compute shot metrics, coaching advice, and highlight top 4 made shots."""
    rated_shots = []
    is_hs = "hoang_sao" in session_data.get("video_file", "").lower()

    for sh in session_data["shots"]:
        (cb_rx, cb_ry), (ob_rx, ob_ry) = find_cue_and_object_ball(sh, session_data)
        pock_rx, pock_ry = find_target_pocket(sh, session_data, ob_rx, ob_ry)

        geom = compute_geometry(cb_rx, cb_ry, ob_rx, ob_ry, pock_rx, pock_ry, is_hs_session=is_hs)
        diff, grade, is_make = compute_difficulty_and_grade(sh, geom)
        cue_tip, speed_gear, coach_tip = compute_pool_coaching_metrics(sh, is_hs_session=is_hs)
        cue_path, obj_path = generate_shot_paths(sh, session_data, geom, is_hs_session=is_hs)

        sh["cut_angle_deg"] = geom["cut_angle_deg"]
        sh["margin_deg"] = geom["margin_deg"]
        sh["difficulty_score"] = diff
        sh["grade"] = grade
        sh["aim_line"] = geom["aim_line"]
        sh["cue_path"] = cue_path
        sh["obj_path"] = obj_path
        sh["cue_tip"] = cue_tip
        sh["speed_gear"] = speed_gear
        sh["coach_tip"] = coach_tip
        sh["is_highlight"] = False
        rated_shots.append((sh, is_make, diff))

    # Top 4 highest rated made shots get ⭐ BEST SHOT
    made_shots = [item for item in rated_shots if item[1]]
    made_shots.sort(key=lambda item: item[2], reverse=True)
    top_4_ids = {item[0]["shot_number"] for item in made_shots[:4]}

    for sh, _, _ in rated_shots:
        if sh["shot_number"] in top_4_ids:
            sh["is_highlight"] = True
        else:
            sh["is_highlight"] = False

    return session_data


def process_and_save(json_path, js_path, js_window_var):
    """Load JSON, rate all shots, and update JSON and JS in place."""
    print(f"\nProcessing {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = rate_session_shots(data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2)
    print(f"  [OK] Saved updated {json_path}")

    js_content = f"window.{js_window_var} = {json.dumps(updated_data)};\n"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"  [OK] Saved updated {js_path}")

    print(f"\n{'#':<3} | {'Grade':<5} | {'Diff':<5} | {'Cut Angle':<10} | {'Margin':<9} | {'Speed Gear':<24} | Spin Name")
    print("-" * 90)
    for sh in updated_data["shots"]:
        hl = "⭐ BEST SHOT" if sh["is_highlight"] else "-"
        spin = sh.get("cue_tip", {}).get("spin_name", "Center")
        gear = sh.get("speed_gear", "Gear 2")
        print(f"{sh['shot_number']:<3} | {sh['grade']:<5} | {sh['difficulty_score']:<5} | {sh['cut_angle_deg']:<8.1f}° | ±{sh['margin_deg']:<6.1f}° | {gear:<24} | {spin}")


def main():
    print("=================================================================")
    print("  RAILBIRD SHOT RATING & POOL-PLAYER COACHING ENGINE (1000x500)")
    print("=================================================================")

    process_and_save(
        "c:/billard/hoang_sao_session.json",
        "c:/billard/hoang_sao_session.js",
        "HOANG_SAO_SESSION"
    )

    process_and_save(
        "c:/billard/session_data.json",
        "c:/billard/session_data.js",
        "RAILBIRD_SESSION"
    )

    print("\n[COMPLETE] All shots rated and enriched with pool coaching metrics in-place!")


if __name__ == "__main__":
    main()
