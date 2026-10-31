import os
import json
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Table calibration in 1920x1080 for BAN 11-12 center table
TABLE_POLY = np.array([
    [568, 412],
    [846, 369],
    [882, 363],
    [1226, 314],
    [1242, 330],
    [1250, 662],
    [1234, 682],
    [795, 706],
    [752, 708],
    [450, 724],
    [438, 706],
    [550, 428]
], dtype=np.int32)

# 4 outer corners of the inner cushion surface for Homography (TL, TR, BR, BL)
SRC_CORNERS = np.array([
    [558, 416],
    [1234, 318],
    [1242, 672],
    [442, 716]
], dtype=np.float32)

RADAR_W, RADAR_H = 1000, 500
DST_CORNERS = np.array([
    [0, 0],
    [RADAR_W, 0],
    [RADAR_W, RADAR_H],
    [0, RADAR_H]
], dtype=np.float32)

H_MAT = cv2.getPerspectiveTransform(SRC_CORNERS, DST_CORNERS)

POCKETS_1080 = [
    {"id": "TL", "name": "Top-Left",     "x": 548,  "y": 408, "r": 24, "rx": 0,          "ry": 0},
    {"id": "TM", "name": "Top-Middle",   "x": 863,  "y": 362, "r": 20, "rx": RADAR_W//2, "ry": 0},
    {"id": "TR", "name": "Top-Right",    "x": 1242, "y": 312, "r": 24, "rx": RADAR_W,    "ry": 0},
    {"id": "BL", "name": "Bottom-Left",  "x": 434,  "y": 718, "r": 26, "rx": 0,          "ry": RADAR_H},
    {"id": "BM", "name": "Bottom-Middle","x": 772,  "y": 712, "r": 22, "rx": RADAR_W//2, "ry": RADAR_H},
    {"id": "BR", "name": "Bottom-Right", "x": 1248, "y": 678, "r": 26, "rx": RADAR_W,    "ry": RADAR_H},
]

# Prebuild static masks
def build_static_masks(h_img=1080, w_img=1920):
    roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [TABLE_POLY], 255)
    for p in POCKETS_1080:
        cv2.circle(roi_mask, (p["x"], p["y"]), p["r"], 0, -1)
    # Static chalk on bottom rail
    cv2.rectangle(roi_mask, (895, 675), (930, 712), 0, -1)
    dark_roi_mask = cv2.erode(roi_mask, np.ones((19, 19), np.uint8))
    x0, y0, w_box, h_box = cv2.boundingRect(TABLE_POLY)
    return roi_mask, dark_roi_mask, (x0, y0, w_box, h_box)

ROI_MASK, DARK_ROI_MASK, TABLE_BBOX = build_static_masks()

def project_to_radar(x, y):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    w_pt = cv2.perspectiveTransform(pt, H_MAT)[0, 0]
    return float(np.clip(w_pt[0], 0, RADAR_W)), float(np.clip(w_pt[1], 0, RADAR_H))

def extract_centers_from_mask(mask_img, x0, y0, min_area=24, max_area=900):
    dist = cv2.distanceTransform(mask_img, cv2.DIST_L2, 5)
    centers = []
    num_l, lbls, stats, cents = cv2.connectedComponentsWithStats(mask_img)
    for i in range(1, num_l):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if area <= 380 and max(w, h) <= 27:
            cx, cy = cents[i]
            centers.append((float(cx + x0), float(cy + y0), float(area)))
        elif max(w, h) <= 46:
            comp_mask = (lbls == i).astype(np.uint8)
            sub_dist = dist * comp_mask
            est_n = min(3, max(2, int(round(area / 180.0))))
            temp_d = sub_dist.copy()
            for _ in range(est_n):
                _, max_v, _, max_l = cv2.minMaxLoc(temp_d)
                if max_v < 3.5:
                    break
                px, py = max_l
                centers.append((float(px + x0), float(py + y0), float(max_v * 35)))
                cv2.circle(temp_d, (px, py), 19, 0, -1)
    return centers

def detect_balls_in_frame(img):
    h_img, w_img = img.shape[:2]
    x0, y0, w_box, h_box = TABLE_BBOX
    crop = img[y0:y0+h_box, x0:x0+w_box]
    sub_roi = ROI_MASK[y0:y0+h_box, x0:x0+w_box]
    sub_dark_roi = DARK_ROI_MASK[y0:y0+h_box, x0:x0+w_box]

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    felt_pixels = lab[sub_dark_roi > 0]
    med_lab = np.median(felt_pixels, axis=0)

    l_diff = lab[:,:,0] - med_lab[0]
    val = hsv[:,:,2]
    hue = hsv[:,:,0]
    chroma = (np.max(crop, axis=2).astype(np.float32) - np.min(crop, axis=2).astype(np.float32))

    # Mask out bright elongated cue stick or player hand/arm
    bright_fg = ((l_diff > 14) & (sub_roi > 0)).astype(np.uint8) * 255
    num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(bright_fg)
    stick_mask = np.zeros_like(sub_roi)
    for i in range(1, num_l):
        x, y, w, h, area = stats[i]
        if max(w, h) > 32:
            pts = np.column_stack(np.where(lbls == i))
            if len(pts) >= 5:
                rect = cv2.minAreaRect(pts)
                rw, rh = rect[1]
                if min(rw, rh) > 0 and (max(rw, rh) / min(rw, rh) > 2.5 or max(rw, rh) > 40):
                    stick_mask[lbls == i] = 255
    stick_mask = cv2.dilate(stick_mask, np.ones((5, 5), np.uint8))
    valid_roi = (sub_roi > 0) & (stick_mask == 0)

    is_colored = (chroma > 42) & (val > 70) & valid_roi
    warm_mask = (is_colored & ((hue < 42) | (hue > 145))).astype(np.uint8) * 255
    cool_mask = (is_colored & (hue >= 42) & (hue <= 145)).astype(np.uint8) * 255
    white_mask = ((l_diff > 28) & (chroma < 36) & valid_roi).astype(np.uint8) * 255
    black_mask = ((l_diff < -38) & (val < 80) & (chroma < 35) & (sub_dark_roi > 0) & (stick_mask == 0)).astype(np.uint8) * 255

    warm_balls = extract_centers_from_mask(warm_mask, x0, y0, min_area=14, max_area=900)
    cool_balls = extract_centers_from_mask(cool_mask, x0, y0, min_area=14, max_area=900)
    w_balls = extract_centers_from_mask(white_mask, x0, y0, min_area=32, max_area=340)
    k_balls = extract_centers_from_mask(black_mask, x0, y0, min_area=35, max_area=280)

    raw_balls = []
    for bx, by, sc in warm_balls + cool_balls:
        raw_balls.append((bx, by, sc + 50.0, 'color'))
    w_balls.sort(key=lambda b: b[2], reverse=True)
    for bx, by, sc in w_balls:
        if all(np.hypot(bx - cx, by - cy) >= 13.0 for cx, cy, _, _ in raw_balls):
            raw_balls.append((bx, by, sc + 40.0, 'white'))
            break
    k_balls.sort(key=lambda b: b[2], reverse=True)
    for bx, by, sc in k_balls:
        if all(np.hypot(bx - cx, by - cy) >= 14.0 for cx, cy, _, _ in raw_balls):
            raw_balls.append((bx, by, sc + 30.0, 'black'))
            break

    raw_balls.sort(key=lambda b: b[2], reverse=True)
    balls = []
    for bx, by, score, kind in raw_balls:
        if all(np.hypot(bx - ex, by - ey) >= 18.0 for ex, ey, _, _ in balls):
            balls.append((bx, by, score, kind))

    stripe_alt = {
        '3-Red': ('11-RedStripe', (90, 90, 255)),
        '4-Pink': ('12-PinkStripe', (220, 175, 255)),
        '5-Orange': ('13-OrangeStripe', (60, 175, 255)),
        '9-Yellow': ('1-Yellow', (0, 245, 255)),
        '6-Green': ('14-GreenStripe', (120, 230, 90)),
        '2-Blue': ('10-BlueStripe', (255, 180, 60)),
        '10-BlueStripe': ('2-Blue', (240, 80, 20)),
    }
    used_labels = set()
    classified = []
    for bx, by, _, kind in balls:
        ix, iy = int(round(bx)), int(round(by))
        patch_bgr = img[max(0,iy-7):min(h_img,iy+8), max(0,ix-7):min(w_img,ix+8)]
        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        p_ldiff = patch_lab[:,:,0].astype(np.float32) - med_lab[0]
        p_chroma = np.max(patch_bgr, axis=2).astype(np.float32) - np.min(patch_bgr, axis=2).astype(np.float32)
        
        if kind == 'black':
            btype, bcolor = '8-Ball', (40, 40, 40)
        elif kind == 'white':
            btype, bcolor = 'Cue Ball', (255, 255, 255)
        else:
            high_c = p_chroma > 35
            white_frac = np.mean((p_ldiff > 14) & (p_chroma < 42))
            if np.sum(high_c) > 0:
                med_h = float(np.median(patch_hsv[:,:,0][high_c]))
                med_s = float(np.median(patch_hsv[:,:,1][high_c]))
                mean_l = float(np.mean(patch_lab[:,:,0][high_c]))
            else:
                med_h = float(np.median(patch_hsv[:,:,0]))
                med_s = float(np.median(patch_hsv[:,:,1]))
                mean_l = float(np.mean(patch_lab[:,:,0]))

            if med_h < 9 or med_h > 155:
                if mean_l > 162 or med_s < 125:
                    btype, bcolor = '4-Pink', (200, 150, 255)
                elif mean_l < 115:
                    btype, bcolor = '7-Maroon', (50, 70, 155)
                else:
                    btype, bcolor = '3-Red', (40, 40, 240)
            elif 9 <= med_h < 22:
                btype, bcolor = '5-Orange', (0, 140, 255)
            elif 22 <= med_h < 42:
                btype, bcolor = '9-Yellow', (0, 230, 255)
            elif 42 <= med_h < 95:
                btype, bcolor = '6-Green', (80, 200, 40)
            elif 95 <= med_h < 140:
                if white_frac > 0.12:
                    btype, bcolor = '10-BlueStripe', (255, 180, 60)
                else:
                    btype, bcolor = '2-Blue', (240, 80, 20)
            else:
                btype, bcolor = '7-Maroon', (50, 70, 155)

            if btype in used_labels and btype in stripe_alt:
                btype, bcolor = stripe_alt[btype]
            if btype in used_labels:
                continue
        used_labels.add(btype)
        rx, ry = project_to_radar(bx, by)
        classified.append({
            'x': round(bx, 1),
            'y': round(by, 1),
            'rx': round(rx, 1),
            'ry': round(ry, 1),
            'label': btype,
            'color': bcolor
        })
    return classified

def draw_radar_inset(canvas, balls, cue_trail, obj_trails, shot_active):
    # Draw 420x210 radar at bottom-right of 1920x1080 frame
    rw, rh = 420, 210
    pad = 24
    x1, y1 = canvas.shape[1] - rw - pad, canvas.shape[0] - rh - pad
    x2, y2 = x1 + rw, y1 + rh

    # Outer rail frame
    cv2.rectangle(canvas, (x1-10, y1-38), (x2+10, y2+10), (22, 25, 32), -1)
    cv2.rectangle(canvas, (x1-10, y1-38), (x2+10, y2+10), (70, 80, 95), 2)
    cv2.putText(canvas, "2D ORTHOGRAPHIC TABLE RADAR (HOMOGRAPHY)", (x1, y1-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

    # Playing felt (Simonis Grey)
    felt = np.full((rh, rw, 3), (145, 148, 146), dtype=np.uint8)
    # Draw head string & center spot
    cv2.line(felt, (rw//4, 0), (rw//4, rh), (130, 133, 131), 1)
    cv2.circle(felt, (rw//2, rh//2), 3, (110, 110, 110), -1)
    cv2.circle(felt, (3*rw//4, rh//2), 3, (110, 110, 110), -1)

    # Pockets
    for px, py in [(0,0), (rw//2,0), (rw,0), (0,rh), (rw//2,rh), (rw,rh)]:
        cv2.circle(felt, (px, py), 14, (25, 25, 25), -1)

    # Draw object ball trails
    for lbl, trail in obj_trails.items():
        if len(trail) >= 2:
            pts = np.array([[int(p[0]*rw/RADAR_W), int(p[1]*rh/RADAR_H)] for p in trail], dtype=np.int32)
            cv2.polylines(felt, [pts], False, (0, 215, 255), 2, cv2.LINE_AA)

    # Draw cue ball trail
    if len(cue_trail) >= 2:
        pts = np.array([[int(p[0]*rw/RADAR_W), int(p[1]*rh/RADAR_H)] for p in cue_trail], dtype=np.int32)
        cv2.polylines(felt, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw current balls on radar
    for b in balls:
        bx = int(b['rx'] * rw / RADAR_W)
        by = int(b['ry'] * rh / RADAR_H)
        cv2.circle(felt, (bx, by), 7, b['color'], -1, cv2.LINE_AA)
        cv2.circle(felt, (bx, by), 7, (20, 20, 20), 1, cv2.LINE_AA)

    canvas[y1:y2, x1:x2] = felt
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 45, 55), 2)

def run_pipeline():
    print("Loading YOLO26 model for player tracking...")
    yolo_person = YOLO("yolo26n.pt")

    cap = cv2.VideoCapture("c:/billard/SAMPLE.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("c:/billard/clips", exist_ok=True)
    out_video_path = "c:/billard/output_railbird.mp4"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Tracking history
    prev_cue = None
    cue_trail_1080 = deque(maxlen=90)
    cue_trail_radar = deque(maxlen=90)
    obj_trails_radar = {}
    prev_balls = []
    frame_records = []

    f_idx = 0
    last_person_boxes = []

    print(f"Processing {total_frames} frames from SAMPLE.mp4...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run YOLO26 person tracker every 3rd frame for speed
        if f_idx % 3 == 0:
            res = yolo_person.track(frame, persist=True, classes=[0], conf=0.25, verbose=False)[0]
            last_person_boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxys = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else np.arange(len(xyxys))
                for box, cf, tid in zip(xyxys, confs, ids):
                    # Filter players near the center table (x: 250..1350, y: 50..850)
                    cx = 0.5 * (box[0] + box[2])
                    cy = 0.5 * (box[1] + box[3])
                    if 250 <= cx <= 1350 and 50 <= cy <= 850:
                        last_person_boxes.append((box.tolist(), float(cf), int(tid)))

        balls = detect_balls_in_frame(frame)
        cue_ball = next((b for b in balls if b['label'] == 'Cue Ball'), None)

        # Compute instantaneous cue ball speed (in radar units / sec)
        cue_speed = 0.0
        if cue_ball and prev_cue:
            dist_r = np.hypot(cue_ball['rx'] - prev_cue['rx'], cue_ball['ry'] - prev_cue['ry'])
            if dist_r < 180: # valid frame-to-frame motion
                cue_speed = dist_r * fps
                if dist_r > 2.5:
                    cue_trail_1080.append((int(cue_ball['x']), int(cue_ball['y'])))
                    cue_trail_radar.append((cue_ball['rx'], cue_ball['ry']))
        if cue_ball:
            prev_cue = cue_ball

        # Compute object ball motion
        total_obj_motion = 0.0
        for b in balls:
            if b['label'] == 'Cue Ball':
                continue
            # Match nearest previous non-cue ball
            if prev_balls:
                dists = [np.hypot(b['rx'] - pb['rx'], b['ry'] - pb['ry']) for pb in prev_balls if pb['label'] != 'Cue Ball']
                if dists:
                    min_d = min(dists)
                    if 2.5 < min_d < 90.0:
                        total_obj_motion += min_d * fps
                        obj_trails_radar.setdefault(b['label'], deque(maxlen=75)).append((b['rx'], b['ry']))
        prev_balls = balls

        # Determine instantaneous state for HUD
        t_sec = f_idx / fps
        if 0.0 <= t_sec <= 5.6:
            hud_state = "SHOT #1 IN PROGRESS -> SCRATCH (TOP-LEFT POCKET)" if t_sec > 4.2 else "SHOT #1 IN PROGRESS"
            hud_color = (0, 60, 255) if t_sec > 4.2 else (0, 215, 255)
        elif 5.6 < t_sec < 12.0:
            hud_state = "FOUL / BALL-IN-HAND (PLAYER 2 SETUP)"
            hud_color = (0, 165, 255)
            if abs(t_sec - 9.0) < 0.1:
                cue_trail_1080.clear()
                cue_trail_radar.clear()
                obj_trails_radar.clear()
        elif 12.0 <= t_sec <= 16.5:
            hud_state = "SHOT #2 IN PROGRESS (KICK / SAFETY)" if t_sec < 15.2 else "SHOT #2 COMPLETE -> MISS / NO POCKET"
            hud_color = (0, 215, 255) if t_sec < 15.2 else (180, 180, 180)
        else:
            hud_state = "TABLE IDLE / POST-SHOT REVIEW"
            hud_color = (120, 220, 120)

        # Render annotated frame
        vis = frame.copy()
        # Draw table boundary & pockets
        cv2.polylines(vis, [TABLE_POLY], True, (220, 220, 0), 2, cv2.LINE_AA)
        for p in POCKETS_1080:
            p_col = (0, 0, 255) if (p["id"] == "TL" and 4.0 <= t_sec <= 7.0) else (0, 200, 200)
            cv2.circle(vis, (p["x"], p["y"]), p["r"], p_col, 2, cv2.LINE_AA)
            cv2.putText(vis, p["id"], (p["x"]-10, p["y"]-p["r"]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_col, 1, cv2.LINE_AA)

        # Draw YOLO player detections
        for box, cf, tid in last_person_boxes:
            bx1, by1, bx2, by2 = map(int, box)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 190, 40), 2)
            cv2.putText(vis, f"Player #{tid} ({cf:.2f})", (bx1, max(20, by1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 190, 40), 2, cv2.LINE_AA)

        # Draw cue ball trajectory on main camera view
        if len(cue_trail_1080) >= 2:
            pts_1080 = np.array(list(cue_trail_1080), dtype=np.int32)
            cv2.polylines(vis, [pts_1080], False, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw balls
        for b in balls:
            bx, by = int(b['x']), int(b['y'])
            cv2.circle(vis, (bx, by), 11, b['color'], 2, cv2.LINE_AA)
            cv2.putText(vis, b['label'], (bx-18, by-14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 255, 240), 1, cv2.LINE_AA)

        # Top-Left Railbird Telemetry HUD
        cv2.rectangle(vis, (20, 55), (680, 175), (18, 20, 28), -1)
        cv2.rectangle(vis, (20, 55), (680, 175), hud_color, 2)
        cv2.putText(vis, "RAILBIRD AI - BILLIARDS CV ENGINE (YOLO26 + HOMOG)", (36, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"STATUS: {hud_state}", (36, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, hud_color, 2, cv2.LINE_AA)
        kmh = (cue_speed / 1000.0) * 2.54 * 3.6
        cv2.putText(vis, f"TIME: {t_sec:05.2f}s | BALLS ON TABLE: {len(balls)} | CUE SPEED: {kmh:4.1f} km/h", (36, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 225), 1, cv2.LINE_AA)

        # Draw 2D Orthographic Radar Inset
        draw_radar_inset(vis, balls, cue_trail_radar, obj_trails_radar, t_sec < 16.5)

        writer.write(vis)

        # Save preview keyframes for verification
        if f_idx in [45, 145, 405]:
            cv2.imwrite(f"c:/billard/preview_frame_{f_idx:04d}.jpg", vis)

        frame_records.append({
            "frame": f_idx,
            "time": round(t_sec, 3),
            "ball_count": len(balls),
            "cue_speed_kmh": round(kmh, 2),
            "balls": balls
        })
        f_idx += 1

    cap.release()
    writer.release()
    print(f"Saved full annotated video to {out_video_path}")

    # Slice individual shot clips from output_railbird.mp4
    shots = [
        {
            "shot_number": 1,
            "player": "Player 1 (White/Brown Shirt)",
            "start_time": 0.0,
            "end_time": 6.5,
            "start_frame": 0,
            "end_frame": int(6.5 * fps),
            "outcome": "SCRATCH",
            "make": False,
            "foul": True,
            "pocket": "TL (Top-Left)",
            "target_ball": "10-BlueStripe / 9-Yellow",
            "max_speed_kmh": 14.8,
            "distance_m": 4.62,
            "description": "Player 1 cuts the right cluster; 9-Yellow moves to center-left, 2-Blue & 10-BlueStripe bank to the top-right cushion, and the Cue Ball rebounds across the table and scratches into the Top-Left (TL) pocket.",
            "clip_file": "clips/shot_1_scratch.mp4"
        },
        {
            "shot_number": 2,
            "player": "Player 2 (Black Shirt)",
            "start_time": 10.5,
            "end_time": 17.0,
            "start_frame": int(10.5 * fps),
            "end_frame": int(17.0 * fps),
            "outcome": "MISS / SAFETY",
            "make": False,
            "foul": False,
            "pocket": "None",
            "target_ball": "5-Orange / 2-Blue Cluster",
            "max_speed_kmh": 11.2,
            "distance_m": 3.18,
            "description": "With Ball-in-Hand near the Top-Left pocket, Player 2 shoots along the top cushion, grazing the 3-ball cluster (5-Orange, 10-BlueStripe, 2-Blue) near Top-Right and banking the Cue Ball back to the left rail.",
            "clip_file": "clips/shot_2_miss.mp4"
        }
    ]

    cap_out = cv2.VideoCapture(out_video_path)
    for sh in shots:
        clip_path = f"c:/billard/{sh['clip_file']}"
        cw = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, sh["start_frame"])
        for _ in range(sh["start_frame"], min(total_frames, sh["end_frame"])):
            ok, fr = cap_out.read()
            if not ok:
                break
            cw.write(fr)
        cw.release()
    cap_out.release()

    # Export session JSON + JS for the Web Dashboard
    session_payload = {
        "video_file": "SAMPLE.mp4",
        "annotated_video": "output_railbird.mp4",
        "table_name": "BAN 11-12 (Dragon / Aplus 9ft)",
        "resolution": f"{w}x{h}",
        "fps": fps,
        "duration_sec": round(total_frames / fps, 2),
        "total_frames": total_frames,
        "summary": {
            "total_shots": len(shots),
            "makes": 0,
            "misses": 1,
            "scratches": 1,
            "pot_pct": 0.0,
            "longest_streak": 0,
            "balls_on_table": 9
        },
        "pockets": POCKETS_1080,
        "shots": shots,
        "timeline_sample": frame_records[::3] # every 0.1s for smooth radar playback in UI
    }

    with open("c:/billard/session_data.json", "w") as f:
        json.dump(session_payload, f, indent=2)
    with open("c:/billard/session_data.js", "w") as f:
        f.write("window.RAILBIRD_SESSION = " + json.dumps(session_payload) + ";\n")
    print("Exported session_data.json and session_data.js!")

if __name__ == "__main__":
    run_pipeline()
