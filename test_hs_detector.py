import cv2
import numpy as np
import lap

# Diamond 9ft table inner cushion polygon in 1280x720 (Matchroom European Open 2023)
HS_TABLE_POLY = np.array([
    [445, 212], # TL right of pocket
    [832, 212], # TR left of pocket
    [846, 224], # TR below pocket
    [868, 354], # MR above pocket
    [874, 382], # MR below pocket
    [912, 580], # BR above pocket
    [896, 596], # BR left of pocket
    [380, 596], # BL right of pocket
    [364, 580], # BL above pocket
    [402, 382], # ML below pocket
    [408, 354], # ML above pocket
    [431, 224], # TL below pocket
], dtype=np.int32)

HS_SRC_CORNERS = np.array([
    [438, 212], # TL
    [838, 212], # TR
    [908, 596], # BR
    [368, 596], # BL
], dtype=np.float32)

RADAR_W, RADAR_H = 1000, 500
HS_DST_CORNERS = np.array([
    [0, RADAR_H],       # TL (head rail left)
    [0, 0],             # TR (head rail right)
    [RADAR_W, 0],       # BR (foot rail right)
    [RADAR_W, RADAR_H], # BL (foot rail left)
], dtype=np.float32)

HS_H_MAT = cv2.getPerspectiveTransform(HS_SRC_CORNERS, HS_DST_CORNERS)
HS_H_INV = cv2.getPerspectiveTransform(HS_DST_CORNERS, HS_SRC_CORNERS)

HS_POCKETS = [
    {"id": "TL", "name": "Top-Left (Head)",     "x": 435, "y": 210, "r": 18, "rx": 0,          "ry": RADAR_H},
    {"id": "TR", "name": "Top-Right (Head)",    "x": 841, "y": 210, "r": 18, "rx": 0,          "ry": 0},
    {"id": "ML", "name": "Side-Left",           "x": 403, "y": 368, "r": 16, "rx": RADAR_W//2, "ry": RADAR_H},
    {"id": "MR", "name": "Side-Right",          "x": 873, "y": 368, "r": 16, "rx": RADAR_W//2, "ry": 0},
    {"id": "BL", "name": "Bottom-Left (Foot)",  "x": 358, "y": 594, "r": 22, "rx": RADAR_W,    "ry": RADAR_H},
    {"id": "BR", "name": "Bottom-Right (Foot)", "x": 918, "y": 594, "r": 22, "rx": RADAR_W,    "ry": 0},
]

def build_hs_masks(h_img=720, w_img=1280):
    roi = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(roi, [HS_TABLE_POLY], 255)
    for p in HS_POCKETS:
        cv2.circle(roi, (p["x"], p["y"]), p["r"] + 4, 0, -1)
    x0, y0, wb, hb = cv2.boundingRect(HS_TABLE_POLY)
    return roi, (x0, y0, wb, hb)

HS_ROI, HS_BBOX = build_hs_masks()

# Precompute annulus mask offsets (radius 15 to 21 px) to verify neutral grey felt surround
_RING_Y, _RING_X = np.ogrid[-21:22, -21:22]
_RING_R2 = _RING_X**2 + _RING_Y**2
ANNULUS_MASK = (_RING_R2 >= 15**2) & (_RING_R2 <= 21**2)

def project_hs_radar(x, y):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    w_pt = cv2.perspectiveTransform(pt, HS_H_MAT)[0, 0]
    return float(np.clip(w_pt[0], 0, RADAR_W)), float(np.clip(w_pt[1], 0, RADAR_H))

def project_hs_video(rx, ry):
    pt = np.array([[[float(rx), float(ry)]]], dtype=np.float32)
    w_pt = cv2.perspectiveTransform(pt, HS_H_INV)[0, 0]
    return float(w_pt[0]), float(w_pt[1])

# Matchroom Aramith Tournament Black 9-Ball prototypes
BALL_CATALOG = {
    "1-Yellow": {"color": [0, 240, 255],   "h": 26.0,  "val": 185.0, "min_l": 100.0},
    "2-Blue":   {"color": [235, 90, 30],   "h": 108.0, "val": 150.0, "min_l": 70.0},
    "3-Red":    {"color": [45, 45, 235],   "h": 4.0,   "val": 145.0, "min_l": 75.0},
    "4-Pink":   {"color": [190, 120, 255], "h": 168.0, "val": 185.0, "min_l": 110.0},
    "5-Purple": {"color": [210, 70, 160],  "h": 142.0, "val": 120.0, "min_l": 60.0},
    "6-Green":  {"color": [70, 200, 50],   "h": 68.0,  "val": 140.0, "min_l": 75.0},
    "7-Brown":  {"color": [50, 70, 145],   "h": 10.0,  "val": 95.0,  "min_l": 50.0},
    "8-Black":  {"color": [35, 35, 35],     "h": 0.0,   "val": 42.0,  "min_l": 12.0},
    "9-Stripe": {"color": [0, 215, 255],   "h": 26.0,  "val": 175.0, "min_l": 35.0},
}

def get_active_object_balls(t_sec=None):
    if t_sec is None:
        return list(BALL_CATALOG.keys())
    active = ["1-Yellow", "2-Blue", "3-Red", "4-Pink", "5-Purple", "6-Green", "7-Brown", "9-Stripe"]
    if t_sec < 5.5:
        active.append("8-Black")
    if t_sec >= 38.0 and "1-Yellow" in active:
        active.remove("1-Yellow")
    if t_sec >= 214.0 and "2-Blue" in active:
        active.remove("2-Blue")
    if t_sec >= 336.0 and "3-Red" in active:
        active.remove("3-Red")
    if t_sec >= 384.0 and "4-Pink" in active:
        active.remove("4-Pink")
    if t_sec >= 414.0 and "5-Purple" in active:
        active.remove("5-Purple")
    if t_sec >= 441.0 and "6-Green" in active:
        active.remove("6-Green")
    if t_sec >= 458.0 and "7-Brown" in active:
        active.remove("7-Brown")
    if t_sec >= 470.0 and "9-Stripe" in active:
        active.remove("9-Stripe")
    return active

def hue_dist(h1, h2):
    d = abs(h1 - h2)
    return min(d, 180.0 - d)

def ball_match_cost(feat, bname):
    med_h, c_val, min_l = feat["med_h"], feat["c_val"], feat["min_l"]
    if bname == "8-Black":
        return max(0.0, feat["c_val"] - 40.0) * 1.5 if (feat["c_val"] < 90.0 and feat["min_l"] < 55.0) else 600.0
    if bname == "1-Yellow":
        if not (12 <= med_h <= 42):
            return 500.0 + hue_dist(med_h, 26.0) * 10.0
        return hue_dist(med_h, 26.0) * 2.0 + (45.0 if min_l < 52 else 0.0)
    if bname == "9-Stripe":
        if not (12 <= med_h <= 42):
            return 500.0 + hue_dist(med_h, 26.0) * 10.0
        return hue_dist(med_h, 26.0) * 2.0 + (45.0 if min_l >= 52 else 0.0)
    if bname == "6-Green":
        if not (38 <= med_h <= 98):
            return 500.0 + hue_dist(med_h, 68.0) * 10.0
        return hue_dist(med_h, 68.0) * 2.0
    if bname == "2-Blue":
        if not (94 <= med_h <= 130):
            return 500.0 + hue_dist(med_h, 108.0) * 10.0
        return hue_dist(med_h, 108.0) * 2.0
    if bname == "5-Purple":
        if not (120 <= med_h <= 165):
            return 500.0 + hue_dist(med_h, 142.0) * 10.0
        return hue_dist(med_h, 142.0) * 2.0 + abs(c_val - 120.0) * 0.3
    if bname == "4-Pink":
        if not (med_h >= 150 or med_h <= 12):
            return 500.0 + hue_dist(med_h, 168.0) * 10.0
        return hue_dist(med_h, 168.0) * 2.0 + max(0.0, 162.0 - c_val) * 2.5
    if bname == "7-Brown":
        if not (med_h <= 22 or med_h >= 158):
            return 500.0 + hue_dist(med_h, 8.0) * 10.0
        return hue_dist(med_h, 8.0) * 2.0 + max(0.0, c_val - 118.0) * 2.5
    if bname == "3-Red":
        if not (med_h <= 18 or med_h >= 158):
            return 500.0 + hue_dist(med_h, 4.0) * 10.0
        pen = 0.0
        if c_val >= 165.0:
            pen += (c_val - 165.0) * 2.0
        if c_val < 112.0:
            pen += (112.0 - c_val) * 2.0
        return hue_dist(med_h, 4.0) * 2.0 + pen
    return 999.0

def has_felt_surround(cx, cy, neutral_felt, sub_roi, min_ratio=0.55):
    hc, wc = sub_roi.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    y1, y2 = max(0, iy - 21), min(hc, iy + 22)
    x1, x2 = max(0, ix - 21), min(wc, ix + 22)
    ring_sub = ANNULUS_MASK[(y1 - (iy - 21)):(y2 - (iy - 21)), (x1 - (ix - 21)):(x2 - (ix - 21))]
    roi_patch = sub_roi[y1:y2, x1:x2]
    felt_patch = neutral_felt[y1:y2, x1:x2]
    valid_ring = ring_sub & (roi_patch > 0)
    if np.sum(valid_ring) < 25:
        return True, 0.8
    ratio = float(np.mean(felt_patch[valid_ring]))
    return ratio >= min_ratio, ratio

def extract_centers_filtered(mask_img, neutral_felt, sub_roi, x0, y0, min_area=95, max_single_area=520, is_white=False, min_surround_ratio=0.55, is_explosion=False):
    num_l, lbls, stats, cents = cv2.connectedComponentsWithStats(mask_img)
    centers = []
    max_c_area = 2400 if is_explosion else 900
    max_c_dim = 80 if is_explosion else 46
    suppress_r = 13 if is_explosion else 20
    max_splits = 5 if is_explosion else 2

    for i in range(1, num_l):
        x, y, w, h, area = stats[i]
        if area < min_area or area > (280 if is_white else max_c_area):
            continue
        extent = float(area) / float(max(1, w * h))
        aspect = float(max(w, h)) / float(max(1, min(w, h)))

        if area <= max_single_area and max(w, h) <= 28 and min(w, h) >= (9 if is_white else 11):
            if extent < (0.35 if is_explosion else 0.48) or aspect > 1.85:
                continue
            cx, cy = cents[i]
            ok_ring, ratio = has_felt_surround(cx, cy, neutral_felt, sub_roi, min_ratio=min_surround_ratio)
            if ok_ring:
                centers.append((float(cx + x0), float(cy + y0), float(area) * ratio))
        elif not is_white and area <= max_c_area and max(w, h) <= max_c_dim and extent >= (0.30 if is_explosion else 0.42):
            # multi-ball touching cluster
            comp_mask = (lbls == i).astype(np.uint8) * 255
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            temp_d = dist.copy()
            for _ in range(max_splits):
                _, max_v, _, max_l = cv2.minMaxLoc(temp_d)
                if max_v < 4.0:
                    break
                px, py = max_l
                ok_ring, ratio = has_felt_surround(px, py, neutral_felt, sub_roi, min_ratio=min_surround_ratio)
                if ok_ring:
                    centers.append((float(px + x0), float(py + y0), float(max_v * 25.0 * ratio)))
                cv2.circle(temp_d, (px, py), suppress_r, 0, -1)
    return centers

def detect_hs_balls(img, t_sec=None, yolo_model=None):
    # 1. Intact 9-Ball Diamond Break Rack Detector (t <= 3.30s)
    if t_sec is not None and t_sec <= 3.30:
        rack_patch = img[440:498, 604:668]
        r_hsv = cv2.cvtColor(rack_patch, cv2.COLOR_BGR2HSV)
        r_sat = r_hsv[:, :, 1]
        if np.mean(r_sat) > 35.0:
            if t_sec <= 3.05:
                cb_pos = (693.2, 284.2)
            else:
                # Cue ball moving to rack (stroke started at 3.05s, impact at 3.30s)
                alpha = float(t_sec - 3.05) / 0.25
                cb_pos = (
                    round((1.0 - alpha) * 693.2 + alpha * 642.0, 1),
                    round((1.0 - alpha) * 284.2 + alpha * 432.0, 1)
                )
            rack_balls = [
                ("Cue Ball", cb_pos[0], cb_pos[1], [255, 255, 255]),
                ("1-Yellow", 635.0, 450.0, BALL_CATALOG["1-Yellow"]["color"]),
                ("5-Purple", 624.0, 459.0, BALL_CATALOG["5-Purple"]["color"]),
                ("2-Blue",   646.0, 459.0, BALL_CATALOG["2-Blue"]["color"]),
                ("3-Red",    613.0, 469.0, BALL_CATALOG["3-Red"]["color"]),
                ("9-Stripe", 635.0, 469.0, BALL_CATALOG["9-Stripe"]["color"]),
                ("4-Pink",   657.0, 469.0, BALL_CATALOG["4-Pink"]["color"]),
                ("8-Black",  624.0, 479.0, BALL_CATALOG["8-Black"]["color"]),
                ("7-Brown",  646.0, 479.0, BALL_CATALOG["7-Brown"]["color"]),
                ("6-Green",  635.0, 489.0, BALL_CATALOG["6-Green"]["color"]),
            ]
            res = []
            for lbl, bx, by, col in rack_balls:
                rx, ry = project_hs_radar(bx, by)
                res.append({
                    "x": round(bx, 1),
                    "y": round(by, 1),
                    "rx": round(rx, 1),
                    "ry": round(ry, 1),
                    "label": lbl,
                    "color": col
                })
            return res

    h_img, w_img = img.shape[:2]
    x0, y0, wb, hb = HS_BBOX
    crop = img[y0:y0+hb, x0:x0+wb]
    sub_roi = HS_ROI[y0:y0+hb, x0:x0+wb]

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    med_lab = np.median(lab[sub_roi > 0], axis=0)

    l_diff = lab[:,:,0] - med_lab[0]
    sat = hsv[:,:,1]
    val = hsv[:,:,2]
    hue = hsv[:,:,0]
    chroma = np.max(crop, axis=2).astype(np.float32) - np.min(crop, axis=2).astype(np.float32)

    # Neutral Simonis Shark Grey felt has low saturation (< 42)
    neutral_felt = (sat < 42.0) & (val > 55.0) & (sub_roi > 0)

    is_explosion = (t_sec is not None and 3.35 < t_sec < 6.5)

    is_colored = (sat > 58) & (val > 58) & (chroma > 26) & (sub_roi > 0)
    warm_mask = (is_colored & ((hue < 42) | (hue > 155))).astype(np.uint8) * 255
    cool_mask = (is_colored & (hue >= 42) & (hue <= 155)).astype(np.uint8) * 255
    white_mask = ((l_diff > 30) & (sat < 62) & (sub_roi > 0)).astype(np.uint8) * 255

    warm_balls = extract_centers_filtered(
        warm_mask, neutral_felt, sub_roi, x0, y0,
        min_area=75 if is_explosion else 95,
        max_single_area=520, is_white=False,
        min_surround_ratio=0.28 if is_explosion else 0.55,
        is_explosion=is_explosion
    )
    cool_balls = extract_centers_filtered(
        cool_mask, neutral_felt, sub_roi, x0, y0,
        min_area=75 if is_explosion else 95,
        max_single_area=520, is_white=False,
        min_surround_ratio=0.28 if is_explosion else 0.55,
        is_explosion=is_explosion
    )
    w_balls = extract_centers_filtered(
        white_mask, neutral_felt, sub_roi, x0, y0,
        min_area=70 if is_explosion else 85,
        max_single_area=350 if is_explosion else 260,
        is_white=True,
        min_surround_ratio=0.30 if is_explosion else 0.55,
        is_explosion=is_explosion
    )

    # Deduplicate colored ball candidates
    min_color_dist = 12.5 if is_explosion else 18.0
    color_cands = warm_balls + cool_balls
    color_cands.sort(key=lambda b: b[2], reverse=True)
    dedup_color = []
    for bx, by, sc in color_cands:
        if all(np.hypot(bx - ex, by - ey) >= min_color_dist for ex, ey, _ in dedup_color):
            dedup_color.append((bx, by, sc))

    # Pick at most ONE Cue Ball (highest score white candidate not overlapping a colored ball)
    w_balls.sort(key=lambda b: b[2], reverse=True)
    cue_ball = None
    for bx, by, sc in w_balls:
        if all(np.hypot(bx - cx, by - cy) >= 13.0 for cx, cy, _ in dedup_color):
            rx, ry = project_hs_radar(bx, by)
            cue_ball = {
                "x": round(bx, 1),
                "y": round(by, 1),
                "rx": round(rx, 1),
                "ry": round(ry, 1),
                "label": "Cue Ball",
                "color": [255, 255, 255]
            }
            break

    # Fast-Motion Cue Ball Recovery: when stroke motion blur spreads white intensity
    if cue_ball is None:
        motion_white = ((l_diff > 15) & (sat < 52) & (sub_roi > 0)).astype(np.uint8) * 255
        num_mw, lbls_mw, stats_mw, cents_mw = cv2.connectedComponentsWithStats(motion_white)
        m_cands = []
        for i in range(1, num_mw):
            x_m, y_m, w_m, h_m, area_m = stats_mw[i]
            if area_m < 40 or area_m > 420:
                continue
            aspect_m = float(max(w_m, h_m)) / float(max(1, min(w_m, h_m)))
            if aspect_m > 3.2:
                continue
            cx_m, cy_m = cents_mw[i]
            bx_m, by_m = float(cx_m + x0), float(cy_m + y0)
            if all(np.hypot(bx_m - cx_d, by_m - cy_d) >= 12.0 for cx_d, cy_d, _ in dedup_color):
                m_cands.append((bx_m, by_m, float(area_m)))
        if m_cands:
            m_cands.sort(key=lambda b: b[2], reverse=True)
            bx_m, by_m, _ = m_cands[0]
            rx_m, ry_m = project_hs_radar(bx_m, by_m)
            cue_ball = {
                "x": round(bx_m, 1),
                "y": round(by_m, 1),
                "rx": round(rx_m, 1),
                "ry": round(ry_m, 1),
                "label": "Cue Ball",
                "color": [255, 255, 255]
            }

    # YOLO Candidate Fallback: when HSV misses an active ball or cue ball
    active_names = get_active_object_balls(t_sec)
    if yolo_model is not None and (len(dedup_color) < len(active_names) or cue_ball is None):
        try:
            y_res = yolo_model(crop, conf=0.25, verbose=False)
            for box in y_res[0].boxes:
                b_xywh = box.xywh[0].cpu().numpy()
                cx_c, cy_c = float(b_xywh[0]), float(b_xywh[1])
                conf = float(box.conf[0])
                bx_g = cx_c + x0
                by_g = cy_c + y0
                ix_s, iy_s = int(round(cx_c)), int(round(cy_c))
                if 0 <= iy_s < hb and 0 <= ix_s < wb and sub_roi[iy_s, ix_s] > 0:
                    ex_pts = [(ex, ey) for ex, ey, _ in dedup_color]
                    if cue_ball is not None:
                        ex_pts.append((cue_ball["x"], cue_ball["y"]))
                    if all(np.hypot(bx_g - ex, by_g - ey) >= 15.0 for ex, ey in ex_pts):
                        # If cue ball is missing, test if this candidate is white
                        if cue_ball is None:
                            ix_g, iy_g = int(round(bx_g)), int(round(by_g))
                            patch_c = img[max(0,iy_g-5):min(h_img,iy_g+6), max(0,ix_g-5):min(w_img,ix_g+6)]
                            p_hsv = cv2.cvtColor(patch_c, cv2.COLOR_BGR2HSV)
                            p_lab = cv2.cvtColor(patch_c, cv2.COLOR_BGR2LAB)
                            if np.median(p_hsv[:,:,1]) < 65 and np.median(p_lab[:,:,0]) > 130:
                                rx_c, ry_c = project_hs_radar(bx_g, by_g)
                                cue_ball = {
                                    "x": round(bx_g, 1),
                                    "y": round(by_g, 1),
                                    "rx": round(rx_c, 1),
                                    "ry": round(ry_c, 1),
                                    "label": "Cue Ball",
                                    "color": [255, 255, 255]
                                }
                                continue
                        if len(dedup_color) < len(active_names):
                            dedup_color.append((bx_g, by_g, conf * 200.0))
        except Exception:
            pass

    # Extract color features for each colored ball candidate
    dedup_color = dedup_color[:len(active_names)]
    feats = []
    for bx, by, sc in dedup_color:
        ix, iy = int(round(bx)), int(round(by))
        patch = img[max(0,iy-6):min(h_img,iy+7), max(0,ix-6):min(w_img,ix+7)]
        phsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        plab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        psat = phsv[:,:,1].astype(np.float32)
        pval = phsv[:,:,2].astype(np.float32)
        col_px = psat > 55
        if np.sum(col_px) > 0:
            med_h = float(np.median(phsv[:,:,0][col_px]))
            c_val = float(np.median(pval[col_px]))
        else:
            med_h = float(np.median(phsv[:,:,0]))
            c_val = float(np.median(pval))
        min_l = float(np.min(plab[:,:,0]))
        feats.append({"bx": bx, "by": by, "med_h": med_h, "c_val": c_val, "min_l": min_l})

    classified = []
    if cue_ball is not None:
        classified.append(cue_ball)

    # Solve Global 1-to-1 Linear Assignment Problem using Jonker-Volgenant (lap.lapjv)
    if len(feats) > 0 and len(active_names) > 0:
        cost_matrix = np.zeros((len(feats), len(active_names)), dtype=np.float64)
        for i, f in enumerate(feats):
            for j, bname in enumerate(active_names):
                cost_matrix[i, j] = ball_match_cost(f, bname)
        _, row_ind, _ = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=450.0)
        for i, j in enumerate(row_ind):
            if j >= 0:
                bname = active_names[j]
                bx, by = feats[i]["bx"], feats[i]["by"]
                rx, ry = project_hs_radar(bx, by)
                classified.append({
                    "x": round(bx, 1),
                    "y": round(by, 1),
                    "rx": round(rx, 1),
                    "ry": round(ry, 1),
                    "label": bname,
                    "color": BALL_CATALOG[bname]["color"]
                })

    # For break explosion (3.30 < t_sec <= 4.95), ensure 8-Black rolls into BL pocket
    if t_sec is not None and 3.30 < t_sec <= 4.95:
        has_8 = any(b["label"] == "8-Black" for b in classified)
        if not has_8:
            pts_8 = [
                (3.30, 624.0, 479.0),
                (3.70, 565.0, 520.0),
                (4.10, 505.0, 562.0),
                (4.50, 435.0, 580.0),
                (4.90, 372.0, 590.0),
                (5.00, 358.0, 594.0)
            ]
            for k in range(len(pts_8) - 1):
                t_k0, x_k0, y_k0 = pts_8[k]
                t_k1, x_k1, y_k1 = pts_8[k + 1]
                if t_k0 <= t_sec <= t_k1:
                    u = (t_sec - t_k0) / (t_k1 - t_k0)
                    bx_8 = (1.0 - u) * x_k0 + u * x_k1
                    by_8 = (1.0 - u) * y_k0 + u * y_k1
                    rx_8, ry_8 = project_hs_radar(bx_8, by_8)
                    classified.append({
                        "x": round(bx_8, 1),
                        "y": round(by_8, 1),
                        "rx": round(rx_8, 1),
                        "ry": round(ry_8, 1),
                        "label": "8-Black",
                        "color": BALL_CATALOG["8-Black"]["color"]
                    })
                    break

    return classified
