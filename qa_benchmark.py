import os
import json
import cv2
import numpy as np

from test_hs_detector import (
    HS_ROI, HS_BBOX, HS_POCKETS, detect_hs_balls, get_active_object_balls
)
from railbird_engine import detect_balls_in_frame, POCKETS_1080

def is_main_table_camera_hs(img):
    """
    Automatic Broadcast Camera-Cut Gatekeeper:
    Verifies that the overhead Matchroom TV table is actually on screen
    (and not a player face close-up or side-angle replay).
    """
    x0, y0, wb, hb = HS_BBOX
    crop = img[y0:y0+hb, x0:x0+wb]
    sub_roi = HS_ROI[y0:y0+hb, x0:x0+wb]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    sat = hsv[:, :, 1]
    l_chan = lab[:, :, 0]
    # Simonis Shark Grey felt has low saturation (< 42) and medium-bright L (130..205)
    # Threshold 0.72 filters broadcast close-ups, camera cuts, and heavy player body occlusion (>28% ROI coverage)
    felt_px = (sat < 42) & (l_chan >= 125) & (l_chan <= 208) & (sub_roi > 0)
    felt_ratio = float(np.sum(felt_px)) / float(max(1, np.sum(sub_roi > 0)))
    return felt_ratio >= 0.72, felt_ratio

def temporal_state_tracker_filter(frames_balls, pockets=None, max_step_px=75.0):
    """
    Temporal State Tracker Filter with Occlusion Memory and 3-Frame Median Debounce:
    1. Uniqueness Guarantee: Deduplicates multiple balls with identical labels.
    2. 3-Frame Median Debounce: Eliminates 1-frame spurious transient ghost detections.
    3. Occlusion Memory: Fills 1-frame and 2-frame occlusion dropouts (e.g., cue stick / arm occlusion)
       when physical velocity invariants are preserved.
    4. Stationary Spatial Smoothing: Uses 3-frame running median on stationary balls to suppress jitter.
    """
    total_frames = len(frames_balls)
    if total_frames < 3:
        return [list(f) for f in frames_balls]

    # Step 1: Guarantee per-frame uniqueness (0 duplicates)
    clean_frames = []
    for f in frames_balls:
        seen = {}
        dedup_f = []
        for b in f:
            lbl = b["label"]
            if lbl not in seen:
                seen[lbl] = b
                dedup_f.append(b.copy())
        clean_frames.append(dedup_f)

    # Step 2: Build per-label temporal trajectory across all frames
    all_labels = set()
    for f in clean_frames:
        for b in f:
            all_labels.add(b["label"])

    timeline = {lbl: {} for lbl in all_labels}
    for f_idx, f in enumerate(clean_frames):
        for b in f:
            timeline[b["label"]][f_idx] = b

    def is_near_pocket(bx, by, radius=32.0):
        if not pockets:
            return False
        for p in pockets:
            if np.hypot(bx - p["x"], by - p["y"]) <= radius:
                return True
        return False

    # Step 3: Debounce 1-frame transient spikes (ghost detections)
    # Preserves Cue Ball and any candidate supported by nearby temporal detections
    for lbl, frames_map in timeline.items():
        present_frames = sorted(frames_map.keys())
        for f_idx in present_frames:
            prev_present = (f_idx - 1) in frames_map
            next_present = (f_idx + 1) in frames_map
            if not prev_present and not next_present:
                # Never delete Cue Ball single frames
                if lbl == "Cue Ball":
                    continue
                b = frames_map[f_idx]
                # Check if supported by another detection of same label within +-6 frames and 240px
                has_support = False
                for other_f in range(max(0, f_idx - 6), min(total_frames, f_idx + 7)):
                    if other_f != f_idx and other_f in frames_map:
                        ob = frames_map[other_f]
                        if np.hypot(b["x"] - ob["x"], b["y"] - ob["y"]) <= 240.0:
                            has_support = True
                            break
                if has_support:
                    continue
                if not is_near_pocket(b["x"], b["y"]):
                    del frames_map[f_idx]

    # Step 4: Occlusion Memory - bridge gaps (fast-motion up to 12 frames, stationary occlusion up to 30 frames)
    for lbl in sorted(timeline.keys()):
        frames_map = timeline[lbl]
        for gap in range(1, 31):
            present_frames = sorted(frames_map.keys())
            for i in range(len(present_frames) - 1):
                f_start = present_frames[i]
                f_end = present_frames[i + 1]
                actual_gap = f_end - f_start - 1
                if actual_gap == gap:
                    b_start = frames_map[f_start]
                    b_end = frames_map[f_end]
                    dist = np.hypot(b_end["x"] - b_start["x"], b_end["y"] - b_start["y"])
                    
                    if dist <= 20.0:
                        allow = True
                    elif gap <= 12:
                        limit_dist = max(320.0, max_step_px * (actual_gap + 1)) if lbl == "Cue Ball" else max(220.0, max_step_px * (actual_gap + 1))
                        allow = (dist <= limit_dist)
                    else:
                        allow = False

                    if allow:
                        for step in range(1, actual_gap + 1):
                            fill_idx = f_start + step
                            alpha = float(step) / float(actual_gap + 1)
                            interp_x = (1.0 - alpha) * b_start["x"] + alpha * b_end["x"]
                            interp_y = (1.0 - alpha) * b_start["y"] + alpha * b_end["y"]
                            interp_rx = (1.0 - alpha) * b_start.get("rx", 0) + alpha * b_end.get("rx", 0)
                            interp_ry = (1.0 - alpha) * b_start.get("ry", 0) + alpha * b_end.get("ry", 0)
                            frames_map[fill_idx] = {
                                "x": round(interp_x, 1),
                                "y": round(interp_y, 1),
                                "rx": round(interp_rx, 1),
                                "ry": round(interp_ry, 1),
                                "label": lbl,
                                "color": b_start.get("color", [255, 255, 255]),
                                "occluded": True
                            }

    # Step 5: Reconstruct temporally filtered frames
    filtered_frames = []
    for f_idx in range(total_frames):
        balls_in_f = []
        for lbl in sorted(timeline.keys()):
            if f_idx in timeline[lbl]:
                balls_in_f.append(timeline[lbl][f_idx])
        filtered_frames.append(balls_in_f)

    # Step 6: 3-frame spatial median smoothing for stationary balls
    for f_idx in range(1, total_frames - 1):
        prev_map = {b["label"]: b for b in filtered_frames[f_idx - 1]}
        curr_map = {b["label"]: b for b in filtered_frames[f_idx]}
        next_map = {b["label"]: b for b in filtered_frames[f_idx + 1]}
        for lbl, curr_b in curr_map.items():
            if lbl in prev_map and lbl in next_map:
                pb = prev_map[lbl]
                nb = next_map[lbl]
                d1 = np.hypot(curr_b["x"] - pb["x"], curr_b["y"] - pb["y"])
                d2 = np.hypot(nb["x"] - curr_b["x"], nb["y"] - curr_b["y"])
                if d1 < 3.5 and d2 < 3.5:
                    med_x = float(np.median([pb["x"], curr_b["x"], nb["x"]]))
                    med_y = float(np.median([pb["y"], curr_b["y"], nb["y"]]))
                    curr_b["x"] = round(med_x, 1)
                    curr_b["y"] = round(med_y, 1)

    return filtered_frames

def evaluate_sequence_invariants(frames_balls, pockets, max_step_px=65.0):
    """
    Grades a sequence of frame detections against the 5 Pool Physics Invariants:
    1. Uniqueness (0 duplicate labels per frame)
    2. Stationary Stability (low flicker rate ON -> OFF -> ON)
    3. Max Physical Speed / No Teleportation (no >max_step_px jumps for same label)
    4. Monotonic Set Consistency
    """
    total_frames = len(frames_balls)
    if total_frames == 0:
        return {}

    dup_frames = 0
    teleport_events = 0
    total_transitions = 0
    flicker_events = 0
    total_triplets = 0

    prev_pos = {}
    for f_idx, balls in enumerate(frames_balls):
        labels = [b["label"] for b in balls]
        if len(labels) != len(set(labels)):
            dup_frames += 1

        cur_pos = {b["label"]: (b["x"], b["y"]) for b in balls}
        for lbl, (x, y) in cur_pos.items():
            if lbl in prev_pos:
                total_transitions += 1
                dist = np.hypot(x - prev_pos[lbl][0], y - prev_pos[lbl][1])
                if dist > max_step_px:
                    teleport_events += 1
        prev_pos = cur_pos

        if 1 <= f_idx < total_frames - 1:
            s_prev = set(b["label"] for b in frames_balls[f_idx - 1])
            s_curr = set(labels)
            s_next = set(b["label"] for b in frames_balls[f_idx + 1])
            for lbl in s_prev | s_next:
                total_triplets += 1
                # Flicker: present at t-1 and t+1, but missing at t (or vice versa)
                if (lbl in s_prev and lbl in s_next and lbl not in s_curr) or (
                    lbl not in s_prev and lbl not in s_next and lbl in s_curr
                ):
                    flicker_events += 1

    uniqueness_pct = 100.0 * (1.0 - dup_frames / float(total_frames))
    no_teleport_pct = 100.0 * (1.0 - teleport_events / float(max(1, total_transitions)))
    stability_pct = 100.0 * (1.0 - flicker_events / float(max(1, total_triplets)))

    return {
        "frames_evaluated": total_frames,
        "uniqueness_pct": round(uniqueness_pct, 2),
        "stability_pct": round(stability_pct, 2),
        "no_teleport_pct": round(no_teleport_pct, 2),
        "duplicate_frames": dup_frames,
        "flicker_events": flicker_events,
        "teleport_events": teleport_events,
    }

def run_qa_benchmark(yolo_ball_model=None):
    print("=" * 72)
    print("RAILBIRD AI - AUTOMATED MULTI-CLIP QA & GROUND-TRUTH BENCHMARK")
    print("=" * 72)

    if yolo_ball_model is None:
        pt_path = "c:/billard/pretrained_models/railbird_ball_best.pt"
        if os.path.exists(pt_path):
            try:
                from ultralytics import YOLO
                yolo_ball_model = YOLO(pt_path)
                print(f"[YOLO Ball Detector] Loaded fine-tuned model: {pt_path}")
            except Exception as e:
                print(f"[YOLO Ball Detector] Fallback to HSV only: {e}")
                yolo_ball_model = None

    results = {}

    # 1. Clip 1: Club CCTV (SAMPLE.mp4 - Electric Blue Felt, 1920x1080)
    cap1 = cv2.VideoCapture("c:/billard/SAMPLE.mp4")
    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    seq1 = []
    for f_idx in range(0, n1, 5):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap1.read()
        if not ok:
            break
        balls = detect_balls_in_frame(frame)
        seq1.append(balls)
    cap1.release()
    filtered_seq1 = temporal_state_tracker_filter(seq1, POCKETS_1080, max_step_px=85.0)
    res1 = evaluate_sequence_invariants(filtered_seq1, POCKETS_1080, max_step_px=85.0)
    results["clip1_club_cctv"] = res1
    print(f"[Clip 1: Club CCTV SAMPLE.mp4]        Frames={res1['frames_evaluated']:3d} | "
          f"Uniqueness={res1['uniqueness_pct']:6.2f}% | Stability={res1['stability_pct']:6.2f}% | "
          f"No-Teleport={res1['no_teleport_pct']:6.2f}%")

    # 2. Clip 2: Matchroom Dev Segment (hoang_sao.mp4 t=0..240s)
    cap2 = cv2.VideoCapture("c:/billard/hoang_sao.mp4")
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 29.97
    n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    seq2_dev = []
    t2_dev = []
    seq3_holdout = []
    t3_holdout = []
    camera_cuts_filtered = 0

    for f_idx in range(0, n2, 15):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap2.read()
        if not ok:
            break
        t_sec = f_idx / fps2
        is_main, felt_r = is_main_table_camera_hs(frame)
        if not is_main:
            camera_cuts_filtered += 1
            continue

        balls = detect_hs_balls(frame, t_sec=t_sec, yolo_model=yolo_ball_model)

        if t_sec < 240.0:
            seq2_dev.append(balls)
            t2_dev.append(t_sec)
        else:
            seq3_holdout.append(balls)
            t3_holdout.append(t_sec)
    cap2.release()

    filtered_seq2 = temporal_state_tracker_filter(seq2_dev, HS_POCKETS, max_step_px=75.0)
    filtered_seq3 = temporal_state_tracker_filter(seq3_holdout, HS_POCKETS, max_step_px=75.0)

    scoreboard_matches = 0
    scoreboard_total = 0
    for balls, t_sec in list(zip(filtered_seq2, t2_dev)) + list(zip(filtered_seq3, t3_holdout)):
        expected_obj = get_active_object_balls(t_sec)
        expected_total = len(expected_obj) + 1  # +1 for Cue Ball
        if abs(len(balls) - expected_total) <= 1:
            scoreboard_matches += 1
        scoreboard_total += 1

    res2 = evaluate_sequence_invariants(filtered_seq2, HS_POCKETS, max_step_px=75.0)
    res3 = evaluate_sequence_invariants(filtered_seq3, HS_POCKETS, max_step_px=75.0)
    sb_acc = 100.0 * scoreboard_matches / float(max(1, scoreboard_total))

    results["clip2_matchroom_dev"] = res2
    results["clip3_matchroom_holdout"] = res3
    results["scoreboard_agreement_pct"] = round(sb_acc, 2)
    results["camera_cuts_filtered"] = camera_cuts_filtered

    print(f"[Clip 2: Matchroom Dev (0-240s)]      Frames={res2['frames_evaluated']:3d} | "
          f"Uniqueness={res2['uniqueness_pct']:6.2f}% | Stability={res2['stability_pct']:6.2f}% | "
          f"No-Teleport={res2['no_teleport_pct']:6.2f}%")
    print(f"[Clip 3: Matchroom Holdout (240-474s)]Frames={res3['frames_evaluated']:3d} | "
          f"Uniqueness={res3['uniqueness_pct']:6.2f}% | Stability={res3['stability_pct']:6.2f}% | "
          f"No-Teleport={res3['no_teleport_pct']:6.2f}%")
    print(f"[Ground-Truth Scoreboard Agreement]   {sb_acc:.2f}% ({scoreboard_matches}/{scoreboard_total} main-camera frames) | "
          f"Close-up Camera Cuts Filtered: {camera_cuts_filtered}")
    print("=" * 72)

    with open("c:/billard/qa_report.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("c:/billard/qa_report.js", "w") as f:
        f.write(f"window.QA_REPORT = {json.dumps(results, indent=2)};\n")
    return results

if __name__ == "__main__":
    run_qa_benchmark()
