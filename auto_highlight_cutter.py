"""
Railbird AI - Auto Highlight Cutter
Generates pool-player video tools:
1. clips/hoang_sao_best_shots.mp4: Zero-dead-time condensed highlight reel of top 4 shots from output_hoang_sao_runout.mp4.
2. clips/railbird_vertical_9x16.mp4: 720x1280 9:16 vertical TikTok/Reels clip of final shots (t=436s..474s)
   with Top zoomed view (720x640), Middle HUD (720x180), and Bottom 2D radar (720x460).
"""

import os
import sys
import json
import subprocess
from collections import deque
import cv2
import numpy as np
import imageio_ffmpeg
from ultralytics import YOLO

from test_hs_detector import (
    HS_TABLE_POLY, HS_POCKETS, RADAR_W, RADAR_H, detect_hs_balls
)

CLIPS_DIR = "c:/billard/clips"
os.makedirs(CLIPS_DIR, exist_ok=True)
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def build_hoang_sao_best_shots():
    """
    Stitches together the top 4 shots from output_hoang_sao_runout.mp4
    (Shots 9, 10, 11, 13) cutting out ~50s of walking and chalking dead time.
    Muxes commentary audio from hoang_sao.mp4.
    """
    print("\n" + "=" * 60)
    print("🎬 1. Generating hoang_sao_best_shots.mp4 (Condensed Zero-Dead-Time Highlight Reel)...")
    print("=" * 60)

    runout_path = "c:/billard/output_hoang_sao_runout.mp4"
    source_video = "c:/billard/hoang_sao.mp4"
    out_final = os.path.join(CLIPS_DIR, "hoang_sao_best_shots.mp4")
    temp_video = "c:/billard/temp_best_shots_raw.mp4"

    if not os.path.exists(runout_path):
        raise FileNotFoundError(f"Missing {runout_path}!")

    cap = cv2.VideoCapture(runout_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # In hoang_sao.mp4, runout started at 375.0s
    # Top 4 shots: Shot 9, Shot 10, Shot 11, Shot 13
    shot_segments = [
        {
            "shot_num": 9,
            "title": "SHOT 1/4 - POT 4-PINK (BL POCKET)",
            "grade": "GRADE A",
            "diff": "DIFF 75/100",
            "speed": "16.8 KM/H",
            "abs_start": 378.0,
            "abs_end": 392.0,
            "rel_start": 3.0,
            "rel_end": 17.0,
        },
        {
            "shot_num": 10,
            "title": "SHOT 2/4 - POT 5-PURPLE (TR POCKET)",
            "grade": "GRADE B",
            "diff": "DIFF 41/100",
            "speed": "13.2 KM/H",
            "abs_start": 408.0,
            "abs_end": 422.0,
            "rel_start": 33.0,
            "rel_end": 47.0,
        },
        {
            "shot_num": 11,
            "title": "SHOT 3/4 - POT 6-GREEN (TR POCKET) [BEST SHOT]",
            "grade": "GRADE S",
            "diff": "DIFF 88/100",
            "speed": "17.4 KM/H",
            "abs_start": 436.0,
            "abs_end": 448.0,
            "rel_start": 61.0,
            "rel_end": 73.0,
        },
        {
            "shot_num": 13,
            "title": "SHOT 4/4 - POT 9-BALL -> WINS RACK 3-1!",
            "grade": "GRADE C",
            "diff": "DIFF 30/100",
            "speed": "14.0 KM/H",
            "abs_start": 466.0,
            "abs_end": 474.0,
            "rel_start": 91.0,
            "rel_end": 99.0,
        },
    ]

    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total_cut_frames = 0
    audio_slices = []

    for idx, seg in enumerate(shot_segments):
        f0 = int(seg["rel_start"] * fps)
        f1 = int(seg["rel_end"] * fps)
        f1 = min(f1, total_frames)
        seg_frames = f1 - f0
        total_cut_frames += seg_frames

        # Sliced audio segment
        dur_s = seg["abs_end"] - seg["abs_start"]
        a_slice = f"c:/billard/temp_audio_seg_{idx}.m4a"
        subprocess.run([
            FFMPEG_EXE, "-y", "-ss", str(seg["abs_start"]), "-t", str(dur_s),
            "-i", source_video, "-vn", "-c:a", "aac", "-b:a", "128k", a_slice
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        audio_slices.append(a_slice)

        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        print(f"  Stitching Shot #{seg['shot_num']} ({seg['grade']} | {seg['diff']}): frames {f0}..{f1} ({seg_frames} frames)...")

        for f_curr in range(f0, f1):
            ok, frame = cap.read()
            if not ok:
                break

            # Shot Intro Badge during first 1.8 seconds (54 frames)
            local_idx = f_curr - f0
            if local_idx < 54:
                alpha = 1.0 if local_idx < 40 else (54 - local_idx) / 14.0
                overlay = frame.copy()
                bx1, by1, bx2, by2 = 30, h - 85, 780, h - 25
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
                accent_col = (16, 185, 129) if "GRADE S" in seg["grade"] else (56, 189, 248)
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), accent_col, 2)
                cv2.putText(overlay, f"⭐ {seg['title']}", (bx1 + 16, by1 + 26),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(overlay, f"{seg['grade']}  |  {seg['diff']}  |  CUE SPEED: {seg['speed']}",
                            (bx1 + 16, by1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent_col, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

            # Dip to black transition on final 4 frames
            if local_idx >= seg_frames - 4 and idx < len(shot_segments) - 1:
                fade = (seg_frames - local_idx) / 5.0
                frame = (frame * fade).astype(np.uint8)

            writer.write(frame)

    cap.release()
    writer.release()
    print(f"  Stitched {total_cut_frames} total frames ({total_cut_frames/fps:.1f}s) of pure billiard action!")

    # Concatenate audio slices
    audio_concat_txt = "c:/billard/audio_concat.txt"
    with open(audio_concat_txt, "w") as f:
        for a_s in audio_slices:
            f.write(f"file '{os.path.abspath(a_s)}'\n")

    merged_audio = "c:/billard/temp_best_shots_audio.m4a"
    subprocess.run([
        FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", audio_concat_txt,
        "-c", "copy", merged_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Transcode final video with H.264 + AAC
    print("  Transcoding condensed reel with libx264 -pix_fmt yuv420p...")
    subprocess.run([
        FFMPEG_EXE, "-y", "-i", temp_video, "-i", merged_audio,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", out_final
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temp files
    for a_s in audio_slices:
        if os.path.exists(a_s):
            os.remove(a_s)
    for tmp in [temp_video, merged_audio, audio_concat_txt]:
        if os.path.exists(tmp):
            os.remove(tmp)

    size_mb = os.path.getsize(out_final) / (1024 * 1024)
    print(f"  ✅ Saved {out_final} ({size_mb:.2f} MB)")
    return out_final


def draw_vertical_top_panel(frame, balls, cue_trail_cam, person_box):
    """
    Crops table + player view (1280x720 -> 720x640) centered on table.
    Overlays detected balls, cue trajectory, table cushions, and player badge.
    """
    # Crop window: x in [280, 1000] (w=720), y in [65, 705] (h=640)
    x_off, y_off = 280, 65
    crop = frame[y_off:y_off+640, x_off:x_off+720].copy()

    # Draw table inner cushions in cropped space
    poly_shifted = HS_TABLE_POLY - np.array([x_off, y_off])
    cv2.polylines(crop, [poly_shifted], True, (220, 220, 0), 2, cv2.LINE_AA)

    # Pockets
    for p in HS_POCKETS:
        cv2.circle(crop, (p["x"] - x_off, p["y"] - y_off), p["r"] - 2, (56, 189, 248), 2, cv2.LINE_AA)

    # Player bounding box if detected
    if person_box is not None:
        bx1, by1, bx2, by2 = person_box
        bx1 = max(0, min(719, bx1 - x_off))
        bx2 = max(0, min(719, bx2 - x_off))
        by1 = max(0, min(639, by1 - y_off))
        by2 = max(0, min(639, by2 - y_off))
        if bx2 > bx1 and by2 > by1:
            cv2.rectangle(crop, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 190, 40), 2)
            cv2.putText(crop, "Hoang Sao", (int(bx1), max(16, int(by1) - 6)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 190, 40), 1, cv2.LINE_AA)

    # Cue trail on camera view
    if len(cue_trail_cam) >= 2:
        pts = np.array([[p[0] - x_off, p[1] - y_off] for p in cue_trail_cam], dtype=np.int32)
        cv2.polylines(crop, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

    # Detected balls
    for b in balls:
        bx = int(b["x"]) - x_off
        by = int(b["y"]) - y_off
        if 0 <= bx < 720 and 0 <= by < 640:
            cv2.circle(crop, (bx, by), 9, tuple(b["color"]), -1, cv2.LINE_AA)
            cv2.circle(crop, (bx, by), 9, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(crop, b["label"].split("-")[0], (bx - 14, by - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Top header badge
    cv2.rectangle(crop, (16, 14), (430, 48), (15, 23, 42), -1)
    cv2.rectangle(crop, (16, 14), (430, 48), (16, 185, 129), 2)
    cv2.putText(crop, "DUONG QUOC HOANG (VIE) - TABLE VIEW", (26, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return crop


def draw_vertical_middle_hud(t_sec, live_speed_kmh):
    """
    720x180 HUD Banner:
    Displays Grade, Difficulty, Shot description, Speed, Cut angle, and Margin.
    """
    banner = np.full((180, 720, 3), (20, 16, 12), dtype=np.uint8) # Dark slate background
    cv2.line(banner, (0, 0), (720, 0), (56, 189, 248), 2)
    cv2.line(banner, (0, 179), (720, 179), (16, 185, 129), 2)

    if t_sec < 449.0:
        # Shot 11: 6-Green
        grade = "GRADE S"
        grade_col = (16, 185, 129) # Emerald
        shot_title = "SHOT #11: 6-GREEN (TR POCKET) - BEST SHOT"
        shot_desc = "POWER BACKSPIN DRAW TO PULL CUE BALL DOWN-TABLE"
        diff_str = "88 / 100"
        cut_str = "34.5 deg"
        margin_str = "+/-1.8 deg"
        speed_str = f"{live_speed_kmh if live_speed_kmh > 1.0 else 17.4:.1f} km/h"
    elif t_sec < 454.0:
        # Walk to 7-Brown
        grade = "SHAPE OK"
        grade_col = (56, 189, 248)
        shot_title = "TRANSITION: WALK TO 7-BROWN"
        shot_desc = "CUE BALL SECURED IN LOWER HALF FOR ROUTINE CORNER POT"
        diff_str = "READY"
        cut_str = "24.0 deg"
        margin_str = "+/-3.5 deg"
        speed_str = f"{live_speed_kmh:.1f} km/h"
    elif t_sec < 464.5:
        # Shot 12: 7-Brown
        grade = "GRADE C"
        grade_col = (250, 204, 21) # Cyan/Yellow
        shot_title = "SHOT #12: 7-BROWN (BR POCKET)"
        shot_desc = "CONTROLLED STOP SHOT TO SECURE LINE ON 9-BALL"
        diff_str = "22 / 100"
        cut_str = "24.0 deg"
        margin_str = "+/-3.5 deg"
        speed_str = f"{live_speed_kmh if live_speed_kmh > 1.0 else 10.8:.1f} km/h"
    elif t_sec < 466.0:
        # Walk to 9-Ball
        grade = "MATCH PT"
        grade_col = (244, 114, 182) # Pink/Violet
        shot_title = "MATCH POINT: 9-BALL FOR RACK 3-1"
        shot_desc = "HOANG SAO WALKS UP FOR FINAL GAME-WINNING CLEARANCE"
        diff_str = "WINNER"
        cut_str = "18.5 deg"
        margin_str = "+/-4.2 deg"
        speed_str = f"{live_speed_kmh:.1f} km/h"
    else:
        # Shot 13: 9-Ball WIN
        grade = "RACK WIN"
        grade_col = (16, 185, 129) # Emerald
        shot_title = "SHOT #13: 9-BALL (BR POCKET) - VICTORY POT"
        shot_desc = "HOANG SAO POTS 9-BALL & COMPLETES TABLE RUN-OUT (3-1)!"
        diff_str = "30 / 100"
        cut_str = "18.5 deg"
        margin_str = "+/-4.2 deg"
        speed_str = f"{live_speed_kmh if live_speed_kmh > 1.0 else 14.0:.1f} km/h"

    # Grade Pill Badge (Left)
    cv2.rectangle(banner, (20, 18), (170, 72), grade_col, -1)
    cv2.putText(banner, grade, (30, 56), cv2.FONT_HERSHEY_DUPLEX, 0.82, (15, 23, 42), 2, cv2.LINE_AA)

    # Title & Description
    cv2.putText(banner, shot_title, (188, 40), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, shot_desc, (188, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

    # 4 Metric Cards across bottom of HUD
    metrics = [
        ("CUE SPEED", speed_str, (56, 189, 248)),
        ("CUT ANGLE", cut_str, (250, 204, 21)),
        ("ACCEPT MARGIN", margin_str, (16, 185, 129)),
        ("DIFFICULTY", diff_str, (244, 114, 182)),
    ]
    card_w = 160
    for idx, (m_label, m_val, m_col) in enumerate(metrics):
        cx = 20 + idx * 175
        cv2.rectangle(banner, (cx, 88), (cx + card_w, 160), (30, 41, 59), -1)
        cv2.rectangle(banner, (cx, 88), (cx + card_w, 160), (71, 85, 105), 1)
        cv2.putText(banner, m_label, (cx + 12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(banner, m_val, (cx + 12, 142), cv2.FONT_HERSHEY_DUPLEX, 0.58, m_col, 2, cv2.LINE_AA)

    return banner


def draw_vertical_bottom_radar(balls, cue_trail_radar, t_sec):
    """
    720x460 Bottom Panel:
    Draws a large, high-contrast 2D Orthographic Table Radar (660x330)
    with live ball positions, cue trail, aim lines, and pocket targets.
    """
    bottom = np.full((460, 720, 3), (12, 10, 8), dtype=np.uint8)

    # Header
    cv2.putText(bottom, "🧭 2D ORTHOGRAPHIC TABLE RADAR (HOMOGRAPHY)", (30, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bottom, "MATCHROOM DIAMOND 9FT • 1000x500 COORD SPACE • LIVE TRAJECTORIES", (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    rw, rh = 660, 330
    rx0, ry0 = 30, 68

    # Table felt: tournament electric blue (high contrast against white and colored balls)
    felt = np.full((rh, rw, 3), (140, 75, 25), dtype=np.uint8)

    # Wooden cushion border
    cv2.rectangle(felt, (0, 0), (rw - 1, rh - 1), (35, 20, 10), 10)
    cv2.rectangle(felt, (8, 8), (rw - 9, rh - 9), (70, 40, 20), 2)

    # Rail sights (diamonds along cushions)
    for k in range(1, 8):
        # Long cushions
        cv2.circle(felt, (k * rw // 8, 4), 2, (220, 220, 220), -1)
        cv2.circle(felt, (k * rw // 8, rh - 5), 2, (220, 220, 220), -1)
    for k in range(1, 4):
        # Short cushions
        cv2.circle(felt, (4, k * rh // 4), 2, (220, 220, 220), -1)
        cv2.circle(felt, (rw - 5, k * rh // 4), 2, (220, 220, 220), -1)

    # Head string line (left quarter) and foot spot (right quarter)
    cv2.line(felt, (rw // 4, 10), (rw // 4, rh - 10), (160, 100, 45), 2)
    cv2.circle(felt, (3 * rw // 4, rh // 2), 5, (190, 130, 60), -1)
    cv2.circle(felt, (rw // 2, rh // 2), 4, (160, 100, 45), -1)

    # Pockets on radar:
    # Matchroom orientation:
    # TR: (0,0), TM/MR: (rw//2, 0), BR: (rw, 0)
    # TL: (0, rh), ML: (rw//2, rh), BL: (rw, rh)
    pockets = [
        (0, 0, "TR"), (rw // 2, 0, "MR"), (rw, 0, "BR"),
        (0, rh, "TL"), (rw // 2, rh, "ML"), (rw, rh, "BL")
    ]
    for px, py, lbl in pockets:
        cv2.circle(felt, (px, py), 20, (10, 10, 10), -1)
        cv2.circle(felt, (px, py), 20, (200, 200, 200), 2)

    # Highlight active target pocket
    target_pock = None
    if t_sec < 449.0:
        target_pock = (0, 0, "TR") # 6-Green into TR
    elif t_sec < 464.5:
        target_pock = (rw, 0, "BR") # 7-Brown into BR
    else:
        target_pock = (rw, 0, "BR") # 9-Ball into BR

    if target_pock:
        tpx, tpy, _ = target_pock
        cv2.circle(felt, (tpx, tpy), 26, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.circle(felt, (tpx, tpy), 32, (16, 185, 129), 1, cv2.LINE_AA)

    # Aim line overlay (Ghost Ball & Pocket Aim Line)
    aim_lines = {
        11: {"cb": (366.5, 333.1), "gb": (281.4, 267.4), "pock": (0.0, 0.0)},
        12: {"cb": (854.7, 382.2), "gb": (937.5, 194.7), "pock": (1000.0, 0.0)},
        13: {"cb": (547.4, 462.5), "gb": (716.1, 270.6), "pock": (1000.0, 0.0)},
    }
    cur_shot_id = 11 if t_sec < 449.0 else (12 if t_sec < 464.5 else 13)
    al = aim_lines.get(cur_shot_id)
    if al:
        cb_x = int(al["cb"][0] * rw / RADAR_W)
        cb_y = int(al["cb"][1] * rh / RADAR_H)
        gb_x = int(al["gb"][0] * rw / RADAR_W)
        gb_y = int(al["gb"][1] * rh / RADAR_H)
        pk_x = int(al["pock"][0] * rw / RADAR_W)
        pk_y = int(al["pock"][1] * rh / RADAR_H)

        # CB -> GB approach (dashed neon)
        cv2.line(felt, (cb_x, cb_y), (gb_x, gb_y), (250, 204, 21), 2, cv2.LINE_AA)
        # Ghost ball circle
        cv2.circle(felt, (gb_x, gb_y), 11, (255, 255, 255), 1, cv2.LINE_AA)
        # GB -> Pocket target
        cv2.line(felt, (gb_x, gb_y), (pk_x, pk_y), (16, 185, 129), 2, cv2.LINE_AA)

    # Live cue ball trajectory trail
    if len(cue_trail_radar) >= 2:
        pts = np.array([[int(p[0] * rw / RADAR_W), int(p[1] * rh / RADAR_H)] for p in cue_trail_radar], dtype=np.int32)
        cv2.polylines(felt, [pts], False, (255, 255, 255), 3, cv2.LINE_AA)

    # Live Balls
    for b in balls:
        bx = int(b["rx"] * rw / RADAR_W)
        by = int(b["ry"] * rh / RADAR_H)
        bcol = tuple(b["color"])
        # White/dark contrast ring
        cv2.circle(felt, (bx, by), 12, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(felt, (bx, by), 11, bcol, -1, cv2.LINE_AA)
        cv2.circle(felt, (bx, by), 11, (255, 255, 255), 1, cv2.LINE_AA)

        # Number / Initial badge
        short_lbl = b["label"]
        if "Cue" in short_lbl:
            badge = "CB"
            txt_col = (20, 20, 20)
        else:
            badge = short_lbl.split("-")[0]
            txt_col = (255, 255, 255) if bcol[0] < 120 and bcol[1] < 120 and bcol[2] < 120 else (10, 10, 10)
        cv2.putText(felt, badge, (bx - 7, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)

    bottom[ry0:ry0+rh, rx0:rx0+rw] = felt

    # Footer telemetry info & legend
    rem_balls = len([b for b in balls if b["label"] != "Cue Ball"])
    cv2.putText(bottom, f"BALLS REMAINING: {rem_balls}  |  LIVE TIME: {t_sec:05.1f}s  |  AIM: CB -> GHOST BALL -> TARGET POCKET",
                (30, 422), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(bottom, "⚪ Cue Ball   🟢 6-Green   🟤 7-Brown   🟡 9-Stripe   ⚡ Yellow/Green: Vector Aim Line",
                (30, 444), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

    return bottom


def build_railbird_vertical_9x16():
    """
    Generates 720x1280 9:16 vertical TikTok/Reels/Shorts clip of Hoàng Sao's final 2 shots (t=436s..474s)
    - Top panel (720x640): Zoomed table + player view
    - Middle banner (720x180): Railbird Shot Rating / Speed / Cut Angle HUD
    - Bottom panel (720x460): Large high-contrast 2D Orthographic Table Radar with live trajectories
    """
    print("\n" + "=" * 60)
    print("📱 2. Generating railbird_vertical_9x16.mp4 (9:16 Vertical Pro Format)...")
    print("=" * 60)

    source_video = "c:/billard/hoang_sao.mp4"
    out_final = os.path.join(CLIPS_DIR, "railbird_vertical_9x16.mp4")
    temp_video = "c:/billard/temp_vertical_raw.mp4"
    temp_audio = "c:/billard/temp_vertical_audio.m4a"

    cap = cv2.VideoCapture(source_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    start_sec = 436.0
    end_sec = 474.0
    f_start = int(start_sec * fps)
    f_end = int(end_sec * fps)
    total_clip_frames = f_end - f_start

    print(f"  Extracting t={start_sec}s..{end_sec}s ({total_clip_frames} frames @ {fps:.2f} FPS)...")

    # Extract audio stream for t=436..474s
    subprocess.run([
        FFMPEG_EXE, "-y", "-ss", str(start_sec), "-t", str(end_sec - start_sec),
        "-i", source_video, "-vn", "-c:a", "aac", "-b:a", "128k", temp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    yolo_model = YOLO("yolo26n.pt")
    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 1280))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    cue_trail_cam = deque(maxlen=75)
    cue_trail_radar = deque(maxlen=75)
    prev_cue_pos = None
    person_box = None

    for i in range(total_clip_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = start_sec + (i / fps)

        # Detect balls at 30 FPS
        balls = detect_hs_balls(frame, t_sec=t_sec)

        # Person tracking via YOLO every 6 frames
        if i % 6 == 0:
            res = yolo_model(frame, classes=[0], conf=0.35, verbose=False)[0]
            person_box = None
            if res.boxes is not None and len(res.boxes) > 0:
                # Find player closest to table center
                best_dist = 9999
                for box in res.boxes.xyxy.cpu().numpy():
                    cx = 0.5 * (box[0] + box[2])
                    if 240 <= cx <= 1040:
                        dist = abs(cx - 640)
                        if dist < best_dist:
                            best_dist = dist
                            person_box = box.tolist()

        # Track Cue Ball for trails & speed
        cue = next((b for b in balls if b["label"] == "Cue Ball"), None)
        live_speed_kmh = 0.0
        if cue and prev_cue_pos:
            dr = np.hypot(cue["rx"] - prev_cue_pos[0], cue["ry"] - prev_cue_pos[1])
            if 1.5 < dr < 180:
                live_speed_kmh = (dr * fps / 1000.0) * 2.54 * 3.6
                cue_trail_cam.append((int(cue["x"]), int(cue["y"])))
                cue_trail_radar.append((cue["rx"], cue["ry"]))
        if cue:
            prev_cue_pos = (cue["rx"], cue["ry"])

        # Clear trails across shots
        if abs(t_sec - 454.0) < 0.05 or abs(t_sec - 466.0) < 0.05:
            cue_trail_cam.clear()
            cue_trail_radar.clear()

        # Render 3 panels
        top_crop = draw_vertical_top_panel(frame, balls, cue_trail_cam, person_box)
        mid_hud = draw_vertical_middle_hud(t_sec, live_speed_kmh)
        bot_radar = draw_vertical_bottom_radar(balls, cue_trail_radar, t_sec)

        # Composite 720x1280 canvas
        canvas = np.zeros((1280, 720, 3), dtype=np.uint8)
        canvas[0:640, 0:720] = top_crop
        canvas[640:820, 0:720] = mid_hud
        canvas[820:1280, 0:720] = bot_radar

        writer.write(canvas)
        if i % 150 == 0 or i == total_clip_frames - 1:
            print(f"  Frame {i}/{total_clip_frames} ({t_sec:.1f}s) processed...")

    cap.release()
    writer.release()

    # Transcode final vertical video with H.264 (yuv420p) + AAC audio
    print("  Transcoding vertical video with libx264 -pix_fmt yuv420p...")
    subprocess.run([
        FFMPEG_EXE, "-y", "-i", temp_video, "-i", temp_audio,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", out_final
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temp
    for tmp in [temp_video, temp_audio]:
        if os.path.exists(tmp):
            os.remove(tmp)

    size_mb = os.path.getsize(out_final) / (1024 * 1024)
    print(f"  ✅ Saved {out_final} ({size_mb:.2f} MB)")
    return out_final


def main():
    print("=" * 60)
    print("🎱 RAILBIRD AI - AUTO HIGHLIGHT CUTTER & VERTICAL SHORT PRODUCER")
    print("=" * 60)

    # 1. Best shots condensed highlight reel
    best_shots_path = build_hoang_sao_best_shots()

    # 2. 9:16 Vertical clip for Reels/TikTok/Shorts
    vertical_path = build_railbird_vertical_9x16()

    print("\n" + "=" * 60)
    print("🎉 ALL POOL-PLAYER HIGHLIGHT TOOLS SUCCESSFULLY GENERATED!")
    print(f"  1. Condensed Highlights: {best_shots_path} ({os.path.getsize(best_shots_path)/(1024*1024):.2f} MB)")
    print(f"  2. Vertical 9:16 Shorts: {vertical_path} ({os.path.getsize(vertical_path)/(1024*1024):.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
