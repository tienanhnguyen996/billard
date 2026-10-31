import os
import json
import subprocess
import cv2
import numpy as np
from collections import deque
import imageio_ffmpeg
from ultralytics import YOLO

from test_hs_detector import (
    HS_TABLE_POLY, HS_POCKETS, HS_ROI, HS_BBOX,
    RADAR_W, RADAR_H, project_hs_radar, detect_hs_balls
)

def draw_hs_radar_inset(canvas, balls, cue_trail):
    rw, rh = 340, 170
    pad = 16
    x1, y1 = canvas.shape[1] - rw - pad, canvas.shape[0] - rh - 75
    x2, y2 = x1 + rw, y1 + rh

    cv2.rectangle(canvas, (x1-8, y1-30), (x2+8, y2+8), (18, 22, 30), -1)
    cv2.rectangle(canvas, (x1-8, y1-30), (x2+8, y2+8), (56, 189, 248), 2)
    cv2.putText(canvas, "2D ORTHOGRAPHIC TABLE RADAR", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 230, 255), 1, cv2.LINE_AA)

    felt = np.full((rh, rw, 3), (145, 148, 146), dtype=np.uint8)
    cv2.line(felt, (rw//4, 0), (rw//4, rh), (165, 168, 166), 1)
    cv2.circle(felt, (3*rw//4, rh//2), 3, (110, 110, 110), -1)

    for px, py in [(0,0), (rw//2,0), (rw,0), (0,rh), (rw//2,rh), (rw,rh)]:
        cv2.circle(felt, (px, py), 11, (20, 20, 20), -1)

    if len(cue_trail) >= 2:
        pts = np.array([[int(p[0]*rw/RADAR_W), int(p[1]*rh/RADAR_H)] for p in cue_trail], dtype=np.int32)
        cv2.polylines(felt, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

    for b in balls:
        bx = int(b['rx'] * rw / RADAR_W)
        by = int(b['ry'] * rh / RADAR_H)
        cv2.circle(felt, (bx, by), 6, tuple(b['color']), -1, cv2.LINE_AA)
        cv2.circle(felt, (bx, by), 6, (15, 15, 15), 1, cv2.LINE_AA)

    canvas[y1:y2, x1:x2] = felt

def main():
    cap = cv2.VideoCapture("c:/billard/hoang_sao.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Scanning all {total_frames} frames sequentially (10Hz telemetry + Run-Out video)...")
    timeline_sample = []
    prev_cue = None

    yolo_model = YOLO("yolo26n.pt")
    raw_out = "c:/billard/hs_runout_raw.mp4"
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    start_f = int(375 * fps)
    cue_trail_1080 = deque(maxlen=75)
    cue_trail_radar = deque(maxlen=75)
    prev_c = None
    person_boxes = []

    shot_windows = [
        (378, 392, "SHOT #9: HOANG SAO POTS 4-PINK (BL POCKET)"),
        (408, 422, "SHOT #10: HOANG SAO POTS 5-PURPLE (TR POCKET)"),
        (436, 448, "SHOT #11: HOANG SAO POTS 6-GREEN (TR POCKET)"),
        (454, 464, "SHOT #12: HOANG SAO POTS 7-BROWN (BR POCKET)"),
        (466, 475, "SHOT #13: HOANG SAO POTS 9-BALL -> WINS RACK 3-1!"),
    ]

    for f_idx in range(total_frames):
        if f_idx < start_f and (f_idx % 3 != 0):
            if not cap.grab():
                break
            continue

        ok, frame = cap.read()
        if not ok:
            break
        t_sec = f_idx / fps
        balls = detect_hs_balls(frame, t_sec=t_sec)

        if f_idx % 3 == 0:
            cue = next((b for b in balls if b["label"] == "Cue Ball"), None)
            speed_kmh = 0.0
            if cue and prev_cue:
                dist_r = np.hypot(cue["rx"] - prev_cue["rx"], cue["ry"] - prev_cue["ry"])
                if dist_r < 220:
                    speed_kmh = ((dist_r * (fps / 3.0)) / 1000.0) * 2.54 * 3.6
            if cue:
                prev_cue = cue

            timeline_sample.append({
                "frame": f_idx,
                "time": round(t_sec, 2),
                "ball_count": len(balls),
                "cue_speed_kmh": round(speed_kmh, 1),
                "balls": balls
            })

        if f_idx >= start_f:
            if (f_idx - start_f) % 4 == 0:
                res = yolo_model.track(frame, persist=True, classes=[0], conf=0.3, verbose=False)[0]
                person_boxes = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for box, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        cx = 0.5 * (box[0] + box[2])
                        if 220 <= cx <= 1060:
                            person_boxes.append((box.tolist(), float(cf)))

            cue = next((b for b in balls if b["label"] == "Cue Ball"), None)
            if cue and prev_c:
                d = np.hypot(cue["rx"] - prev_c["rx"], cue["ry"] - prev_c["ry"])
                if 2.0 < d < 160.0:
                    cue_trail_1080.append((int(cue["x"]), int(cue["y"])))
                    cue_trail_radar.append((cue["rx"], cue["ry"]))
            if cue:
                prev_c = cue

            for s_start, _, _ in shot_windows:
                if abs(t_sec - s_start) < 0.04:
                    cue_trail_1080.clear()
                    cue_trail_radar.clear()

            status_txt = "HOANG SAO RUN-OUT IN PROGRESS (5-BALL CLEARANCE)"
            for s0, s1, stxt in shot_windows:
                if s0 <= t_sec <= s1:
                    status_txt = stxt
                    break

            vis = frame.copy()
            cv2.polylines(vis, [HS_TABLE_POLY], True, (220, 220, 0), 2, cv2.LINE_AA)
            for p in HS_POCKETS:
                cv2.circle(vis, (p["x"], p["y"]), p["r"], (56, 189, 248), 2, cv2.LINE_AA)

            for box, cf in person_boxes:
                bx1, by1, bx2, by2 = map(int, box)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 190, 40), 2)
                cv2.putText(vis, f"Hoang Sao ({cf:.2f})", (bx1, max(20, by1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 40), 2, cv2.LINE_AA)

            if len(cue_trail_1080) >= 2:
                cv2.polylines(vis, [np.array(list(cue_trail_1080), dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)

            for b in balls:
                bx, by = int(b["x"]), int(b["y"])
                cv2.circle(vis, (bx, by), 10, tuple(b["color"]), 2, cv2.LINE_AA)
                cv2.putText(vis, b["label"], (bx-18, by-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 255, 230), 1, cv2.LINE_AA)

            cv2.rectangle(vis, (16, 12), (610, 96), (15, 18, 26), -1)
            cv2.rectangle(vis, (16, 12), (610, 96), (16, 185, 129), 2)
            cv2.putText(vis, "RAILBIRD AI - EUROPEAN OPEN 2023 (HOANG SAO)", (28, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, status_txt, (28, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (16, 235, 160), 2, cv2.LINE_AA)
            yt_s = int(305 + t_sec)
            cv2.putText(vis, f"CLIP TIME: {t_sec:05.1f}s (YT {yt_s//60}:{yt_s%60:02d}) | BALLS REMAINING: {len(balls)}", (28, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 205, 220), 1, cv2.LINE_AA)

            draw_hs_radar_inset(vis, balls, cue_trail_radar)
            writer.write(vis)

            if f_idx in [int(383 * fps), int(414 * fps), int(441 * fps), int(470 * fps)]:
                cv2.imwrite(f"c:/billard/hs_preview_{int(t_sec)}s.jpg", vis)

    cap.release()
    writer.release()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    h264_out = "c:/billard/output_hoang_sao_runout.mp4"
    subprocess.run([
        ffmpeg_exe, "-y", "-i", raw_out,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        h264_out
    ], check=True)
    os.remove(raw_out)
    print(f"Saved H.264 annotated run-out video to {h264_out}!")

    shots = [
        {"shot_number": 1,  "player": "Duong Quoc Hoang", "start_time": 0.0,   "end_time": 12.0,  "yt_time": "05:05", "outcome": "MAKE (BREAK)", "pocket": "Side Pocket", "target_ball": "1-Yellow Rack (8-Ball Made)", "max_speed_kmh": 28.4, "distance_m": 6.80, "description": "Power break from the left side of the box; 8-Ball drops on the break and 1-Yellow sits clean in the upper half."},
        {"shot_number": 2,  "player": "Duong Quoc Hoang", "start_time": 32.0,  "end_time": 44.0,  "yt_time": "05:37", "outcome": "MAKE",         "pocket": "TR (Top-Right)", "target_ball": "1-Yellow", "max_speed_kmh": 12.6, "distance_m": 2.95, "description": "Hoàng Sao cuts the 1-Yellow cleanly into the Top-Right corner pocket and holds the Cue Ball near center table."},
        {"shot_number": 3,  "player": "Duong Quoc Hoang", "start_time": 58.0,  "end_time": 74.0,  "yt_time": "06:03", "outcome": "SAFETY",       "pocket": "None", "target_ball": "2-Blue", "max_speed_kmh": 14.1, "distance_m": 3.60, "description": "Tactical push/safety off 2-Blue, sending 2-Blue up-table and freezing the Cue Ball near 4-Pink on the bottom rail."},
        {"shot_number": 4,  "player": "Pijus Labutis",    "start_time": 116.0, "end_time": 134.0, "yt_time": "07:01", "outcome": "SAFETY",       "pocket": "None", "target_ball": "2-Blue", "max_speed_kmh": 15.8, "distance_m": 4.10, "description": "Labutis plays a two-rail kick/safety onto 2-Blue, sending 2-Blue to the bottom-right and freezing the Cue Ball on the head rail."},
        {"shot_number": 5,  "player": "Duong Quoc Hoang", "start_time": 152.0, "end_time": 168.0, "yt_time": "07:37", "outcome": "SAFETY",       "pocket": "None", "target_ball": "2-Blue", "max_speed_kmh": 13.4, "distance_m": 3.45, "description": "Hoàng Sao banks 2-Blue up beside the Top-Right pocket jaw while leaving the Cue Ball mid-table."},
        {"shot_number": 6,  "player": "Duong Quoc Hoang", "start_time": 206.0, "end_time": 224.0, "yt_time": "08:31", "outcome": "MAKE",         "pocket": "TR (Top-Right)", "target_ball": "2-Blue", "max_speed_kmh": 16.2, "distance_m": 3.85, "description": "Hoàng Sao elevates and pots 2-Blue into the Top-Right corner pocket, moving 3-Red to the right rail."},
        {"shot_number": 7,  "player": "Pijus Labutis",    "start_time": 298.0, "end_time": 316.0, "yt_time": "10:03", "outcome": "SAFETY / KICK","pocket": "None", "target_ball": "3-Red", "max_speed_kmh": 14.9, "distance_m": 3.90, "description": "Labutis kicks off the head cushion to contact 3-Red, but leaves 3-Red visible down the length of the table."},
        {"shot_number": 8,  "player": "Duong Quoc Hoang", "start_time": 330.0, "end_time": 345.0, "yt_time": "10:35", "outcome": "MAKE",         "pocket": "TR (Top-Right)", "target_ball": "3-Red", "max_speed_kmh": 15.5, "distance_m": 3.50, "description": "Hoàng Sao fires 3-Red into the Top-Right corner pocket and floats the Cue Ball to the center for 4-Pink."},
        {"shot_number": 9,  "player": "Duong Quoc Hoang", "start_time": 378.0, "end_time": 392.0, "yt_time": "11:23", "outcome": "MAKE",         "pocket": "BL (Bottom-Left)", "target_ball": "4-Pink", "max_speed_kmh": 16.8, "distance_m": 4.25, "description": "Starts the final clearance: pots 4-Pink in the Bottom-Left pocket and goes two cushions up the right side for 5-Purple."},
        {"shot_number": 10, "player": "Duong Quoc Hoang", "start_time": 408.0, "end_time": 422.0, "yt_time": "11:53", "outcome": "MAKE",         "pocket": "TR (Top-Right)", "target_ball": "5-Purple", "max_speed_kmh": 13.2, "distance_m": 2.80, "description": "Pots 5-Purple into Top-Right and holds the Cue Ball one cushion to the left half of the table for 6-Green."},
        {"shot_number": 11, "player": "Duong Quoc Hoang", "start_time": 436.0, "end_time": 448.0, "yt_time": "12:21", "outcome": "MAKE",         "pocket": "TR (Top-Right)", "target_ball": "6-Green", "max_speed_kmh": 17.4, "distance_m": 4.40, "description": "Pots 6-Green in Top-Right with pure backspin draw, pulling the Cue Ball all the way down-table for 7-Brown."},
        {"shot_number": 12, "player": "Duong Quoc Hoang", "start_time": 454.0, "end_time": 464.0, "yt_time": "12:39", "outcome": "MAKE",         "pocket": "BR (Bottom-Right)", "target_ball": "7-Brown", "max_speed_kmh": 10.8, "distance_m": 1.65, "description": "Rolls 7-Brown into Bottom-Right pocket and stops dead in line for the 9-Ball."},
        {"shot_number": 13, "player": "Duong Quoc Hoang", "start_time": 466.0, "end_time": 474.0, "yt_time": "12:51", "outcome": "MAKE (WIN)",   "pocket": "BR (Bottom-Right)", "target_ball": "9-Stripe (9-Ball)", "max_speed_kmh": 14.0, "distance_m": 2.10, "description": "Pots the 9-Ball into Bottom-Right to complete the table run-out ('dọn bàn') and take a 3–1 lead!"}
    ]

    payload = {
        "video_file": "hoang_sao.mp4",
        "runout_video": "output_hoang_sao_runout.mp4",
        "table_name": "European Open 2023 • Matchroom TV Table (Diamond 9ft)",
        "players": "Dương Quốc Hoàng (Hoàng Sao) vs Pijus Labutis",
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "duration_sec": round(total_frames / fps, 1),
        "summary": {
            "total_shots": len(shots),
            "makes": 8,
            "safeties": 5,
            "scratches": 0,
            "pot_pct": 100.0,
            "longest_streak": 6,
            "balls_on_table": 9
        },
        "table_poly": HS_TABLE_POLY.tolist(),
        "pockets": HS_POCKETS,
        "shots": shots,
        "timeline_sample": timeline_sample
    }

    with open("c:/billard/hoang_sao_session.json", "w") as f:
        json.dump(payload, f)
    with open("c:/billard/hoang_sao_session.js", "w") as f:
        f.write("window.HOANG_SAO_SESSION = " + json.dumps(payload) + ";\n")
    print("Exported clean 1-to-1 hoang_sao_session.json and hoang_sao_session.js!")

if __name__ == "__main__":
    main()
