import os
import yaml
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from test_hs_detector import HS_BBOX, HS_ROI, detect_hs_balls, get_active_object_balls
from railbird_engine import TABLE_BBOX, detect_balls_in_frame
from qa_benchmark import is_main_table_camera_hs

DATASET_DIR = "c:/billard/datasets/railbird_table_balls"

def build_table_crop_dataset():
    for split in ["train", "val"]:
        os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)

    train_count = 0
    val_count = 0

    # 1. Extract verified table crops from Hoang Sao (European Open 2023 - Grey Felt)
    cap_hs = cv2.VideoCapture("c:/billard/hoang_sao.mp4")
    fps_hs = cap_hs.get(cv2.CAP_PROP_FPS) or 29.97
    n_hs = int(cap_hs.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, wb, hb = HS_BBOX

    for f_idx in range(0, n_hs, 45):
        cap_hs.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap_hs.read()
        if not ok:
            break
        t_sec = f_idx / fps_hs
        is_main, _ = is_main_table_camera_hs(frame)
        if not is_main:
            continue
        balls = detect_hs_balls(frame, t_sec=t_sec)
        expected_n = len(get_active_object_balls(t_sec)) + 1
        # Only use high-confidence frames where ball count matches Matchroom scoreboard
        if len(balls) < 2 or abs(len(balls) - expected_n) > 1:
            continue

        crop = frame[y0:y0+hb, x0:x0+wb]
        split = "train" if (f_idx % 225 != 0) else "val"
        fname = f"hs_{f_idx:05d}"
        cv2.imwrite(f"{DATASET_DIR}/images/{split}/{fname}.jpg", crop)

        lines = []
        for b in balls:
            cx = (b["x"] - x0) / float(wb)
            cy = (b["y"] - y0) / float(hb)
            bw = 20.0 / float(wb)
            bh = 20.0 / float(hb)
            if 0.01 < cx < 0.99 and 0.01 < cy < 0.99:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(f"{DATASET_DIR}/labels/{split}/{fname}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        if split == "train":
            train_count += 1
        else:
            val_count += 1
    cap_hs.release()

    # 2. Extract verified table crops from SAMPLE.mp4 (Club CCTV - Blue Felt)
    cap_s = cv2.VideoCapture("c:/billard/SAMPLE.mp4")
    n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    sx0, sy0, swb, shb = TABLE_BBOX

    for f_idx in range(0, n_s, 10):
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap_s.read()
        if not ok:
            break
        balls = detect_balls_in_frame(frame)
        if len(balls) < 5:
            continue
        crop = frame[sy0:sy0+shb, sx0:sx0+swb]
        split = "train" if (f_idx % 50 != 0) else "val"
        fname = f"club_{f_idx:05d}"
        cv2.imwrite(f"{DATASET_DIR}/images/{split}/{fname}.jpg", crop)

        lines = []
        for b in balls:
            cx = (b["x"] - sx0) / float(swb)
            cy = (b["y"] - sy0) / float(shb)
            bw = 22.0 / float(swb)
            bh = 22.0 / float(shb)
            if 0.01 < cx < 0.99 and 0.01 < cy < 0.99:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(f"{DATASET_DIR}/labels/{split}/{fname}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        if split == "train":
            train_count += 1
        else:
            val_count += 1
    cap_s.release()

    yaml_path = f"{DATASET_DIR}/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path": DATASET_DIR,
            "train": "images/train",
            "val": "images/val",
            "names": {0: "ball"}
        }, f, sort_keys=False)

    print(f"Built unified table-crop dataset: {train_count} train, {val_count} val images -> {yaml_path}")
    return yaml_path

def train_and_validate():
    yaml_path = build_table_crop_dataset()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device={device} (CUDA available={torch.cuda.is_available()})")

    # Fine-tune starting from pre-trained jchadwick_billiards_yolov8n.pt or yolo26n.pt
    base_weights = "c:/billard/pretrained_models/jchadwick_billiards_yolov8n.pt"
    if not os.path.exists(base_weights):
        base_weights = "yolo26n.pt"

    model = YOLO(base_weights)
    results = model.train(
        data=yaml_path,
        epochs=25,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        project="c:/billard/runs/detect",
        name="railbird_ball_v2",
        exist_ok=True,
        verbose=True,
    )

    best_pt = "c:/billard/runs/detect/railbird_ball_v2/weights/best.pt"
    target_pt = "c:/billard/pretrained_models/railbird_ball_best.pt"
    if os.path.exists(best_pt):
        import shutil
        shutil.copy2(best_pt, target_pt)
        print(f"Saved fine-tuned model to {target_pt}")

        val_model = YOLO(target_pt)
        metrics = val_model.val(data=yaml_path, imgsz=640, device=device)
        print(f"Validation mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    train_and_validate()
