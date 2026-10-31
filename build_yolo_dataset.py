import os
import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset, visualize_image_annotations

# Import our calibrated detector
import sys
sys.path.append('C:/Users/tiena/.gemini/antigravity/brain/45b1672c-afe1-4c24-8c97-aa8c4c6e2c37/scratch')
from test_detector import detect_and_classify_balls, table_poly

# Table crop region in 1920x1080
CROP_X0, CROP_Y0 = 380, 270
CROP_X1, CROP_Y1 = 1320, 780
CROP_W = CROP_X1 - CROP_X0
CROP_H = CROP_Y1 - CROP_Y0

CLASS_MAP = {
    'Cue Ball': 0,
    '8-Ball': 1,
    '2-Blue': 2,
    '3-Red': 3,
    '4-Pink': 4,
    '5-Orange': 5,
    '6-Green': 6,
    '9-Yellow': 7,
    '10-BlueStripe': 2,
    'Object Ball': 2,
}

CLASS_NAMES = {
    0: 'cue_ball',
    1: 'eight_ball',
    2: 'blue_ball',
    3: 'red_ball',
    4: 'pink_ball',
    5: 'orange_ball',
    6: 'green_ball',
    7: 'yellow_ball',
}

ds_root = 'c:/billard/datasets/pool_balls'
for split in ['train', 'val']:
    os.makedirs(f'{ds_root}/images/{split}', exist_ok=True)
    os.makedirs(f'{ds_root}/labels/{split}', exist_ok=True)

cap = cv2.VideoCapture('c:/billard/SAMPLE.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Sample 40 frames across the video: first 32 for train, last 8 for val
frame_indices = np.linspace(0, total_frames - 10, 40, dtype=int)

for idx, f_no in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_no))
    ok, frame = cap.read()
    if not ok:
        continue
    balls = detect_and_classify_balls(frame)
    crop = frame[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    split = 'train' if idx < 32 else 'val'
    stem = f'frame_{f_no:04d}'
    img_path = f'{ds_root}/images/{split}/{stem}.jpg'
    lbl_path = f'{ds_root}/labels/{split}/{stem}.txt'
    cv2.imwrite(img_path, crop)

    lines = []
    for b in balls:
        bx = b['x'] - CROP_X0
        by = b['y'] - CROP_Y0
        if 0 <= bx < CROP_W and 0 <= by < CROP_H:
            cls_id = CLASS_MAP.get(b['label'], 2)
            # Ball radius is ~10px -> box 22x22px
            cx = np.clip(bx / CROP_W, 0.0, 1.0)
            cy = np.clip(by / CROP_H, 0.0, 1.0)
            bw = 22.0 / CROP_W
            bh = 22.0 / CROP_H
            lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    with open(lbl_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

cap.release()

yaml_content = f"""path: {ds_root}
train: images/train
val: images/val
names:
  0: cue_ball
  1: eight_ball
  2: blue_ball
  3: red_ball
  4: pink_ball
  5: orange_ball
  6: green_ball
  7: yellow_ball
"""

yaml_path = f'{ds_root}/data.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

info = check_det_dataset(yaml_path)
print('Dataset validated successfully! Classes:', info['names'])
