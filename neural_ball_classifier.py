"""
Railbird AI - Stage-2 Neural Ball Identity Classifier Helper
Provides fast GPU inference for 32x32 ball crops alongside railbird_ball_best.pt and lap.lapjv.
"""

import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

CLASSES = [
    "Cue_Ball",
    "1_Yellow",
    "2_Blue",
    "3_Red",
    "4_Pink",
    "5_Purple",
    "5_Orange",
    "6_Green",
    "7_Brown",
    "8_Ball",
    "9_Stripe",
    "10_BlueStripe"
]

DISPLAY_NAMES = {
    "Cue_Ball": "Cue Ball",
    "1_Yellow": "1-Yellow",
    "2_Blue": "2-Blue",
    "3_Red": "3-Red",
    "4_Pink": "4-Pink",
    "5_Purple": "5-Purple",
    "5_Orange": "5-Orange",
    "6_Green": "6-Green",
    "7_Brown": "7-Brown",
    "8_Ball": "8-Ball",
    "9_Stripe": "9-Stripe",
    "10_BlueStripe": "10-BlueStripe"
}

BALL_COLORS_BGR = {
    "Cue_Ball": [255, 255, 255],
    "1_Yellow": [0, 245, 255],
    "2_Blue": [255, 180, 60],
    "3_Red": [60, 60, 240],
    "4_Pink": [220, 160, 240],
    "5_Purple": [220, 80, 140],
    "5_Orange": [40, 150, 255],
    "6_Green": [100, 220, 80],
    "7_Brown": [40, 80, 180],
    "8_Ball": [40, 40, 40],
    "9_Stripe": [40, 220, 255],
    "10_BlueStripe": [240, 80, 20]
}

DEFAULT_WEIGHTS = "c:/billard/pretrained_models/railbird_ball_cls.pt"

_MODEL_CACHE = None
_DEVICE_CACHE = None


class BallClassifierCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        out = self.classifier(feat)
        return out


def get_neural_classifier(weights_path=DEFAULT_WEIGHTS):
    """Returns cached BallClassifierCNN model on GPU/CPU."""
    global _MODEL_CACHE, _DEVICE_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _DEVICE_CACHE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BallClassifierCNN(num_classes=len(CLASSES))

    if os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"[AI] Loaded neural ball classifier from {weights_path} onto {device}")
        except Exception as e:
            print(f"[WARN] Failed loading weights from {weights_path}: {e}")
    else:
        print(f"[WARN] Weights {weights_path} not found. Running with initialized weights.")

    model.to(device)
    model.eval()
    _MODEL_CACHE = model
    _DEVICE_CACHE = device
    return _MODEL_CACHE, _DEVICE_CACHE


def extract_single_crop(frame, cx, cy, crop_size=32):
    """Extracts a 32x32 RGB crop around (cx, cy)."""
    h_img, w_img = frame.shape[:2]
    half = crop_size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)

    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(w_img, x2)
    y2_c = min(h_img, y2)

    crop = frame[y1_c:y2_c, x1_c:x2_c]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop_rgb


_NORM_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_ball_crops_neural(img, centers, weights_path=DEFAULT_WEIGHTS):
    """
    Classifies a list of ball centers in img (BGR numpy array).
    centers: list of (cx, cy) tuples or dicts with 'x' and 'y' (or 'cx' and 'cy').
    Returns: list of classified ball dictionaries.
    """
    if not centers or img is None or img.size == 0:
        return []

    model, device = get_neural_classifier(weights_path)

    # Normalize input center coordinates
    pts = []
    for c in centers:
        if isinstance(c, (tuple, list)):
            pts.append((float(c[0]), float(c[1])))
        elif isinstance(c, dict):
            cx = float(c.get("x", c.get("cx", 0.0)))
            cy = float(c.get("y", c.get("cy", 0.0)))
            pts.append((cx, cy))

    # Build crop batch
    crop_tensors = []
    for cx, cy in pts:
        crop_rgb = extract_single_crop(img, cx, cy, crop_size=32)
        pil_img = Image.fromarray(crop_rgb)
        t = _NORM_TRANSFORM(pil_img)
        crop_tensors.append(t)

    batch = torch.stack(crop_tensors, dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    preds_np = preds.cpu().numpy()
    confs_np = confidences.cpu().numpy()

    classified_balls = []
    for idx, (cx, cy) in enumerate(pts):
        cls_idx = int(preds_np[idx])
        conf = float(confs_np[idx])
        cls_name = CLASSES[cls_idx]
        disp_name = DISPLAY_NAMES.get(cls_name, cls_name)
        color = BALL_COLORS_BGR.get(cls_name, [255, 255, 255])

        classified_balls.append({
            "label": disp_name,
            "raw_class": cls_name,
            "class_idx": cls_idx,
            "confidence": round(conf, 3),
            "x": round(cx, 1),
            "y": round(cy, 1),
            "color": color
        })

    return classified_balls
