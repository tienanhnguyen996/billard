"""
Railbird AI - Stage-2 Neural Ball Identity Classifier Training
1. Extracts 32x32 RGB ball crops from hoang_sao.mp4 and SAMPLE.mp4 based on verified session JSON labels.
2. Saves crops into datasets/ball_crops/{train,val}/{class_name}/.
3. Trains a fast PyTorch CNN on cuda:0 (GTX 1650) for 18 epochs.
4. Saves trained weights to pretrained_models/railbird_ball_cls.pt.
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import json
import random
import time
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

LABEL_MAP = {
    "Cue Ball": "Cue_Ball",
    "1-Yellow": "1_Yellow",
    "2-Blue": "2_Blue",
    "3-Red": "3_Red",
    "4-Pink": "4_Pink",
    "5-Purple": "5_Purple",
    "5-Orange": "5_Orange",
    "6-Green": "6_Green",
    "7-Brown": "7_Brown",
    "8-Ball": "8_Ball",
    "9-Stripe": "9_Stripe",
    "9-Yellow": "1_Yellow",
    "10-BlueStripe": "10_BlueStripe",
    "13-OrangeStripe": "5_Orange",
}

BASE_DIR = "c:/billard"
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "ball_crops")
MODELS_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODELS_DIR, exist_ok=True)


class BallClassifierCNN(nn.Module):
    """
    Lightweight, high-accuracy PyTorch CNN for 32x32 ball crops.
    Architecture:
    3 Conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool) + 2 Linear layers with Dropout.
    """
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 -> 4x4
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


def extract_ball_crop(frame, cx, cy, crop_size=32):
    """Extracts a square RGB crop centered at (cx, cy)."""
    h_img, w_img = frame.shape[:2]
    half = crop_size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Padding if near edge
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

    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop_rgb


def extract_all_crops():
    """Extracts crops across hoang_sao.mp4 and SAMPLE.mp4 and splits into train/val."""
    print("=" * 60)
    print("📸 Step 1: Extracting 32x32 Ball Crops from Session Ground Truth...")
    print("=" * 60)

    # Initialize directories
    for split in ["train", "val"]:
        for c in CLASSES:
            os.makedirs(os.path.join(DATASET_DIR, split, c), exist_ok=True)

    extracted_per_class = {c: [] for c in CLASSES}

    # 1. hoang_sao.mp4
    hs_json = os.path.join(BASE_DIR, "hoang_sao_session.json")
    hs_video = os.path.join(BASE_DIR, "hoang_sao.mp4")
    if os.path.exists(hs_json) and os.path.exists(hs_video):
        print("  Processing hoang_sao.mp4...")
        with open(hs_json, "r") as f:
            hs_data = json.load(f)
        samples = hs_data.get("timeline_sample", [])
        cap = cv2.VideoCapture(hs_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97

        # Sample every 5th sample (every 15 frames = ~0.5s) to get diverse angles/positions
        selected_samples = samples[::5]
        for s in selected_samples:
            f_idx = s["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            for b in s.get("balls", []):
                raw_lbl = b.get("label", "")
                mapped_cls = LABEL_MAP.get(raw_lbl)
                if mapped_cls and mapped_cls in extracted_per_class:
                    crop = extract_ball_crop(frame, b["x"], b["y"], crop_size=32)
                    extracted_per_class[mapped_cls].append(crop)
        cap.release()

    # 2. SAMPLE.mp4
    sample_json = os.path.join(BASE_DIR, "session_data.json")
    sample_video = os.path.join(BASE_DIR, "SAMPLE.mp4")
    if os.path.exists(sample_json) and os.path.exists(sample_video):
        print("  Processing SAMPLE.mp4...")
        with open(sample_json, "r") as f:
            sm_data = json.load(f)
        samples = sm_data.get("timeline_sample", [])
        cap = cv2.VideoCapture(sample_video)
        # Sample every 2nd sample
        selected_samples = samples[::2]
        for s in selected_samples:
            f_idx = s["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            for b in s.get("balls", []):
                raw_lbl = b.get("label", "")
                mapped_cls = LABEL_MAP.get(raw_lbl)
                if mapped_cls and mapped_cls in extracted_per_class:
                    crop = extract_ball_crop(frame, b["x"], b["y"], crop_size=32)
                    extracted_per_class[mapped_cls].append(crop)
        cap.release()

    # Save images to train/val with data balancing / augmentation
    total_train = 0
    total_val = 0
    print("\n  Class Distribution & Saving:")
    for c in CLASSES:
        crops = extracted_per_class[c]
        random.seed(42)
        random.shuffle(crops)

        # Ensure at least 60 samples per class via geometric rotation/flip augmentation
        if len(crops) < 60 and len(crops) > 0:
            aug_crops = []
            while len(crops) + len(aug_crops) < 60:
                for src in crops:
                    k = random.choice([1, 2, 3])
                    rot = np.rot90(src, k).copy()
                    if random.random() > 0.5:
                        rot = np.fliplr(rot).copy()
                    aug_crops.append(rot)
                    if len(crops) + len(aug_crops) >= 60:
                        break
            crops.extend(aug_crops)

        # 80/20 train/val split
        n_val = max(5, int(len(crops) * 0.20))
        val_crops = crops[:n_val]
        train_crops = crops[n_val:]

        for idx, img in enumerate(train_crops):
            p = os.path.join(DATASET_DIR, "train", c, f"{c}_{idx:04d}.jpg")
            Image.fromarray(img).save(p, quality=95)
        for idx, img in enumerate(val_crops):
            p = os.path.join(DATASET_DIR, "val", c, f"{c}_{idx:04d}.jpg")
            Image.fromarray(img).save(p, quality=95)

        total_train += len(train_crops)
        total_val += len(val_crops)
        print(f"    {c:<14}: {len(train_crops):>4} train | {len(val_crops):>3} val")

    print(f"\n  ✅ Total Extracted Crops: {total_train} train, {total_val} val ({total_train+total_val} total)")


class BallCropsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for c in CLASSES:
            c_dir = os.path.join(root_dir, c)
            if not os.path.exists(c_dir):
                continue
            idx = CLASS_TO_IDX[c]
            for f in os.listdir(c_dir):
                if f.endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(c_dir, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train_classifier():
    """Trains the BallClassifierCNN on GPU (cuda:0)."""
    print("\n" + "=" * 60)
    print("🚀 Step 2: Training PyTorch BallClassifierCNN on cuda:0 (GTX 1650)...")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = BallCropsDataset(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_ds = BallCropsDataset(os.path.join(DATASET_DIR, "val"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = BallClassifierCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)

    best_val_acc = 0.0
    best_weights_path = os.path.join(MODELS_DIR, "railbird_ball_cls.pt")

    epochs = 18
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = (correct / max(1, total)) * 100.0
        train_loss = total_loss / max(1, total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / max(1, val_total)) * 100.0

        if val_acc > best_val_acc or ep == epochs:
            best_val_acc = val_acc
            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "val_acc": val_acc,
                "epoch": ep
            }
            torch.save(checkpoint, best_weights_path)

        if ep % 3 == 0 or ep == epochs:
            print(f"  Epoch [{ep:>2}/{epochs}] | Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    dur = time.time() - t_start
    print(f"\n  ✅ Training completed in {dur:.1f}s! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  ✅ Saved model weights to: {best_weights_path} ({os.path.getsize(best_weights_path)/(1024*1024):.2f} MB)")
    return best_weights_path


def main():
    extract_all_crops()
    train_classifier()


if __name__ == "__main__":
    main()
