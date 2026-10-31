"""
Railbird AI - Download Monitor
Monitors C:\\Users\\tiena\\Downloads recursively (including subfolders like 'F1 22 [FitGirl Repack] rar')
for best_pool_1280.pt, best*.pt, or active *.crdownload from Google Colab.
When download completes, automatically copies to c:\\billard\\pretrained_models\\best_pool_1280.pt.
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path

DOWNLOADS_DIR = Path(os.environ.get("USERPROFILE", "C:/Users/tiena")) / "Downloads"
TARGET_DIR = Path("c:/billard/pretrained_models")
STATUS_FILE = Path("c:/billard/DOWNLOAD_STATUS.json")

def check_download():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    
    # Search root and all subdirectories (including 'F1 22 [FitGirl Repack] rar', etc.)
    # 1. Check for active downloading .crdownload files (modified within last 2 hours)
    crdownloads = []
    for pattern in ["*.crdownload", "*pool*.crdownload", "best*.crdownload"]:
        for f in DOWNLOADS_DIR.rglob(pattern):
            if (now - f.stat().st_mtime) < 7200:
                crdownloads.append(f)
                
    if crdownloads:
        crdownloads.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        active = crdownloads[0]
        size_mb = active.stat().st_size / (1024 * 1024)
        return {
            "status": "downloading",
            "file": active.name,
            "path": str(active),
            "size_mb": round(size_mb, 2),
            "message": f"Downloading in progress: {active.name} in {active.parent.name} ({size_mb:.1f} MB so far)"
        }
        
    # 2. Check for completed .pt files (modified within last 4 hours)
    candidates = []
    patterns = ["best_pool_1280*.pt", "best*.pt", "*pool*.pt", "*.pt"]
    for p in patterns:
        for f in DOWNLOADS_DIR.rglob(p):
            # Ignore unrelated pytorch files if any, unless best or pool
            fname = f.name.lower()
            if "best" in fname or "pool" in fname or "yolo" in fname:
                if (now - f.stat().st_mtime) < 14400: # 4 hours
                    candidates.append(f)
                    
    # Deduplicate by path
    candidates = list({str(c): c for c in candidates}.values())
    
    if candidates:
        # Sort by most recently modified
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        chosen = candidates[0]
        size_mb = chosen.stat().st_size / (1024 * 1024)
        
        # Verify file is not still being written to disk
        s0 = chosen.stat().st_size
        time.sleep(1.0)
        s1 = chosen.stat().st_size
        if s0 != s1 or s1 < 10000: # minimum valid .pt model is > 1MB
            return {
                "status": "downloading",
                "file": chosen.name,
                "path": str(chosen),
                "size_mb": round(size_mb, 2),
                "message": f"File is currently being written to disk: {chosen.name} ({size_mb:.1f} MB)"
            }
            
        # Copy to pretrained_models
        dest = TARGET_DIR / "best_pool_1280.pt"
        shutil.copy2(chosen, dest)
        
        info = {
            "status": "completed",
            "source": str(chosen),
            "destination": str(dest),
            "size_mb": round(size_mb, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"SUCCESS! {chosen.name} ({size_mb:.1f} MB) found in '{chosen.parent.name}', installed to {dest}"
        }
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return info
        
    return {
        "status": "waiting",
        "search_roots": [
            str(DOWNLOADS_DIR),
            str(DOWNLOADS_DIR / "F1 22 [FitGirl Repack] rar")
        ],
        "message": "Recursive watcher active across Downloads and all subfolders. No .pt or .crdownload detected yet."
    }

if __name__ == "__main__":
    result = check_download()
    print(json.dumps(result, indent=2))
