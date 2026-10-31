---
description: Compact Codebase Graph & Memory Map for Railbird AI Clone (saves context tokens)
---
# Railbird AI Clone — Project Graph & Memory (`c:\billard`)

## 1. Core Pipeline Graph
- `test_hs_detector.py` → Matchroom 9ft Diamond Table homography (`HS_TABLE_POLY`, `HS_POCKETS`, `RADAR_W=1000, RADAR_H=500`), felt-ring surround filter (`ANNULUS_MASK`), and `lap.lapjv` 1-to-1 Hungarian ball assignment (`detect_hs_balls`).
- `process_hoang_sao.py` → Processes `hoang_sao.mp4` (`1280x720` @ `29.97 FPS`, European Open 2023), tracks players with `YOLO("yolo26n.pt")`, exports `output_hoang_sao_runout.mp4`, `hoang_sao_session.json`, and `hoang_sao_session.js` (`window.HOANG_SAO_SESSION`).
- `railbird_engine.py` → Processes club CCTV `SAMPLE.mp4` (`1920x1080` @ `30 FPS`), exports `output_railbird.mp4`, `clips/shot_1_scratch.mp4`, `clips/shot_2_miss.mp4`, `session_data.json`, and `session_data.js` (`window.RAILBIRD_SESSION`).
- `train_pool_detector.py` → Builds table-cropped YOLO dataset (`datasets/railbird_table_balls/data.yaml`) and trains `pretrained_models/railbird_ball_best.pt` on local GTX 1650 (`CUDA 12.4`).
- `qa_benchmark.py` → Automated multi-clip QA validator (`qa_report.json`): checks Uniqueness (`0%` duplicates), Flicker Stability, No-Teleport velocity bounds, Camera-Cut Gatekeeper (`is_main_table_camera_hs`), and Matchroom Scoreboard Ground Truth.
- `shot_rating_engine.py` → 2D vector geometry engine computing `cut_angle_deg`, `margin_deg`, `difficulty_score` (`0-100`), `grade` (`S/A/B/C/F`), `is_highlight`, and `aim_line` (`cb_rx, cb_ry, gb_rx, gb_ry, pock_rx, pock_ry`).
- `index.html` → Standalone Dual-Session Railbird Web App consuming `hoang_sao_session.js`, `session_data.js`, and `qa_report.json`.
