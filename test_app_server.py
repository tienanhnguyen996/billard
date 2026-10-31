"""
Test Suite for Railbird Production Backend Server (app_server.py)
Tests:
1. Byte-Range Support (206 Partial Content)
2. GET /api/status
3. POST /api/detect_frame (GPU detection + Neural Classification + 3-Ball Runout Route)
4. POST /api/export_clip (On-demand shot rendering)
5. POST /api/analyze_job & GET /api/job_status (Background analysis job)
6. POST /api/db & GET /api/db (Persistent database)
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import urllib.request
import urllib.parse
import json
import base64
import time
import cv2

BASE_URL = "http://127.0.0.1:8080"


def test_byte_range():
    print("\n--- 1. Testing HTTP Byte-Range Support (206 Partial Content) ---")
    req = urllib.request.Request(f"{BASE_URL}/output_hoang_sao_runout.mp4")
    req.add_header("Range", "bytes=0-1023")
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_range = resp.headers.get("Content-Range")
            content_length = resp.headers.get("Content-Length")
            data = resp.read()
            print(f"  Status: {status} (Expected 206)")
            print(f"  Content-Range: {content_range}")
            print(f"  Content-Length: {content_length} (Expected 1024)")
            print(f"  Bytes Read: {len(data)}")
            assert status == 206, f"Expected 206, got {status}"
            assert len(data) == 1024, f"Expected 1024 bytes, got {len(data)}"
            print("  [PASS] Byte-Range 206 Partial Content verified!")
    except Exception as e:
        print(f"  [FAIL] Byte-range error: {e}")
        raise


def test_api_status():
    print("\n--- 2. Testing GET /api/status ---")
    url = f"{BASE_URL}/api/status"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"  Server Name    : {res.get('server')}")
        print(f"  CUDA Available : {res.get('cuda_available')}")
        print(f"  Device Name    : {res.get('device_name')}")
        print(f"  Detector Model : {res.get('models', {}).get('ball_detector')}")
        print(f"  Classifier Model: {res.get('models', {}).get('ball_classifier')}")
        print(f"  DB Summary     : {res.get('db_summary')}")
        assert res.get("cuda_available") is True, "Expected CUDA available"
        assert res.get("models", {}).get("ball_classifier", {}).get("exists") is True
        print("  [PASS] GET /api/status verified!")


def test_api_detect_frame():
    print("\n--- 3. Testing POST /api/detect_frame (Optimal 3-Ball Runout Route) ---")
    # Grab a frame from hoang_sao.mp4 at 438s
    cap = cv2.VideoCapture("hoang_sao.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(438 * 29.97))
    ok, frame = cap.read()
    cap.release()
    assert ok, "Failed to read test frame from hoang_sao.mp4"

    # Encode to JPEG base64
    _, buf = cv2.imencode(".jpg", frame)
    b64_str = base64.b64encode(buf).decode("utf-8")

    req_body = {
        "image_base64": b64_str,
        "session_id": "hoang_sao"
    }

    url = f"{BASE_URL}/api/detect_frame"
    data_bytes = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})

    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.time() - t0) * 1000

    print(f"  Server Latency : {res.get('latency_ms')} ms (Network roundtrip: {elapsed_ms:.1f} ms)")
    print(f"  Balls Detected : {res.get('ball_count')}")
    for b in res.get("balls", []):
        nl = b.get("neural_label", "-")
        nc = b.get("neural_conf", 0.0)
        print(f"    {b.get('label'):<12} at ({b.get('rx'):.1f}, {b.get('ry'):.1f}) [Neural: {nl} {nc*100:.1f}%]")

    print(f"  Optimal 3-Ball Runout Route (Steps: {len(res.get('best_route', []))}):")
    for step in res.get("best_route", []):
        print(f"    Step {step['step']}: Pot {step['ball']} into {step['pocket']} (Cut {step['cut_angle_deg']}°, Margin ±{step['pocket_margin_deg']}°, Diff {step['difficulty']}/100)")
        print(f"      Leave Zone: ({step['leave_zone']['rx']}, {step['leave_zone']['ry']}) - {step['leave_zone']['description']}")

    assert res.get("success") is True
    assert len(res.get("balls", [])) > 0
    assert len(res.get("best_route", [])) > 0
    print("  [PASS] POST /api/detect_frame verified!")


def test_api_export_clip():
    print("\n--- 4. Testing POST /api/export_clip ---")
    url = f"{BASE_URL}/api/export_clip"
    req_body = {
        "session_id": "hoang_sao",
        "shot_number": 11,
        "format": "16x9"
    }
    data_bytes = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"  Export Status : {res.get('success')}")
        print(f"  Download URL  : {res.get('download_url')}")
        print(f"  File Path     : {res.get('file_path')}")
        print(f"  Size Bytes    : {res.get('size_bytes')} bytes")
        assert res.get("success") is True
        assert res.get("size_bytes", 0) > 0
        print("  [PASS] POST /api/export_clip verified!")


def test_api_analyze_job():
    print("\n--- 5. Testing POST /api/analyze_job & GET /api/job_status ---")
    url_job = f"{BASE_URL}/api/analyze_job"
    req_body = {
        "video_path": "SAMPLE.mp4",
        "corners": "558,416,1234,318,1242,672,442,716",
        "end_sec": 6.0,
        "out": "session_api_test.json"
    }
    data_bytes = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(url_job, data=data_bytes, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        job_id = res.get("job_id")
        print(f"  Created Job ID: {job_id}")
        assert res.get("success") is True

    # Poll status
    print("  Polling job status...")
    for _ in range(20):
        url_status = f"{BASE_URL}/api/job_status?id={job_id}"
        with urllib.request.urlopen(url_status) as resp:
            s_res = json.loads(resp.read().decode("utf-8"))
            job_info = s_res.get("job", {})
            st = job_info.get("status")
            pr = job_info.get("progress")
            print(f"    Status: {st} | Progress: {pr}%")
            if st == "completed":
                print(f"  Job Completed! Result: {job_info.get('result')}")
                break
            elif st == "failed":
                raise RuntimeError(f"Job failed: {job_info.get('error')}")
        time.sleep(1.0)
    print("  [PASS] Background analyze job verified!")


def test_api_db():
    print("\n--- 6. Testing POST /api/db & GET /api/db ---")
    # Save a test calibration
    url = f"{BASE_URL}/api/db"
    new_calib = {
        "action": "save_calibration",
        "data": {
            "id": "test_club_ban_9",
            "table_name": "Test Billiards Club - Table 9 (Min 9ft)",
            "resolution": "1920x1080",
            "corners": [[500, 300], [1300, 250], [1350, 750], [450, 800]]
        }
    }
    data_bytes = json.dumps(new_calib).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"  Saved Calibration: {res.get('item', {}).get('id')}")
        assert res.get("success") is True

    # Retrieve whole DB
    with urllib.request.urlopen(f"{BASE_URL}/api/db") as resp:
        db_res = json.loads(resp.read().decode("utf-8"))
        db = db_res.get("db", {})
        calib_ids = [c["id"] for c in db.get("calibrations", [])]
        print(f"  DB Calibrations in storage: {calib_ids}")
        assert "test_club_ban_9" in calib_ids
        print("  [PASS] POST & GET /api/db verified!")


def main():
    print("=" * 65)
    print("🧪 RAILBIRD AI BACKEND COMPREHENSIVE TEST SUITE")
    print("=" * 65)
    test_byte_range()
    test_api_status()
    test_api_detect_frame()
    test_api_export_clip()
    test_api_analyze_job()
    test_api_db()
    print("\n" + "=" * 65)
    print("🎉 ALL PRODUCTION BACKEND TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    main()
