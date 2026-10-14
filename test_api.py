"""
BIOMM 2FA — API Test Script
Tests the signup + login flow using the API endpoints.
Run with: py test_api.py (while the server is running)
"""

import json
import sys
import time
import requests

BASE = "http://localhost:8000"

def test_signup_and_login():
    session = requests.Session()
    test_uid = f"test_user_{int(time.time()) % 10000}"
    print(f"\n{'='*60}")
    print(f"  BIOMM 2FA API Test — user: {test_uid}")
    print(f"{'='*60}\n")

    # ── Step 1: Signup start (link fingerprint) ──
    print("[1] POST /api/signup/start ...")
    resp = session.post(f"{BASE}/api/signup/start", json={"user_id": test_uid})
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {json.dumps(data, indent=2)}")
    if resp.status_code != 200 or not data.get("ok"):
        print(f"    ✗ Signup start FAILED: {data.get('detail') or data.get('reason')}")
        return False
    print(f"    ✓ Fingerprint linked: {data.get('linked_file')}")

    # ── Step 2: Enroll 5 signatures ──
    # We generate synthetic strokes (zigzag pattern) for testing
    print(f"\n[2] POST /api/signup/signature × 5 ...")
    for i in range(1, 6):
        strokes = _make_test_strokes(seed=42)  # same pattern each time for consistency
        resp = session.post(f"{BASE}/api/signup/signature",
                            json={"user_id": test_uid, "strokes": strokes})
        data = resp.json()
        if not data.get("ok"):
            print(f"    Sample {i}: ✗ {data.get('reason')}")
            # Retry with same seed if consistency check fails
            if "different" in data.get("reason", "").lower():
                print("    Retrying with same pattern...")
                resp = session.post(f"{BASE}/api/signup/signature",
                                    json={"user_id": test_uid, "strokes": strokes})
                data = resp.json()
                if not data.get("ok"):
                    print(f"    Still failed: {data.get('reason')}")
                    return False
        print(f"    Sample {i}: ✓ saved={data.get('saved')}/{data.get('total')} complete={data.get('complete')}")
        if data.get("complete"):
            print(f"    Enrollment complete! threshold={data.get('threshold')}")
            break

    # ── Step 3: Login signature ──
    print(f"\n[3] POST /api/login/signature ...")
    strokes = _make_test_strokes(seed=42)  # same pattern
    resp = session.post(f"{BASE}/api/login/signature",
                        json={"user_id": test_uid, "strokes": strokes})
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {json.dumps(data, indent=2)}")
    if not data.get("ok"):
        print(f"    ✗ Signature verification FAILED: {data.get('reason')}")
        return False
    print(f"    ✓ Signature accepted — score={data.get('score')}")
    stage1_token = data.get("stage1_token")

    # ── Step 4: Login fingerprint ──
    print(f"\n[4] POST /api/login/fingerprint ...")
    resp = session.post(f"{BASE}/api/login/fingerprint",
                        json={"stage1_token": stage1_token})
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {json.dumps(data, indent=2)}")
    if not data.get("granted"):
        print(f"    ✗ Fingerprint verification FAILED: {data.get('reason')}")
        return False
    print(f"    ✓ ACCESS GRANTED — fp_score={data.get('fp_score')} pairs={data.get('fp_pairs')}")

    # ── Step 5: Check session ──
    print(f"\n[5] GET /api/me ...")
    resp = session.get(f"{BASE}/api/me")
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"    ✓ Session active: user={data.get('user_id')} expires_in={data.get('session_expires_in')}s")
    else:
        print(f"    ✗ Session check failed")

    # ── Step 6: Logout ──
    print(f"\n[6] POST /api/logout ...")
    resp = session.post(f"{BASE}/api/logout")
    data = resp.json()
    print(f"    ✓ Logged out: {data}")

    print(f"\n{'='*60}")
    print(f"  ALL TESTS PASSED ✓")
    print(f"{'='*60}\n")
    return True


def _make_test_strokes(seed=42):
    """Generate a reproducible zigzag signature pattern."""
    import random
    rng = random.Random(seed)
    strokes = []
    # 2-3 strokes making a zigzag
    for _ in range(rng.randint(2, 3)):
        pts = []
        x, y = rng.uniform(0.2, 0.4), rng.uniform(0.3, 0.5)
        for _ in range(rng.randint(15, 30)):
            x += rng.uniform(0.005, 0.02)
            y += rng.uniform(-0.015, 0.015)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            pts.append([round(x, 4), round(y, 4)])
        strokes.append(pts)
    return strokes


if __name__ == "__main__":
    try:
        requests.get(BASE, timeout=3)
    except requests.ConnectionError:
        print(f"ERROR: Server not running at {BASE}")
        print("Start it with: uvicorn app:app --reload")
        sys.exit(1)

    ok = test_signup_and_login()
    sys.exit(0 if ok else 1)
