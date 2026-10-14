# BIOMM 2FA - FastAPI Backend
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from biometrics import (
    SIG_CFG, SYS, REGISTRY,
    sanitize_user, get_matcher, normalize_strokes,
    render_strokes, EnrollmentStore, signup_from_pool,
    load_signature_profiles, get_auto_capture, verify_fingerprint,
    on_access_granted,
)

app = FastAPI(title="BIOMM 2FA", docs_url="/docs")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Session store (in-memory) ──
_sessions: dict[str, dict] = {}
SESSION_MAX_AGE = 600  # 10 minutes

# Stage1 tokens (one-use, valid 2 min)
_stage1_tokens: dict[str, dict] = {}

# Enrollment state (in-memory per-user)
_enrollment_state: dict[str, dict] = {}


def _clean_sessions():
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v["last_active"] > SESSION_MAX_AGE]
    for k in expired:
        del _sessions[k]

def _clean_tokens():
    now = time.time()
    expired = [k for k, v in _stage1_tokens.items() if now - v["created"] > 120]
    for k in expired:
        del _stage1_tokens[k]

def _get_session(request: Request) -> Optional[dict]:
    _clean_sessions()
    sid = request.cookies.get("session_id")
    if sid and sid in _sessions:
        _sessions[sid]["last_active"] = time.time()
        return _sessions[sid]
    return None


# ── Models ──

class SignupStartReq(BaseModel):
    user_id: str

class SignatureReq(BaseModel):
    user_id: str
    strokes: list  # [[[x,y], ...], ...]

class LoginSignatureReq(BaseModel):
    user_id: Optional[str] = None
    strokes: list

class FingerprintReq(BaseModel):
    stage1_token: str


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/safe-area", response_class=HTMLResponse)
async def safe_area_page(request: Request):
    session = _get_session(request)
    if not session:
        return HTMLResponse(
            '<script>window.location.href="/";</script>',
            status_code=200
        )
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Signup API ──

@app.post("/api/signup/start")
async def signup_start(req: SignupStartReq):
    try:
        uid = sanitize_user(req.user_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    rec = REGISTRY.get(uid)
    if rec and rec.get("status") == "ACTIVE":
        raise HTTPException(409, f"User '{uid}' already exists")

    result = signup_from_pool(uid)
    if not result.get("ok"):
        raise HTTPException(400, result.get("reason", "signup failed"))

    # Init enrollment state
    _enrollment_state[uid] = {"saved": 0, "trajectories": [], "stroke_counts": []}

    REGISTRY.log("signup_start", uid)
    return {"ok": True, "user_id": uid, "linked_file": result.get("linked_file")}


@app.post("/api/signup/signature")
async def signup_signature(req: SignatureReq):
    try:
        uid = sanitize_user(req.user_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    state = _enrollment_state.get(uid)
    if state is None:
        raise HTTPException(400, "No active signup for this user. Call /api/signup/start first.")

    # Parse strokes
    strokes = []
    for s in req.strokes:
        arr = np.array(s, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            strokes.append(arr)

    if not strokes or sum(len(s) for s in strokes) < 12:
        return {"ok": False, "reason": "Draw a signature first (too few points)"}

    traj = normalize_strokes(strokes, SIG_CFG.trajectory_points)
    if traj is None:
        return {"ok": False, "reason": "Signature too small - draw again"}

    # Consistency check
    matcher = get_matcher()
    prev_trajs = state["trajectories"]
    if len(prev_trajs) >= 2:
        ratio = matcher.consistency_ratio(traj, prev_trajs)
        if ratio > SIG_CFG.consistency_max_ratio:
            return {"ok": False, "reason": f"Too different from previous samples (x{ratio:.1f}) - draw it again",
                    "consistency_ratio": round(ratio, 2)}

    # Save
    state["saved"] += 1
    state["trajectories"].append(traj)
    state["stroke_counts"].append(len(strokes))

    store = EnrollmentStore(SIG_CFG.enroll_dir, uid)
    image = render_strokes(strokes, 400, 400)
    store.save_sample(traj, image, state["saved"], stroke_count=len(strokes))

    saved = state["saved"]
    total = SIG_CFG.enroll_count

    if saved >= total:
        # Build profile
        profile = matcher.build_profile(state["trajectories"], state["stroke_counts"])
        store.save_calibration(profile)
        REGISTRY.upsert(uid, status="ACTIVE",
                        signature=dict(enrolled=True, samples=len(profile.samples),
                                       threshold=profile.threshold,
                                       enrolled_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        REGISTRY.clear_failures(uid)
        REGISTRY.log("signup_ok", uid, sig_samples=saved)
        del _enrollment_state[uid]
        return {"ok": True, "saved": saved, "total": total, "complete": True,
                "threshold": round(profile.threshold, 3),
                "consistency": round(profile.consistency, 3)}

    return {"ok": True, "saved": saved, "total": total, "complete": False}


# ── Login API ──

@app.post("/api/login/signature")
async def login_signature(req: LoginSignatureReq):
    _clean_tokens()

    claimed = None
    if req.user_id:
        try:
            claimed = sanitize_user(req.user_id)
        except ValueError as e:
            raise HTTPException(400, str(e))

    key = claimed or "__anonymous__"

    wait = REGISTRY.locked_for(key)
    if wait > 0:
        return {"ok": False, "reason": f"Temporarily locked - try again in {wait:.0f}s",
                "locked_seconds": round(wait)}

    if claimed:
        rec = REGISTRY.get(claimed)
        if not rec or rec.get("status") != "ACTIVE":
            REGISTRY.register_failure(key)
            return {"ok": False, "reason": "Unknown or inactive user"}
    elif not REGISTRY.active_users():
        return {"ok": False, "reason": "No active users enrolled"}

    # Parse strokes
    strokes = []
    for s in req.strokes:
        arr = np.array(s, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            strokes.append(arr)

    if not strokes or sum(len(s) for s in strokes) < 12:
        return {"ok": False, "reason": "Draw a signature first (too few points)"}

    traj = normalize_strokes(strokes, SIG_CFG.trajectory_points)
    if traj is None:
        return {"ok": False, "reason": "Signature too small"}

    matcher = get_matcher()

    if claimed:
        # VERIFY mode
        store = EnrollmentStore(SIG_CFG.enroll_dir, claimed)
        samples = store.load_samples()
        if not samples:
            return {"ok": False, "reason": "No enrolled signatures"}
        profile = matcher.build_profile([t for t, _ in samples],
                                         [c for _, c in samples if c is not None])
        ok, score, subscores = matcher.verify(traj, profile, len(strokes))
        uid = claimed
        thr = profile.threshold
    else:
        # IDENTIFY mode
        profiles = load_signature_profiles(REGISTRY.active_users())
        if not profiles:
            return {"ok": False, "reason": "No enrolled users"}
        ok, uid, score, thr, why, subscores = matcher.identify(traj, profiles, len(strokes))

    if ok:
        token = secrets.token_urlsafe(32)
        _stage1_tokens[token] = {"user_id": uid, "created": time.time(), "score": float(score)}
        REGISTRY.log("sig_verified", uid, score=float(score))
        return {"ok": True, "user_id": uid, "score": round(float(score), 4),
                "threshold": round(float(thr), 4), "sub_scores": subscores, "stage1_token": token}
    else:
        REGISTRY.register_failure(key)
        REGISTRY.log("sig_failed", claimed or "?", score=float(score))
        return {"ok": False, "reason": "Signature did not match",
                "score": round(float(score), 4), "threshold": round(float(thr), 4), "sub_scores": subscores}


@app.post("/api/login/fingerprint")
async def login_fingerprint(req: FingerprintReq, response: Response):
    _clean_tokens()

    token_data = _stage1_tokens.pop(req.stage1_token, None)
    if not token_data:
        raise HTTPException(401, "Invalid or expired stage1_token")

    uid = token_data["user_id"]
    sig_score = token_data["score"]

    # Auto-capture
    image_path, allow_identical = get_auto_capture(uid)
    if image_path is None:
        REGISTRY.log("fp_failed", uid, reason="linked fingerprint image not found")
        return {"granted": False, "user_id": uid, "reason": "No fingerprint linked to this account",
                "sig_score": round(sig_score, 4), "fp_score": 0.0, "fp_pairs": 0}

    fp = verify_fingerprint(uid, image_path, allow_identical=allow_identical)

    if not fp["ok"]:
        REGISTRY.register_failure(uid)
        REGISTRY.log("login_denied", uid, reason=f"fingerprint: {fp['reason']}",
                      sig_score=sig_score, fp_score=fp["score"], fp_pairs=fp["pairs"])
        return {"granted": False, "user_id": uid, "reason": fp["reason"],
                "sig_score": round(sig_score, 4), "fp_score": round(fp["score"], 4),
                "fp_pairs": fp["pairs"]}

    # Both factors passed
    REGISTRY.clear_failures(uid)
    REGISTRY.log("login_granted", uid, sig_score=sig_score, fp_score=fp["score"], fp_pairs=fp["pairs"])

    try:
        on_access_granted(uid)
    except Exception as exc:
        print(f"[warn] unlock hook failed: {exc}")

    # Create session
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "user_id": uid,
        "login_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "login_timestamp": time.time(),
        "last_active": time.time(),
        "sig_score": round(sig_score, 4),
        "fp_score": round(fp["score"], 4),
        "fp_pairs": fp["pairs"],
    }

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
    )

    return {"granted": True, "user_id": uid,
            "sig_score": round(sig_score, 4),
            "fp_score": round(fp["score"], 4),
            "fp_pairs": fp["pairs"]}


@app.get("/api/me")
async def me(request: Request):
    session = _get_session(request)
    if not session:
        raise HTTPException(401, "Not authenticated")

    activity = REGISTRY.recent_activity(session["user_id"], n=5)
    return {
        "user_id": session["user_id"],
        "login_time": session["login_time"],
        "sig_score": session["sig_score"],
        "fp_score": session["fp_score"],
        "fp_pairs": session["fp_pairs"],
        "session_expires_in": round(SESSION_MAX_AGE - (time.time() - session["last_active"])),
        "recent_activity": activity,
    }


@app.post("/api/logout")
async def logout(request: Request, response: Response):
    sid = request.cookies.get("session_id")
    if sid and sid in _sessions:
        uid = _sessions[sid]["user_id"]
        REGISTRY.log("logout", uid)
        del _sessions[sid]
    response.delete_cookie("session_id")
    return {"ok": True}
