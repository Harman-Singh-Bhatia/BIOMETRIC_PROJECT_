# Registry, lockout, audit log - extracted from BIOMM_2FA_PROJECT notebook (SECTIONS 7, 8c, 9)
from __future__ import annotations

import csv
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import SIG_CFG, SYS
from .signature import (
    EnrollmentStore, SignatureMatcher, SiameseEncoder, Profile,
    normalize_strokes, render_strokes
)

# ── Helpers ──

def sanitize_user(user: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in (user or "").strip())
    if not cleaned:
        raise ValueError("user_id cannot be empty")
    return cleaned

# ── Registry ──

class Registry:
    """users.json = {users: {id: {...}}, lockouts: {key: {...}}}
    status: PENDING | ACTIVE."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "users.json"
        self.audit_path = self.root / "audit_log.csv"

    def _load(self) -> dict:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"users": {}, "lockouts": {}}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, uid: str) -> Optional[dict]:
        return self._load()["users"].get(uid)

    def upsert(self, uid: str, **fields) -> dict:
        d = self._load()
        rec = d["users"].setdefault(uid, {"user_id": uid,
                                          "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                          "status": "PENDING"})
        rec.update(fields)
        self._save(d)
        return rec

    def active_users(self) -> List[str]:
        return sorted(u for u, r in self._load()["users"].items() if r.get("status") == "ACTIVE")

    def fp_dir(self, uid: str) -> Path:
        return self.root / "fingerprints" / uid

    # ---- lockout ----
    def locked_for(self, key: str) -> float:
        rec = self._load()["lockouts"].get(key)
        return max(0.0, rec["locked_until"] - time.time()) if rec else 0.0

    def register_failure(self, key: str) -> None:
        d = self._load()
        rec = d["lockouts"].setdefault(key, {"fails": 0, "locked_until": 0.0})
        rec["fails"] += 1
        if rec["fails"] >= SYS.MAX_FAILS:
            rec["locked_until"] = time.time() + SYS.LOCKOUT_SEC
            rec["fails"] = 0
        self._save(d)

    def clear_failures(self, key: str) -> None:
        d = self._load()
        if d["lockouts"].pop(key, None) is not None:
            self._save(d)

    # ---- audit ----
    def log(self, event: str, user: Optional[str], **detail) -> None:
        new = not self.audit_path.is_file()
        with self.audit_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "event", "user", "detail"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, user or "",
                        json.dumps(detail, default=float)])

    def recent_activity(self, user: Optional[str] = None, n: int = 5) -> List[dict]:
        """Return the last n audit log entries (optionally filtered by user)."""
        if not self.audit_path.is_file():
            return []
        rows = []
        with self.audit_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if user and row.get("user", "") != user:
                    continue
                rows.append(row)
        return rows[-n:]


REGISTRY = Registry(SYS.DATA_DIR)

# ── Matcher singleton ──

_MATCHER: Optional[SignatureMatcher] = None

def get_matcher() -> SignatureMatcher:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = SignatureMatcher(SIG_CFG, SiameseEncoder(SIG_CFG.siamese_weights, SIG_CFG.img_w, SIG_CFG.img_h))
    return _MATCHER

# ── Signature profiles ──

def load_signature_profiles(uids: Sequence[str]) -> Dict[str, Profile]:
    matcher, profiles = get_matcher(), {}
    for uid in uids:
        samples = EnrollmentStore(SIG_CFG.enroll_dir, uid).load_samples()
        if samples:
            profiles[uid] = matcher.build_profile([t for t, _ in samples],
                                                  [c for _, c in samples if c is not None])
    return profiles

# ── Pool-based signup ──

def _sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def list_fingerprint_pool(image_dir=None):
    from .config import BASE_DIR
    image_dir = Path(image_dir or SYS.FP_POOL_DIR)
    if not image_dir.exists() and (BASE_DIR / "Fingerprint").exists():
        image_dir = BASE_DIR / "Fingerprint"
    if not image_dir.exists():
        raise FileNotFoundError(f"Fingerprint folder not found: {image_dir}")
    pat = re.compile(r"^fingerprint(\d+)\.(png|bmp|jpg|jpeg|tif|tiff)$", re.IGNORECASE)
    files = [f.name for f in image_dir.iterdir() if pat.match(f.name)]
    files.sort(key=lambda n: int(pat.match(n).group(1)))
    return image_dir, files

def _linked_pool_hashes(exclude_uid=None):
    from .fingerprint import load_fp_templates
    hashes = set()
    for uid in REGISTRY._load()["users"]:
        if uid == exclude_uid:
            continue
        for t in load_fp_templates(uid):
            hashes.add(t["sha256"])
    return hashes

def signup_from_pool(user_id, image_path=None, image_dir=None, overwrite=False):
    """Link ONE fingerprint image from the pool folder to a user."""
    from .fingerprint import build_fp_template, fp_quality_problems
    uid = sanitize_user(user_id)
    rec = REGISTRY.get(uid)
    if rec and rec.get("status") == "ACTIVE" and not overwrite:
        return dict(ok=False, reason="user already exists")

    image_dir, files = list_fingerprint_pool(image_dir)
    if image_path is None:
        used = _linked_pool_hashes(exclude_uid=uid)
        chosen = next((image_dir / f for f in files if _sha256_file(image_dir / f) not in used), None)
        if chosen is None:
            return dict(ok=False, reason="no unused image left in the fingerprint folder")
        image_path = str(chosen)

    # Build template
    try:
        tpl = build_fp_template(image_path)
    except Exception as exc:
        return dict(ok=False, reason=f"fingerprint processing failed: {exc}")

    bad = fp_quality_problems(tpl)
    if bad:
        return dict(ok=False, reason="poor quality: " + "; ".join(bad))

    # Save template
    d = REGISTRY.fp_dir(uid)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("template_*.json"):
        old.unlink()
    (d / "template_01.json").write_text(json.dumps(tpl), encoding="utf-8")

    REGISTRY.upsert(uid, fingerprint=dict(
        enrolled=True, samples=1,
        sources=[tpl["source"]],
        paths=[tpl.get("path", "")],
        enrolled_at=time.strftime("%Y-%m-%d %H:%M:%S")
    ))
    REGISTRY.upsert(uid, status="PENDING")

    return dict(ok=True, linked_file=Path(image_path).name, user_id=uid)

# ── Access granted hook ──

def on_access_granted(uid: str) -> None:
    """Hook: connect the locker unlock here (Arduino/ESP32 relay)."""
    print(f"[access] UNLOCK signal -> user '{uid}'")
