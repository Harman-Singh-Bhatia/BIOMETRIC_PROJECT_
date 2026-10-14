# Signature module - extracted from BIOMM_2FA_PROJECT notebook (SECTIONS 2, 3)
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.feature import hog

from .config import Config, SIG_CFG

try:
    import tensorflow as tf
except ImportError:
    tf = None

# ── Geometry helpers ──

def landmark_xy(hand, idx: int) -> np.ndarray:
    p = hand[idx]
    return np.array([p.x, p.y], dtype=np.float32)

def landmark_xyz(hand, idx: int) -> np.ndarray:
    p = hand[idx]
    return np.array([p.x, p.y, p.z], dtype=np.float32)

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def polyline_length(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float32)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg))).astype(np.float32)

def resample(points: np.ndarray, n: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points
    if len(points) < 2:
        return np.repeat(points[:1], n, axis=0)
    s = polyline_length(points)
    if s[-1] < 1e-8:
        return np.repeat(points[:1], n, axis=0)
    t = np.linspace(0.0, float(s[-1]), n)
    return np.column_stack(
        (np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1]))
    ).astype(np.float32)

def gaussian_smooth(points: np.ndarray, radius: int = 2, sigma: float = 1.0) -> np.ndarray:
    if len(points) < 2 * radius + 1:
        return points.copy()
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    padded = np.pad(points, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(points)
    for d in range(points.shape[1]):
        out[:, d] = np.convolve(padded[:, d], k, mode="valid")
    return out

# ── Trajectory normalization ──

def normalize_strokes(strokes: Sequence[np.ndarray], n_points: int) -> Optional[np.ndarray]:
    valid = [np.asarray(s, dtype=np.float32) for s in strokes if len(s) >= 2]
    if not valid:
        return None
    pts = np.concatenate(valid, axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(hi - lo))
    if span < 1e-6:
        return None
    lengths = np.array([max(float(polyline_length(s)[-1]), 1e-6) for s in valid])
    alloc = np.maximum(6, np.round(n_points * lengths / lengths.sum())).astype(int)
    pieces = [gaussian_smooth(resample((s - lo) / span, int(n))) for s, n in zip(valid, alloc)]
    traj = resample(np.concatenate(pieces, axis=0), n_points)
    lo2, hi2 = traj.min(axis=0), traj.max(axis=0)
    span2 = float(max(max(hi2 - lo2), 1e-6))
    centre = (lo2 + hi2) / 2
    traj = (traj - centre) / span2 + 0.5
    return np.clip(traj, 0, 1).astype(np.float32)

def _render_polylines(polys: Sequence[np.ndarray], width: int, height: int,
                      thickness: int, margin: int) -> np.ndarray:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    polys = [p for p in polys if len(p) >= 2]
    if not polys:
        return canvas
    box = min(width, height) - 2 * margin
    offset = ((width - box) / 2, (height - box) / 2)
    scale = 4
    big = np.full((height * scale, width * scale), 255, dtype=np.uint8)
    pts = [((p * box + offset) * scale).round().astype(np.int32).reshape(-1, 1, 2) for p in polys]
    cv2.polylines(big, pts, False, 0, thickness * scale, cv2.LINE_AA)
    return cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)

def render_signature(traj: np.ndarray, width: int, height: int,
                     thickness: int = 4, margin: int = 12) -> np.ndarray:
    if traj is None:
        traj = np.empty((0, 2), dtype=np.float32)
    return _render_polylines([traj], width, height, thickness, margin)

def render_strokes(strokes: Sequence[np.ndarray], width: int, height: int,
                   thickness: int = 6, margin: int = 30) -> np.ndarray:
    valid = [np.asarray(s, dtype=np.float32) for s in strokes if len(s) >= 2]
    if not valid:
        return _render_polylines([], width, height, thickness, margin)
    pts = np.concatenate(valid, axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(max(hi - lo), 1e-6))
    centre = (lo + hi) / 2
    polys = [gaussian_smooth((s - centre) / span + 0.5) for s in valid]
    return _render_polylines(polys, width, height, thickness, margin)

def preprocess_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_AREA)
    return (binary.astype(np.float32) / 255.0)[..., None]

# ── Features ──

def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 12) -> float:
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = max(1, i - band), min(m, i + band)
        for j in range(j0, j1 + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))

def trajectory_descriptor(traj: np.ndarray) -> np.ndarray:
    d = np.gradient(traj, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return np.concatenate([traj, 0.5 * d], axis=1).astype(np.float32)

def hog_feature(img: np.ndarray) -> np.ndarray:
    f = hog(img[..., 0], orientations=9, pixels_per_cell=(11, 11),
            cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-8)

def hu_feature(img: np.ndarray) -> np.ndarray:
    m = cv2.moments((img[..., 0] * 255).astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return np.nan_to_num(hu).astype(np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ── Siamese Encoder ──

class SiameseEncoder:
    """Optional CNN embedding. Silently disabled if TF/weights are missing."""

    def __init__(self, weights: str, img_w: int, img_h: int):
        self.model = None
        if tf is None:
            print("[info] tensorflow not installed - Siamese embedding disabled")
            return
        if not os.path.isfile(weights):
            print(f"[info] Siamese weights not found ({weights}) - embedding disabled")
            return
        try:
            self.model = self._build(img_w, img_h)
            self.model.load_weights(weights)
            print("[info] Siamese encoder loaded")
        except Exception as exc:
            self.model = None
            print(f"[warn] could not load Siamese weights ({exc}) - embedding disabled")

    @staticmethod
    def _build(img_w: int, img_h: int):
        L = tf.keras.layers
        inputs = tf.keras.Input(shape=(img_h, img_w, 1))
        x = inputs
        for filters, k, pool in ((32, 5, True), (64, 3, True), (96, 3, True), (128, 3, False)):
            x = L.Conv2D(filters, k, padding="same")(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            if pool:
                x = L.MaxPooling2D(2)(x)
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(256, activation="relu")(x)
        x = L.Dropout(0.25)(x)
        x = L.Dense(128)(x)
        outputs = L.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
        return tf.keras.Model(inputs, outputs, name="signature_encoder")

    @property
    def available(self) -> bool:
        return self.model is not None

    def embed(self, img: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        e = self.model.predict(img[None], verbose=0)[0]
        return (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)

# ── Data classes ──

@dataclass
class Sample:
    traj: np.ndarray
    desc: np.ndarray
    hog: np.ndarray
    hu: np.ndarray
    embed: Optional[np.ndarray]
    aspect: float = 1.0

@dataclass
class Profile:
    samples: List[Sample]
    dtw_scale: float
    threshold: float
    weights: Dict[str, float] = field(default_factory=dict)
    stroke_counts: List[int] = field(default_factory=list)
    consistency: float = 0.0

# ── SignatureMatcher ──

class SignatureMatcher:
    def __init__(self, cfg: Config, encoder: SiameseEncoder):
        self.cfg = cfg
        self.encoder = encoder

    def make_sample(self, traj: np.ndarray) -> Sample:
        img = preprocess_image(render_signature(traj, self.cfg.img_w, self.cfg.img_h),
                               self.cfg.img_w, self.cfg.img_h)
        lo, hi = traj.min(axis=0), traj.max(axis=0)
        w_span = max(float(hi[0] - lo[0]), 1e-4)
        h_span = max(float(hi[1] - lo[1]), 1e-4)
        aspect = w_span / h_span
        return Sample(
            traj=traj,
            desc=trajectory_descriptor(traj),
            hog=hog_feature(img),
            hu=hu_feature(img),
            embed=self.encoder.embed(img),
            aspect=aspect,
        )

    def _weights(self) -> Dict[str, float]:
        w = {"dtw": self.cfg.w_dtw, "embed": self.cfg.w_embed,
             "hog": self.cfg.w_hog, "hu": self.cfg.w_hu}
        if not self.encoder.available:
            w["embed"] = 0.0
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def pair_score_details(self, q: Sample, r: Sample, dtw_scale: float, w: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        s_dtw = float(np.exp(-dtw_distance(q.desc, r.desc) / max(dtw_scale, 1e-6)))
        s_hog = max(0.0, float((cosine(q.hog, r.hog) + 1) / 2))
        s_hu = float(np.exp(-np.linalg.norm(q.hu - r.hu) / 3.0))
        s_aspect = float(np.exp(-abs(np.log(q.aspect / max(r.aspect, 1e-4))) / 0.8))
        s_embed = 0.0

        # Weighted combination incorporating shape aspect matching
        w_dtw = w.get("dtw", 0.50) * 0.90
        w_hog = w.get("hog", 0.18)
        w_hu = w.get("hu", 0.12)
        w_aspect = 0.10

        score = w_dtw * s_dtw + w_hog * s_hog + w_hu * s_hu + w_aspect * s_aspect
        if w.get("embed", 0) > 0 and q.embed is not None and r.embed is not None:
            s_embed = max(0.0, float((cosine(q.embed, r.embed) + 1) / 2))
            score += w["embed"] * s_embed

        subscores = {
            "s_dtw": round(s_dtw, 4),
            "s_hog": round(s_hog, 4),
            "s_hu": round(s_hu, 4),
            "s_aspect": round(s_aspect, 4),
            "s_embed": round(s_embed, 4),
        }
        return float(score), subscores

    def pair_score(self, q: Sample, r: Sample, dtw_scale: float, w: Dict[str, float]) -> float:
        sc, _ = self.pair_score_details(q, r, dtw_scale, w)
        return sc

    def consistency_ratio(self, traj: np.ndarray, prev: Sequence[np.ndarray]) -> float:
        if len(prev) < 2:
            return 1.0
        descs = [trajectory_descriptor(p) for p in prev]
        q = trajectory_descriptor(traj)
        n = len(descs)
        mutual = np.median([dtw_distance(descs[i], descs[j])
                            for i in range(n) for j in range(i + 1, n)])
        to_new = np.median([dtw_distance(q, d) for d in descs])
        return float(to_new / max(mutual, 1e-6))

    def build_profile(self, trajs: Sequence[np.ndarray],
                      stroke_counts: Sequence[int] = ()) -> Profile:
        samples = [self.make_sample(t) for t in trajs]
        w = self._weights()
        n = len(samples)
        dists = [dtw_distance(samples[i].desc, samples[j].desc)
                 for i in range(n) for j in range(i + 1, n)]
        # Calibrated scale based on median distance among enrolled samples
        dtw_scale = float(np.clip(np.median(dists) * 0.95, 0.12, 0.35)) if dists else 0.15
        loo = []
        for i in range(n):
            others = [s for j, s in enumerate(samples) if j != i]
            if others:
                loo.append(self._top_k_mean(samples[i], others, dtw_scale, w))
        if loo:
            thr = float(np.mean(loo) - self.cfg.threshold_margin_std * (np.std(loo) + 0.01))
            if len(loo) <= 2:
                thr -= 0.05
        else:
            thr = self.cfg.threshold_floor
        thr = float(np.clip(thr, self.cfg.threshold_floor, self.cfg.threshold_ceiling))
        return Profile(samples=samples, dtw_scale=dtw_scale, threshold=thr, weights=w,
                       stroke_counts=list(stroke_counts),
                       consistency=float(np.mean(loo)) if loo else 0.0)

    def _top_k_mean(self, q: Sample, refs: Sequence[Sample], dtw_scale: float,
                    w: Dict[str, float], k: int = 2) -> float:
        scores = sorted((self.pair_score(q, r, dtw_scale, w) for r in refs), reverse=True)
        return float(np.mean(scores[:min(k, len(scores))]))

    def verify_sample(self, q: Sample, profile: Profile,
                      stroke_count: Optional[int] = None) -> Tuple[bool, float, Dict[str, float]]:
        pair_results = [self.pair_score_details(q, r, profile.dtw_scale, profile.weights)
                        for r in profile.samples]
        pair_results.sort(key=lambda x: x[0], reverse=True)
        top_k = pair_results[:min(2, len(pair_results))]
        score = float(np.mean([sc for sc, _ in top_k]))
        best_subscores = top_k[0][1].copy() if top_k else {}

        if stroke_count is not None and profile.stroke_counts:
            expected = int(np.median(profile.stroke_counts))
            if abs(stroke_count - expected) > self.cfg.stroke_count_tolerance:
                score *= self.cfg.stroke_count_penalty
                best_subscores["stroke_penalty"] = self.cfg.stroke_count_penalty

        accepted = score >= profile.threshold
        print(f"[SIG VERIFY] Final Score: {score:.4f} | Threshold: {profile.threshold:.4f} | Accepted: {accepted} | Sub-scores: {best_subscores}")
        return accepted, score, best_subscores

    def verify(self, traj: np.ndarray, profile: Profile,
               stroke_count: Optional[int] = None) -> Tuple[bool, float, Dict[str, float]]:
        return self.verify_sample(self.make_sample(traj), profile, stroke_count)

    def identify(self, traj: np.ndarray, profiles: Dict[str, Profile],
                 stroke_count: Optional[int] = None) -> Tuple[bool, Optional[str], float, float, str, Dict[str, float]]:
        """1:N match against all enrolled users."""
        q = self.make_sample(traj)
        cands = []
        for uid, prof in profiles.items():
            ok, score, subs = self.verify_sample(q, prof, stroke_count)
            cands.append((score, uid, ok, prof.threshold, subs))
        cands.sort(key=lambda x: x[0], reverse=True)
        passing = [c for c in cands if c[2]]
        if not passing:
            best = cands[0]
            return False, None, best[0], best[3], "no profile matched", best[4]
        if len(passing) > 1 and passing[0][0] - passing[1][0] < self.cfg.ident_margin:
            return False, None, passing[0][0], passing[0][3], "ambiguous (two users too close)", passing[0][4]
        score, uid, _, thr, subs = passing[0]
        return True, uid, score, thr, "ok", subs

# ── Storage ──

class EnrollmentStore:
    def __init__(self, root: Path, user_id: str):
        self.dir = root / user_id
        self.meta = self.dir / "profile.json"

    def count(self) -> int:
        return len(list(self.dir.glob("signature_*.npy"))) if self.dir.exists() else 0

    def reset(self) -> int:
        removed = 0
        if self.dir.exists():
            for p in list(self.dir.glob("signature_*.*")) + [self.meta]:
                if p.is_file():
                    p.unlink()
                    removed += 1
        return removed

    def save_sample(self, traj: np.ndarray, image: np.ndarray, index: int,
                    stroke_count: int = 0) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / f"signature_{index:02d}.npy", traj)
        png = self.dir / f"signature_{index:02d}.png"
        cv2.imwrite(str(png), image)
        (self.dir / f"signature_{index:02d}.json").write_text(json.dumps({
            "strokes": stroke_count,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }))
        return png

    def load_trajectories(self) -> List[np.ndarray]:
        return [t for t, _ in self.load_samples()]

    def load_samples(self) -> List[Tuple[np.ndarray, Optional[int]]]:
        out = []
        for p in sorted(self.dir.glob("signature_*.npy")) if self.dir.exists() else []:
            t = np.load(p)
            if t.ndim != 2 or t.shape[1] != 2 or len(t) < 2:
                continue
            meta = p.with_suffix(".json")
            strokes = None
            if meta.is_file():
                try:
                    strokes = int(json.loads(meta.read_text()).get("strokes", 0)) or None
                except (ValueError, json.JSONDecodeError):
                    strokes = None
            out.append((t.astype(np.float32), strokes))
        return out

    def save_calibration(self, profile: Profile) -> None:
        self.meta.write_text(json.dumps({
            "version": 2,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": profile.threshold,
            "dtw_scale": profile.dtw_scale,
            "weights": profile.weights,
            "samples": len(profile.samples),
            "stroke_counts": profile.stroke_counts,
            "consistency": profile.consistency,
        }, indent=2))

    def log_verification(self, ok: bool, score: float, threshold: float, strokes: int) -> None:
        path = self.dir / "verify_log.csv"
        new = not path.is_file()
        with path.open("a", encoding="utf-8") as f:
            if new:
                f.write("time,result,score,threshold,strokes\n")
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{'MATCH' if ok else 'NO_MATCH'},"
                    f"{score:.4f},{threshold:.4f},{strokes}\n")

    def load_calibration(self) -> Optional[dict]:
        if self.meta.is_file():
            return json.loads(self.meta.read_text())
        return None
