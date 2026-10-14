# Fingerprint pipeline - extracted from BIOMM_2FA_PROJECT notebook (SECTIONS 5, 6)
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .config import FP_CFG, SYS

# ── Image processing pipeline ──

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def segment(gray, block):
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (block, block))
    sq   = cv2.blur(g * g, (block, block))
    std  = np.sqrt(np.maximum(sq - mean * mean, 0))
    std8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, _ = cv2.threshold(std8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (std8 > 0.6 * t).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab == big).astype(np.uint8)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block + 1, block + 1)))
    return m

def fix_polarity(gray, mask, min_bg_frac=0.05):
    bg = gray[mask == 0]
    if bg.size > min_bg_frac * gray.size:
        if bg.mean() < gray[mask > 0].mean():
            return 255 - gray, True
    return gray, False

def normalize(gray, mask, sigma=12):
    g = gray.astype(np.float32)
    w = mask.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma) + 1e-6
    mu  = cv2.GaussianBlur(g * w, (0, 0), sigma) / den
    var = cv2.GaussianBlur(((g - mu) ** 2) * w, (0, 0), sigma) / den
    gstd = g[mask > 0].std() + 1e-6
    out = (g - mu) / (np.sqrt(var) + 0.2 * gstd)
    out[mask == 0] = 0
    return out

def orientation_field(norm, mask, block):
    sm = cv2.GaussianBlur(norm, (0, 0), 1.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    s = block * 0.6
    Gxx = cv2.GaussianBlur(gx * gx, (0, 0), s)
    Gyy = cv2.GaussianBlur(gy * gy, (0, 0), s)
    Gxy = cv2.GaussianBlur(gx * gy, (0, 0), s)
    num = 2 * Gxy
    den = Gxx - Gyy
    mag = np.sqrt(num ** 2 + den ** 2)
    coherence = (mag / (Gxx + Gyy + 1e-6)).astype(np.float32)
    coherence[mask == 0] = 0
    c = cv2.GaussianBlur(den / (mag + 1e-6), (0, 0), block * 0.4)
    s2 = cv2.GaussianBlur(num / (mag + 1e-6), (0, 0), block * 0.4)
    theta_grad = 0.5 * np.arctan2(s2, c)
    ridge_theta = np.mod(theta_grad + np.pi / 2, np.pi).astype(np.float32)
    return ridge_theta, coherence

def estimate_ridge_frequency(norm, mask, default_f, B=32, step=16, nfft=128):
    h, w = norm.shape
    win = np.outer(np.hanning(B), np.hanning(B)).astype(np.float32)
    fx = np.fft.fftshift(np.fft.fftfreq(nfft))
    r = np.hypot(*np.meshgrid(fx, fx))
    band = (r >= 1 / 25) & (r <= 1 / 4)
    freqs = []
    for y in range(0, h - B, step):
        for x in range(0, w - B, step):
            if mask[y:y + B, x:x + B].mean() < 0.95:
                continue
            patch = norm[y:y + B, x:x + B] * win
            F = np.abs(np.fft.fftshift(np.fft.fft2(patch, s=(nfft, nfft))))
            Fb = np.where(band, F, 0)
            idx = np.argmax(Fb)
            if Fb.flat[idx] > 3 * (F[band].mean() + 1e-6):
                freqs.append(r.flat[idx])
    if len(freqs) < 5:
        return default_f
    return float(np.median(freqs))

def make_gabor(phi, freq, s_across, s_along):
    half = int(np.ceil(2.5 * max(s_across, s_along)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    c, s = np.cos(phi), np.sin(phi)
    v = x * c + y * s
    u = -x * s + y * c
    g = np.exp(-0.5 * (u ** 2 / s_across ** 2 + v ** 2 / s_along ** 2)) * np.cos(2 * np.pi * freq * u)
    g -= g.mean()
    g /= np.abs(g).sum()
    return g.astype(np.float32)

def gabor_enhance(norm, ridge_theta, mask, freq, n_orient):
    period = 1.0 / freq
    s_across, s_along = 0.45 * period, 0.8 * period
    resp = np.empty((n_orient,) + norm.shape, np.float32)
    for k in range(n_orient):
        phi = k * np.pi / n_orient
        resp[k] = cv2.filter2D(norm, cv2.CV_32F, make_gabor(phi, freq, s_across, s_along),
                               borderType=cv2.BORDER_REFLECT)
    pos = ridge_theta / (np.pi / n_orient)
    k0 = np.floor(pos).astype(int) % n_orient
    k1 = (k0 + 1) % n_orient
    a = (pos - np.floor(pos)).astype(np.float32)
    r0 = np.take_along_axis(resp, k0[None], 0)[0]
    r1 = np.take_along_axis(resp, k1[None], 0)[0]
    out = (1 - a) * r0 + a * r1
    out[mask == 0] = 0
    return out

def remove_small_blobs(b, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab].astype(np.uint8)

def binarize_ridges(enh, mask, period):
    ridge = ((enh < 0) & (mask > 0)).astype(np.uint8)
    amin = max(8, int(0.7 * period * period))
    ridge = remove_small_blobs(ridge, amin)
    ridge = 1 - remove_small_blobs(1 - ridge, amin)
    return ridge * (mask > 0)

_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def crossing_number(sk):
    p = np.pad(sk.astype(np.int8), 1)
    H, W = sk.shape
    ring = [p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in _RING]
    cn = np.zeros(sk.shape, np.int8)
    for i in range(8):
        cn += np.abs(ring[i] - ring[(i + 1) % 8])
    return cn // 2

def prune_spurs(sk, max_len, passes=2):
    sk = sk.astype(np.uint8).copy()
    H, W = sk.shape
    for _ in range(passes):
        cn = crossing_number(sk)
        ends = np.argwhere((sk == 1) & (cn == 1))
        changed = False
        for y, x in ends:
            if sk[y, x] == 0:
                continue
            path, cur, hit_junction = [(y, x)], (y, x), False
            while len(path) <= max_len:
                cy, cx = cur
                nb = [(cy + dy, cx + dx) for dy, dx in _RING
                      if 0 <= cy + dy < H and 0 <= cx + dx < W
                      and sk[cy + dy, cx + dx] and (cy + dy, cx + dx) not in path]
                if not nb:
                    break
                if len(nb) > 1 or cn[nb[0]] >= 3:
                    hit_junction = True
                    break
                cur = nb[0]
                path.append(cur)
            if len(path) <= max_len:
                for py, px in path:
                    sk[py, px] = 0
                changed = True
        if not changed:
            break
    return sk

def _branch_starts(sk, y, x):
    H, W = sk.shape
    vals = [1 if (0 <= y + dy < H and 0 <= x + dx < W and sk[y + dy, x + dx]) else 0 for dy, dx in _RING]
    starts = []
    for i in range(8):
        if vals[i] and not vals[i - 1]:
            run = [i]
            j = (i + 1) % 8
            while vals[j] and j != i:
                run.append(j)
                j = (j + 1) % 8
            mid = run[len(run) // 2]
            starts.append((y + _RING[mid][0], x + _RING[mid][1]))
    return starts

def _trace(sk, origin, start, steps):
    H, W = sk.shape
    oy, ox = origin
    visited = {origin} | {(oy + dy, ox + dx) for dy, dx in _RING}
    cur, n = start, 1
    for _ in range(steps - 1):
        cy, cx = cur
        nxt = [(cy + dy, cx + dx) for dy, dx in _RING
               if 0 <= cy + dy < H and 0 <= cx + dx < W and sk[cy + dy, cx + dx]
               and (cy + dy, cx + dx) not in visited]
        if not nxt:
            break
        cur = nxt[0]
        visited.add(cur)
        n += 1
    return cur, n

def minutia_direction(sk, x, y, kind, steps):
    starts = _branch_starts(sk, y, x)
    angs, lens = [], []
    for st in starts:
        end, n = _trace(sk, (y, x), st, steps)
        angs.append(np.arctan2(end[0] - y, end[1] - x))
        lens.append(n)
    if not angs:
        return 0.0, 0
    if kind == "Ending" or len(angs) < 3:
        return float(angs[0]), int(lens[0])
    best, pair = 10, (0, 1)
    for i in range(len(angs)):
        for j in range(i + 1, len(angs)):
            d = abs((angs[i] - angs[j] + np.pi) % (2 * np.pi) - np.pi)
            if d < best:
                best, pair = d, (i, j)
    a, b = angs[pair[0]], angs[pair[1]]
    return float(np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))), int(min(lens))

def _clip01(v):
    return float(np.clip(v, 0.0, 1.0))

def extract_minutiae(sk, mask, coherence, clarity, period, cfg):
    H, W = sk.shape
    cn = crossing_number(sk)
    dt = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    margin = cfg.ROI_MARGIN_P * period
    pts = np.argwhere((sk == 1) & ((cn == 1) | (cn == 3)) & (dt > margin))
    out = []
    for y, x in pts:
        kind = "Ending" if cn[y, x] == 1 else "Bifurcation"
        ang, tl = minutia_direction(sk, x, y, kind, cfg.TRACE_STEPS)
        q_coh   = _clip01(coherence[y, x] / 0.8)
        q_clar  = _clip01(clarity[y, x])
        q_dist  = _clip01(dt[y, x] / (3 * period))
        q_trace = _clip01(tl / cfg.TRACE_STEPS)
        q = 0.40 * q_coh + 0.30 * q_clar + 0.15 * q_dist + 0.15 * q_trace
        out.append(dict(x=int(x), y=int(y), type=kind, angle=ang, quality=q,
                        coherence=float(coherence[y, x]), border_dist=float(dt[y, x])))
    return out

def remove_false_minutiae(mins, period, cfg):
    stats = {"raw": len(mins)}
    mins = [m for m in mins if m["coherence"] >= cfg.MIN_COH and m["quality"] >= cfg.MIN_QUALITY]
    stats["after_quality"] = len(mins)
    if len(mins) < 2:
        stats["final"] = len(mins)
        return mins, stats
    P = period
    xy = np.array([[m["x"], m["y"]] for m in mins], float)
    tree = cKDTree(xy)
    drop = set()
    for i, j in sorted(tree.query_pairs(1.5 * P)):
        a, b = mins[i], mins[j]
        d = np.hypot(a["x"] - b["x"], a["y"] - b["y"])
        ta, tb = a["type"], b["type"]
        v = np.array([b["x"] - a["x"], b["y"] - a["y"]], float)
        if d < 0.4 * P:
            drop.add(i if a["quality"] < b["quality"] else j)
        elif ta == tb == "Ending" and d < 1.5 * P:
            da = np.array([np.cos(a["angle"]), np.sin(a["angle"])])
            db = np.array([np.cos(b["angle"]), np.sin(b["angle"])])
            if da @ v < 0 and db @ (-v) < 0:
                drop.update((i, j))
        elif ta != tb and d < 0.7 * P:
            drop.update((i, j))
        elif ta == tb == "Bifurcation" and d < 0.8 * P:
            drop.update((i, j))
    kept = [m for k, m in enumerate(mins) if k not in drop]
    stats["after_pair_rules"] = len(kept)
    if len(kept) > 3:
        xy = np.array([[m["x"], m["y"]] for m in kept], float)
        cnt = np.array([len(n) - 1 for n in cKDTree(xy).query_ball_point(xy, 2.0 * P)])
        kept = [m for m, c in zip(kept, cnt) if c <= 4]
    stats["final"] = len(kept)
    return kept, stats

def process_fingerprint(gray, cfg=FP_CFG):
    r = {}
    mask = segment(gray, cfg.BLOCK)
    gray, inverted = fix_polarity(gray, mask)
    norm = normalize(gray, mask)
    ridge_theta, coherence = orientation_field(norm, mask, cfg.BLOCK)
    freq = estimate_ridge_frequency(norm, mask, cfg.DEFAULT_RIDGE_FREQ)
    period = 1.0 / freq
    enh = gabor_enhance(norm, ridge_theta, mask, freq, cfg.N_ORIENT)
    binary = binarize_ridges(enh, mask, period)
    skel = skeletonize(binary).astype(np.uint8)
    skel[mask == 0] = 0
    skel = prune_spurs(skel, max_len=max(4, int(cfg.SPUR_PERIODS * period)))
    amp = cv2.GaussianBlur(np.abs(enh), (0, 0), cfg.BLOCK * 0.5)
    ref = np.percentile(amp[mask > 0], 90) + 1e-6 if mask.any() else 1.0
    clarity = np.clip(amp / ref, 0, 1)
    raw = extract_minutiae(skel, mask, coherence, clarity, period, cfg)
    final, fstats = remove_false_minutiae(raw, period, cfg)
    roi = mask > 0
    roi_area = int(roi.sum())
    mean_coh = float(coherence[roi].mean()) if roi_area else 0.0
    mean_clar = float(clarity[roi].mean()) if roi_area else 0.0
    good_area = float((coherence[roi] > 0.3).mean()) if roi_area else 0.0
    n = len(final)
    n_end = sum(m["type"] == "Ending" for m in final)
    n_bif = n - n_end
    q_img = 100 * (0.35 * _clip01(mean_coh / 0.6) + 0.25 * mean_clar
                   + 0.25 * good_area + 0.15 * _clip01(n / 20))
    grade = "Good" if q_img >= 65 else ("Fair" if q_img >= 40 else "Poor")
    scores = dict(
        n_minutiae=n, n_endings=n_end, n_bifurcations=n_bif,
        end_bif_ratio=round(n_end / max(1, n_bif), 2),
        mean_minutia_quality=round(float(np.mean([m["quality"] for m in final])) if final else 0.0, 3),
        mean_coherence=round(mean_coh, 3), mean_clarity=round(mean_clar, 3),
        good_area_ratio=round(good_area, 3),
        roi_area_ratio=round(roi_area / mask.size, 3),
        ridge_period_px=round(period, 2),
        n_core=0, n_delta=0,
        image_quality=round(q_img, 1), quality_grade=grade,
        polarity_inverted=inverted,
    )
    r.update(gray=gray, mask=mask, norm=norm, ridge_theta=ridge_theta, coherence=coherence,
             enhanced=enh, binary=binary, skeleton=skel, raw_minutiae=raw, minutiae=final,
             filter_stats=fstats, period=period, scores=scores)
    return r

# ── Templates & matching ──

def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_fp_template(image_path, cfg=FP_CFG):
    res = process_fingerprint(load_gray(str(image_path)), cfg)
    s = res["scores"]
    return dict(
        source=os.path.basename(str(image_path)),
        path=str(Path(image_path).resolve()),
        sha256=_sha256_file(image_path),
        period=float(res["period"]),
        image_quality=float(s["image_quality"]),
        grade=s["quality_grade"],
        n_minutiae=int(s["n_minutiae"]),
        minutiae=[dict(x=int(m["x"]), y=int(m["y"]), type=m["type"],
                       angle=float(m["angle"]), quality=float(m["quality"])) for m in res["minutiae"]],
    )

def fp_quality_problems(tpl):
    problems = []
    if tpl["n_minutiae"] < SYS.FP_MIN_MINUTIAE:
        problems.append(f"only {tpl['n_minutiae']} minutiae found (at least {SYS.FP_MIN_MINUTIAE} required)")
    if tpl["image_quality"] < SYS.FP_MIN_IMAGE_QUALITY:
        problems.append(f"image quality {tpl['image_quality']:.0f} < {SYS.FP_MIN_IMAGE_QUALITY:.0f}")
    return problems

def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _to_array(mins):
    return np.array([[m["x"], m["y"], m["angle"], 1.0 if m["type"] == "Ending" else 0.0,
                      m.get("quality", 1.0)] for m in mins], dtype=np.float64).reshape(-1, 5)

def _candidate_transforms(Q, R, period, max_rot, n_cand=8):
    tb, ab = period, np.radians(10)
    rows = []
    for t in (0.0, 1.0):
        q, r = Q[Q[:, 3] == t], R[R[:, 3] == t]
        if len(q) == 0 or len(r) == 0:
            continue
        dth = _wrap_pi(r[:, None, 2] - q[None, :, 2])
        c, s = np.cos(dth), np.sin(dth)
        tx = r[:, None, 0] - (q[None, :, 0] * c - q[None, :, 1] * s)
        ty = r[:, None, 1] - (q[None, :, 0] * s + q[None, :, 1] * c)
        ok = np.abs(dth) <= max_rot
        rows.append(np.column_stack([dth[ok], tx[ok], ty[ok]]))
    if not rows:
        return []
    V = np.vstack(rows)
    if len(V) == 0:
        return []
    K = np.column_stack([np.round(V[:, 1] / tb), np.round(V[:, 2] / tb), np.round(V[:, 0] / ab)]).astype(int)
    uk, cnt = np.unique(K, axis=0, return_counts=True)
    cands = []
    for idx in np.argsort(-cnt)[: n_cand * 3]:
        near = (np.abs(K - uk[idx]) <= 1).all(axis=1)
        cands.append((int(near.sum()), V[near].mean(axis=0)))
    cands.sort(key=lambda z: -z[0])
    out = []
    for _, m in cands:
        if all(abs(m[0] - o[0]) > ab / 2 or np.hypot(m[1] - o[1], m[2] - o[2]) > tb / 2 for o in out):
            out.append(m)
        if len(out) >= n_cand:
            break
    return out

def _apply(Q, th, tx, ty):
    c, s = np.cos(th), np.sin(th)
    out = Q.copy()
    out[:, 0] = Q[:, 0] * c - Q[:, 1] * s + tx
    out[:, 1] = Q[:, 0] * s + Q[:, 1] * c + ty
    out[:, 2] = _wrap_pi(Q[:, 2] + th)
    return out

def _pair(Qt, R, tol_d, tol_a):
    D = np.hypot(Qt[:, None, 0] - R[None, :, 0], Qt[:, None, 1] - R[None, :, 1])
    A = np.abs(_wrap_pi(Qt[:, None, 2] - R[None, :, 2]))
    T = Qt[:, None, 3] == R[None, :, 3]
    cost = np.where((D <= tol_d) & (A <= tol_a) & T, D, 1e6)
    ri, ci = linear_sum_assignment(cost)
    good = cost[ri, ci] < 1e5
    return ri[good], ci[good]

def _refine(Qt, R, ri, ci):
    P, T = Qt[ri, :2], R[ci, :2]
    pc, tc = P.mean(0), T.mean(0)
    p, t = P - pc, T - tc
    dth = np.arctan2((p[:, 0] * t[:, 1] - p[:, 1] * t[:, 0]).sum(), (p * t).sum())
    c, s = np.cos(dth), np.sin(dth)
    out = Qt.copy()
    x, y = Qt[:, 0] - pc[0], Qt[:, 1] - pc[1]
    out[:, 0] = x * c - y * s + tc[0]
    out[:, 1] = x * s + y * c + tc[1]
    out[:, 2] = _wrap_pi(Qt[:, 2] + dth)
    return out

def _in_region(P, ref_xy, margin):
    if len(ref_xy) < 3:
        return len(P)
    hull = cv2.convexHull(ref_xy.astype(np.float32))
    return int(sum(cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -margin for x, y in P[:, :2]))

def match_minutiae(q_mins, r_mins, period,
                   max_rot_deg=None, tol_periods=None, tol_angle_deg=None):
    max_rot = np.radians(SYS.FP_MAX_ROTATION_DEG if max_rot_deg is None else max_rot_deg)
    tol_d = (SYS.FP_TOL_PERIODS if tol_periods is None else tol_periods) * period
    tol_a = np.radians(SYS.FP_TOL_ANGLE_DEG if tol_angle_deg is None else tol_angle_deg)
    best = dict(score=0.0, pairs=0, n_query=len(q_mins), n_ref=len(r_mins), rotation_deg=0.0)
    if len(q_mins) < 3 or len(r_mins) < 3:
        return best
    Q, R = _to_array(q_mins), _to_array(r_mins)
    for th, tx, ty in _candidate_transforms(Q, R, period, max_rot):
        Qt = _apply(Q, th, tx, ty)
        ri, ci = _pair(Qt, R, tol_d, tol_a)
        if len(ri) < 3:
            continue
        Qt = _refine(Qt, R, ri, ci)
        ri, ci = _pair(Qt, R, tol_d, tol_a)
        m = len(ri)
        if m < 3:
            continue
        nq = max(_in_region(Qt, R[:, :2], tol_d), m)
        nr = max(_in_region(R, Qt[:, :2], tol_d), m)
        score = (m * m) / float(nq * nr)
        if score > best["score"]:
            best.update(score=float(score), pairs=int(m), rotation_deg=float(np.degrees(th)),
                        n_query=int(nq), n_ref=int(nr))
    return best

def match_fp(q_tpl, r_tpl):
    period = float(np.median([q_tpl["period"], r_tpl["period"]]))
    return match_minutiae(q_tpl["minutiae"], r_tpl["minutiae"], period)

def fp_accept(result):
    return result["score"] >= SYS.FP_THRESHOLD and result["pairs"] >= SYS.FP_MIN_PAIRS

# ── High-level functions ──

def load_fp_templates(uid: str) -> List[dict]:
    from .registry import REGISTRY
    d = REGISTRY.fp_dir(uid)
    import json as _json
    return [_json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("template_*.json"))] if d.exists() else []

def _pool_dir(image_dir=None) -> Path:
    from .config import BASE_DIR
    image_dir = Path(image_dir or SYS.FP_POOL_DIR)
    if not image_dir.exists() and (BASE_DIR / "Fingerprint").exists():
        image_dir = BASE_DIR / "Fingerprint"
    if not image_dir.exists():
        raise FileNotFoundError(f"Fingerprint folder not found: {image_dir}")
    return image_dir

def linked_fingerprint_image(uid):
    from .registry import REGISTRY
    import json as _json
    fp = (REGISTRY.get(uid) or {}).get("fingerprint", {})
    candidates = [Path(p) for p in fp.get("paths", []) if p]
    try:
        candidates += [_pool_dir() / s for s in fp.get("sources", [])]
    except FileNotFoundError:
        pass
    return next((str(p) for p in candidates if p.is_file()), None)

def make_test_capture(image_path, out_path=None, seed=0):
    img = load_gray(str(image_path))
    rng = np.random.default_rng(seed)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderValue=255)
    out = np.clip(out.astype(np.float32) + rng.normal(0, 5, out.shape), 0, 255).astype(np.uint8)
    out_path = out_path or (str(Path(image_path).with_name(Path(image_path).stem + "_testcapture.png")))
    cv2.imwrite(out_path, out)
    return out_path

def get_auto_capture(uid: str) -> Tuple[Optional[str], bool]:
    linked = linked_fingerprint_image(uid)
    if linked is None:
        return None, False
    if SYS.FP_AUTO_CAPTURE == "simulate":
        out_dir = SYS.DATA_DIR / "tmp_captures"
        out_dir.mkdir(parents=True, exist_ok=True)
        capture = make_test_capture(linked, str(out_dir / f"{uid}_capture.png"), seed=int(time.time()) % 100000)
        return capture, False
    return linked, True

def verify_fingerprint(uid: str, image_path: str, allow_identical: bool = False) -> dict:
    templates = load_fp_templates(uid)
    if not templates:
        return dict(ok=False, reason="no fingerprint template", score=0.0, pairs=0)
    try:
        q = build_fp_template(image_path)
    except Exception as exc:
        return dict(ok=False, reason=f"image read/process error: {exc}", score=0.0, pairs=0)
    if not allow_identical and q["sha256"] in {t["sha256"] for t in templates}:
        return dict(ok=False, reason="identical to an enrolment image (a fresh capture is required)", score=0.0, pairs=0)
    bad = fp_quality_problems(q)
    if bad:
        return dict(ok=False, reason="poor quality: " + "; ".join(bad), score=0.0, pairs=0)
    best = max((match_fp(q, t) for t in templates), key=lambda r: r["score"])
    return dict(ok=fp_accept(best), reason="ok" if fp_accept(best) else "below threshold",
                score=best["score"], pairs=best["pairs"], rotation_deg=best["rotation_deg"])
