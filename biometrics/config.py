# Configuration - extracted from BIOMM_2FA_PROJECT notebook (SECTION 1)
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

def _base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()

BASE_DIR = _base_dir()

@dataclass
class Config:
    hand_model: str = os.environ.get(
        "HAND_MODEL", str(BASE_DIR / "hand_landmarker.task")
    )
    siamese_weights: str = os.environ.get(
        "SIGNATURE_WEIGHTS", str(BASE_DIR / "signature_encoder.weights.h5")
    )
    enroll_dir: Path = BASE_DIR / "biometric_2fa_data" / "air_enrollments"

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    detect_width: int = 960

    enroll_count: int = 5
    trajectory_points: int = 128

    img_w: int = 155
    img_h: int = 220

    stroke_color: Tuple[int, int, int] = (0, 0, 0)
    stroke_halo: Tuple[int, int, int] = (255, 255, 255)
    stroke_thickness: int = 5

    palm_stable_frames: int = 5
    palm_cooldown_frames: int = 25
    tail_trim_points: int = 6
    banner_frames: int = 60

    consistency_max_ratio: float = 2.2
    stroke_count_tolerance: int = 3
    stroke_count_penalty: float = 0.95

    w_dtw: float = 0.50
    w_embed: float = 0.20
    w_hog: float = 0.18
    w_hu: float = 0.12

    threshold_floor: float = 0.38
    threshold_ceiling: float = 0.70
    threshold_margin_std: float = 1.2

    # ---- 2FA login ----
    login_max_tries: int = 3
    ident_margin: float = 0.02
    show_scores: bool = False

# Fingerprint pipeline config
FP_CFG = SimpleNamespace(
    BLOCK        = 16,
    ROI_MARGIN_P = 1.5,
    MIN_COH      = 0.25,
    MIN_QUALITY  = 0.30,
    SPUR_PERIODS = 1.0,
    N_ORIENT     = 16,
    TRACE_STEPS  = 10,
    DEFAULT_RIDGE_FREQ = 1/9,
)

# 2FA system config
SYS = SimpleNamespace(
    DATA_DIR = BASE_DIR / "biometric_2fa_data",
    FP_MIN_ENROLL_SAMPLES  = 1,       # this dataset has only one image per identity
    FP_MIN_MINUTIAE        = 12,
    FP_MIN_IMAGE_QUALITY   = 40.0,
    FP_THRESHOLD           = 0.10,
    FP_MIN_PAIRS           = 6,
    FP_ENROLL_CONSISTENCY  = 0.6,
    FP_MAX_ROTATION_DEG    = 60,
    FP_TOL_PERIODS         = 1.2,
    FP_TOL_ANGLE_DEG       = 25,
    MAX_FAILS   = 3,
    LOCKOUT_SEC = 60,
    DEBUG       = os.environ.get("DEBUG", "true").lower() in ("1", "true", "yes"),
    FP_AUTO_CAPTURE = os.environ.get("FP_AUTO_CAPTURE", "same"),
)

# Pool directory for fingerprints
SYS.FP_POOL_DIR = Path(os.environ.get("FP_POOL_DIR", str(BASE_DIR / "Fingerprint")))

SIG_CFG = Config()
SYS.DATA_DIR.mkdir(parents=True, exist_ok=True)
