# BIOMM 2FA - Biometrics module
# Extracted from BIOMM_2FA_PROJECT notebook

from .config import Config, FP_CFG, SYS, SIG_CFG
from .signature import (
    normalize_strokes, render_signature, render_strokes,
    SignatureMatcher, SiameseEncoder, EnrollmentStore,
    Sample, Profile, polyline_length, resample, gaussian_smooth,
    preprocess_image, dtw_distance, trajectory_descriptor,
    hog_feature, hu_feature, cosine
)
from .fingerprint import (
    process_fingerprint, build_fp_template, fp_quality_problems,
    match_minutiae, match_fp, fp_accept, verify_fingerprint,
    load_fp_templates, get_auto_capture, linked_fingerprint_image,
    make_test_capture
)
from .registry import (
    Registry, sanitize_user, REGISTRY, get_matcher,
    signup_from_pool, load_signature_profiles, on_access_granted
)
