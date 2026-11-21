"""
edge/features.py
----------------
Turns one 150-sample window into a small, explainable feature vector.

The first nine features are the Stage 1 set. Stage 2 adds the three that carry
most of the physics of a fall, and that a plain "was there a big spike?" rule
cannot see:

  freefall_depth   how far the magnitude dropped below 1 g just before impact
  tilt_change_deg  how much the gravity direction rotated, before vs after
  post_stillness   how motionless the wrist is after the impact

Lying down also changes orientation, and a hand hitting a table also spikes,
but only a fall combines a spike, a rotation and then stillness.
"""

import numpy as np

from ..config import FS, PEAK_IDX

STAGE1_FEATURES = ["mean", "std", "min", "max", "range", "rms", "sma", "max_jerk", "energy"]
STAGE2_FEATURES = ["freefall_depth", "tilt_change_deg", "post_stillness", "post_energy", "pre_activity"]
FEATURE_NAMES = STAGE1_FEATURES + STAGE2_FEATURES


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_features(window: np.ndarray) -> dict:
    """window: (150, 3) acceleration in g. Returns a dict keyed by FEATURE_NAMES."""
    mag = np.linalg.norm(window, axis=1)
    pre_mag = mag[:PEAK_IDX - 10]
    post_mag = mag[-60:]
    pre_vec = window[:35].mean(axis=0)
    post_vec = window[-40:].mean(axis=0)

    return {
        # Stage 1 features
        "mean": float(mag.mean()),
        "std": float(mag.std()),
        "min": float(mag.min()),
        "max": float(mag.max()),
        "range": float(mag.max() - mag.min()),
        "rms": float(np.sqrt(np.mean(mag ** 2))),
        "sma": float(np.mean(np.abs(window).sum(axis=1))),          # signal magnitude area
        "max_jerk": float(np.max(np.abs(np.diff(mag))) * FS),
        "energy": float(np.sum((mag - 1.0) ** 2) / len(mag)),
        # Stage 2 features
        "freefall_depth": float(max(0.0, 1.0 - mag[PEAK_IDX - 25:PEAK_IDX].min())),
        "tilt_change_deg": _angle_deg(pre_vec, post_vec),
        "post_stillness": float(np.mean(np.abs(post_mag - 1.0) < 0.1)),
        "post_energy": float(np.mean((post_mag - 1.0) ** 2)),
        "pre_activity": float(pre_mag.std()),
    }


def feature_matrix(windows: np.ndarray, feature_names=FEATURE_NAMES) -> np.ndarray:
    return np.array([[extract_features(w)[n] for n in feature_names] for w in windows])
