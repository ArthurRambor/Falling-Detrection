"""
edge/fall_model.py
------------------
The on-device classifier of Pipeline 1.

Two stages, the way a real wearable would run it:
  1. a nearly free trigger: is there any sample above TRIGGER_G?
  2. only then, the Random Forest scores the 3-second window.

Gating matters for false alarms per day. Without it the classifier would have
to be right on every one of the 86,400 seconds in a day. With it, it is asked
about roughly 40 candidate events.
"""

import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..config import PEAK_IDX, TRIGGER_G
from .features import FEATURE_NAMES, extract_features, feature_matrix


def triggered_mask(windows: np.ndarray) -> np.ndarray:
    """True for windows whose magnitude crosses TRIGGER_G near the impact sample."""
    mag = np.linalg.norm(windows[:, PEAK_IDX - 10:PEAK_IDX + 20, :], axis=2)
    return mag.max(axis=1) >= TRIGGER_G


def make_classifier(seed: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed,
    )


def train(dataset: dict, feature_names=FEATURE_NAMES, seed: int = 42) -> dict:
    """Fit on triggered windows only and return a self-describing model bundle."""
    keep = triggered_mask(dataset["windows"])
    X = feature_matrix(dataset["windows"][keep], feature_names)
    y = dataset["labels"][keep]
    clf = make_classifier(seed).fit(X, y)
    return {
        "model": clf,
        "feature_names": list(feature_names),
        "trigger_g": TRIGGER_G,
        "n_train_windows": int(keep.sum()),
        "n_train_subjects": int(len(set(dataset["subject_ids"][keep]))),
        "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(),
        "data_source": "documented synthetic simulator (sim/fall_data.py)",
    }


def predict_window(bundle: dict, window: np.ndarray) -> float:
    """Probability that one (150, 3) window is a fall."""
    feats = extract_features(window)
    x = np.array([[feats[n] for n in bundle["feature_names"]]])
    return float(bundle["model"].predict_proba(x)[0, 1])


def feature_importance(bundle: dict) -> list:
    imp = bundle["model"].feature_importances_
    return sorted(zip(bundle["feature_names"], imp.tolist()), key=lambda kv: -kv[1])


def save(bundle: dict, path) -> None:
    joblib.dump(bundle, path)


def load(path) -> dict:
    bundle = joblib.load(path)
    bundle["model"].n_jobs = 1          # single-sample inference is faster without a thread pool
    return bundle


def timed_predict(bundle: dict, window: np.ndarray):
    t0 = time.perf_counter()
    p = predict_window(bundle, window)
    return p, (time.perf_counter() - t0) * 1000.0
