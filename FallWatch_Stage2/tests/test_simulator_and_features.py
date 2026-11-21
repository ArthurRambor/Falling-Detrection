import numpy as np
from fallwatch.config import PEAK_IDX, WINDOW_N
from fallwatch.edge.features import FEATURE_NAMES, extract_features
from fallwatch.sim.fall_data import build_dataset


def test_shapes_and_labels(small_ds):
    w, y = small_ds["windows"], small_ds["labels"]
    assert w.shape[1:] == (WINDOW_N, 3)
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == 30 * 10


def test_deterministic_by_seed():
    a = build_dataset(n_subjects=3, falls_per_subject=2, adl_per_subject=5, seed=5)["windows"]
    b = build_dataset(n_subjects=3, falls_per_subject=2, adl_per_subject=5, seed=5)["windows"]
    assert np.array_equal(a, b)


def test_hard_negatives_present(small_ds):
    kinds = set(small_ds["event_types"])
    assert {"sit_down", "plop_sit", "stairs_down", "arm_hit", "lie_down", "fall_soft"} <= kinds


def test_feature_vector_complete_and_finite(small_ds):
    f = extract_features(small_ds["windows"][0])
    assert list(f) == FEATURE_NAMES
    assert all(np.isfinite(v) for v in f.values())


def test_falls_rotate_more_than_sitting(small_ds):
    ev, w = small_ds["event_types"], small_ds["windows"]
    tilt = lambda kind: np.mean([extract_features(x)["tilt_change_deg"] for x in w[ev == kind][:60]])
    assert tilt("fall_hard") > tilt("sit_down") + 15
