import numpy as np
from fallwatch.cloud import mobility as mob


def series(weeks, level=(5000, 35, 4.0), drop_from=None, drop=0.5, seed=0, cv=0.12):
    rng = np.random.default_rng(seed)
    d = np.array(level)[None, :] * (1 + rng.normal(0, cv, (weeks * 7, 3)))
    if drop_from is not None:
        d[drop_from * 7:] *= drop
    return d


def test_stable_patient_is_not_flagged():
    assert mob.detect(series(20, seed=1))["alert_week"] is None


def test_sustained_decline_is_flagged_within_two_weeks():
    r = mob.detect(series(20, drop_from=10, drop=0.5, seed=2))
    assert r["alert_week"] is not None and 10 <= r["alert_week"] <= 12


def test_single_bad_week_is_not_flagged():
    d = series(20, seed=3); d[12 * 7:13 * 7] *= 0.5
    assert mob.detect(d)["alert_week"] is None


def test_std_floor_prevents_absurd_z():
    d = np.tile([5000.0, 35.0, 4.0], (140, 1))            # perfectly regular person
    d[10 * 7:] *= 0.9                                      # a 10% dip
    z = mob.detect(d)["z"]
    assert np.nanmin(z) > -4                               # Stage 1 gave -8.9 for this kind of case
