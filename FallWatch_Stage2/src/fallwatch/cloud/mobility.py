"""
cloud/mobility.py
-----------------
Pipeline 2 detector: is this person moving less than THEY used to?

  1. Aggregate the daily data to weekly means.
  2. The first BASELINE_WEEKS weeks are the person's own baseline (mean, std).
  3. Each later week gets one z-score per metric, averaged into a composite.
  4. Alert when the composite is below Z_THRESHOLD for SUSTAIN_WEEKS weeks in a row.

Fixes over Stage 1: the baseline std has a floor (5% of the baseline mean), so
a very regular person does not produce z = -9 from a normal week; and the
detector reports the week it fired, so lead time can be measured.
"""

import numpy as np

from ..config import BASELINE_WEEKS, STD_FLOOR_FRAC, SUSTAIN_WEEKS, Z_THRESHOLD


def weekly_means(daily: np.ndarray) -> np.ndarray:
    """daily (days, metrics) -> (weeks, metrics)."""
    weeks = daily.shape[0] // 7
    return daily[:weeks * 7].reshape(weeks, 7, -1).mean(axis=1)


def composite_z(weekly: np.ndarray, baseline_weeks: int = BASELINE_WEEKS,
                floor_frac: float = STD_FLOOR_FRAC) -> np.ndarray:
    """Composite z per week (NaN for baseline weeks). weekly: (weeks, metrics)."""
    base = weekly[:baseline_weeks]
    mu = base.mean(axis=0)
    sd = np.maximum(base.std(axis=0, ddof=1), floor_frac * mu)
    z = (weekly - mu) / sd
    out = z.mean(axis=1)
    out[:baseline_weeks] = np.nan
    return out


def detect(daily: np.ndarray, z_threshold: float = Z_THRESHOLD,
           sustain: int = SUSTAIN_WEEKS) -> dict:
    wk = weekly_means(daily)
    z = composite_z(wk)
    run, alert_week = 0, None
    for w in range(BASELINE_WEEKS, len(z)):
        run = run + 1 if z[w] < z_threshold else 0
        if run >= sustain:
            alert_week = w                      # alert is raised at the end of this week
            break
    return {"weekly": wk, "z": z, "alert_week": alert_week,
            "min_z": float(np.nanmin(z)) if len(z) > BASELINE_WEEKS else float("nan")}


def detect_cohort(cohort, data, **kw) -> list:
    rows = []
    for p, d in zip(cohort, data):
        r = detect(d, **kw)
        rows.append({"pid": p.pid, "frailty": p.frailty, "kind": p.kind,
                     "onset_week": p.onset_week, "alert_week": r["alert_week"],
                     "min_z": r["min_z"]})
    return rows
