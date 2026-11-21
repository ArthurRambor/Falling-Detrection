"""
sim/mobility_data.py
--------------------
Daily mobility aggregates for Pipeline 2 (steps, sit-to-stand transitions,
hours out of the house), one row per patient-day.

Why this simulator is harder than Stage 1's:
  * Stage 1 had almost no week-to-week variation, so a z-score of -8.9 was
    meaningless. Here each patient has day-of-week effects, week-level random
    effects (a rainy week, a visiting grandchild) and day noise.
  * Stable patients can have a transient dip (a cold, a bad week). A good
    detector must not alert on one bad week.
  * Decliners drift down gradually from an onset week, not in a step.
  * Some stable patients drift slowly downward with age (no clinical event).
    They are labelled stable, which is the hard case for any baseline method.
"""

from dataclasses import dataclass

import numpy as np

N_WEEKS = 20
DAYS = N_WEEKS * 7
METRICS = ("steps", "transitions", "hours_out")


@dataclass
class MobilityPatient:
    pid: str
    frailty: str
    base: dict
    kind: str                     # "stable" | "transient_dip" | "slow_drift" | "decliner"
    onset_week: int = -1          # only for decliners
    drop_per_week: float = 0.0    # fractional loss per week after onset


_BASE = {   # (steps/day, transitions/day, hours out) by frailty group
    "robust": (5200, 38, 4.5),
    "pre-frail": (3400, 30, 3.0),
    "frail": (1900, 22, 1.6),
}


def make_cohort(n_patients: int = 60, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    cohort = []
    for i in range(n_patients):
        frailty = rng.choice(["robust", "pre-frail", "frail"], p=[0.5, 0.3, 0.2])
        b = _BASE[str(frailty)]
        scale = rng.lognormal(0, 0.15)
        base = {m: v * scale for m, v in zip(METRICS, b)}
        u = rng.random()
        if u < 0.35:
            kind, onset, drop = "decliner", int(rng.integers(8, 18)), float(rng.uniform(0.03, 0.12))
        elif u < 0.55:
            kind, onset, drop = "transient_dip", int(rng.integers(8, 17)), 0.0
        elif u < 0.70:
            kind, onset, drop = "slow_drift", -1, 0.0
        else:
            kind, onset, drop = "stable", -1, 0.0
        cohort.append(MobilityPatient(f"P{i:03d}", str(frailty), base, kind, onset, drop))
    return cohort


def simulate_patient(p: MobilityPatient, seed: int) -> np.ndarray:
    """Return array (DAYS, 3): steps, transitions, hours_out."""
    rng = np.random.default_rng(seed)
    out = np.zeros((DAYS, 3))
    dow = rng.normal(0, 0.06, 7)                       # personal weekday pattern
    week_fx = rng.normal(0, 0.09, N_WEEKS)             # week-level variation shared by all metrics
    for k, m in enumerate(METRICS):
        for d in range(DAYS):
            w = d // 7
            level = 1.0 + dow[d % 7] + week_fx[w] + rng.normal(0, 0.10)
            if p.kind == "decliner" and w >= p.onset_week:
                level *= max(0.35, 1.0 - p.drop_per_week * (w - p.onset_week + 1))
            elif p.kind == "transient_dip" and w == p.onset_week:
                level *= 0.72
            elif p.kind == "slow_drift":
                level *= 1.0 - 0.012 * w
            out[d, k] = max(0.0, p.base[m] * level)
    out[:, 1] = np.round(out[:, 1])
    out[:, 0] = np.round(out[:, 0])
    return out


def simulate_cohort(n_patients: int = 60, seed: int = 7):
    cohort = make_cohort(n_patients, seed)
    data = np.stack([simulate_patient(p, seed * 1000 + i) for i, p in enumerate(cohort)])
    return cohort, data                                # data: (n_patients, DAYS, 3)
