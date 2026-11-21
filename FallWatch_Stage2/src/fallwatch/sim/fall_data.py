"""
sim/fall_data.py
----------------
Documented simulator for a wrist-worn tri-axial accelerometer (Stage 1 decision:
no physical wearable yet, so the data are synthetic and said so everywhere).

What changed since Stage 1
  * Stage 1 only had easy negatives (walking, standing), so every metric was
    1.00. That told us the signal was separable, not that the model worked.
  * Stage 2 adds the situations that really confuse fall detectors: abrupt
    lying down, plopping into a chair, a stumble that is recovered, a hand
    hitting a table, descending stairs.
  * Each simulated person has a frailty group. Frail people fall "softer"
    (lower impact, little free fall), which is how real elderly falls differ
    from the young volunteers in public datasets. This lets us audit the model
    by group, the same exercise as Class 11.

Every window has 150 samples (1 s before, 2 s after the impact peak) at 50 Hz.
The peak is always at sample 50, because the edge trigger centres windows on it.
"""

from dataclasses import dataclass

import numpy as np

from ..config import FS, PEAK_IDX, WINDOW_N

T = (np.arange(WINDOW_N) - PEAK_IDX) / FS          # seconds relative to impact
FRAILTY_GROUPS = ("robust", "pre-frail", "frail")

# Expected triggering-level activities per person-day. Used both to sample the
# negative class and to turn a false-positive RATE into false alarms PER DAY.
ADL_PER_DAY = {
    "sit_down": 12.0,
    "stand_up": 12.0,
    "stairs_down": 8.0,
    "arm_hit": 6.0,
    "plop_sit": 2.0,
    "lie_down": 2.0,
    "stumble_recover": 0.6,
}
ADL_EVENTS_PER_DAY = sum(ADL_PER_DAY.values())


@dataclass
class Subject:
    sid: int
    age: int
    frailty: str
    cadence: float          # steps per second
    gait_amp: float         # g, amplitude of walking oscillation
    noise_sd: float         # g, sensor noise
    impact_mean: float      # g, average impact peak of this person's falls
    freefall_depth: float   # 1 - minimum magnitude just before impact


_FRAILTY_PARAMS = {
    #            age range   cadence  gait_amp  impact  freefall
    "robust":    ((65, 78), (1.90, 0.10), (0.35, 0.05), (4.2, 0.5), (0.55, 0.10)),
    "pre-frail": ((68, 85), (1.70, 0.10), (0.28, 0.05), (3.5, 0.5), (0.42, 0.10)),
    "frail":     ((74, 92), (1.40, 0.10), (0.20, 0.04), (2.9, 0.5), (0.30, 0.10)),
}


def make_subject(sid: int, rng: np.random.Generator) -> Subject:
    frailty = str(rng.choice(FRAILTY_GROUPS, p=[0.5, 0.3, 0.2]))
    (a0, a1), cad, amp, imp, ff = _FRAILTY_PARAMS[frailty]
    return Subject(
        sid=sid,
        age=int(rng.integers(a0, a1 + 1)),
        frailty=frailty,
        cadence=float(rng.normal(*cad)),
        gait_amp=float(max(0.08, rng.normal(*amp))),
        noise_sd=float(rng.uniform(0.02, 0.05)),
        impact_mean=float(max(1.8, rng.normal(*imp))),
        freefall_depth=float(np.clip(rng.normal(*ff), 0.05, 0.8)),
    )


# ------------------------------------------------------------------ physics helpers
def _unit(v):
    return v / np.linalg.norm(v)


def _rotate(v, axis, angle):
    """Rodrigues rotation of vector v (3,) about unit axis by angle (N,) -> (N,3)."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _smooth(t, t0, dur):
    x = np.clip((t - t0) / dur, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _random_axis(rng, horizontal=False):
    if horizontal:
        th = rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(th), np.sin(th), 0.0])
    return _unit(rng.normal(size=3))


def _gravity(rng, tilt_deg, t0, dur):
    """Gravity vector over the window; it rotates by tilt_deg between t0 and t0+dur."""
    pose = rng.normal(0, np.radians(20))                      # wrist is never perfectly upright
    g0 = _rotate(np.array([0.0, 0.0, 1.0]), _random_axis(rng), np.array([pose]))[0]
    angle = np.radians(tilt_deg) * _smooth(T, t0, dur)
    return _rotate(g0, _random_axis(rng, horizontal=True), angle)


def _gait(rng, amp, cadence, mask):
    """Walking oscillation, only where mask is True."""
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    vert = np.sin(2 * np.pi * cadence * T + ph1)
    lat = 0.5 * np.sin(np.pi * cadence * T + ph2)
    out = np.stack([0.3 * vert + 0.8 * lat, 0.3 * vert + 0.2 * lat, vert], axis=1)
    return amp * out * mask[:, None]


def _build(rng, s, *, tilt, tilt_t0, tilt_dur, peak, width=0.05, freefall=0.0,
           gait_pre=0.0, gait_post=0.0, ring=0.0, extra_peaks=()):
    """Assemble one 150-sample window from its physical ingredients."""
    g = _gravity(rng, tilt, tilt_t0, tilt_dur)
    if freefall > 0:
        g = g * (1 - freefall * np.exp(-0.5 * ((T + 0.12) / 0.05) ** 2))[:, None]
    acc = g.copy()
    acc += _gait(rng, gait_pre * s.gait_amp, s.cadence, T < -0.15)
    acc += _gait(rng, gait_post * s.gait_amp, s.cadence, T > 0.6)

    def pulse(t_c, amp, w, direction):
        return (amp * np.exp(-0.5 * ((T - t_c) / w) ** 2))[:, None] * direction

    direction = _unit(g[PEAK_IDX] + rng.normal(0, 0.25, 3))
    acc += pulse(0.0, max(peak - 1.0, 0.05), width, direction)
    for t_c, amp in extra_peaks:
        acc += pulse(t_c, amp, width * 1.5, _unit(g[PEAK_IDX] + rng.normal(0, 0.4, 3)))
    if ring > 0:                                              # post-impact ringing
        env = np.where(T > 0, np.exp(-T / 0.12), 0.0)
        acc += (ring * env * np.sin(2 * np.pi * rng.uniform(6, 10) * T))[:, None] * _random_axis(rng)
    acc += rng.normal(0, s.noise_sd, acc.shape)
    return acc.astype(np.float32)


# ------------------------------------------------------------------ activities
def _sit_down(rng, s):
    return _build(rng, s, tilt=rng.uniform(10, 35), tilt_t0=-0.3, tilt_dur=0.8,
                  peak=rng.uniform(1.5, 2.3), gait_pre=0.8, gait_post=0.0)


def _stand_up(rng, s):
    return _build(rng, s, tilt=rng.uniform(10, 30), tilt_t0=-0.4, tilt_dur=0.7,
                  peak=rng.uniform(1.4, 2.0), gait_pre=0.0, gait_post=0.7)


def _plop_sit(rng, s):
    return _build(rng, s, tilt=rng.uniform(5, 20), tilt_t0=-0.2, tilt_dur=0.4,
                  peak=rng.uniform(2.2, 3.4), width=0.05, freefall=rng.uniform(0.1, 0.25),
                  gait_pre=0.8, ring=0.3)


def _stairs_down(rng, s):
    base = rng.uniform(1.9, 2.8)
    return _build(rng, s, tilt=rng.uniform(0, 5), tilt_t0=0.0, tilt_dur=1.0, peak=base,
                  gait_pre=1.1, gait_post=1.1,
                  extra_peaks=[(-0.5, 0.8 * (base - 1)), (0.5, 0.8 * (base - 1))])


def _stumble_recover(rng, s):
    return _build(rng, s, tilt=rng.uniform(5, 20), tilt_t0=-0.1, tilt_dur=0.3,
                  peak=rng.uniform(2.4, 3.9), freefall=rng.uniform(0.3, 0.5),
                  gait_pre=1.0, gait_post=1.0, ring=0.2)


def _lie_down(rng, s):
    if rng.random() < 0.25:                                   # abrupt lying, the hard variant
        dur, peak = rng.uniform(0.5, 0.9), rng.uniform(1.8, 2.6)
    else:
        dur, peak = rng.uniform(1.2, 2.5), rng.uniform(1.2, 2.1)
    return _build(rng, s, tilt=rng.uniform(70, 100), tilt_t0=-0.8, tilt_dur=dur,
                  peak=peak, gait_pre=0.5, gait_post=0.0)


def _arm_hit(rng, s):
    return _build(rng, s, tilt=rng.uniform(0, 8), tilt_t0=-0.1, tilt_dur=0.3,
                  peak=rng.uniform(2.4, 4.6), width=0.03, gait_pre=rng.uniform(0, 0.5),
                  gait_post=rng.uniform(0, 0.3), ring=rng.uniform(0.3, 0.8))


_ADL = {
    "sit_down": _sit_down, "stand_up": _stand_up, "plop_sit": _plop_sit,
    "stairs_down": _stairs_down, "stumble_recover": _stumble_recover,
    "lie_down": _lie_down, "arm_hit": _arm_hit,
}


def generate_adl(kind: str, s: Subject, rng: np.random.Generator) -> np.ndarray:
    return _ADL[kind](rng, s)


def generate_fall(s: Subject, rng: np.random.Generator):
    """Returns (window, subtype). Frail people have more soft slumping falls."""
    soft_prob = {"robust": 0.05, "pre-frail": 0.15, "frail": 0.30}[s.frailty]
    if rng.random() < soft_prob:
        win = _build(rng, s, tilt=rng.uniform(55, 100), tilt_t0=-0.4, tilt_dur=rng.uniform(0.8, 1.6),
                     peak=rng.uniform(1.6, 2.6), freefall=0.05, gait_pre=rng.uniform(0.3, 0.8),
                     gait_post=0.0)
        return win, "fall_soft"
    peak = float(np.clip(rng.normal(s.impact_mean, 0.8), 1.5, 8.0))
    rises = rng.random() < 0.25                                # tries to get up inside the window
    win = _build(rng, s, tilt=rng.uniform(55, 110), tilt_t0=-0.15, tilt_dur=rng.uniform(0.25, 0.6),
                 peak=peak, width=rng.uniform(0.04, 0.08),
                 freefall=float(max(0.0, rng.normal(s.freefall_depth, 0.12))),
                 gait_pre=rng.uniform(0.4, 1.0), gait_post=0.5 if rises else 0.0,
                 ring=rng.uniform(0.5, 1.5))
    return win, "fall_hard"


# ------------------------------------------------------------------ datasets
def build_dataset(n_subjects: int = 90, falls_per_subject: int = 14,
                  adl_per_subject: int = 140, seed: int = 42) -> dict:
    """
    Labelled windows for training and evaluation.
    ADL types are sampled in proportion to ADL_PER_DAY, so the false-positive
    rate measured on this set can be converted into false alarms per day.
    """
    rng = np.random.default_rng(seed)
    subjects = [make_subject(i, rng) for i in range(n_subjects)]
    kinds = list(ADL_PER_DAY)
    probs = np.array([ADL_PER_DAY[k] for k in kinds]) / ADL_EVENTS_PER_DAY
    windows, labels, sids, types = [], [], [], []
    for s in subjects:
        for _ in range(falls_per_subject):
            w, sub = generate_fall(s, rng)
            windows.append(w); labels.append(1); sids.append(s.sid); types.append(sub)
        for k in rng.choice(kinds, size=adl_per_subject, p=probs):
            windows.append(generate_adl(str(k), s, rng)); labels.append(0)
            sids.append(s.sid); types.append(str(k))
    return {
        "windows": np.stack(windows),
        "labels": np.array(labels),
        "subject_ids": np.array(sids),
        "event_types": np.array(types),
        "subjects": subjects,
        "seed": seed,
    }


# ------------------------------------------------------------------ continuous stream for the live demo
def _filler(rng, s, dur_s, kind):
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    if kind == "lying":
        g = _rotate(np.array([0.0, 0.0, 1.0]), _random_axis(rng, horizontal=True),
                    np.full(n, np.radians(85)))
        return (g + rng.normal(0, s.noise_sd * 0.6, (n, 3))).astype(np.float32)
    g = np.tile([0.0, 0.0, 1.0], (n, 1))
    amp = 0.0 if kind == "quiet" else min(s.gait_amp, 0.4)
    osc = amp * np.sin(2 * np.pi * s.cadence * t + rng.uniform(0, 6.28))
    g[:, 2] += osc
    g[:, 0] += 0.3 * osc
    return (g + rng.normal(0, s.noise_sd, (n, 3))).astype(np.float32)


def build_stream(rng, s: Subject, script):
    """
    script: list of ("walk"|"quiet"|"lying", seconds) or ("event", kind) where
    kind is an ADL name or "fall". Returns (samples (n,3), truth [(kind, t_impact_s)]).
    """
    parts, truth, cursor = [], [], 0
    for item, arg in script:
        if item == "event":
            if arg == "fall":
                seg, _ = generate_fall(s, rng)
            else:
                seg = generate_adl(arg, s, rng)
            truth.append((arg, (cursor + PEAK_IDX) / FS))
        else:
            seg = _filler(rng, s, float(arg), item)
        parts.append(seg)
        cursor += len(seg)
    return np.concatenate(parts), truth
