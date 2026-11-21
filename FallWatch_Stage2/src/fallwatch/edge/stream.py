"""
edge/stream.py
--------------
Replays a continuous accelerometer stream the way the wearable would see it:
find a trigger, cut the window around the impact peak, wait for the 2 seconds
after it, classify, and hand the result to the gateway.

The latency of one detection is therefore at least POST_S (2 s) by design.
Everything else is processing time, which we measure instead of assuming.
"""

from dataclasses import dataclass, field

import numpy as np

from ..config import (FS, PEAK_SEARCH_S, POST_N, PRE_N, REFRACTORY_S,
                      TRIGGER_G, WINDOW_N)
from .fall_model import timed_predict


@dataclass
class EdgeEvent:
    t_impact: float          # seconds from stream start
    t_ready: float           # when the 2 s after the impact have been recorded
    peak_g: float
    prob: float              # classifier probability of fall
    proc_ms: float           # measured feature + inference time
    window: np.ndarray = field(repr=False, default=None)


def find_peaks(samples: np.ndarray) -> list:
    """Sample indices of impact peaks: first crossing of TRIGGER_G, then local maximum."""
    mag = np.linalg.norm(samples, axis=1)
    peaks, i, n = [], 0, len(mag)
    refractory = int(REFRACTORY_S * FS)
    search = int(PEAK_SEARCH_S * FS)
    while i < n:
        if mag[i] >= TRIGGER_G:
            p = i + int(np.argmax(mag[i:i + search]))
            peaks.append(p)
            i = p + refractory
        else:
            i += 1
    return peaks


def run_edge(bundle: dict, samples: np.ndarray, t_offset: float = 0.0) -> list:
    """Score every trigger in the stream. Triggers too close to either end are skipped."""
    events = []
    mag = np.linalg.norm(samples, axis=1)
    for p in find_peaks(samples):
        lo, hi = p - PRE_N, p + POST_N
        if lo < 0 or hi > len(samples):
            continue
        win = samples[lo:hi]
        assert win.shape == (WINDOW_N, 3)
        prob, ms = timed_predict(bundle, win)
        events.append(EdgeEvent(t_impact=t_offset + p / FS, t_ready=t_offset + hi / FS,
                                peak_g=float(mag[p]), prob=prob, proc_ms=ms, window=win))
    return events
