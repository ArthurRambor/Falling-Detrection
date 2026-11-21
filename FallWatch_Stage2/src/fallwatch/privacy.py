"""
privacy.py
----------
LGPD-oriented helpers (Lei 13.709/2018). Three ideas, all small and testable:

  1. Pseudonymisation: the cloud never sees the patient's real identifier,
     only an HMAC-SHA256 pseudonym. Without the secret key it cannot be reversed
     or recomputed, so the key stays with the health unit (UBS).
  2. Consent: no event is stored or forwarded for a patient who has not consented.
  3. Data minimisation: raw accelerometer windows never leave the wearable. The
     cloud receives an event (time, probability, level) or a daily aggregate.
"""

import hashlib
import hmac
import os

DEFAULT_KEY = os.environ.get("FALLWATCH_KEY", "demo-key-change-me").encode()


def pseudonymize(patient_id: str, key: bytes = DEFAULT_KEY) -> str:
    digest = hmac.new(key, str(patient_id).encode(), hashlib.sha256).hexdigest()
    return "p-" + digest[:16]


class ConsentRegistry:
    """Who agreed to monitoring. In production this lives in the UBS system."""

    def __init__(self):
        self._granted = {}

    def grant(self, patient_id: str):
        self._granted[patient_id] = True

    def revoke(self, patient_id: str):
        self._granted[patient_id] = False

    def has_consent(self, patient_id: str) -> bool:
        return self._granted.get(patient_id, False)


def minimise_event(event) -> dict:
    """What the cloud is allowed to see about a fall event. No raw samples."""
    return {"t_impact_s": round(float(event.t_impact), 2),
            "peak_g": round(float(event.peak_g), 2),
            "prob": round(float(event.prob), 3)}
