"""
pipeline.py
-----------
Glues the tiers together for the live demo and the dashboard.

  run_fall_scenario    edge -> fog -> cloud store -> FHIR Condition
  run_mobility_cohort  daily data -> weekly z -> alert -> FHIR Observation
"""

from datetime import date, datetime, timezone

import numpy as np

from .cloud import mobility as mob
from .cloud.store import Store
from .config import BASELINE_WEEKS
from .edge.stream import run_edge
from .fhir import export as fx
from .fog.gateway import Gateway
from .privacy import ConsentRegistry, minimise_event, pseudonymize
from .sim.fall_data import build_stream, make_subject

DEMO_SCRIPT = [
    ("walk", 12), ("event", "sit_down"), ("quiet", 8),
    ("event", "stand_up"), ("walk", 10),
    ("event", "arm_hit"), ("walk", 8),
    ("event", "fall"), ("lying", 12),
    ("walk", 6),
]


def run_fall_scenario(bundle, patient_id="maria-silva", seed=11, script=None, store=None,
                      link_down_for_first_alert=False, start=None, cancel_uncertain=False):
    """Run one continuous stream through all three tiers. Returns a dict of everything."""
    rng = np.random.default_rng(seed)
    subject = make_subject(0, rng)
    samples, truth = build_stream(rng, subject, script or DEMO_SCRIPT)
    start = start or datetime(2026, 11, 28, 9, 0, 0, tzinfo=timezone.utc)

    consent = ConsentRegistry()
    consent.grant(patient_id)
    pseudo = pseudonymize(patient_id)
    store = store or Store()
    notified, resources = [], [fx.build_patient(pseudo), fx.build_device()]

    def to_cloud(p, d):
        cond = fx.build_fall_condition(p, d, start)
        resources.append(cond)
        store.add_fall(p, d, cond)

    gw = Gateway(notifier=lambda p, d: notified.append(d), cloud_sink=to_cloud,
                 link_up=not link_down_for_first_alert)

    if not consent.has_consent(patient_id):
        return {"error": "no consent: nothing is processed or stored"}

    events = run_edge(bundle, samples)
    decisions = []
    for ev in events:
        decisions.append((ev, gw.handle(pseudo, ev, cancelled=cancel_uncertain)))
    flushed = gw.restore_link() if link_down_for_first_alert else 0

    return {"patient": pseudo, "samples": samples, "truth": truth, "events": events,
            "decisions": decisions, "notified": notified, "resources": resources,
            "store": store, "flushed": flushed, "minimised": [minimise_event(e) for e in events]}


def run_mobility_cohort(n_patients=60, seed=7, store=None, start_date=date(2026, 1, 5)):
    from .sim.mobility_data import simulate_cohort
    cohort, data = simulate_cohort(n_patients, seed)
    store = store or Store()
    resources, rows = [], []
    for p, daily in zip(cohort, data):
        pseudo = pseudonymize(p.pid)
        r = mob.detect(daily)
        for w, (wk, z) in enumerate(zip(r["weekly"], r["z"])):
            store.add_week(pseudo, w, wk[0], wk[1], wk[2], z)
        if r["alert_week"] is not None:
            w = r["alert_week"]
            obs = fx.build_mobility_observation(pseudo, w, start_date, r["weekly"][w], r["z"][w], r["min_z"])
            store.add_mobility_alert(pseudo, w, r["min_z"], obs)
            resources.append(obs)
        rows.append({"pid": p.pid, "pseudo": pseudo, "frailty": p.frailty, "kind": p.kind,
                     "onset_week": p.onset_week, "alert_week": r["alert_week"], "min_z": r["min_z"]})
    return {"cohort": cohort, "data": data, "rows": rows, "store": store, "observations": resources}
