"""
evaluation/mobility_eval.py
---------------------------
Pipeline 2 evaluation at the PATIENT level (did we flag the people who declined,
and leave alone the people who did not?), plus lead time, a threshold sweep and
a machine-learning comparator.

Ground truth comes from the simulator: "decliner" has a real onset week;
"stable", "transient_dip" and "slow_drift" do not. The slow-drift group is the
awkward one (a gentle downward trend with no clinical event); it is reported
separately so nobody has to guess how it was counted.
"""

import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from ..config import BASELINE_WEEKS, BENCHMARK_MOBILITY, OUTPUTS_DIR, TARGET_SENSITIVITY, TARGET_SPECIFICITY
from ..cloud import mobility as mob
from ..sim.mobility_data import simulate_cohort

KINDS = ("decliner", "stable", "transient_dip", "slow_drift")


def _patient_metrics(rows):
    dec = [r for r in rows if r["kind"] == "decliner"]
    non = [r for r in rows if r["kind"] != "decliner"]
    nonslow = [r for r in non if r["kind"] != "slow_drift"]
    det = [r for r in dec if r["alert_week"] is not None]
    lead = [r["alert_week"] - r["onset_week"] for r in det]
    return {
        "sensitivity": len(det) / len(dec),
        "specificity_all_non_decliners": 1 - sum(r["alert_week"] is not None for r in non) / len(non),
        "specificity_excl_slow_drift": 1 - sum(r["alert_week"] is not None for r in nonslow) / len(nonslow),
        "false_alert_by_kind": {k: float(np.mean([r["alert_week"] is not None for r in rows if r["kind"] == k]))
                                for k in KINDS if any(r["kind"] == k for r in rows)},
        "median_lead_weeks": float(np.median(lead)) if lead else None,
        "n_decliners": len(dec), "n_non_decliners": len(non),
    }


def _week_features(daily):
    wk = mob.weekly_means(daily)
    base = wk[:BASELINE_WEEKS]
    mu, sd = base.mean(0), np.maximum(base.std(0, ddof=1), 0.05 * base.mean(0))
    z = (wk - mu) / sd                                   # (weeks, 3)
    feats = []
    for w in range(BASELINE_WEEKS, len(wk)):
        prev = z[w - 1] if w > BASELINE_WEEKS else z[w]
        feats.append(np.concatenate([z[w], prev, [z[w].mean(), (z[w].mean() + prev.mean()) / 2]]))
    return np.array(feats)


def ml_comparator(cohort, data, seed=0, sustain=2):
    """Gradient boosting on weekly z-features, patient-wise CV. Label = week at/after onset."""
    X, y, g, idx = [], [], [], []
    for i, (p, d) in enumerate(zip(cohort, data)):
        f = _week_features(d)
        weeks = np.arange(BASELINE_WEEKS, BASELINE_WEEKS + len(f))
        X.append(f); g += [i] * len(f); idx += [(i, w) for w in weeks]
        y += [int(p.kind == "decliner" and w >= p.onset_week) for w in weeks]
    X, y, g = np.vstack(X), np.array(y), np.array(g)
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        prob[te] = GradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    rows = []
    for i, p in enumerate(cohort):
        pw = prob[g == i]
        run, alert = 0, None
        for k, v in enumerate(pw):
            run = run + 1 if v >= 0.5 else 0
            if run >= sustain:
                alert = BASELINE_WEEKS + k
                break
        rows.append({"pid": p.pid, "kind": p.kind, "onset_week": p.onset_week, "alert_week": alert})
    return _patient_metrics(rows)


def run(n_patients=200, seed=7, save=True, verbose=True):
    cohort, data = simulate_cohort(n_patients, seed)
    res = {"n_patients": n_patients, "seed": seed}
    res["zscore_default"] = _patient_metrics(mob.detect_cohort(cohort, data))
    res["sweep"] = []
    for z in (-1.0, -1.5, -2.0, -2.5):
        for s in (2, 3):
            m = _patient_metrics(mob.detect_cohort(cohort, data, z_threshold=z, sustain=s))
            res["sweep"].append({"z_threshold": z, "sustain_weeks": s, "sensitivity": m["sensitivity"],
                                 "specificity_excl_slow_drift": m["specificity_excl_slow_drift"],
                                 "median_lead_weeks": m["median_lead_weeks"]})
    res["ml_gradient_boosting"] = ml_comparator(cohort, data)
    res["benchmark"] = BENCHMARK_MOBILITY
    res["targets"] = {"sensitivity": TARGET_SENSITIVITY, "specificity": TARGET_SPECIFICITY}
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        (OUTPUTS_DIR / "mobility_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    if verbose:
        print_report(res)
    return res


def print_report(r):
    z, ml = r["zscore_default"], r["ml_gradient_boosting"]
    print(f"\nPipeline 2: mobility decline ({r['n_patients']} simulated patients, patient-level)")
    print(f"{'method':26s}{'sensitivity':>13s}{'specificity*':>14s}{'specificity**':>15s}{'lead (wk)':>11s}")
    for name, m in (("z-score (z<-1.5, 2 wk)", z), ("gradient boosting", ml)):
        print(f"{name:26s}{m['sensitivity']:>13.1%}{m['specificity_excl_slow_drift']:>14.1%}"
              f"{m['specificity_all_non_decliners']:>15.1%}{m['median_lead_weeks']:>11.1f}")
    print("  * excluding slow drift   ** all non-decliners incl. slow drift")
    print("False-alert rate by group (z-score):", {k: f"{v:.0%}" for k, v in z["false_alert_by_kind"].items()})
    print("Sweep:", [(s["z_threshold"], s["sustain_weeks"], f"{s['sensitivity']:.0%}",
                      f"{s['specificity_excl_slow_drift']:.0%}") for s in r["sweep"]])


if __name__ == "__main__":
    run()
