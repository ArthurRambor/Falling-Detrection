"""
evaluation/fall_eval.py
-----------------------
Pipeline 1 evaluation, built to answer the three questions a reviewer asks:

  * Is it good on people the model has never seen?   -> subject-wise GroupKFold
  * Is that number honest end to end?                -> trigger misses count as missed falls
  * Is it usable in a home?                          -> false alarms per patient-day

Everything runs on the documented simulator, so the absolute numbers say how
the method behaves on OUR synthetic data, not how it would behave on real
older adults. docs/MODEL_CARD.md says so too.
"""

import json
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from ..config import (BENCHMARK_REAL_WORLD, OUTPUTS_DIR, PEAK_IDX, TARGET_FALSE_ALARMS_PER_DAY,
                      TARGET_SENSITIVITY, TARGET_SPECIFICITY)
from ..edge.fall_model import make_classifier, triggered_mask, train, predict_window
from ..edge.features import FEATURE_NAMES, STAGE1_FEATURES, feature_matrix
from ..sim.fall_data import ADL_EVENTS_PER_DAY, FRAILTY_GROUPS, build_dataset


def oof_probabilities(ds, feature_names, folds=5, seed=42, X_all=None, trig=None):
    """Out-of-fold P(fall) for every window; untriggered windows get 0.0."""
    trig = triggered_mask(ds["windows"]) if trig is None else trig
    if X_all is None:
        X_all = feature_matrix(ds["windows"][trig], FEATURE_NAMES)
    cols = [FEATURE_NAMES.index(n) for n in feature_names]
    X, y, g = X_all[:, cols], ds["labels"][trig], ds["subject_ids"][trig]
    p_trig = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        p_trig[te] = make_classifier(seed, n_jobs=-1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    p = np.zeros(len(ds["labels"]))
    p[trig] = p_trig
    return p


def _counts(pred, ds):
    """Per-subject confusion counts, so the bootstrap can resample subjects cheaply."""
    y, sid = ds["labels"], ds["subject_ids"]
    n = int(sid.max()) + 1
    out = np.zeros((n, 4))                 # tp, falls, fp, adl
    for s in range(n):
        m = sid == s
        out[s] = [(pred[m] & (y[m] == 1)).sum(), (y[m] == 1).sum(),
                  (pred[m] & (y[m] == 0)).sum(), (y[m] == 0).sum()]
    return out


def _metrics(c):
    tp, falls, fp, adl = c.sum(axis=0)
    sens, fpr = tp / falls, fp / adl
    return {"sensitivity": sens, "specificity": 1 - fpr, "fp_rate": fpr,
            "false_alarms_per_day": fpr * ADL_EVENTS_PER_DAY}


def cluster_bootstrap(counts, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(counts)
    samples = [_metrics(counts[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return {k: [float(np.percentile([s[k] for s in samples], 2.5)),
                float(np.percentile([s[k] for s in samples], 97.5))] for k in samples[0]}


def summarize(pred, ds, n_boot=1000):
    c = _counts(pred, ds)
    m = {k: float(v) for k, v in _metrics(c).items()}
    m["ci95"] = cluster_bootstrap(c, n_boot)
    return m


def run(n_subjects=90, seed=42, n_boot=1000, save=True, verbose=True):
    ds = build_dataset(n_subjects=n_subjects, seed=seed)
    trig = triggered_mask(ds["windows"])
    y, ev = ds["labels"], ds["event_types"]
    X_all = feature_matrix(ds["windows"][trig], FEATURE_NAMES)

    res = {"n_subjects": n_subjects, "seed": seed, "n_windows": int(len(y)),
           "n_falls": int(y.sum()), "adl_events_per_day": ADL_EVENTS_PER_DAY,
           "trigger_recall": float(trig[y == 1].mean()),
           "adl_trigger_rate": float(trig[y == 0].mean())}

    # ---- three models, same protocol
    p2 = oof_probabilities(ds, FEATURE_NAMES, seed=seed, X_all=X_all, trig=trig)
    p1 = oof_probabilities(ds, STAGE1_FEATURES, seed=seed, X_all=X_all, trig=trig)
    peak = np.linalg.norm(ds["windows"][:, PEAK_IDX - 10:PEAK_IDX + 20, :], axis=2).max(axis=1)
    res["models"] = {
        "threshold_only_2.5g": summarize(peak >= 2.5, ds, n_boot),
        "stage1_features": summarize(p1 >= 0.5, ds, n_boot),
        "stage2_features": summarize(p2 >= 0.5, ds, n_boot),
    }

    # ---- final model: by frailty group and by activity type
    pred = p2 >= 0.5
    frailty = np.array([ds["subjects"][s].frailty for s in ds["subject_ids"]])
    res["by_frailty"] = {}
    for g in FRAILTY_GROUPS:
        m = frailty == g
        falls, adl = (m & (y == 1)).sum(), (m & (y == 0)).sum()
        sens = (pred[m] & (y[m] == 1)).sum() / falls
        fpr = (pred[m] & (y[m] == 0)).sum() / adl
        res["by_frailty"][g] = {"n_subjects": int(len({s for s, f in zip(ds["subject_ids"], frailty) if f == g})),
                                "n_falls": int(falls), "sensitivity": float(sens),
                                "false_alarms_per_day": float(fpr * ADL_EVENTS_PER_DAY)}
    res["by_activity"] = {}
    for k in sorted(set(ev)):
        m = ev == k
        res["by_activity"][k] = {"n": int(m.sum()), "triggered": float(trig[m].mean()),
                                 "flagged": float(pred[m].mean())}
    soft = ev == "fall_soft"
    res["soft_fall_sensitivity"] = float(pred[soft].mean())
    res["hard_fall_sensitivity"] = float(pred[ev == "fall_hard"].mean())

    # ---- trade-off curve (sensitivity vs false alarms per day)
    res["sweep"] = []
    for t in np.round(np.arange(0.10, 0.96, 0.05), 2):
        m = _metrics(_counts(p2 >= t, ds))
        res["sweep"].append({"threshold": float(t), "sensitivity": float(m["sensitivity"]),
                             "false_alarms_per_day": float(m["false_alarms_per_day"])})

    # ---- compute cost of one decision on this machine
    bundle = train(ds, seed=seed)
    wins = ds["windows"][np.where(trig)[0][:150]]
    times = []
    for w in wins:
        t0 = time.perf_counter(); bundle["model"].n_jobs = 1; predict_window(bundle, w)
        times.append((time.perf_counter() - t0) * 1000)
    res["inference_ms"] = {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95))}
    res["alert_latency_s_immediate"] = float(2.0 + np.mean(times) / 1000 + 0.2)

    # ---- targets and benchmarks
    f = res["models"]["stage2_features"]
    res["targets"] = {
        "sensitivity": {"target": TARGET_SENSITIVITY, "value": f["sensitivity"],
                        "met": f["sensitivity"] >= TARGET_SENSITIVITY},
        "specificity": {"target": TARGET_SPECIFICITY, "value": f["specificity"],
                        "met": f["specificity"] >= TARGET_SPECIFICITY},
        "false_alarms_per_day": {"target": TARGET_FALSE_ALARMS_PER_DAY, "value": f["false_alarms_per_day"],
                                 "met": f["false_alarms_per_day"] < TARGET_FALSE_ALARMS_PER_DAY},
    }
    res["benchmark_real_world"] = BENCHMARK_REAL_WORLD
    res["importance"] = sorted(zip(FEATURE_NAMES, make_classifier(seed).fit(
        X_all, y[trig]).feature_importances_.tolist()), key=lambda kv: -kv[1])

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        (OUTPUTS_DIR / "fall_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    if verbose:
        print_report(res)
    return res


def print_report(r):
    print(f"\nPipeline 1: fall detection ({r['n_subjects']} simulated subjects, {r['n_falls']} falls, "
          f"{r['n_windows'] - r['n_falls']} daily-activity events)")
    print(f"Trigger recall {r['trigger_recall']:.1%}; daily-activity events passing the trigger "
          f"{r['adl_trigger_rate']:.1%}")
    print(f"{'model':24s}{'sensitivity':>22s}{'specificity':>22s}{'false alarms/day':>22s}")
    for name, m in r["models"].items():
        ci = m["ci95"]
        print(f"{name:24s}"
              f"{m['sensitivity']:>8.1%} [{ci['sensitivity'][0]:.1%},{ci['sensitivity'][1]:.1%}]"
              f"{m['specificity']:>9.1%} [{ci['specificity'][0]:.1%},{ci['specificity'][1]:.1%}]"
              f"{m['false_alarms_per_day']:>8.2f} [{ci['false_alarms_per_day'][0]:.2f},{ci['false_alarms_per_day'][1]:.2f}]")
    print("By frailty group:", {g: f"{v['sensitivity']:.1%} sens, {v['false_alarms_per_day']:.2f} FA/day"
                                for g, v in r["by_frailty"].items()})
    print(f"Hard falls {r['hard_fall_sensitivity']:.1%}, soft slump falls {r['soft_fall_sensitivity']:.1%}")
    print(f"Inference {r['inference_ms']['mean']:.1f} ms; immediate-alert latency {r['alert_latency_s_immediate']:.2f} s")
    for k, v in r["targets"].items():
        print(f"  target {k}: {v['value']:.3f} vs {v['target']} -> {'met' if v['met'] else 'NOT met'}")


if __name__ == "__main__":
    run()
