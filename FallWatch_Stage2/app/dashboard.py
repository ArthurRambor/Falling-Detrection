"""
app/dashboard.py  (Leonardo's module)
-------------------------------------
Caregiver and clinician view of the prototype.

    streamlit run app/dashboard.py

Four tabs: how well it works (evaluation), a live fall scenario, mobility trends
per patient, and the privacy / FHIR evidence. All data is synthetic and the page
says so at the top.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fallwatch.cloud import mobility as mob                                  # noqa: E402
from fallwatch.config import (BASELINE_WEEKS, FS, OUTPUTS_DIR, TRIGGER_G, Z_THRESHOLD)   # noqa: E402
from fallwatch.edge import fall_model as fm                                  # noqa: E402
from fallwatch.fhir.export import build_bundle                               # noqa: E402
from fallwatch.fhir.validate import validate_bundle                          # noqa: E402
from fallwatch.pipeline import run_fall_scenario, run_mobility_cohort        # noqa: E402
from fallwatch.sim.fall_data import build_dataset                            # noqa: E402

INK, MUTED, ACCENT, ALERT, GOOD = "#1f2933", "#8a949e", "#2f6fdb", "#d1481f", "#2a8a5b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

st.set_page_config(page_title="FallWatch dashboard", layout="wide")


@st.cache_resource
def get_bundle():
    path = ROOT / "models" / "fall_rf.joblib"
    if path.exists():
        return fm.load(path)
    b = fm.train(build_dataset())
    path.parent.mkdir(exist_ok=True)
    fm.save(b, path)
    return fm.load(path)


@st.cache_data
def get_metrics(name):
    p = OUTPUTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def get_cohort():
    r = run_mobility_cohort()
    return r["cohort"], r["data"], r["rows"], r["observations"]


st.title("FallWatch: fall detection and mobility monitoring")
st.warning("All data on this page is synthetic (documented simulator). Numbers describe the method on our "
           "simulated people, not performance on real older adults.", icon=None)

tab_eval, tab_live, tab_mob, tab_priv = st.tabs(
    ["Evaluation", "Live fall scenario", "Mobility trends", "Privacy and FHIR"])

# ------------------------------------------------------------------ evaluation
with tab_eval:
    fall, mobm = get_metrics("fall_metrics.json"), get_metrics("mobility_metrics.json")
    if not fall or not mobm:
        st.info("No evaluation results yet. Run `python demo.py evaluate` and reload this page.")
    else:
        st.subheader("Pipeline 1: fall detection (subject-wise cross-validation)")
        f = fall["models"]["stage2_features"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensitivity, end to end (target 80%)", f"{f['sensitivity']:.1%}")
        c2.metric("Specificity (target 85%)", f"{f['specificity']:.1%}")
        c3.metric("False alarms per patient-day (target < 1)", f"{f['false_alarms_per_day']:.2f}")
        c4.metric("Alert latency (target < 5 s)", f"{fall['alert_latency_s_immediate']:.2f} s")
        st.caption(f"95% CIs come from a bootstrap over subjects: sensitivity "
                   f"{f['ci95']['sensitivity'][0]:.1%} to {f['ci95']['sensitivity'][1]:.1%}. "
                   f"Sensitivity counts falls the 1.8 g trigger never saw ({1 - fall['trigger_recall']:.1%}).")

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            names = {"threshold_only_2.5g": ("Peak ≥ 2.5 g only", (-8, 10), "right"),
                     "stage1_features": ("Stage 1 features", (8, -14), "left"),
                     "stage2_features": ("Stage 2 features", (-8, 10), "right")}
            clip = lambda v: max(v, 0.05)
            sw = fall["sweep"]
            ax.plot([clip(s["false_alarms_per_day"]) for s in sw], [s["sensitivity"] for s in sw],
                    color=ACCENT, lw=1.2, alpha=.5)
            for k, m in fall["models"].items():
                label, off, ha = names[k]
                x_, y_ = clip(m["false_alarms_per_day"]), m["sensitivity"]
                ax.scatter(x_, y_, s=60, color=ACCENT if k == "stage2_features" else MUTED, zorder=3)
                ax.annotate(label, (x_, y_), xytext=off, textcoords="offset points", fontsize=9, color=INK, ha=ha)
            ax.axvline(1.0, color=ALERT, ls="--", lw=1)
            ax.text(1.08, 0.52, "limit: 1 false alarm/day", color=ALERT, fontsize=8, rotation=90)
            ax.set_xscale("log"); ax.set_xlim(0.05, 20)
            ax.set_xticks([0.05, 0.1, 1, 10]); ax.set_xticklabels(["<0.05", "0.1", "1", "10"])
            ax.set_xlabel("False alarms per patient-day (log scale)"); ax.set_ylabel("Sensitivity")
            ax.set_ylim(0.5, 1.0)
            st.pyplot(fig)
        with right:
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            groups = list(fall["by_frailty"])
            sens = [fall["by_frailty"][g]["sensitivity"] for g in groups]
            bars = ax.bar(groups, sens, color=[GOOD if s >= .8 else ALERT for s in sens], width=.55)
            for b_, s in zip(bars, sens):
                ax.text(b_.get_x() + b_.get_width() / 2, s + .01, f"{s:.0%}", ha="center", fontsize=10)
            ax.axhline(0.8, color=INK, ls="--", lw=1)
            ax.set_ylim(0, 1.08); ax.set_ylabel("Sensitivity")
            ax.set_title("Fairness audit by frailty group", fontsize=10, loc="left")
            st.pyplot(fig)
        frail = fall["by_frailty"]["frail"]["sensitivity"]
        st.markdown(f"**Where it is weak.** Frail simulated wearers are caught {frail:.0%} of the time, below the 80% "
                    f"target, and soft slump falls only {fall['soft_fall_sensitivity']:.0%}. The wearer who falls "
                    f"gently is the one most likely to be missed. Fixing this is the first item of future work.")

        st.subheader("Pipeline 2: mobility decline (patient level)")
        z, ml = mobm["zscore_default"], mobm["ml_gradient_boosting"]
        st.table({"Method": ["Personal z-score (z < -1.5, 2 weeks)", "Gradient boosting (ML tier)"],
                  "Sensitivity": [f"{z['sensitivity']:.0%}", f"{ml['sensitivity']:.0%}"],
                  "Specificity (excl. slow drift)": [f"{z['specificity_excl_slow_drift']:.0%}", f"{ml['specificity_excl_slow_drift']:.0%}"],
                  "Median lead time (weeks)": [z["median_lead_weeks"], ml["median_lead_weeks"]]})
        st.caption("Literature reference (Frontiers in Public Health 2026, 20 studies): pooled sensitivity 0.55, "
                   "specificity 0.89, ML AUC 0.90 vs statistical 0.79.")

# ------------------------------------------------------------------ live scenario
with tab_live:
    st.subheader("One continuous stream through edge, fog and cloud")
    cA, cB, cC = st.columns(3)
    seed = cA.number_input("Simulation seed", 1, 999, 11)
    link_down = cB.checkbox("Cut the internet at the fall", value=False)
    cancel = cC.checkbox("Wearer presses 'I am OK' on uncertain events", value=False)
    r = run_fall_scenario(get_bundle(), seed=int(seed), link_down_for_first_alert=link_down, cancel_uncertain=cancel)

    t = np.arange(len(r["samples"])) / FS
    mag = np.linalg.norm(r["samples"], axis=1)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, mag, color=MUTED, lw=.8)
    ax.axhline(TRIGGER_G, color=ACCENT, ls=":", lw=1)
    ax.text(t[-1], TRIGGER_G + .05, "trigger 1.8 g", ha="right", color=ACCENT, fontsize=8)
    for kind, ti in r["truth"]:
        ax.annotate(kind.replace("_", " "), (ti, mag[int(ti * FS)]), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    for ev, d in r["decisions"]:
        ax.scatter(ev.t_impact, ev.peak_g, s=70, color=ALERT if d.action == "alert" else ACCENT, zorder=3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|acceleration| (g)")
    st.pyplot(fig)
    st.caption("Red dot: alert sent. Blue dot: passed the trigger but dropped or cancelled. No dot: never reached the classifier.")

    rows = []
    for ev, d in r["decisions"]:
        truth = min(r["truth"], key=lambda kt: abs(kt[1] - ev.t_impact))[0]
        rows.append({"time (s)": round(ev.t_impact, 1), "what really happened": truth,
                     "peak (g)": round(ev.peak_g, 2), "P(fall)": round(ev.prob, 2), "decision": d.action,
                     "latency (s)": d.latency_s, "buffered offline": d.buffered})
    st.dataframe(rows, hide_index=True)
    if link_down:
        st.info(f"Link was down: the local caregiver was notified at once, and {r['flushed']} event(s) were "
                f"flushed to the cloud when it came back.")
    st.metric("Caregiver notifications", len(r["notified"]))

# ------------------------------------------------------------------ mobility
with tab_mob:
    cohort, data, rows, obs = get_cohort()
    st.subheader("Weekly mobility against each person's own baseline")
    pick = st.selectbox("Patient", [f"{x['pid']}  ({x['frailty']})" for x in rows], index=3)
    i = [f"{x['pid']}  ({x['frailty']})" for x in rows].index(pick)
    res = mob.detect(data[i])
    wk, z = res["weekly"], res["z"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(range(len(wk)), wk[:, 0], color=INK, marker="o", ms=3)
    axes[0].axvspan(-.5, BASELINE_WEEKS - .5, color=ACCENT, alpha=.08)
    axes[0].text(2.5, wk[:, 0].min(), "baseline", color=ACCENT, fontsize=8, ha="center", va="bottom")
    axes[0].set_xlabel("Week"); axes[0].set_ylabel("Steps per day")
    axes[1].plot(range(len(z)), z, color=INK, marker="o", ms=3)
    axes[1].axhline(Z_THRESHOLD, color=ALERT, ls="--", lw=1)
    axes[1].set_xlabel("Week"); axes[1].set_ylabel("Composite z-score")
    if res["alert_week"] is not None:
        axes[1].axvline(res["alert_week"], color=ALERT, lw=1.5)
        axes[1].text(res["alert_week"] - .2, np.nanmax(z), "alert", color=ALERT, ha="right", fontsize=9)
    st.pyplot(fig)
    x = rows[i]
    st.write(f"Simulated truth: **{x['kind'].replace('_', ' ')}**"
             + (f", onset week {x['onset_week']}" if x["onset_week"] >= 0 else "")
             + (f". Alert raised at the end of week {x['alert_week']}." if x["alert_week"] is not None else ". No alert."))
    n_alert = sum(r_["alert_week"] is not None for r_ in rows)
    st.caption(f"{n_alert} of {len(rows)} patients in this simulated cohort were flagged. A sustained fall of the "
               f"composite below {Z_THRESHOLD} for 2 weeks in a row is a screening signal for the ACS or UBS team, not a diagnosis.")

# ------------------------------------------------------------------ privacy / FHIR
with tab_priv:
    st.subheader("What leaves the wearer, and in what format")
    r = run_fall_scenario(get_bundle(), seed=11)
    cond = next(x for x in r["resources"] if x["resourceType"] == "Condition")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Privacy controls**")
        st.markdown("- Raw accelerometer windows stay on the wearable. The cloud stores the event, its probability "
                    "and the time.\n- Patients appear as an HMAC-SHA256 pseudonym; the key stays with the health unit.\n"
                    "- Nothing is processed for a patient without consent.\n- The Condition is marked "
                    "*unconfirmed*: the device detects, a clinician diagnoses.")
        problems = validate_bundle(build_bundle(r["resources"]))
        if problems:
            st.error(str(problems))
        else:
            st.success("FHIR R4 validation: valid (fhir.resources)")
        st.write("Event stored in the cloud database:", {k: v for k, v in r["store"].rows("fall_events")[0].items() if k != "fhir"})
    with c2:
        st.markdown("**FHIR Condition sent to the EHR**")
        st.json(cond, expanded=2)
