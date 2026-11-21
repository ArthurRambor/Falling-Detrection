"""
demo.py - one entry point for the whole prototype.

    python demo.py train       train the fall model and save it
    python demo.py evaluate    full evaluation of both pipelines (about 40 s)
    python demo.py live        one continuous stream through edge -> fog -> cloud -> FHIR
    python demo.py live --link-down     same, with the internet cut at the moment of the fall
    python demo.py cohort      Pipeline 2 on a simulated cohort, with a FHIR Observation
    python demo.py all         train + live + cohort (the fast path for the walkthrough)

Dashboard:  streamlit run app/dashboard.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fallwatch.config import MODELS_DIR, OUTPUTS_DIR, DATA_DIR            # noqa: E402
from fallwatch.edge import fall_model as fm                               # noqa: E402
from fallwatch.fhir.export import build_bundle                            # noqa: E402
from fallwatch.fhir.validate import validate_bundle                       # noqa: E402
from fallwatch.pipeline import run_fall_scenario, run_mobility_cohort     # noqa: E402
from fallwatch.sim.fall_data import build_dataset                         # noqa: E402

MODEL_PATH = MODELS_DIR / "fall_rf.joblib"


def line(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def cmd_train(_):
    line("TRAIN  (edge tier: Random Forest on 14 hand-crafted features)")
    ds = build_dataset()
    bundle = fm.train(ds)
    MODELS_DIR.mkdir(exist_ok=True)
    fm.save(bundle, MODEL_PATH)
    print(f"trained on {bundle['n_train_windows']} triggered windows from {bundle['n_train_subjects']} simulated subjects")
    print("top features:", ", ".join(f"{n} ({v:.2f})" for n, v in fm.feature_importance(bundle)[:5]))
    print(f"saved {MODEL_PATH.relative_to(Path(__file__).parent)}")


def _bundle():
    if not MODEL_PATH.exists():
        cmd_train(None)
    return fm.load(MODEL_PATH)


def cmd_evaluate(_):
    from fallwatch.evaluation import fall_eval, mobility_eval
    fall_eval.run()
    mobility_eval.run()
    print(f"\nmetrics written to {OUTPUTS_DIR.name}/fall_metrics.json and mobility_metrics.json")


def cmd_live(a):
    line("LIVE  (one continuous 68 s stream, 4 events: 3 look-alikes then 1 real fall)")
    r = run_fall_scenario(_bundle(), link_down_for_first_alert=a.link_down)
    print("ground truth :", [(k, f"t={t:.0f}s") for k, t in r["truth"]])
    print(f"stream       : {len(r['samples'])} samples at 50 Hz, {len(r['events'])} passed the 1.8 g trigger\n")
    for ev, d in r["decisions"]:
        what = min(r["truth"], key=lambda kt: abs(kt[1] - ev.t_impact))[0]
        extra = f"  -> ALERT ({d.level}), latency {d.latency_s} s, within 5 s: {d.within_target}" if d.action == "alert" \
            else f"  -> {d.action}"
        print(f"  t={ev.t_impact:5.1f}s  peak {ev.peak_g:4.2f} g  P(fall)={ev.prob:4.2f}  [{what}]{extra}")
    print(f"\ncaregiver notifications: {len(r['notified'])}")
    if a.link_down:
        print(f"link was down: alert went to the local caregiver anyway; {r['flushed']} buffered event(s) flushed to the cloud on reconnect")
    cond = [x for x in r["resources"] if x["resourceType"] == "Condition"]
    if cond:
        c = cond[0]
        print("\nFHIR Condition sent to the cloud (full JSON: outputs/example_condition.json):")
        print(f"  code          SNOMED {c['code']['coding'][0]['code']} + ICD-10 {c['code']['coding'][1]['code']}")
        print(f"  status        {c['clinicalStatus']['coding'][0]['code']} / {c['verificationStatus']['coding'][0]['code']}")
        print(f"  onset         {c['onsetDateTime']}")
        print(f"  subject       {c['subject']['reference']}  (pseudonym, no name)")
        print(f"  note          {c['note'][0]['text']}")
        OUTPUTS_DIR.mkdir(exist_ok=True)
        (OUTPUTS_DIR / "example_condition.json").write_text(json.dumps(c, indent=2))
    problems = validate_bundle(build_bundle(r["resources"]))
    print("\nFHIR validation (fhir.resources R4B):", "VALID" if not problems else problems)
    print("raw accelerometer samples stored in the cloud: 0 (only the event and its probability)")


def cmd_cohort(_):
    line("COHORT  (Pipeline 2: personal-baseline z-score on 60 simulated patients, 20 weeks)")
    r = run_mobility_cohort()
    alerted = [x for x in r["rows"] if x["alert_week"] is not None]
    print(f"{len(alerted)} of {len(r['rows'])} patients alerted\n")
    for x in alerted[:8]:
        onset = f"onset wk {x['onset_week']}" if x["onset_week"] >= 0 else "no onset"
        print(f"  {x['pid']} ({x['frailty']:9s}) truth={x['kind']:13s} {onset:11s} alert at wk {x['alert_week']}  min z {x['min_z']:.1f}")
    if r["observations"]:
        o = r["observations"][0]
        print("\nFHIR Observation for the first alert (full JSON: outputs/example_observation.json):")
        print(f"  code          LOINC {o['code']['coding'][0]['code']}  value {o['valueQuantity']['value']} {o['valueQuantity']['unit']}")
        print(f"  period        {o['effectivePeriod']['start']} to {o['effectivePeriod']['end']}  (real ISO dates)")
        print(f"  z-score       {o['component'][0]['valueQuantity']['value']}")
        OUTPUTS_DIR.mkdir(exist_ok=True)
        (OUTPUTS_DIR / "example_observation.json").write_text(json.dumps(o, indent=2))
    problems = validate_bundle(build_bundle(r["observations"]))
    print("\nFHIR validation:", "VALID" if not problems else problems)


def cmd_all(a):
    cmd_train(a); cmd_live(a); cmd_cohort(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "evaluate", "live", "cohort", "all"])
    ap.add_argument("--link-down", action="store_true", help="cut the internet during the live scenario")
    args = ap.parse_args()
    {"train": cmd_train, "evaluate": cmd_evaluate, "live": cmd_live, "cohort": cmd_cohort, "all": cmd_all}[args.command](args)
