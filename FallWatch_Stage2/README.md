# FallWatch: fall detection and mobility monitoring (Stage 2 prototype)

Thematic Seminar, Stage 2. Group: Arthur, Eduardo, Leonardo. Everything runs on **synthetic data** from a documented simulator.

```
wearable (edge)           phone / hub (fog)          cloud + EHR
accelerometer 50 Hz  -->  confirm, ask, dedupe,  --> SQLite store, FHIR R4
trigger >= 1.8 g          buffer if offline          Condition (fall)
Random Forest, 14 feat.   notify caregiver           Observation (mobility)
                          daily aggregates  ------>  personal z-score detector
```

## Quick start
```bash
pip install -r requirements.txt
python demo.py all                 # train, live fall scenario, mobility cohort (about 15 s)
python demo.py live --link-down    # same stream, internet cut at the fall
python demo.py evaluate            # full evaluation, writes outputs/*.json (about 40 s)
streamlit run app/dashboard.py     # dashboard (needs `demo.py evaluate` once for the first tab)
python -m pytest -q                # 23 tests
```

## Who owns what
| Owner | Files |
|---|---|
| Arthur: edge, fog, simulator | `sim/fall_data.py`, `edge/features.py`, `edge/fall_model.py`, `edge/stream.py`, `fog/gateway.py` |
| Eduardo: cloud, FHIR | `cloud/mobility.py`, `cloud/store.py`, `sim/mobility_data.py`, `fhir/export.py`, `fhir/validate.py` |
| Leonardo: dashboard, privacy, evaluation | `app/dashboard.py`, `privacy.py`, `evaluation/*.py`, `docs/MODEL_CARD.md` |
| Shared | `config.py` (every constant), `pipeline.py`, `demo.py`, `tests/` |

## What changed since Stage 1
- Hard negatives and frailty groups in the simulator, so results are not a trivial 1.00.
- Trigger-gated classifier; sensitivity counts trigger misses; false alarms reported per patient-day.
- Orientation-change and post-impact features (sensitivity 80% to 91% at the same protocol).
- Std floor in the mobility baseline (Stage 1 produced z = -8.9 from a normal week).
- Valid FHIR dates (Stage 1 wrote "day-84"); Condition marked unconfirmed; validated with `fhir.resources`.
- Audit by frailty group and a model card.

## Honest status
Numbers come from our own simulator and describe the method, not real older adults. The weak spots (soft falls 46%, frail group 79%, mobility specificity 80% at the default threshold) are reported, not tuned away. See `docs/MODEL_CARD.md`. `data/sisfall_loader.py` is written and unit-tested on a synthetic SisFall-format file but has not been run on the real dataset.
