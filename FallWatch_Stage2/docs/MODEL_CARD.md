# Model card: FallWatch fall classifier and mobility detector

Written for the Thematic Seminar, Stage 2 (Arthur, Eduardo, Leonardo). Format follows the Class 11 suggestion of a model card plus a minimal audit by group.

## Intended use
Support a caregiver or a UBS/ACS team of older adults living at home: (1) raise an alert when a wrist wearable probably recorded a fall, and (2) flag a sustained fall in a person's own mobility so a human can follow up. It is a screening aid. It does not diagnose, and every output is marked "unconfirmed" in FHIR. Under ANVISA RDC 657/2022 a product with this purpose would have to be assessed as software as a medical device; this prototype has not been.

## Not intended for
Replacing an emergency call button, clinical decisions without a clinician, children, or any population not represented in the training data.

## Data
Everything is synthetic, produced by `src/fallwatch/sim/fall_data.py` and `sim/mobility_data.py`, written by the team. 90 simulated subjects (robust 50%, pre-frail 30%, frail 20%), 14 falls and 140 daily-activity events each, 50 Hz, wrist accelerometer only. Hard negatives are included (sitting, standing, plopping into a chair, stairs, stumble and recover, lying down, hitting an arm on a table). Roughly 5 to 30% of falls (by frailty) are soft slumps with a low impact. **No real older adult is in this data.** `data/sisfall_loader.py` can score the same pipeline on SisFall, but has not been run on the real files.

## Model
Trigger (any sample at or above 1.8 g) then a Random Forest (300 trees, depth 10) on 14 features computed over a 3 s window: nine from Stage 1 plus freefall depth, tilt change, post-impact stillness, post-impact energy and pre-impact activity. Tilt change is the most important feature (about 0.5 of the total importance). Fog rule: below 0.50 drop, 0.50 to 0.80 ask the wearer for 20 s, at or above 0.80 alert at once.

## Evaluation (subject-wise 5-fold, 1,000-resample bootstrap over subjects)
| Model | Sensitivity | Specificity | False alarms per patient-day |
|---|---|---|---|
| Peak at or above 2.5 g only | 79.2% | 76.3% | 10.09 |
| Stage 1 features | 80.2% | 97.7% | 1.00 |
| Stage 2 features (final) | 91.4% [89.1, 93.5] | 99.4% [99.3, 99.6] | 0.24 [0.19, 0.29] |

Sensitivity is end to end: falls that never crossed the 1.8 g trigger (4.7%) count as missed. False alarms per day assume about 42.6 trigger-capable activities per day, taken from the simulator's activity mix. Alert latency is 2.0 s (waiting for the post-impact window) + about 9 ms of compute + 0.2 s simulated link = 2.21 s for confident alerts, and 22 s when the wearer is asked first.

Mobility (200 simulated patients, patient level): the personal z-score rule (z below -1.5 for 2 weeks) flags 92% of simulated decliners with a median lead time of 2 weeks, but leaves 20% of stable patients and 82% of slow-drift patients flagged; specificity excluding slow drift is 80%, **below the 85% target**. A gradient boosting comparator reaches 99% specificity at 73% sensitivity. Requiring 3 weeks instead of 2 gives 84% sensitivity and 97% specificity in the sweep (`outputs/mobility_metrics.json`). Choosing between them is a clinical decision about the cost of a missed decline versus a needless visit.

## Audit by group (frailty)
| Group | Sensitivity | False alarms per day |
|---|---|---|
| Robust | 97.3% | 0.24 |
| Pre-frail | 88.2% | 0.19 |
| Frail | 79.1% | 0.37 |

Hard falls are detected 99.0% of the time; **soft slump falls only 46.1%**. The group that falls most gently, and is most at risk, is served worst. This is an equal-opportunity gap in the Class 11 sense and the main limitation to fix before any real use.

## Limitations and known gaps
- Synthetic data authored by the same team that wrote the classifier. The simulator encodes our assumptions about what a fall looks like, so high scores are expected and are not evidence about real people. The real-world benchmark (Bagala et al. 2012: mean sensitivity 57%, best 82.8%, on 29 real falls) is the realistic reference.
- Accelerometer only; no gyroscope, no barometer, no heart rate.
- The false alarm estimate depends on the assumed activity mix.
- Mobility signal is simulated; real step counts are affected by weather, illness and wearing habits.
- No standard FHIR/LOINC code exists for "decline against personal baseline"; the weekly step count uses LOINC 41950-7 and the z-score is a free-text component.
- No clinical validation, no ANVISA assessment, no user study.

## Privacy
Raw samples never leave the wearable. The cloud stores events and weekly aggregates under an HMAC-SHA256 pseudonym; the key stays with the health unit. Processing requires recorded consent (LGPD art. 7 and 11 would apply to health data; art. 20 gives the right to review of automated decisions).

## Next steps
1. Soften the miss on slow, low-impact falls (add post-fall stillness over a longer window, add a gyroscope) and re-audit by frailty group.
2. Replace the simulator with SisFall, then UP-Fall, and finally a small supervised pilot.
3. Add a clinician-in-the-loop review of every mobility alert and log overrides.
