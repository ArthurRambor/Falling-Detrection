"""
config.py
---------
Every number that shapes the system lives here, so the code walkthrough can
point to one file when someone asks "why 50 Hz?" or "why 1.8 g?".
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------- sensing
FS = 50                       # wearable sampling rate, Hz
PRE_S = 1.0                   # seconds kept BEFORE the impact peak
POST_S = 2.0                  # seconds kept AFTER the impact peak
PRE_N = int(FS * PRE_S)       # 50 samples
POST_N = int(FS * POST_S)     # 100 samples
WINDOW_N = PRE_N + POST_N     # 150 samples = the 3-second window from Stage 1
PEAK_IDX = PRE_N              # the impact peak is always sample 50 of a window

# ---------------------------------------------------------------- edge
TRIGGER_G = 1.8               # a window is only scored if |a| crosses this
REFRACTORY_S = 5.0            # ignore new triggers for this long after one
PEAK_SEARCH_S = 0.3           # look this far past the first crossing for the true peak

# ---------------------------------------------------------------- fog
CONFIRM_P = 0.50              # below this probability the edge event is dropped
IMMEDIATE_P = 0.80            # at or above this the alert goes out at once
CANCEL_WINDOW_S = 20.0        # between CONFIRM_P and IMMEDIATE_P the wearer may cancel
ALERT_DEDUP_S = 60.0          # one alert per fall, not one per bounce
LINK_DELAY_S = 0.2            # simulated gateway to cloud latency
LATENCY_TARGET_S = 5.0        # slide 5: alert within 5 seconds of impact

# ---------------------------------------------------------------- mobility
BASELINE_WEEKS = 6            # first six weeks define the personal baseline
Z_THRESHOLD = -1.5            # composite weekly z below this counts as low
SUSTAIN_WEEKS = 2             # ...and it must last this many consecutive weeks
STD_FLOOR_FRAC = 0.05         # baseline std never below 5% of baseline mean
MOBILITY_METRICS = ("steps", "transitions", "hours_out")

# ---------------------------------------------------------------- Stage 2 targets (slide 7)
TARGET_SENSITIVITY = 0.80
TARGET_SPECIFICITY = 0.85
TARGET_FALSE_ALARMS_PER_DAY = 1.0
BENCHMARK_REAL_WORLD = {      # Bagala et al., 2012, PLOS ONE, 29 real falls
    "mean_sensitivity": 0.57,
    "best_sensitivity": 0.828,
    "mean_specificity": 0.83,
    "best_specificity": 0.98,
}
BENCHMARK_MOBILITY = {        # Frontiers in Public Health, 2026 meta-analysis
    "pooled_sensitivity": 0.55,
    "pooled_specificity": 0.89,
    "auc": 0.85,
    "auc_ml": 0.90,
    "auc_statistical": 0.79,
}

# ---------------------------------------------------------------- codes (verified)
SNOMED_FALL = ("http://snomed.info/sct", "217082002", "Accidental fall")
ICD10_FALL = ("http://hl7.org/fhir/sid/icd-10", "W19", "Unspecified fall")
LOINC_DAILY_STEPS = ("http://loinc.org", "41950-7", "Number of steps in 24 hour Measured")
