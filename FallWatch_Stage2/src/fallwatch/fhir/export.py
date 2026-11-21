"""
fhir/export.py
--------------
Both pipelines end here. A fall event becomes a FHIR R4 Condition; a mobility
decline becomes a FHIR R4 Observation. Both reference a Patient that carries
only a pseudonym, so the bundle can go to an EHR (for example the RNDS or a
UBS system) without exposing identity.

Changes since Stage 1:
  * effectivePeriod uses real ISO dates (Stage 1 wrote "day-84", which is not
    a valid FHIR dateTime).
  * The fall Condition is "unconfirmed": a wearable detects, a clinician diagnoses.
  * The mobility trend has no standard LOINC code, so the weekly step count uses
    LOINC 41950-7 and the personal-baseline z-score is carried as a component
    with text only. That gap is documented in docs/MODEL_CARD.md.
"""

from datetime import date, datetime, timedelta, timezone

from ..config import ICD10_FALL, LOINC_DAILY_STEPS, SNOMED_FALL

DEVICE_ID = "wrist-wearable-01"


def _coding(system, code, display):
    return {"system": system, "code": code, "display": display}


def iso_utc(seconds_from_start: float, start: datetime) -> str:
    return (start + timedelta(seconds=float(seconds_from_start))).astimezone(timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def build_patient(pseudo: str, frailty: str = None) -> dict:
    res = {"resourceType": "Patient", "id": pseudo,
           "identifier": [{"system": "urn:fallwatch:pseudonym", "value": pseudo}],
           "meta": {"tag": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue",
                             "code": "PSEUDED", "display": "pseudonymized"}]}}
    return res


def build_device() -> dict:
    return {"resourceType": "Device", "id": DEVICE_ID, "status": "active",
            "deviceName": [{"name": "Wrist accelerometer, 50 Hz (simulated)", "type": "user-friendly-name"}],
            "type": {"text": "Wearable accelerometer"}}


def build_fall_condition(pseudo: str, decision, start: datetime) -> dict:
    when = iso_utc(decision.t_impact, start)
    return {
        "resourceType": "Condition",
        "id": f"fall-{pseudo}-{int(decision.t_impact * 1000)}",
        "clinicalStatus": {"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/condition-clinical", "active", "Active")]},
        "verificationStatus": {"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/condition-ver-status", "unconfirmed", "Unconfirmed")]},
        "category": [{"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/condition-category", "problem-list-item", "Problem List Item")]}],
        "code": {"coding": [_coding(*SNOMED_FALL), _coding(*ICD10_FALL)], "text": "Fall detected by wearable"},
        "subject": {"reference": f"Patient/{pseudo}"},
        "onsetDateTime": when,
        "recordedDate": when,
        "note": [{"text": f"Detected by wearable classifier. Probability {decision.prob:.2f}; "
                          f"alert level {decision.level}; latency {decision.latency_s} s. "
                          f"Needs clinician confirmation."}],
    }


def build_mobility_observation(pseudo: str, week_index: int, start_date: date, weekly_row,
                               z: float, min_z: float = None) -> dict:
    """weekly_row = (steps, transitions, hours_out) means for that week."""
    d0 = start_date + timedelta(days=7 * week_index)
    d1 = d0 + timedelta(days=6)
    steps, transitions, hours_out = (float(x) for x in weekly_row)
    return {
        "resourceType": "Observation",
        "id": f"mob-{pseudo}-w{week_index}",
        "status": "final",
        "category": [{"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/observation-category", "activity", "Activity")]}],
        "code": {"coding": [_coding(*LOINC_DAILY_STEPS)]},
        "subject": {"reference": f"Patient/{pseudo}"},
        "device": {"reference": f"Device/{DEVICE_ID}"},
        "effectivePeriod": {"start": d0.isoformat(), "end": d1.isoformat()},
        "valueQuantity": {"value": round(steps, 1), "unit": "steps/day",
                          "system": "http://unitsofmeasure.org", "code": "/d"},
        "interpretation": [{"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "L", "Low")]}],
        "component": [
            {"code": {"text": "Composite mobility z-score vs personal baseline"},
             "valueQuantity": {"value": round(float(z), 2), "unit": "z-score",
                               "system": "http://unitsofmeasure.org", "code": "{score}"}},
            {"code": {"text": "Sit-to-stand transitions per day"},
             "valueQuantity": {"value": round(transitions, 1), "unit": "per day",
                               "system": "http://unitsofmeasure.org", "code": "/d"}},
            {"code": {"text": "Hours out of the house per day"},
             "valueQuantity": {"value": round(hours_out, 2), "unit": "h/day",
                               "system": "http://unitsofmeasure.org", "code": "h/d"}},
        ],
        "note": [{"text": "Sustained decline against the patient's own baseline. "
                          "Screening signal, not a diagnosis."}],
    }


def build_bundle(resources: list) -> dict:
    return {"resourceType": "Bundle", "type": "collection",
            "entry": [{"resource": r} for r in resources]}
