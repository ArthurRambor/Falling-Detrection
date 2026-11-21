"""
fhir/validate.py
----------------
Validates a resource or bundle against the FHIR R4 data model using the
`fhir.resources` library (R4B package, which is R4-compatible for these types).
Returns a list of problems; empty list means valid.
"""

import json


def validate(resource: dict) -> list:
    try:
        from fhir.resources.R4B import get_fhir_model_class
    except Exception as exc:           # library missing: say so instead of passing silently
        return [f"fhir.resources not available: {exc}"]
    try:
        cls = get_fhir_model_class(resource["resourceType"])
        cls.model_validate(json.loads(json.dumps(resource)))
        return []
    except Exception as exc:
        return [str(exc)[:300]]


def validate_bundle(bundle: dict) -> list:
    problems = validate(bundle)
    for entry in bundle.get("entry", []):
        problems += [f"{entry['resource']['resourceType']}: {p}" for p in validate(entry["resource"])]
    return problems
