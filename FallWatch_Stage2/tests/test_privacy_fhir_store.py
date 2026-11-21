import datetime as dt
import pytest
from fallwatch.cloud.store import Store
from fallwatch.fhir import export as fx
from fallwatch.fhir.validate import validate, validate_bundle
from fallwatch.pipeline import run_fall_scenario, run_mobility_cohort
from fallwatch.privacy import ConsentRegistry, pseudonymize


def test_pseudonym_is_stable_keyed_and_hides_the_id():
    a, b = pseudonymize("maria", b"k1"), pseudonymize("maria", b"k1")
    assert a == b and a != pseudonymize("maria", b"k2") and "maria" not in a


def test_consent_is_required():
    c = ConsentRegistry()
    assert not c.has_consent("x")
    c.grant("x"); assert c.has_consent("x")
    c.revoke("x"); assert not c.has_consent("x")


def test_live_scenario_one_alert_for_the_real_fall(bundle):
    r = run_fall_scenario(bundle, seed=11)
    alerts = [d for _, d in r["decisions"] if d.action == "alert"]
    assert len(alerts) == 1 and alerts[0].latency_s < 5
    fall_t = [t for k, t in r["truth"] if k == "fall"][0]
    assert abs(alerts[0].t_impact - fall_t) < 0.5


def test_no_raw_samples_in_cloud_store(bundle):
    r = run_fall_scenario(bundle, seed=11)
    cols = list(r["store"].rows("fall_events")[0])
    assert not any("sample" in c or "window" in c or "accel" in c for c in cols)


def test_fhir_bundle_valid(bundle):
    r = run_fall_scenario(bundle, seed=11)
    assert validate_bundle(fx.build_bundle(r["resources"])) == []
    m = run_mobility_cohort(n_patients=20)
    assert validate_bundle(fx.build_bundle(m["observations"])) == []


def test_stage1_invalid_date_is_caught():
    obs = fx.build_mobility_observation("p-x", 3, dt.date(2026, 1, 5), (4000, 30, 3), -2.0)
    assert validate(obs) == []
    obs["effectivePeriod"]["start"] = "day-84"
    assert validate(obs) != []


def test_store_roundtrip():
    s = Store()
    s.add_week("p", 1, 1.0, 2.0, 3.0, float("nan"))
    assert s.rows("mobility_weeks")[0]["z"] is None
