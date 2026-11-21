import numpy as np
from fallwatch.config import LATENCY_TARGET_S
from fallwatch.edge.stream import EdgeEvent, find_peaks
from fallwatch.fog.gateway import Gateway


def ev(t, p):
    return EdgeEvent(t_impact=t, t_ready=t + 2.0, peak_g=3.0, prob=p, proc_ms=10.0)


def test_low_probability_is_dropped():
    assert Gateway().handle("p", ev(10, 0.2)).action == "dropped"


def test_confident_alert_is_within_latency_target():
    d = Gateway().handle("p", ev(10, 0.95))
    assert d.action == "alert" and d.level == "immediate" and d.latency_s < LATENCY_TARGET_S


def test_uncertain_event_waits_and_can_be_cancelled():
    gw = Gateway()
    assert gw.handle("p", ev(10, 0.6), cancelled=True).action == "cancelled"
    d = gw.handle("p", ev(100, 0.6))
    assert d.action == "alert" and d.level == "after_timeout" and not d.within_target


def test_bounces_do_not_alert_twice():
    gw = Gateway()
    assert gw.handle("p", ev(10, 0.95)).action == "alert"
    assert gw.handle("p", ev(14, 0.95)).action == "duplicate"
    assert gw.handle("p", ev(100, 0.95)).action == "alert"


def test_offline_alert_reaches_caregiver_and_is_flushed_later():
    seen, cloud = [], []
    gw = Gateway(notifier=lambda p, d: seen.append(d), cloud_sink=lambda p, d: cloud.append(d), link_up=False)
    d = gw.handle("p", ev(10, 0.95))
    assert d.buffered and len(seen) == 1 and len(cloud) == 0
    assert gw.restore_link() == 1 and len(cloud) == 1


def test_trigger_refractory_period():
    x = np.zeros((1000, 3)); x[:, 2] = 1.0
    x[200] = [0, 0, 3.0]; x[220] = [0, 0, 3.0]; x[600] = [0, 0, 3.0]
    assert find_peaks(x) == [200, 600]          # 220 is inside the 5 s refractory window
