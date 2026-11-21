"""
fog/gateway.py
--------------
The fog tier: a phone or home hub next to the wearer. It does what the wrist
cannot (decide, buffer, notify) and what the cloud should not (sit between
the fall and the caregiver when the internet is slow).

Decision rule for each edge event:
  prob <  CONFIRM_P                      drop (logged as a suppressed event)
  CONFIRM_P <= prob < IMMEDIATE_P        ask the wearer; alert unless cancelled
                                         within CANCEL_WINDOW_S
  prob >= IMMEDIATE_P                    alert at once
Repeated triggers inside ALERT_DEDUP_S are the same fall (bounces, rolling),
so only the first one alerts.

If the link to the cloud is down, alerts still go to the local caregiver
notifier and the event is buffered; the buffer is flushed when the link returns.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import (ALERT_DEDUP_S, CANCEL_WINDOW_S, CONFIRM_P, IMMEDIATE_P,
                      LATENCY_TARGET_S, LINK_DELAY_S)


@dataclass
class Decision:
    action: str                         # "alert" | "dropped" | "cancelled" | "duplicate"
    level: Optional[str]                # "immediate" | "after_timeout" | None
    t_impact: float
    prob: float
    latency_s: Optional[float] = None   # impact to caregiver notified
    within_target: Optional[bool] = None
    buffered: bool = False
    reason: str = ""


@dataclass
class Gateway:
    notifier: Optional[Callable] = None     # called with (patient, decision)
    link_up: bool = True
    cloud_sink: Optional[Callable] = None   # called with (patient, decision)
    last_alert_t: dict = field(default_factory=dict)
    buffer: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def handle(self, patient: str, ev, cancelled: bool = False) -> Decision:
        """ev is an EdgeEvent. cancelled simulates the wearer pressing 'I am OK'."""
        base = (ev.t_ready - ev.t_impact) + ev.proc_ms / 1000.0        # 2 s window + compute

        if ev.prob < CONFIRM_P:
            return self._record(patient, Decision("dropped", None, ev.t_impact, ev.prob,
                                                  reason="below confirmation threshold"))

        last = self.last_alert_t.get(patient)
        if last is not None and ev.t_impact - last < ALERT_DEDUP_S:
            return self._record(patient, Decision("duplicate", None, ev.t_impact, ev.prob,
                                                  reason="same fall as the previous alert"))

        if ev.prob >= IMMEDIATE_P:
            level, latency = "immediate", base + LINK_DELAY_S
        else:
            if cancelled:
                return self._record(patient, Decision("cancelled", None, ev.t_impact, ev.prob,
                                                      reason="wearer pressed 'I am OK'"))
            level, latency = "after_timeout", base + CANCEL_WINDOW_S + LINK_DELAY_S

        self.last_alert_t[patient] = ev.t_impact
        d = Decision("alert", level, ev.t_impact, ev.prob, latency_s=round(latency, 3),
                     within_target=latency <= LATENCY_TARGET_S, buffered=not self.link_up)
        if self.notifier:
            self.notifier(patient, d)          # local caregiver is notified even if the link is down
        if self.link_up:
            if self.cloud_sink:
                self.cloud_sink(patient, d)
        else:
            self.buffer.append((patient, d))
        return self._record(patient, d)

    def restore_link(self) -> int:
        """Flush buffered events to the cloud. Returns how many were sent."""
        self.link_up = True
        sent = 0
        while self.buffer:
            patient, d = self.buffer.pop(0)
            if self.cloud_sink:
                self.cloud_sink(patient, d)
            sent += 1
        return sent

    def _record(self, patient, d):
        self.log.append((patient, d))
        return d
