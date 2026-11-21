"""
cloud/store.py
--------------
SQLite as a stand-in for the cloud database. Only pseudonymised, minimised
data goes in: events and weekly aggregates, never raw accelerometer samples.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS fall_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT, t_impact REAL, prob REAL,
  level TEXT, latency_s REAL, within_target INTEGER, buffered INTEGER, fhir TEXT);
CREATE TABLE IF NOT EXISTS mobility_weeks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT, week INTEGER, steps REAL,
  transitions REAL, hours_out REAL, z REAL);
CREATE TABLE IF NOT EXISTS mobility_alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT, week INTEGER, min_z REAL, fhir TEXT);
"""


class Store:
    def __init__(self, path=":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def add_fall(self, patient, d, fhir_resource=None):
        self.db.execute(
            "INSERT INTO fall_events (patient,t_impact,prob,level,latency_s,within_target,buffered,fhir)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (patient, d.t_impact, d.prob, d.level, d.latency_s, int(bool(d.within_target)),
             int(d.buffered), json.dumps(fhir_resource) if fhir_resource else None))
        self.db.commit()

    def add_week(self, patient, week, steps, transitions, hours_out, z):
        self.db.execute(
            "INSERT INTO mobility_weeks (patient,week,steps,transitions,hours_out,z) VALUES (?,?,?,?,?,?)",
            (patient, int(week), float(steps), float(transitions), float(hours_out),
             None if z != z else float(z)))
        self.db.commit()

    def add_mobility_alert(self, patient, week, min_z, fhir_resource=None):
        self.db.execute("INSERT INTO mobility_alerts (patient,week,min_z,fhir) VALUES (?,?,?,?)",
                        (patient, int(week), float(min_z),
                         json.dumps(fhir_resource) if fhir_resource else None))
        self.db.commit()

    def rows(self, table, where="", args=()):
        cur = self.db.execute(f"SELECT * FROM {table} {where}", args)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
