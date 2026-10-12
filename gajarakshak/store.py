"""Event / audit store.

  EventStore     SQLite, zero setup. Tests, benchmark, laptop demo.
  PgEventStore   PostgreSQL. Uses PostGIS (geometry for towers and zones, spatial queries) and
                 TimescaleDB (events as a hypertable) when the extensions are available — the
                 docker-compose image (timescale/timescaledb-ha) ships both. Without them it
                 still works as plain Postgres.

Writes never block the simulation/safety loop: PgEventStore queues them and a writer thread
flushes in batches. Reads (dashboard, assistant) go straight to the database.
"""
from __future__ import annotations

import csv
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("gajarakshak.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_ts  REAL NOT NULL,
    clock   TEXT NOT NULL,
    topic   TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_topic_ts ON events(topic, sim_ts);
"""


class EventStore:
    kind = "sqlite"

    def __init__(self, path: str | None = None):
        self.path = path or ":memory:"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.now = 0.0
        self.clock = ""
        self._pending = 0

    def on_bus(self, topic: str, msg: dict) -> None:
        self.write(topic, msg)

    def write(self, topic: str, msg: dict) -> None:
        self.db.execute("INSERT INTO events(sim_ts, clock, topic, payload) VALUES (?,?,?,?)",
                        (self.now, self.clock, topic, json.dumps(msg, default=str)))
        self._pending += 1
        if self._pending >= 200:
            self.db.commit()
            self._pending = 0

    def recent(self, limit: int = 100, topic_prefix: str | None = None) -> list[dict]:
        self.db.commit()
        q = "SELECT sim_ts, clock, topic, payload FROM events"
        args: tuple = ()
        if topic_prefix:
            q += " WHERE topic LIKE ?"
            args = (topic_prefix + "%",)
        q += " ORDER BY id DESC LIMIT ?"
        rows = self.db.execute(q, args + (limit,)).fetchall()
        return [{"t": r[0], "clock": r[1], "topic": r[2], "data": json.loads(r[3])} for r in rows]

    def export_csv(self, out: str | Path) -> Path:
        self.db.commit()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sim_ts", "clock", "topic", "payload"])
            for row in self.db.execute("SELECT sim_ts, clock, topic, payload FROM events ORDER BY id"):
                w.writerow(row)
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "path": self.path}

    def close(self) -> None:
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass


PG_DDL = """
CREATE TABLE IF NOT EXISTS towers (
    id text PRIMARY KEY, chainage_m double precision NOT NULL, lateral_m double precision NOT NULL, link text);
CREATE TABLE IF NOT EXISTS zones (
    id text PRIMARY KEY, start_m double precision NOT NULL, end_m double precision NOT NULL, hotspot boolean NOT NULL);
CREATE TABLE IF NOT EXISTS hotspots (
    id text PRIMARY KEY, name text NOT NULL, chainage_m double precision NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id bigint GENERATED ALWAYS AS IDENTITY, ts timestamptz NOT NULL DEFAULT now(), run_id text NOT NULL,
    sim_ts double precision NOT NULL, clock text NOT NULL, topic text NOT NULL, payload jsonb NOT NULL,
    PRIMARY KEY (id, ts));
CREATE INDEX IF NOT EXISTS ix_events_topic ON events (topic, ts DESC);
CREATE TABLE IF NOT EXISTS alerts (
    run_id text NOT NULL, id text NOT NULL, zone text NOT NULL, line text NOT NULL, level text NOT NULL,
    max_level text NOT NULL, status text NOT NULL, reason text, opened_sim double precision,
    cleared_sim double precision, controller_ack_sim double precision, log jsonb,
    updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (run_id, id));
CREATE TABLE IF NOT EXISTS tower_status (
    tower text PRIMARY KEY, status text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now());
"""

# Section coordinates: x = chainage (m), y = lateral offset (m), SRID 0 (local linear reference).
# Production uses surveyed coordinates (EPSG:32643, UTM zone 43N); the queries do not change.
PG_GIS = """
ALTER TABLE towers ADD COLUMN IF NOT EXISTS geom geometry(Point, 0);
ALTER TABLE zones  ADD COLUMN IF NOT EXISTS geom geometry(Polygon, 0);
CREATE INDEX IF NOT EXISTS ix_towers_geom ON towers USING gist (geom);
CREATE INDEX IF NOT EXISTS ix_zones_geom  ON zones  USING gist (geom);
UPDATE towers SET geom = ST_SetSRID(ST_MakePoint(chainage_m, lateral_m), 0);
UPDATE zones  SET geom = ST_SetSRID(ST_MakeEnvelope(start_m, -200, end_m, 200), 0);
"""


class PgEventStore:
    kind = "postgres"
    FLUSH_S = 0.5

    def __init__(self, url: str, cfg, run_id: str):
        import psycopg  # optional dependency

        self.url, self.run_id = url, run_id
        self.now, self.clock = 0.0, ""
        self.features = {"postgis": False, "timescale": False}
        self.stats = {"written": 0, "dropped": 0, "errors": 0}
        self.conn = psycopg.connect(url, autocommit=True)           # reads (event-loop thread)
        self._init_schema(cfg)
        self._q: queue.Queue = queue.Queue(maxsize=100_000)
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._run_writer, name="pg-writer", daemon=True)
        self._writer.start()

    # schema ------------------------------------------------------------------
    def _try(self, sql: str, params=None) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
            return True
        except Exception as e:
            log.info("optional SQL skipped (%s): %s", type(e).__name__, str(e).splitlines()[0][:120])
            return False

    def _init_schema(self, cfg) -> None:
        self.features["postgis"] = self._try("CREATE EXTENSION IF NOT EXISTS postgis")
        self.features["timescale"] = self._try("CREATE EXTENSION IF NOT EXISTS timescaledb")
        with self.conn.cursor() as cur:
            cur.execute(PG_DDL)
            for t in cfg.towers:
                cur.execute("INSERT INTO towers (id, chainage_m, lateral_m, link) VALUES (%s,%s,%s,%s) "
                            "ON CONFLICT (id) DO UPDATE SET chainage_m=EXCLUDED.chainage_m, "
                            "lateral_m=EXCLUDED.lateral_m, link=EXCLUDED.link",
                            (t["id"], t["x"], t["y"], t.get("link")))
            for z in cfg.zones:
                cur.execute("INSERT INTO zones (id, start_m, end_m, hotspot) VALUES (%s,%s,%s,%s) "
                            "ON CONFLICT (id) DO UPDATE SET start_m=EXCLUDED.start_m, end_m=EXCLUDED.end_m, "
                            "hotspot=EXCLUDED.hotspot", (z.id, z.start, z.end, z.hotspot))
            for h in cfg.hotspots:
                cur.execute("INSERT INTO hotspots (id, name, chainage_m) VALUES (%s,%s,%s) "
                            "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, chainage_m=EXCLUDED.chainage_m",
                            (h["id"], h["name"], h["x"]))
        if self.features["postgis"] and not self._try(PG_GIS):
            self.features["postgis"] = False
        if self.features["timescale"]:
            ok = (self._try("SELECT create_hypertable('events', 'ts', if_not_exists => TRUE, migrate_data => TRUE)")
                  or self._try("SELECT create_hypertable('events', by_range('ts'), if_not_exists => TRUE)"))
            self.features["timescale"] = ok
        log.info("Postgres store ready: %s", self.features)

    # writes (never block the caller) -----------------------------------------
    def on_bus(self, topic: str, msg: dict) -> None:
        self.write(topic, msg)

    def write(self, topic: str, msg: dict) -> None:
        try:
            self._q.put_nowait((self.now, self.clock, topic, json.dumps(msg, default=str), msg))
        except queue.Full:
            self.stats["dropped"] += 1

    def _run_writer(self) -> None:
        import psycopg
        conn = None
        while not self._stop.is_set() or not self._q.empty():
            batch = []
            deadline = time.monotonic() + self.FLUSH_S
            while len(batch) < 1000 and time.monotonic() < deadline:
                try:
                    batch.append(self._q.get(timeout=0.1))
                except queue.Empty:
                    if self._stop.is_set():
                        break
            if not batch:
                continue
            try:
                if conn is None or conn.closed:
                    conn = psycopg.connect(self.url)
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO events (run_id, sim_ts, clock, topic, payload) VALUES (%s,%s,%s,%s,%s::jsonb)",
                        [(self.run_id, s, c, t, p) for s, c, t, p, _ in batch])
                    for s, c, t, p, m in batch:
                        if t == "core/alerts":
                            a = m["alert"]
                            cur.execute(
                                "INSERT INTO alerts (run_id, id, zone, line, level, max_level, status, reason, "
                                "opened_sim, cleared_sim, controller_ack_sim, log, updated_at) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb, now()) "
                                "ON CONFLICT (run_id, id) DO UPDATE SET level=EXCLUDED.level, "
                                "max_level=EXCLUDED.max_level, status=EXCLUDED.status, reason=EXCLUDED.reason, "
                                "cleared_sim=EXCLUDED.cleared_sim, controller_ack_sim=EXCLUDED.controller_ack_sim, "
                                "log=EXCLUDED.log, updated_at=now()",
                                (self.run_id, a["id"], a["zone"], a["line"], a["level"], a["max_level"], a["status"],
                                 a["reason"], a["opened_at"], a["cleared_at"], a["controller_ack"], json.dumps(a["log"])))
                        elif t == "agents/health" and m.get("event") == "status":
                            cur.execute("INSERT INTO tower_status (tower, status, updated_at) VALUES (%s,%s,now()) "
                                        "ON CONFLICT (tower) DO UPDATE SET status=EXCLUDED.status, updated_at=now()",
                                        (m["tower"], m["to"]))
                conn.commit()
                self.stats["written"] += len(batch)
            except Exception as e:
                self.stats["errors"] += 1
                log.warning("Postgres write failed (%s); %d events lost", type(e).__name__, len(batch))
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                time.sleep(1.0)
        if conn is not None:
            conn.close()

    # reads ---------------------------------------------------------------------
    def recent(self, limit: int = 100, topic_prefix: str | None = None) -> list[dict]:
        q = "SELECT sim_ts, clock, topic, payload FROM events WHERE run_id = %s"
        args: list = [self.run_id]
        if topic_prefix:
            q += " AND topic LIKE %s"
            args.append(topic_prefix + "%")
        q += " ORDER BY ts DESC, id DESC LIMIT %s"
        args.append(limit)
        with self.conn.cursor() as cur:
            cur.execute(q, args)
            return [{"t": r[0], "clock": r[1], "topic": r[2], "data": r[3]} for r in cur.fetchall()]

    def nearest_towers(self, x: float, y: float, k: int = 3) -> list[dict] | None:
        """PostGIS KNN query. None when PostGIS is not available (caller falls back)."""
        if not self.features["postgis"]:
            return None
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, round(ST_Distance(geom, ST_SetSRID(ST_MakePoint(%s, %s), 0))::numeric) "
                        "FROM towers ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0) LIMIT %s",
                        (x, y, x, y, k))
            return [{"id": r[0], "distance_m": float(r[1])} for r in cur.fetchall()]

    def export_csv(self, out: str | Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with self.conn.cursor() as cur, out.open("w", newline="", encoding="utf-8") as f:
            cur.execute("SELECT sim_ts, clock, topic, payload::text FROM events WHERE run_id=%s ORDER BY id",
                        (self.run_id,))
            w = csv.writer(f)
            w.writerow(["sim_ts", "clock", "topic", "payload"])
            w.writerows(cur.fetchall())
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "run_id": self.run_id, **self.features, **self.stats,
                "queued": self._q.qsize()}

    def close(self) -> None:
        self._stop.set()
        self._writer.join(timeout=3)
        try:
            self.conn.close()
        except Exception:
            pass


def make_store(database_url: str | None, cfg, run_id: str, sqlite_path: str | None = None):
    if database_url:
        return PgEventStore(database_url, cfg, run_id)
    return EventStore(sqlite_path)
