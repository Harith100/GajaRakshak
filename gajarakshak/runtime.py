"""Wires world → edge → cloud core → agents, one tick at a time.

Modes:
  none                no protection (baseline)
  blanket             today's practice: blanket night speed restriction across the section
  gajarakshak         this system, alerts delivered to pilots via SMS / IVR / VHF ladder
  gajarakshak+kavach  phase 3: restriction pushed straight to the loco as a Kavach TSR
"""
from __future__ import annotations

import random
import statistics
import uuid
from collections import deque

from .agents.health import DeviceHealthAgent
from .agents.insights import template_summary
from .agents.verification import GroqVisionVerifier, SimulatedVerifier, VerificationAgent
from .bus import Bus
from .config import Config
from .core.dispatch import Dispatcher
from .core.rail import SimRailGateway
from .core.risk import RiskEngine, TrackStore
from .edge.perception import PerceptionAgent
from .models import RiskLevel
from .sim.scenarios import SCENARIOS
from .sim.sensors import Camera
from .sim.world import World
from .store import EventStore

KMPH = 1 / 3.6
MODES = ("none", "blanket", "gajarakshak", "gajarakshak+kavach")


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


class Runtime:
    def __init__(self, cfg: Config | None = None, seed: int = 7, mode: str = "gajarakshak",
                 auto_spawn: bool = True, db_path: str | None = None, dt: float = 1.0,
                 bus: Bus | None = None, store=None, vision: GroqVisionVerifier | None = None):
        """bus/store/vision default to in-process bus, SQLite and no vision model (tests, benchmark).
        api.py builds them from env: MQTT broker, Postgres, Groq vision."""
        assert mode in MODES, mode
        self.cfg = cfg or Config()
        self.seed, self.mode, self.dt = seed, mode, dt
        self.run_id = uuid.uuid4().hex[:8]
        self.world = World(self.cfg, seed, auto_spawn)
        self.sensor_rng = random.Random(seed * 31 + 1)
        self.sys_rng = random.Random(seed * 17 + 3)
        self.bus = bus or Bus()
        self.cameras = {t["id"]: Camera(t, self.cfg.camera) for t in self.cfg.towers}
        self.edges = {tid: PerceptionAgent(tid, self.bus, self.cfg.risk["publish_min"]) for tid in self.cameras}
        self.tracks = TrackStore()
        self.bus.subscribe("edge/+/tracks", self.tracks.on_msg)
        self.bus.subscribe("edge/+/keyframes", self.tracks.on_keyframe)
        self.risk = RiskEngine(self.cfg)
        self.rail = SimRailGateway(self.world)
        self.dispatch = Dispatcher(self.cfg, self.bus, self.rail, self.sys_rng, mode)
        self.dispatch.clock = self.world.clock
        self.health = DeviceHealthAgent(self.cfg, self.bus)
        self.verifier = VerificationAgent(self.tracks, self.risk, self.bus, SimulatedVerifier(self.sys_rng), vision)
        self.store = store if store is not None else EventStore(db_path)
        for pat in ("core/#", "rail/#", "agents/#"):
            self.bus.subscribe(pat, self.store.on_bus)
        self.feed: deque = deque(maxlen=80)
        for pat in ("core/alerts", "agents/#"):
            self.bus.subscribe(pat, self._to_feed)
        self.risk_map: dict = {}
        self.herd_first_alert: dict[str, float] = {}
        self._seen_collisions = 0
        self.speed = 4.0
        self.paused = False

    # ─────────────── control ───────────────
    def set_mode(self, mode: str) -> None:
        assert mode in MODES, mode
        self.mode = mode
        self.dispatch.mode = mode

    def scenario(self, name: str) -> str:
        msg = SCENARIOS[name](self)
        self.feed.appendleft({"clock": self.world.clock(), "kind": "scenario", "text": msg})
        self.store.write("sim/scenario", {"name": name, "text": msg})
        return msg

    def _to_feed(self, topic: str, msg: dict) -> None:
        clk = self.world.clock()
        if topic == "core/alerts":
            a = msg["alert"]
            text = f"{a['id']} {msg['event'].upper()} {a['level']} · {a['zone']}/{a['line']} · {a['reason']}"
            self.feed.appendleft({"clock": clk, "kind": "alert", "level": a["level"], "text": text})
        elif topic == "agents/verification":
            self.feed.appendleft({"clock": clk, "kind": "verify",
                                  "text": f"Verifier: track {msg['track']} (score {msg['score']}) → {msg['verdict']}"})
        elif topic == "agents/health":
            if msg["event"] == "status":
                text = f"Health: {msg['tower']} {msg['from']} → {msg['to']}"
            else:
                text = "Health: blind zones " + (", ".join(msg["zones"]) or "cleared")
            self.feed.appendleft({"clock": clk, "kind": "health", "text": text})

    # ─────────────── tick ───────────────
    def step(self) -> None:
        w, dt = self.world, self.dt
        blanket = self.cfg.risk["blanket_night_kmph"] * KMPH if (self.mode == "blanket" and w.is_night) else None
        w.step(dt, blanket)
        self.store.now, self.store.clock = w.t, w.clock()

        objs = list(w.observables())
        hb_every = self.cfg.health["heartbeat_s"]
        for cid, cam in self.cameras.items():
            dets = cam.detect(objs, w.is_night, w.t, self.sensor_rng)
            self.edges[cid].process(dets, w.t, dt)
            if w.t - cam.last_heartbeat >= hb_every:
                cam.last_heartbeat = w.t
                hb = cam.heartbeat(w.t, self.sensor_rng)
                if hb:
                    self.bus.publish(f"edge/{cid}/heartbeat", hb, 120)

        self.bus.pump()          # MQTT: deliver what the broker has received (this tick's or a live camera's)
        self.tracks.expire(w.t)
        protected = self.mode.startswith("gajarakshak")
        if protected:
            self.verifier.tick(w.t, w.clock())
        self.health.tick(w.t, w.is_night, w.clock())
        self.risk_map = self.risk.evaluate(self.tracks.tracks.values(), self.health.blind_zones, w.t) if protected else {}
        self.dispatch.update(self.risk_map, w.t)

        for a in self.dispatch.active.values():
            for g in a.gt_ids:
                if g.startswith("H"):
                    hid = g.rsplit("-", 1)[0]
                    self.herd_first_alert.setdefault(hid, w.t)
        if len(w.collisions) > self._seen_collisions:
            for c in w.collisions[self._seen_collisions:]:
                self.store.write("sim/collision", c)
                self.feed.appendleft({"clock": w.clock(), "kind": "collision",
                                      "text": f"💥 COLLISION: {c['train']} hit elephant at km {c['x'] / 1000:.2f} ({c['speed_kmph']} km/h)"})
            self._seen_collisions = len(w.collisions)

    def close(self) -> None:
        self.bus.close()
        self.store.close()

    def run_for(self, seconds: float) -> None:
        for _ in range(int(seconds / self.dt)):
            self.step()

    # ─────────────── evaluation / views ───────────────
    def stats(self) -> dict:
        w, d = self.world, self.dispatch
        alerts = list(d.history) + list(d.active.values())
        by_level: dict[str, int] = {}
        for a in alerts:
            by_level[a.max_level.name] = by_level.get(a.max_level.name, 0) + 1
        actionable = [a for a in alerts if a.max_level >= RiskLevel.CAUTION]
        false_alerts = [a for a in actionable if not any(g.startswith("H") for g in a.gt_ids)]
        zones: dict[str, int] = {}
        for a in alerts:
            zones[a.zone_id] = zones.get(a.zone_id, 0) + 1
        det_lat = [self.herd_first_alert[h] - t0 for h, t0 in w.herd_watch.items() if h in self.herd_first_alert]
        exposed = [h for h in w.herd_watch]
        missed = [h for h in exposed if h not in self.herd_first_alert and w.t - w.herd_watch[h] > 120]
        fin = w.finished_trains
        delays = [max(0.0, (t.exited_at - t.spawned_at) - t.ideal_s) for t in fin]
        hours = w.t / 3600
        towers = len(self.cfg.towers)
        cam = self.cfg.camera
        video_gb = towers * cam["streams_per_tower"] * cam["bitrate_mbps_per_stream"] * w.t / 8 / 1000
        return {
            "mode": self.mode, "seed": self.seed, "window": (f"{hours:.1f} h simulated" if hours >= 1 else f"{w.t / 60:.0f} min simulated"),
            "sim_hours": round(hours, 2),
            "collisions": len(w.collisions), "near_misses": len(w.near_misses),
            "herds_exposed": len(exposed), "herds_missed": len(missed),
            "detect_to_alert_p50_s": _pct(det_lat, 0.5), "detect_to_alert_p95_s": _pct(det_lat, 0.95),
            "alerts_total": len(alerts), "alerts_by_level": by_level,
            "actionable_alerts": len(actionable), "false_actionable_alerts": len(false_alerts),
            "hot_zones": sorted(zones.items(), key=lambda kv: -kv[1]),
            "trains_passed": len(fin),
            "train_delay_total_min": round(sum(delays) / 60, 1),
            "train_delay_avg_min": round(statistics.mean(delays) / 60, 2) if delays else 0.0,
            "trains_restricted": d.metrics["acks"], "escalations": d.metrics["escalations"],
            "lead_time_p50_s": _pct(d.metrics["lead_times"], 0.5),
            "ack_p95_s": _pct(d.metrics["ack_latency"], 0.95),
            "verification": dict(self.verifier.stats),
            "towers_total": towers, "towers_online": sum(1 for s in self.health.status.values() if s == "ONLINE"),
            "open_tickets": sum(1 for t in self.health.tickets if t["status"] == "OPEN"),
            "uplink_kb": round(self.bus.bytes.get("edge", 0) / 1024, 1),
            "video_equiv_gb": round(video_gb, 1),
        }

    def insights(self) -> dict:
        s = self.stats()
        return {"stats": s, "summary": template_summary(s)}

    def snapshot(self) -> dict:
        w = self.world
        tracks = []
        for t in self.tracks.tracks.values():
            tracks.append({"id": t.id, "x": t.x, "y": t.y, "vx": t.vx, "vy": t.vy, "score": round(t.score, 2),
                           "cls": self.risk.classify(t), "verified": t.verified, "source": t.source})
        zr = [{"zone": k[0], "line": k[1], "level": v.level.name} for k, v in self.risk_map.items()]
        s = self.stats()
        return {
            "t": w.t, "clock": w.clock(), "night": w.is_night, "mode": self.mode, "speed": self.speed,
            "paused": self.paused, "world": w.snapshot(), "tracks": tracks, "zone_risk": zr,
            "alerts": self.dispatch.alerts_view(15), "feed": list(self.feed)[:40],
            "health": self.health.view(),
            "cameras": {k: {"online": c.online, "quality": c.quality} for k, c in self.cameras.items()},
            "verifier": {"pending": len(self.verifier.pending), "recent": self.verifier.recent[-6:],
                         **self.verifier.describe()},
            "system": {"bus": self.bus.describe(), "store": self.store.describe(), "run_id": self.run_id},
            "kpi": {k: s[k] for k in ("collisions", "near_misses", "alerts_total", "actionable_alerts",
                                       "false_actionable_alerts", "trains_restricted", "lead_time_p50_s",
                                       "ack_p95_s", "detect_to_alert_p50_s", "uplink_kb", "video_equiv_gb",
                                       "towers_online", "towers_total", "train_delay_total_min", "trains_passed")},
        }
