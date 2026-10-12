"""Synthetic ground-truth world: elephant herds, confuser animals, people, thermal artefacts, trains.

Coordinates: x = chainage along the section in metres (0 = Ettimadai, 13000 = Walayar),
y = lateral offset from the track centreline in metres (+ north forest, − south / water side).
Line A runs at y=-20 (towards Palakkad), line B at y=+20 (towards Coimbatore).

Nothing in core/, edge/ or agents/ reads this module's state directly — they only see what
the camera model in sensors.py emits. The world is the "truth" we evaluate against.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config
from ..models import Restriction, RiskLevel

KMPH = 1 / 3.6


# ───────────────────────────── animals ─────────────────────────────

@dataclass
class Herd:
    id: str
    n: int
    cx: float
    cy: float
    waypoints: list           # [(x, y, pause_s)]
    speed: float
    offsets: list
    idx: int = 0
    wait_until: float = 0.0
    alive: list = field(default_factory=list)
    done: bool = False
    first_watch_ts: Optional[float] = None   # first time any member within 150 m of track (eval)

    def members(self):
        for k, (ox, oy) in enumerate(self.offsets):
            if self.alive[k]:
                yield f"{self.id}-{k}", self.cx + ox, self.cy + oy


@dataclass
class Animal:
    id: str
    kind: str          # gaur | boar | cattle | human | hotspot
    x: float
    y: float
    waypoints: list
    speed: float
    idx: int = 0
    ttl: float = 1e9
    done: bool = False


def _move_towards(x, y, tx, ty, step):
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy)
    if d <= step:
        return tx, ty, True
    return x + dx / d * step, y + dy / d * step, False


# ───────────────────────────── trains ─────────────────────────────

@dataclass
class Train:
    id: str
    name: str
    kind: str
    line: str
    direction: int
    front: float
    speed: float
    vmax: float
    accel: float
    decel_service: float
    decel_emergency: float
    length: float
    spawned_at: float
    ideal_s: float
    known: dict = field(default_factory=dict)       # alert_id -> Restriction (acted on by pilot)
    inbox: list = field(default_factory=list)       # messages shown in cab view
    emergency: bool = False
    exited_at: Optional[float] = None
    target: float = 0.0

    def s(self, x: float) -> float:
        """Position along direction of travel."""
        return x * self.direction

    def target_speed(self, blanket_mps: Optional[float]) -> tuple[float, bool]:
        v_t = self.vmax if blanket_mps is None else min(self.vmax, blanket_mps)
        emergency = False
        s_front = self.s(self.front)
        s_rear = s_front - self.length
        for r in self.known.values():
            a, b = sorted((self.s(r.start_x), self.s(r.end_x)))
            if s_rear > b:
                continue                                  # fully past the zone
            if s_front >= a:                              # inside zone
                if r.level == RiskLevel.STOP:
                    v_t, emergency = 0.0, True
                else:
                    v_t = min(v_t, r.limit_mps)
                continue
            margin = 80.0 if r.level == RiskLevel.STOP else 20.0
            d = max(0.0, a - s_front - margin)
            allowed = math.sqrt(r.limit_mps ** 2 + 2 * self.decel_service * d)
            if r.level == RiskLevel.STOP and self.speed > 0.5:
                needed = (self.speed ** 2 - r.limit_mps ** 2) / (2 * max(d, 1.0))
                if needed > self.decel_service * 1.05:
                    emergency = True
            v_t = min(v_t, allowed)
        return v_t, emergency

    def step(self, dt: float, blanket_mps: Optional[float]) -> float:
        prev = self.front
        v_t, emergency = self.target_speed(blanket_mps)
        self.target, self.emergency = v_t, emergency and self.speed > v_t
        if self.speed < v_t:
            self.speed = min(v_t, self.speed + self.accel * dt)
        elif self.speed > v_t:
            dec = self.decel_emergency if self.emergency else self.decel_service
            self.speed = max(v_t, self.speed - dec * dt)
        self.front += self.direction * self.speed * dt
        return prev

    def to_dict(self) -> dict:
        lim = min((r.limit_mps for r in self.known.values()), default=None)
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "line": self.line, "dir": self.direction,
            "front": round(self.front, 1), "length": self.length,
            "speed_kmph": round(self.speed / KMPH, 1), "target_kmph": round(self.target / KMPH, 1),
            "vmax_kmph": round(self.vmax / KMPH), "emergency": self.emergency,
            "restricted": bool(self.known), "limit_kmph": None if lim is None else round(lim / KMPH),
            "inbox": self.inbox[-5:],
        }


TRAIN_NAMES = {
    "express": ["Kerala Exp", "Chennai Mail", "Jayanti Janata", "Shalimar Exp", "Vande Bharat", "Island Exp",
                "Sabari Exp", "Nethravathi"],
    "passenger": ["CBE–PGT MEMU", "Palakkad MEMU", "Coimbatore Pass."],
    "freight": ["BCN Goods", "BTPN Tanker", "Container Rake", "Cement BOXN"],
}


# ───────────────────────────── world ─────────────────────────────

class World:
    def __init__(self, cfg: Config, seed: int = 7, auto_spawn: bool = True):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.t = 0.0
        self.herds: list[Herd] = []
        self.animals: list[Animal] = []
        self.trains: list[Train] = []
        self.finished_trains: list[Train] = []
        self.collisions: list[dict] = []
        self.near_misses: list[dict] = []
        self.herd_watch: dict[str, float] = {}      # herd id -> first time within 150 m of track (eval)
        self.herd_size: dict[str, int] = {}
        self.auto_spawn = auto_spawn
        self._n = 0
        self.next_train = {lid: self.rng.uniform(60, 400) for lid in cfg.lines}
        self.line_y = {lid: l["y"] for lid, l in cfg.lines.items()}

    # time of day ---------------------------------------------------
    @property
    def hour(self) -> float:
        return (self.cfg.world["start_hour"] + self.t / 3600.0) % 24

    @property
    def is_night(self) -> bool:
        return self.hour >= 18.5 or self.hour < 6.5

    def clock(self) -> str:
        h = self.hour
        return f"{int(h):02d}:{int((h % 1) * 60):02d}"

    def _id(self, p: str) -> str:
        self._n += 1
        return f"{p}{self._n}"

    # spawning --------------------------------------------------------
    def spawn_herd(self, hotspot_x: float, n: int | None = None, side: int | None = None,
                   linger_s: float | None = None, speed: float | None = None, start_dist: float = 650) -> Herd:
        r = self.rng
        n = n or r.choice([1, 2, 3, 4, 5, 6, 7])
        if side is None:
            # evening/night: forest (north) → water (south); early morning: back
            side = 1 if (self.hour >= 15 or self.hour < 2) else -1
            if r.random() < 0.25:
                side = -side
        if linger_s is None:
            linger_s = r.uniform(60, 260) if r.random() < 0.35 else r.uniform(0, 20)
        x0 = hotspot_x + r.uniform(-150, 150)
        wps = [
            (x0 + r.uniform(-60, 60), side * r.uniform(110, 160), r.uniform(10, 90)),  # hesitate at forest edge
            (x0 + r.uniform(-80, 80), r.uniform(-12, 12), linger_s),                  # on the tracks
            (x0 + r.uniform(-120, 120), -side * start_dist, 0.0),                     # across to the far side
        ]
        rad = 5 + 2.5 * n
        offsets = [(0.0, 0.0)] + [(r.uniform(-rad, rad), r.uniform(-rad / 2, rad / 2)) for _ in range(n - 1)]
        h = Herd(self._id("H"), n, x0, side * start_dist, wps, speed or r.uniform(0.55, 1.0), offsets,
                 alive=[True] * n)
        self.herds.append(h)
        self.herd_size[h.id] = n
        return h

    def spawn_animal(self, kind: str, x: float | None = None) -> Animal:
        r = self.rng
        x = x if x is not None else r.uniform(800, self.cfg.length - 800)
        if kind == "human":       # track walker / trespasser walking along the cess
            y = r.choice([-6, 6, 30])
            wps = [(x + r.choice([-1, 1]) * r.uniform(600, 1500), y, 0)]
            a = Animal(self._id("P"), kind, x, y, wps, r.uniform(1.0, 1.5))
        elif kind == "hotspot":   # sun-warmed rock / smouldering fire: static thermal blob
            a = Animal(self._id("X"), kind, x, r.uniform(-140, 140), [], 0.0, ttl=self.t + r.uniform(90, 400))
        else:
            side = r.choice([-1, 1])
            spd = {"gaur": 0.8, "boar": 1.4, "cattle": 0.6}[kind]
            wps = [(x + r.uniform(-50, 50), r.uniform(-10, 10), r.uniform(0, 40)),
                   (x + r.uniform(-150, 150), -side * 450, 0)]
            a = Animal(self._id({"gaur": "G", "boar": "B", "cattle": "C"}[kind]), kind, x, side * 450, wps, spd)
        self.animals.append(a)
        return a

    def spawn_train(self, line: str, kind: str | None = None, front: float | None = None,
                    speed_kmph: float | None = None) -> Train:
        r = self.rng
        tcfg = self.cfg.trains["types"]
        if kind is None:
            kinds = list(tcfg)
            kind = r.choices(kinds, weights=[tcfg[k]["weight"] for k in kinds])[0]
        p = tcfg[kind]
        lcfg = self.cfg.lines[line]
        d = lcfg["direction"]
        approach = self.cfg.section["approach_m"]
        if front is None:
            front = -approach if d > 0 else self.cfg.length + approach
        vmax = p["vmax_kmph"] * KMPH
        num = r.randint(12000, 22999) if kind != "freight" else r.randint(100, 999)
        name = f"{num} {r.choice(TRAIN_NAMES[kind])}" if kind != "freight" else f"{r.choice(TRAIN_NAMES[kind])} #{num}"
        dist = self.cfg.length + 2 * approach + p["length_m"]
        tr = Train(self._id("TR"), name, kind, line, d, front,
                   (speed_kmph * KMPH) if speed_kmph is not None else vmax, vmax,
                   p["accel"], p["decel_service"], p["decel_emergency"], p["length_m"], self.t, dist / vmax)
        self.trains.append(tr)
        return tr

    # stepping --------------------------------------------------------
    def _poisson(self, per_hour: float, dt: float) -> bool:
        return self.rng.random() < per_hour * dt / 3600.0

    def _auto_spawn(self, dt: float):
        w = self.cfg.world
        rate = w["herds_per_hour_night"] if self.is_night else w["herds_per_hour_day"]
        if self._poisson(rate, dt):
            self.spawn_herd(self.rng.choice(self.cfg.hotspots)["x"])
        if self._poisson(0.35, dt):
            self.spawn_animal("gaur")
        if self._poisson(1.2, dt):
            self.spawn_animal("boar")
        if not self.is_night and self._poisson(1.0, dt):
            self.spawn_animal("cattle")
        if self._poisson(0.5, dt):
            self.spawn_animal("human")
        if 18 <= self.hour < 22 and self._poisson(1.2, dt):
            self.spawn_animal("hotspot")
        lo, hi = self.cfg.trains["headway_min"]
        for lid in self.cfg.lines:
            if self.t >= self.next_train[lid]:
                self.spawn_train(lid)
                self.next_train[lid] = self.t + self.rng.uniform(lo, hi) * 60

    def step(self, dt: float, blanket_mps: Optional[float] = None):
        self.t += dt
        if self.auto_spawn:
            self._auto_spawn(dt)

        for h in self.herds:
            if h.done:
                continue
            if self.t < h.wait_until:
                h.cx += self.rng.uniform(-0.15, 0.15)       # milling about
                continue
            tx, ty, pause = h.waypoints[h.idx]
            h.cx, h.cy, arrived = _move_towards(h.cx, h.cy, tx, ty, h.speed * dt)
            if arrived:
                h.wait_until = self.t + pause
                h.idx += 1
                if h.idx >= len(h.waypoints):
                    h.done = True
            if h.first_watch_ts is None and any(abs(y) <= self.cfg.camera["range_lateral_m"] for _, _, y in h.members()):
                h.first_watch_ts = self.t
                self.herd_watch[h.id] = self.t

        for a in self.animals:
            if a.done:
                continue
            if self.t > a.ttl:
                a.done = True
                continue
            if not a.waypoints:
                continue
            tx, ty, pause = a.waypoints[a.idx]
            a.x, a.y, arrived = _move_towards(a.x, a.y, tx, ty, a.speed * dt)
            if arrived:
                a.idx += 1
                if a.idx >= len(a.waypoints):
                    a.done = True

        for tr in self.trains:
            prev = tr.step(dt, blanket_mps)
            self._check_collisions(tr, prev)

        exit_hi = self.cfg.length + self.cfg.section["approach_m"]
        exit_lo = -self.cfg.section["approach_m"]
        still = []
        for tr in self.trains:
            rear = tr.front - tr.direction * tr.length
            if (tr.direction > 0 and rear > exit_hi) or (tr.direction < 0 and rear < exit_lo):
                tr.exited_at = self.t
                self.finished_trains.append(tr)
            else:
                still.append(tr)
        self.trains = still
        self.herds = [h for h in self.herds if not h.done]
        self.animals = [a for a in self.animals if not a.done]

    def _check_collisions(self, tr: Train, prev_front: float):
        ly = self.line_y[tr.line]
        lo, hi = sorted((prev_front, tr.front))
        for h in self.herds:
            for k, (ox, oy) in enumerate(h.offsets):
                if not h.alive[k]:
                    continue
                ex, ey = h.cx + ox, h.cy + oy
                if abs(ey - ly) < 3.5 and lo <= ex <= hi and tr.speed > 0.3:
                    h.alive[k] = False
                    self.collisions.append({"t": self.t, "train": tr.name, "line": tr.line, "x": round(ex),
                                            "speed_kmph": round(tr.speed / KMPH, 1), "elephant": f"{h.id}-{k}"})
                elif abs(ey - ly) < 3.5 and tr.speed < 0.3 and 0 < (ex - tr.front) * tr.direction < 250:
                    key = (tr.id, h.id)
                    if not any(n["key"] == key for n in self.near_misses):
                        self.near_misses.append({"key": key, "t": self.t, "train": tr.name,
                                                 "gap_m": round(abs(ex - tr.front))})

    # observation surface for sensors --------------------------------------
    def observables(self):
        for h in self.herds:
            for mid, x, y in h.members():
                yield mid, "elephant", x, y
        for a in self.animals:
            yield a.id, a.kind, a.x, a.y

    def snapshot(self) -> dict:
        return {
            "elephants": [{"id": m, "x": round(x, 1), "y": round(y, 1)} for h in self.herds for m, x, y in h.members()],
            "others": [{"id": a.id, "kind": a.kind, "x": round(a.x, 1), "y": round(a.y, 1)} for a in self.animals],
            "trains": [t.to_dict() for t in self.trains],
        }
