"""Cloud track mirror + deterministic RiskEngine.

Design rule: nothing probabilistic or generative sits between "elephant confirmed" and
"train restricted". The rules below are a small, auditable table that a Railway safety
committee can review line by line. Agents (verification, health, insights) only feed it
inputs or consume its outputs — they never sit on the critical path.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

from ..config import Config
from ..models import RiskLevel


@dataclass
class CloudTrack:
    id: str
    tower: str
    x: float
    y: float
    vx: float
    vy: float
    score: float
    first_seen: float
    last_update: float
    verified: bool | None = None
    verify_requested: bool = False
    candidate_since: float | None = None
    source: str = "sim"            # "sim" (simulated camera) | "camera" (real detector, e.g. perception demo)
    wall_update: float = 0.0
    gt: str | None = None          # SIM ONLY


class TrackStore:
    """Cloud-side mirror of confirmed edge tracks (fed by edge/+/tracks and edge/+/keyframes).

    Simulated tracks expire on simulation time; tracks from a real camera expire on wall-clock
    time, so a live camera keeps working whatever speed the simulation runs at.
    """

    TTL_S = 12.0
    LIVE_TTL_S = 6.0

    def __init__(self):
        self.tracks: dict[str, CloudTrack] = {}
        self.keyframes: dict[str, bytes] = {}   # track id -> JPEG crop for the verifier
        self.now = 0.0

    def on_keyframe(self, topic: str, msg: dict) -> None:
        try:
            self.keyframes[msg["track_id"]] = base64.b64decode(msg["jpeg_b64"])
        except (KeyError, ValueError):
            pass

    def on_msg(self, topic: str, msg: dict) -> None:
        t = msg["track"]
        if msg["event"] == "lost":
            self.tracks.pop(t["id"], None)
            return
        ct = self.tracks.get(t["id"])
        if ct is None:
            ct = CloudTrack(t["id"], t["tower"], t["x"], t["y"], t["vx"], t["vy"], t["score"], t["first_seen"], self.now)
            self.tracks[t["id"]] = ct
        ct.x, ct.y, ct.vx, ct.vy, ct.score, ct.last_update = t["x"], t["y"], t["vx"], t["vy"], t["score"], self.now
        ct.gt = msg.get("_gt")
        ct.source = msg.get("source", "sim")
        ct.wall_update = time.monotonic()

    def expire(self, now: float) -> None:
        self.now = now
        wall = time.monotonic()
        dead = [k for k, v in self.tracks.items()
                if (v.source == "sim" and now - v.last_update > self.TTL_S)
                or (v.source != "sim" and wall - v.wall_update > self.LIVE_TTL_S)]
        for tid in dead:
            del self.tracks[tid]
            self.keyframes.pop(tid, None)
        if len(self.keyframes) > 200:   # keyframes for tracks that were never confirmed
            for tid in [k for k in self.keyframes if k not in self.tracks][:100]:
                del self.keyframes[tid]


@dataclass
class ZoneRisk:
    zone_id: str
    line: str
    level: RiskLevel
    reason: str
    track_ids: list = field(default_factory=list)
    gt_ids: set = field(default_factory=set)
    min_dist: float = 1e9
    count: int = 0


class RiskEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        r = cfg.risk
        self.stop_d, self.caution_d, self.watch_d = r["stop_dist_m"], r["caution_dist_m"], r["watch_dist_m"]
        self.caution_ttt, self.buffer = r["caution_ttt_s"], r["zone_buffer_m"]
        self.conf_high, self.verify_low = r["conf_high"], r["conf_verify_low"]
        self.verify_sla = r.get("verify_sla_s", 8.0)
        self.line_y = {lid: l["y"] for lid, l in cfg.lines.items()}

    def classify(self, t: CloudTrack) -> str:
        if t.verified is False:
            return "rejected"
        if t.verified is True or t.score >= self.conf_high:
            return "confirmed"
        if t.score >= self.verify_low:
            return "candidate"
        return "low"

    def level_for(self, cls: str, d: float, ttt: float, waited: float = 0.0) -> RiskLevel:
        """THE rule table. Keep it this small."""
        if cls == "confirmed":
            if d <= self.stop_d:
                return RiskLevel.STOP
            if d <= self.caution_d or (d <= self.watch_d and ttt <= self.caution_ttt):
                return RiskLevel.CAUTION
            return RiskLevel.ADVISORY
        if cls == "candidate":
            # unverified and on the track, and the verifier has missed its SLA (slow/down):
            # fail safe — slow the trains rather than wait
            if d <= self.stop_d and waited >= self.verify_sla:
                return RiskLevel.CAUTION
            return RiskLevel.ADVISORY
        return RiskLevel.NONE

    def evaluate(self, tracks, blind_zones: set[str], now: float = 0.0) -> dict[tuple[str, str], ZoneRisk]:
        out: dict[tuple[str, str], ZoneRisk] = {}
        for t in tracks:
            cls = self.classify(t)
            if cls == "candidate" and t.candidate_since is None:
                t.candidate_since = now
            if cls in ("rejected", "low"):
                continue
            waited = now - t.candidate_since if t.candidate_since is not None else 0.0
            zones = self.cfg.zones_for(t.x - self.buffer, t.x + self.buffer)
            for line, ly in self.line_y.items():
                off = t.y - ly
                d = abs(off)
                v_app = -(off / d) * t.vy if d > 0.5 else 0.0     # +ve = moving towards this line
                ttt = d / v_app if v_app > 0.1 else float("inf")
                lvl = self.level_for(cls, d, ttt, waited)
                if lvl == RiskLevel.NONE:
                    continue
                for z in zones:
                    key = (z.id, line)
                    zr = out.get(key)
                    if zr is None:
                        zr = out[key] = ZoneRisk(z.id, line, lvl, "")
                    zr.level = max(zr.level, lvl)
                    zr.track_ids.append(t.id)
                    zr.count += 1
                    if t.gt:
                        zr.gt_ids.add(t.gt)
                    if d < zr.min_dist:
                        zr.min_dist = d
                        moving = "approaching" if ttt < 1e8 else "not approaching"
                        tag = "" if cls == "confirmed" else " (unverified)"
                        zr.reason = f"elephant{tag} {d:.0f} m from line {line}, {moving}"
        for zr in out.values():
            if zr.count > 1:
                zr.reason = f"{zr.count} tracks · nearest " + zr.reason
        for zid in blind_zones:
            for line in self.line_y:
                key = (zid, line)
                if key in out:
                    out[key].level = max(out[key].level, RiskLevel.BLIND)
                else:
                    out[key] = ZoneRisk(zid, line, RiskLevel.BLIND, "sensor coverage lost in hotspot at night")
        return out
