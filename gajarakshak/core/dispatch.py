"""AlertDispatcher — deterministic alert lifecycle + delivery with escalation.

  * Upgrades apply immediately; downgrades/clears only after `clear_hold_s` of lower risk
    (asymmetric hysteresis — elephants step back into the bush and come out again).
  * Every restriction sent to a loco pilot must be acknowledged. No ack within
    `ack_timeout_s` → escalate to the next channel: CUG SMS → IVR call → Station Master on VHF.
  * STOP goes out on IVR and VHF in parallel from the start.
  * Kavach mode (phase 3): restriction is pushed as a TSR to the loco, no human in the loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..bus import Bus
from ..config import Config
from ..models import Alert, Restriction, RiskLevel
from .rail import SimRailGateway

KMPH = 1 / 3.6
LADDER = ["cug_sms", "ivr", "vhf_via_sm"]
NOTICE_P = {"cug_sms": 0.55, "ivr": 0.93, "vhf_via_sm": 0.99, "kavach": 1.0}  # pilot actually notices


@dataclass
class Attempt:
    channel: str
    sent_at: float
    deliver_at: float
    noticed: bool
    ack_at: float


@dataclass
class Delivery:
    alert_id: str
    train_id: str
    level: RiskLevel
    restriction: Restriction | None
    created_at: float
    attempts: list = field(default_factory=list)
    acked_at: float | None = None
    release: bool = False
    done: bool = False
    exhausted: bool = False


class Dispatcher:
    def __init__(self, cfg: Config, bus: Bus, rail: SimRailGateway, rng: random.Random, mode: str = "gajarakshak"):
        self.cfg, self.bus, self.rail, self.rng = cfg, bus, rail, rng
        self.mode = mode
        d = cfg.dispatch
        self.clear_hold, self.ack_timeout, self.horizon = d["clear_hold_s"], d["ack_timeout_s"], d["notify_horizon_m"]
        self.chan_latency, self.reaction = d["channels"], d["pilot_reaction_s"]
        r = cfg.risk
        self.limits = {RiskLevel.STOP: 0.0, RiskLevel.CAUTION: r["caution_kmph"] * KMPH,
                       RiskLevel.BLIND: r["blind_kmph"] * KMPH}
        self.active: dict[tuple[str, str], Alert] = {}
        self.history: list[Alert] = []
        self.by_id: dict[str, Alert] = {}
        self.deliveries: dict[tuple[str, str], Delivery] = {}
        self.metrics = {"notifications": 0, "escalations": 0, "acks": 0, "lead_times": [], "ack_latency": []}
        self._n = 0
        self.now = 0.0
        self.clock = lambda: ""

    # ─────────────── alert lifecycle ───────────────
    def update(self, risk_map: dict, now: float) -> None:
        self.now = now
        for key in set(risk_map) | set(self.active):
            zr = risk_map.get(key)
            lvl = zr.level if zr else RiskLevel.NONE
            a = self.active.get(key)
            if a is None:
                if lvl > RiskLevel.NONE:
                    self._open(key, zr)
                continue
            if zr:
                a.track_ids = zr.track_ids
                a.gt_ids |= zr.gt_ids
            if lvl > a.level:
                self._set_level(a, lvl, zr.reason, "upgraded")
                a.pending_level = None
            elif lvl < a.level:
                if a.pending_level is None:
                    a.pending_level, a.pending_since = lvl, now
                else:
                    a.pending_level = max(a.pending_level, lvl)
                if now - a.pending_since >= self.clear_hold:
                    if a.pending_level == RiskLevel.NONE:
                        self._clear(a)
                    else:
                        self._set_level(a, a.pending_level, zr.reason if zr else a.reason, "downgraded")
                    a.pending_level = None
            else:
                a.pending_level = None
                if zr:
                    a.reason = zr.reason
        if self.mode != "none":
            self._sync_deliveries()
            self._process_deliveries()

    def _open(self, key, zr) -> None:
        self._n += 1
        a = Alert(f"AL-{self._n:04d}", key[0], key[1], zr.level, zr.reason, self.now, self.now,
                  track_ids=zr.track_ids, gt_ids=set(zr.gt_ids), max_level=zr.level)
        self.active[key] = a
        self.by_id[a.id] = a
        self._log(a, f"opened {zr.level.name}: {zr.reason}")
        self.bus.publish("core/alerts", {"event": "opened", "alert": a.to_dict()}, 200)
        self._notify_staff(a)

    def _set_level(self, a: Alert, lvl: RiskLevel, reason: str, verb: str) -> None:
        a.level, a.reason, a.updated_at = lvl, reason, self.now
        a.max_level = max(a.max_level, lvl)
        self._log(a, f"{verb} → {lvl.name}: {reason}")
        self.bus.publish("core/alerts", {"event": verb, "alert": a.to_dict()}, 200)
        if verb == "upgraded":
            self._notify_staff(a)

    def _clear(self, a: Alert) -> None:
        a.status, a.cleared_at, a.updated_at = "CLEARED", self.now, self.now
        self._log(a, f"cleared after {self.clear_hold:.0f}s hold")
        del self.active[(a.zone_id, a.line)]
        self.history.append(a)
        self.bus.publish("core/alerts", {"event": "cleared", "alert": a.to_dict()}, 200)

    def _notify_staff(self, a: Alert) -> None:
        to = ["Section Controller (PGT)", f"SM {self.cfg.section['stations'][0]['code']}",
              f"SM {self.cfg.section['stations'][1]['code']}"]
        if a.level >= RiskLevel.CAUTION:
            to.append("Forest RRT, Madukkarai range")
        self._log(a, "notified " + ", ".join(to))
        self.bus.publish("rail/notify", {"alert": a.id, "to": to, "level": a.level.name}, 120)

    def controller_ack(self, alert_id: str) -> bool:
        a = self.by_id.get(alert_id)
        if not a or a.controller_ack:
            return False
        a.controller_ack = self.now
        self._log(a, "acknowledged by Section Controller")
        return True

    def _log(self, a: Alert, text: str) -> None:
        a.log.append(f"[{self.clock()}] {text}")

    # ─────────────── train deliveries ───────────────
    def _zone_bounds(self, a: Alert):
        z = self.cfg.zone_by_id[a.zone_id]
        return z.start, z.end

    def _relevant(self, train, a: Alert) -> bool:
        """Train on that line, not yet past the zone, and within the notification horizon."""
        if train.line != a.line:
            return False
        zs, ze = self._zone_bounds(a)
        s_front = train.front * train.direction
        za, zb = sorted((zs * train.direction, ze * train.direction))
        s_rear = s_front - train.length
        return s_rear <= zb and (za - s_front) <= self.horizon

    def _sync_deliveries(self) -> None:
        restricting = {a.id: a for a in self.active.values() if a.level in self.limits}
        # new / upgraded restrictions
        for a in restricting.values():
            zs, ze = self._zone_bounds(a)
            for tr in self.rail.positions(a.line):
                if not self._relevant(tr, a):
                    continue
                key = (a.id, tr.id)
                d = self.deliveries.get(key)
                if d and not d.release and d.level == a.level:
                    continue
                r = Restriction(a.id, a.zone_id, a.line, zs, ze, self.limits[a.level], a.level)
                nd = Delivery(a.id, tr.id, a.level, r, self.now)
                self.deliveries[key] = nd
                self._send(nd, tr, first=True)
        # releases: alert cleared or no longer restricting
        for key, d in list(self.deliveries.items()):
            if d.release:
                continue
            if d.alert_id not in restricting:
                rd = Delivery(d.alert_id, d.train_id, RiskLevel.NONE, None, self.now, release=True)
                self.deliveries[key] = rd
                tr = self.rail.get(d.train_id)
                if tr is None:
                    del self.deliveries[key]
                    continue
                self._send(rd, tr, first=True)

    def _send(self, d: Delivery, tr, first: bool) -> None:
        if self.mode == "gajarakshak+kavach":
            chans = ["kavach"]
        elif d.release:
            chans = ["cug_sms"]
        elif first and d.level == RiskLevel.STOP:
            chans = ["ivr", "vhf_via_sm"]
        else:
            used = {at.channel for at in d.attempts}
            nxt = next((c for c in LADDER if c not in used), None)
            chans = [nxt] if nxt else []
        for ch in chans:
            lat = 2.0 if ch == "kavach" else self.rng.uniform(*self.chan_latency[ch])
            noticed = d.release or self.rng.random() < NOTICE_P[ch]
            ack = self.now + lat + self.rng.uniform(*self.reaction)
            d.attempts.append(Attempt(ch, self.now, self.now + lat, noticed, ack))
            self.metrics["notifications"] += 1
            if not first:
                self.metrics["escalations"] += 1
            a = self.by_id.get(d.alert_id)
            if a:
                what = "RELEASE" if d.release else d.level.name
                self._log(a, f"{what} → {tr.name} via {ch}" + ("" if first else " (escalation)"))

    def _process_deliveries(self) -> None:
        for key, d in list(self.deliveries.items()):
            tr = self.rail.get(d.train_id)
            if tr is None:
                del self.deliveries[key]
                continue
            if d.done:
                a = self.by_id.get(d.alert_id)
                if a and a.status == "ACTIVE" and not self._relevant(tr, a):
                    del self.deliveries[key]  # train has cleared the zone
                continue
            acked = [at for at in d.attempts if at.noticed and at.ack_at <= self.now]
            if acked:
                at = min(acked, key=lambda x: x.ack_at)
                d.acked_at, d.done = at.ack_at, True
                a = self.by_id.get(d.alert_id)
                stamp = f"[{self.clock()}]"
                if d.release:
                    self.rail.release(tr.id, d.alert_id)
                    self.rail.cab_message(tr.id, f"{stamp} RELEASE {d.alert_id}: normal speed")
                    if d.release and (a is None or a.status == "CLEARED" or a.level not in self.limits):
                        del self.deliveries[key]
                    continue
                self.rail.apply(tr.id, d.restriction)
                lim = round(d.restriction.limit_mps / KMPH)
                txt = "STOP short of zone" if d.level == RiskLevel.STOP else f"max {lim} km/h"
                self.rail.cab_message(tr.id, f"{stamp} {d.level.name} {d.restriction.zone_id} line {d.restriction.line}: {txt} ({at.channel})")
                self.metrics["acks"] += 1
                self.metrics["ack_latency"].append(at.ack_at - d.created_at)
                zs, ze = d.restriction.start_x, d.restriction.end_x
                dist = min(zs * tr.direction, ze * tr.direction) - tr.front * tr.direction
                if dist > 0:
                    self.metrics["lead_times"].append(dist / max(tr.speed, 1.0))
                if a:
                    self._log(a, f"{tr.name} acknowledged via {at.channel} ({at.ack_at - d.created_at:.0f}s)")
                continue
            last = max(at.sent_at for at in d.attempts) if d.attempts else d.created_at
            if self.now - last >= self.ack_timeout:
                used = {at.channel for at in d.attempts}
                if not d.release and self.mode != "gajarakshak+kavach" and any(c not in used for c in LADDER):
                    self._send(d, tr, first=False)
                elif not d.exhausted:
                    d.exhausted = True
                    a = self.by_id.get(d.alert_id)
                    if a:
                        self._log(a, f"⚠ {tr.name} unacknowledged on all channels — controller to intervene")

    # ─────────────── views ───────────────
    def alerts_view(self, limit: int = 30) -> list[dict]:
        act = sorted(self.active.values(), key=lambda a: (-a.level, a.opened_at))
        return [a.to_dict() for a in act] + [a.to_dict() for a in reversed(self.history[-limit:])]
