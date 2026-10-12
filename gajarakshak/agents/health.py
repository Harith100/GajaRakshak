"""DeviceHealthAgent — watches tower heartbeats and turns coverage loss into a safety input.

Key behaviour (fail-safe): if a hotspot zone loses most of its camera coverage at night,
the agent declares it BLIND and the RiskEngine applies a precautionary speed restriction
until coverage is restored. A dead camera must never look like "no elephants".
"""
from __future__ import annotations

from ..bus import Bus
from ..config import Config


def _union_len(intervals):
    total, cur = 0.0, None
    for a, b in sorted(intervals):
        if cur is None or a > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        total += cur[1] - cur[0]
    return total


class DeviceHealthAgent:
    def __init__(self, cfg: Config, bus: Bus):
        self.cfg, self.bus = cfg, bus
        self.offline_after = cfg.health["offline_after_s"]
        self.blind_ratio = cfg.health["blind_coverage_ratio"]
        self.window = cfg.health.get("hotspot_window_m", 350)
        self.last: dict[str, dict] = {}
        self.last_ts: dict[str, float] = {t["id"]: 0.0 for t in cfg.towers}
        self.status: dict[str, str] = {t["id"]: "ONLINE" for t in cfg.towers}
        self.tickets: list[dict] = []
        self.blind_zones: set[str] = set()
        self._n = 0
        bus.subscribe("edge/+/heartbeat", self._on_hb)

    def _on_hb(self, topic: str, msg: dict) -> None:
        self.last[msg["tower"]] = msg
        self.last_ts[msg["tower"]] = msg["ts"]

    def tick(self, now: float, night: bool, clock: str) -> None:
        for tid in self.status:
            hb = self.last.get(tid, {})
            if now - self.last_ts[tid] > self.offline_after:
                st = "OFFLINE"
            elif hb.get("img_quality", 1) < 0.6 or hb.get("battery", 1) < 0.15:
                st = "DEGRADED"
            else:
                st = "ONLINE"
            if st != self.status[tid]:
                self._transition(tid, self.status[tid], st, clock, hb)
                self.status[tid] = st

        healthy = [t for t in self.cfg.towers if self.status[t["id"]] == "ONLINE"]
        along = self.cfg.camera["range_along_m"]
        blind = set()
        if night:
            # coverage is judged on each hotspot's crossing window, not the whole km zone:
            # one dead tower at a crossing is a blind crossing, even if the zone is "83% covered"
            def cov(towers, a, b):
                return _union_len([(max(a, t["x"] - along), min(b, t["x"] + along))
                                   for t in towers if t["x"] + along > a and t["x"] - along < b])
            for h in self.cfg.hotspots:
                a, b = h["x"] - self.window, h["x"] + self.window
                nominal = cov(self.cfg.towers, a, b)
                if nominal > 0 and cov(healthy, a, b) / nominal < self.blind_ratio:
                    blind.update(z.id for z in self.cfg.zones_for(a, b))
        if blind != self.blind_zones:
            self.bus.publish("agents/health", {"event": "blind_zones", "zones": sorted(blind)}, 60)
        self.blind_zones = blind

    def _transition(self, tid, old, new, clock, hb) -> None:
        self.bus.publish("agents/health", {"event": "status", "tower": tid, "from": old, "to": new}, 60)
        if new == "ONLINE":
            for tk in self.tickets:
                if tk["tower"] == tid and tk["status"] == "OPEN":
                    tk["status"], tk["closed"] = "RESOLVED", clock
            return
        self._n += 1
        issue = "no heartbeat (power/link)" if new == "OFFLINE" else \
            ("image quality low — lens fog/dirt" if hb.get("img_quality", 1) < 0.6 else "battery low")
        self.tickets.append({"id": f"MT-{self._n:03d}", "tower": tid, "issue": issue, "opened": clock,
                             "status": "OPEN", "assignee": "Field tech, Madukkarai"})

    def view(self) -> dict:
        return {"status": self.status, "blind_zones": sorted(self.blind_zones), "tickets": self.tickets[-10:],
                "telemetry": {k: {kk: v.get(kk) for kk in ("battery", "img_quality", "link", "temp_c")}
                              for k, v in self.last.items()}}
