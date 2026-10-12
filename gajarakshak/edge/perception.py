"""PerceptionAgent — runs on the tower's edge box (Jetson Orin class).

Production pipeline per camera:  decode → detector (thermal+RGB, TensorRT INT8) → tracker
(ByteTrack) → geo-projection (pixel → chainage/lateral via per-camera homography) → this module.

What this module owns (and the demo implements for real):
  * M-of-N confirmation — a track must be seen 3 times before it is published (kills flicker FPs)
  * per-track elephant evidence (mean elephant confidence across frames)
  * velocity estimate → heading relative to the track
  * edge-side filtering: only elephant-relevant tracks leave the tower (~300 B JSON, not video)
"""
from __future__ import annotations

import json
import math

from ..bus import Bus
from ..models import Detection, Track

GATE_M = 28.0
CONFIRM_HITS = 3
MAX_MISSES = 6


class PerceptionAgent:
    def __init__(self, tower_id: str, bus: Bus, publish_min: float, source: str = "sim"):
        self.tower_id = tower_id
        self.source = source          # "sim" for simulated cameras, "camera" for a real detector
        self.bus = bus
        self.publish_min = publish_min
        self.tracks: dict[str, Track] = {}
        self._n = 0

    def process(self, dets: list[Detection], t: float, dt: float) -> None:
        unmatched = list(dets)
        # greedy nearest-neighbour association (ByteTrack-lite)
        pairs = []
        for tr in self.tracks.values():
            px, py = tr.x + tr.vx * dt, tr.y + tr.vy * dt
            for i, d in enumerate(unmatched):
                dist = math.hypot(d.x - px, d.y - py)
                if dist <= GATE_M:
                    pairs.append((dist, tr.id, i))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, i in pairs:
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid)
            used_d.add(i)
            self._update(self.tracks[tid], unmatched[i], t, dt)
        for tid, tr in list(self.tracks.items()):
            if tid not in used_t:
                tr.misses += 1
                if tr.misses > MAX_MISSES:
                    del self.tracks[tid]
                    if tr.confirmed and tr.elephant_score >= self.publish_min:
                        self.bus.publish(f"edge/{self.tower_id}/tracks", {"event": "lost", "track": tr.to_msg(), "_gt": tr.gt_id, "source": self.source}, 80)
        for i, d in enumerate(unmatched):
            if i not in used_d:
                self._n += 1
                tid = f"{self.tower_id}-{self._n}"
                self.tracks[tid] = Track(tid, self.tower_id, d.x, d.y, hits=1,
                                         elephant_conf_sum=d.conf if d.label == "elephant" else 0.0,
                                         first_seen=t, last_seen=t,
                                         gt_votes={d.gt_id: 1} if d.gt_id else {})

        for tr in self.tracks.values():
            if tr.last_seen == t and tr.hits >= CONFIRM_HITS and tr.elephant_score >= self.publish_min:
                ev = "update" if tr.confirmed else "confirmed"
                tr.confirmed = True
                msg = {"event": ev, "track": tr.to_msg(), "_gt": tr.gt_id, "source": self.source}  # _gt: SIM-ONLY eval tag
                self.bus.publish(f"edge/{self.tower_id}/tracks", msg, len(json.dumps(msg["track"])))

    @staticmethod
    def _update(tr: Track, d: Detection, t: float, dt: float) -> None:
        a = 0.5
        nx, ny = a * d.x + (1 - a) * (tr.x + tr.vx * dt), a * d.y + (1 - a) * (tr.y + tr.vy * dt)
        span = max(t - tr.last_seen, dt)
        b = 0.25
        tr.vx = b * (nx - tr.x) / span + (1 - b) * tr.vx
        tr.vy = b * (ny - tr.y) / span + (1 - b) * tr.vy
        tr.x, tr.y = nx, ny
        tr.hits += 1
        tr.misses = 0
        tr.last_seen = t
        if d.label == "elephant":
            tr.elephant_conf_sum += d.conf
        if d.gt_id:
            tr.gt_votes[d.gt_id] = tr.gt_votes.get(d.gt_id, 0) + 1
