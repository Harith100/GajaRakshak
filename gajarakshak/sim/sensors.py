"""Thermal + RGB camera model per tower.

Models what a fine-tuned detector on a Jetson would plausibly emit: per-frame detections with
class confusion (gaur and thermal artefacts are the realistic elephant confusers), position
noise, distance fall-off, day/night contrast differences, fog/lens degradation and outages.
"""
from __future__ import annotations

import random

from ..models import Detection

# P(predicted label | true kind)
CONFUSION = {
    "elephant": {"elephant": 0.93, "gaur": 0.04, "unknown": 0.03},
    "gaur":     {"elephant": 0.30, "gaur": 0.62, "unknown": 0.08},
    "boar":     {"elephant": 0.02, "boar": 0.90, "unknown": 0.08},
    "cattle":   {"elephant": 0.06, "cattle": 0.86, "unknown": 0.08},
    "human":    {"elephant": 0.005, "human": 0.97, "unknown": 0.025},
    "hotspot":  {"elephant": 0.75, "unknown": 0.25},
}


class Camera:
    def __init__(self, tower: dict, cam_cfg: dict):
        self.id = tower["id"]
        self.x = tower["x"]
        self.link = tower.get("link", "ofc")
        self.along = cam_cfg["range_along_m"]
        self.lateral = cam_cfg["range_lateral_m"]
        self.online = True
        self.quality = 1.0            # 1.0 clean lens; fog / dirt / spider webs reduce it
        self.battery = 0.92
        self.last_heartbeat = -1e9

    def covers(self, x: float, y: float) -> bool:
        return abs(x - self.x) <= self.along and abs(y) <= self.lateral

    def detect(self, objects, night: bool, t: float, rng: random.Random) -> list[Detection]:
        if not self.online:
            return []
        out = []
        base = 0.93 if night else 0.84      # thermal contrast is worse on a hot afternoon
        for gt_id, kind, x, y in objects:
            if not self.covers(x, y):
                continue
            dn = max(abs(x - self.x) / self.along, abs(y) / self.lateral)
            p = base * (1 - 0.35 * dn) * self.quality
            if rng.random() > p:
                continue
            probs = CONFUSION[kind]
            label = rng.choices(list(probs), weights=list(probs.values()))[0]
            correct = (label == kind) or (kind == "hotspot" and label == "unknown")
            mu = 0.84 if correct else 0.56
            conf = min(0.99, max(0.05, rng.gauss(mu, 0.08) * (1 - 0.15 * dn) * (0.6 + 0.4 * self.quality)))
            noise = 4 + 10 * dn
            out.append(Detection(self.id, t, label, conf, x + rng.gauss(0, noise), y + rng.gauss(0, noise), gt_id))
        return out

    def heartbeat(self, t: float, rng: random.Random) -> dict | None:
        if not self.online:
            return None
        self.battery = max(0.05, min(1.0, self.battery + rng.uniform(-0.0004, 0.0003)))
        return {"tower": self.id, "ts": t, "img_quality": round(self.quality, 2), "battery": round(self.battery, 3),
                "link": self.link, "temp_c": round(rng.uniform(28, 41), 1)}
