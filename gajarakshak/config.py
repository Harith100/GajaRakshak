"""Section configuration loading + geometry helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "section_walayar_ettimadai.json"


@dataclass(frozen=True)
class Zone:
    id: str
    start: float
    end: float
    hotspot: bool


class Config:
    def __init__(self, path: str | os.PathLike | None = None):
        path = Path(path or os.environ.get("GR_CONFIG", DEFAULT_CONFIG))
        self.raw = json.loads(path.read_text(encoding="utf-8"))
        s = self.raw["section"]
        self.section = s
        self.length = s["length_m"]
        self.lines = {l["id"]: l for l in s["lines"]}
        self.towers = self.raw["towers"]
        self.camera = self.raw["camera"]
        self.risk = self.raw["risk"]
        self.dispatch = self.raw["dispatch"]
        self.health = self.raw["health"]
        self.world = self.raw["world"]
        self.trains = self.raw["trains"]
        self.hotspots = self.raw["hotspots"]
        self.water_bodies = self.raw["water_bodies"]
        zl = s["zone_length_m"]
        self.zones: list[Zone] = []
        for i in range(int(self.length // zl)):
            a, b = i * zl, (i + 1) * zl
            hot = any(a - 300 <= h["x"] <= b + 300 for h in self.hotspots)
            self.zones.append(Zone(f"Z{i + 1:02d}", a, b, hot))
        self.zone_by_id = {z.id: z for z in self.zones}

    def zones_for(self, x0: float, x1: float) -> list[Zone]:
        return [z for z in self.zones if z.end > x0 and z.start < x1]

    def public(self) -> dict:
        """Static geometry for the dashboard."""
        return {
            "section": self.section, "towers": self.towers, "hotspots": self.hotspots,
            "water_bodies": self.water_bodies, "camera": self.camera,
            "zones": [z.__dict__ for z in self.zones],
        }
