"""RailGateway — the only module that talks to Indian Railways systems.

Production adapters (one class per integration, same interface):
  positions():    RTIS (ISRO-GAGAN loco tracking) / COA feed via the division's control office
  send():         CUG SMS gateway · IVR outbound call (pilot presses 1 to acknowledge)
                  · Station Master relays on VHF · (phase 3) Kavach TSR via Stationary Kavach
The demo adapter below drives the simulated trains in sim/world.py.
"""
from __future__ import annotations

from ..models import Restriction


class SimRailGateway:
    def __init__(self, world):
        self.world = world

    def positions(self, line: str | None = None):
        return [t for t in self.world.trains if line is None or t.line == line]

    def get(self, train_id: str):
        return next((t for t in self.world.trains if t.id == train_id), None)

    def apply(self, train_id: str, r: Restriction) -> bool:
        t = self.get(train_id)
        if t is None:
            return False
        t.known[r.alert_id] = r
        return True

    def release(self, train_id: str, alert_id: str) -> None:
        t = self.get(train_id)
        if t is not None:
            t.known.pop(alert_id, None)

    def cab_message(self, train_id: str, text: str) -> None:
        t = self.get(train_id)
        if t is not None:
            t.inbox.append(text)
