"""Domain models shared across edge, core and agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class RiskLevel(IntEnum):
    NONE = 0
    ADVISORY = 1   # elephant in watch area, moving away / far: inform controller + forest team
    BLIND = 2      # sensor coverage lost in a hotspot at night: precautionary restriction
    CAUTION = 3    # elephant near/approaching track: speed restriction through zone
    STOP = 4       # elephant on/at track: stop short of zone


@dataclass
class Detection:
    """One per-frame detection emitted by an edge camera (pre-tracking)."""
    tower_id: str
    ts: float
    label: str
    conf: float
    x: float
    y: float
    gt_id: Optional[str] = None  # ground truth id — SIMULATION ONLY, used for evaluation, never for decisions


@dataclass
class Track:
    """A confirmed object track maintained on the edge node and mirrored in the cloud."""
    id: str
    tower_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    misses: int = 0
    elephant_conf_sum: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    confirmed: bool = False
    verified: Optional[bool] = None       # set by VerificationAgent
    verify_requested: bool = False
    verified_at: float = 0.0
    gt_votes: dict = field(default_factory=dict)  # SIM ONLY

    @property
    def elephant_score(self) -> float:
        return self.elephant_conf_sum / max(self.hits, 1)

    @property
    def gt_id(self) -> Optional[str]:
        if not self.gt_votes:
            return None
        return max(self.gt_votes.items(), key=lambda kv: kv[1])[0]

    def to_msg(self) -> dict:
        return {
            "id": self.id, "tower": self.tower_id, "x": round(self.x, 1), "y": round(self.y, 1),
            "vx": round(self.vx, 2), "vy": round(self.vy, 2), "score": round(self.elephant_score, 3),
            "hits": self.hits, "verified": self.verified, "first_seen": self.first_seen,
        }


@dataclass
class Restriction:
    alert_id: str
    zone_id: str
    line: str
    start_x: float
    end_x: float
    limit_mps: float
    level: RiskLevel


@dataclass
class Alert:
    id: str
    zone_id: str
    line: str
    level: RiskLevel
    reason: str
    opened_at: float
    updated_at: float
    status: str = "ACTIVE"            # ACTIVE | CLEARED
    pending_level: Optional[RiskLevel] = None
    pending_since: float = 0.0
    cleared_at: Optional[float] = None
    controller_ack: Optional[float] = None
    track_ids: list = field(default_factory=list)
    gt_ids: set = field(default_factory=set)  # SIM ONLY — for false-alert accounting
    max_level: RiskLevel = RiskLevel.NONE
    log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "zone": self.zone_id, "line": self.line, "level": self.level.name,
            "max_level": self.max_level.name, "reason": self.reason, "status": self.status,
            "opened_at": self.opened_at, "updated_at": self.updated_at, "cleared_at": self.cleared_at,
            "controller_ack": self.controller_ack, "tracks": self.track_ids[:6], "log": self.log[-8:],
        }
