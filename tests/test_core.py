"""Safety-path unit tests. These are the tests a Railway safety reviewer would read first."""
from gajarakshak.bus import Bus, topic_matches
from gajarakshak.config import Config
from gajarakshak.core.risk import CloudTrack, RiskEngine
from gajarakshak.models import RiskLevel
from gajarakshak.runtime import Runtime
from gajarakshak.sim.world import KMPH, World

CFG = Config()


def trk(y, vy=0.0, score=0.9, verified=None, x=7900.0):
    return CloudTrack("t1", "T12", x, y, 0.0, vy, score, 0.0, 0.0, verified=verified)


# ── rule table ──────────────────────────────────────────────
def test_confirmed_on_track_is_stop_for_that_line():
    r = RiskEngine(CFG).evaluate([trk(-20)], set(), 0)
    assert r[("Z08", "A")].level == RiskLevel.STOP


def test_between_lines_stops_both_lines():
    r = RiskEngine(CFG).evaluate([trk(0)], set(), 0)
    assert r[("Z08", "A")].level == RiskLevel.STOP and r[("Z08", "B")].level == RiskLevel.STOP


def test_approaching_within_watch_is_caution():
    r = RiskEngine(CFG).evaluate([trk(140, vy=-1.0)], set(), 0)   # 120 m from B, 1 m/s towards it
    assert r[("Z08", "B")].level == RiskLevel.CAUTION


def test_moving_away_is_advisory_only():
    r = RiskEngine(CFG).evaluate([trk(140, vy=+1.0)], set(), 0)
    assert r[("Z08", "B")].level == RiskLevel.ADVISORY


def test_rejected_track_never_raises_risk():
    assert RiskEngine(CFG).evaluate([trk(-20, score=0.4, verified=False)], set(), 0) == {}


def test_unverified_on_track_fails_safe_after_sla():
    eng = RiskEngine(CFG)
    t = trk(-20, score=0.4)
    assert eng.evaluate([t], set(), 0)[("Z08", "A")].level == RiskLevel.ADVISORY      # verifier still inside SLA
    assert eng.evaluate([t], set(), 10)[("Z08", "A")].level == RiskLevel.CAUTION     # SLA blown → slow trains


def test_blind_zone_applies_to_both_lines():
    r = RiskEngine(CFG).evaluate([], {"Z08"}, 0)
    assert r[("Z08", "A")].level == RiskLevel.BLIND and r[("Z08", "B")].level == RiskLevel.BLIND


# ── train physics ───────────────────────────────────────────
def test_train_stops_short_of_stop_zone_when_warned_early():
    from gajarakshak.models import Restriction
    w = World(CFG, auto_spawn=False)
    tr = w.spawn_train("A", kind="freight", front=3000)
    tr.known["x"] = Restriction("x", "Z08", "A", 7000, 8000, 0.0, RiskLevel.STOP)
    for _ in range(900):
        w.step(1.0)
    assert tr.speed == 0 and tr.front < 7000


# ── bus ─────────────────────────────────────────────────────
def test_topic_wildcards():
    assert topic_matches("edge/+/tracks", "edge/T01/tracks")
    assert topic_matches("agents/#", "agents/health")
    assert not topic_matches("edge/+/tracks", "edge/T01/heartbeat")


# ── end to end ──────────────────────────────────────────────
def test_scripted_crossing_no_collision_and_trains_stopped():
    rt = Runtime(auto_spawn=False)
    rt.scenario("herd_crossing")
    min_speed = {}
    for _ in range(900):
        rt.step()
        for t in rt.world.trains:
            min_speed[t.name] = min(min_speed.get(t.name, 99), t.speed / KMPH)
    assert rt.world.collisions == []
    assert all(v < 1 for v in min_speed.values())        # both trains held for the crossing
    assert not rt.dispatch.active or all(a.level <= RiskLevel.ADVISORY for a in rt.dispatch.active.values())


def test_false_positive_does_not_stop_trains():
    rt = Runtime(auto_spawn=False)
    rt.scenario("false_positive")
    rt.run_for(600)
    stops = [a for a in rt.dispatch.history + list(rt.dispatch.active.values()) if a.max_level == RiskLevel.STOP]
    assert not stops


def test_camera_failure_creates_blind_zone_and_ticket():
    rt = Runtime(auto_spawn=False)
    rt.scenario("camera_failure")
    rt.run_for(120)
    assert rt.health.blind_zones and rt.health.tickets
