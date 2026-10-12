"""Scripted scenarios for the live demo. Each injects ground truth / faults into a running world."""
from __future__ import annotations


def herd_crossing(rt) -> str:
    """Herd of 5 crosses at H3 while an express approaches on B line and a freight on A line."""
    w = rt.world
    hx = 7900
    h = w.spawn_herd(hx, n=5, side=1, linger_s=120, speed=0.9, start_dist=230)
    h.waypoints[0] = (h.waypoints[0][0], 135, 15)
    w.spawn_train("B", kind="express", front=hx + 5200)
    w.spawn_train("A", kind="freight", front=hx - 4300)
    return "Herd of 5 approaching H3 (km 7.9) · 1 express on B line, 1 freight on A line inbound"


def false_positive(rt) -> str:
    """Sun-warmed rock + a gaur near the track at H2: should be filtered/verified away, no trains slowed."""
    w = rt.world
    a = w.spawn_animal("hotspot", x=5300)
    a.y, a.ttl = 45.0, w.t + 400
    w.spawn_animal("gaur", x=5450)
    w.spawn_train("A", kind="passenger", front=5300 - 4000)
    return "Thermal hotspot + gaur near H2 (km 5.3) · passenger train inbound on A line"


def camera_failure(rt) -> str:
    """Tower T12 (covering H3) loses power at night → BLIND zone → precautionary 40 km/h."""
    rt.cameras["T12"].online = False
    rt.world.spawn_train("B", kind="express", front=7900 + 4500)
    return "Tower T12 offline (power) · express inbound on B line"


def fog(rt) -> str:
    rt.cameras["T05"].quality = 0.45
    rt.cameras["T06"].quality = 0.5
    return "Lens fog on T05/T06 (image quality ↓)"


def restore(rt) -> str:
    for c in rt.cameras.values():
        c.online, c.quality = True, 1.0
    return "All towers restored"


def rush(rt) -> str:
    w = rt.world
    for hx in (2700, 5300, 10500):
        w.spawn_herd(hx, start_dist=260)
    w.spawn_train("A", front=-2500)
    w.spawn_train("B", front=w.cfg.length + 2500)
    w.spawn_train("A", kind="freight", front=6000)
    return "Three herds at H1/H2/H4 + three trains"


SCENARIOS = {
    "herd_crossing": herd_crossing,
    "false_positive": false_positive,
    "camera_failure": camera_failure,
    "fog": fog,
    "restore": restore,
    "rush": rush,
}
