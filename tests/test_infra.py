"""Real infrastructure tests. Skipped unless the services are reachable:

    GR_TEST_MQTT_URL=mqtt://localhost:1883
    GR_TEST_DATABASE_URL=postgresql://postgres:gajarakshak@localhost:5432/gajarakshak

`docker compose up emqx db` provides both.
"""
import os
import time

import pytest

from gajarakshak.bus import MqttBus
from gajarakshak.config import Config
from gajarakshak.runtime import Runtime
from gajarakshak.store import PgEventStore

MQTT = os.environ.get("GR_TEST_MQTT_URL")
PG = os.environ.get("GR_TEST_DATABASE_URL")


@pytest.mark.skipif(not MQTT, reason="GR_TEST_MQTT_URL not set")
def test_herd_crossing_over_real_mqtt_broker():
    bus = MqttBus(MQTT, client_id="gr-test-core")
    assert bus.connected
    rt = Runtime(auto_spawn=False, bus=bus)
    rt.scenario("herd_crossing")
    for _ in range(900):
        rt.step()
        time.sleep(0.002)               # give the broker round trip a moment per tick
    d = bus.describe()
    assert d["sent"] > 100 and d["received"] > 100
    assert rt.world.collisions == []
    peak = max(a.max_level for a in rt.dispatch.history + list(rt.dispatch.active.values()))
    assert peak.name == "STOP"
    rt.close()


@pytest.mark.skipif(not MQTT, reason="GR_TEST_MQTT_URL not set")
def test_second_process_publisher_reaches_core():
    """Simulates the perception demo: a different MQTT client publishes a track."""
    core = MqttBus(MQTT, client_id="gr-test-core2")
    rt = Runtime(auto_spawn=False, bus=core)
    cam = MqttBus(MQTT, client_id="gr-test-cam", subscribe_remote=False)
    msg = {"event": "confirmed", "source": "camera",
           "track": {"id": "CAM-1", "tower": "T12R", "x": 7900.0, "y": -15.0, "vx": 0, "vy": 0,
                     "score": 0.9, "hits": 5, "first_seen": 0}}
    for _ in range(20):
        cam.publish("edge/T12R/tracks", msg)
        time.sleep(0.05)
        rt.step()
    assert "CAM-1" in rt.tracks.tracks
    assert any(a.level.name == "STOP" for a in rt.dispatch.active.values())
    cam.close()
    rt.close()


@pytest.mark.skipif(not PG, reason="GR_TEST_DATABASE_URL not set")
def test_postgres_store_records_alerts_and_events():
    cfg = Config()
    store = PgEventStore(PG, cfg, run_id="test" + str(int(time.time())))
    rt = Runtime(cfg, auto_spawn=False, store=store)
    rt.scenario("herd_crossing")
    rt.run_for(300)
    time.sleep(1.5)                     # writer thread flushes every 0.5 s
    ev = store.recent(50, "core/alerts")
    assert ev and ev[0]["topic"] == "core/alerts"
    with store.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM alerts WHERE run_id=%s", (store.run_id,))
        assert cur.fetchone()[0] >= 2
        cur.execute("SELECT count(*) FROM towers")
        assert cur.fetchone()[0] == len(cfg.towers)
    near = store.nearest_towers(7900, 0, 2)
    if store.features["postgis"]:
        assert near[0]["id"] in ("T11", "T12")
    else:
        assert near is None
    assert store.describe()["errors"] == 0
    rt.close()
