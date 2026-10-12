"""Message bus with MQTT-style topics.

Topic contract (same in the demo and in production):

    edge/{tower}/tracks       confirmed tracks (JSON, ~300 B)              edge → cloud
    edge/{tower}/heartbeat    device health                                edge → cloud
    edge/{tower}/keyframes    JPEG crop of a borderline track (verifier)   edge → cloud
    core/risk, core/alerts    zone risk / alert lifecycle                  cloud internal
    rail/notify               messages to loco pilots / station masters    cloud → rail gateway
    agents/#                  agent outputs                                cloud internal

Two implementations:
  Bus      in-process, synchronous. Used by tests and the benchmark (deterministic).
  MqttBus  edge/# goes through a REAL MQTT broker (EMQX / Mosquitto). Anything publishing to the
           broker — the simulated towers in this process, or a separate process such as the
           perception demo — is received by the core. Incoming messages are queued by the paho
           network thread and delivered on the simulation thread by pump(), so no locks are
           needed in the core.
"""
from __future__ import annotations

import json
import logging
import queue
import ssl
import time
import uuid
from collections import defaultdict
from typing import Callable
from urllib.parse import urlparse

log = logging.getLogger("gajarakshak.bus")


def topic_matches(pattern: str, topic: str) -> bool:
    p, t = pattern.split("/"), topic.split("/")
    for i, part in enumerate(p):
        if part == "#":
            return True
        if i >= len(t):
            return False
        if part != "+" and part != t[i]:
            return False
    return len(p) == len(t)


class Bus:
    kind = "in-process"

    def __init__(self):
        self._subs: list[tuple[str, Callable[[str, dict], None]]] = []
        self.counters: dict[str, int] = defaultdict(int)
        self.bytes: dict[str, int] = defaultdict(int)

    def subscribe(self, pattern: str, fn: Callable[[str, dict], None]) -> None:
        self._subs.append((pattern, fn))

    def _account(self, topic: str, size_bytes: int) -> None:
        root = topic.split("/")[0]
        self.counters[root] += 1
        self.bytes[root] += size_bytes

    def _deliver(self, topic: str, msg: dict) -> None:
        for pattern, fn in self._subs:
            if topic_matches(pattern, topic):
                fn(topic, msg)

    def publish(self, topic: str, msg: dict, size_bytes: int = 0) -> None:
        self._account(topic, size_bytes)
        self._deliver(topic, msg)

    def pump(self, max_messages: int = 10_000) -> int:
        return 0

    def describe(self) -> dict:
        return {"kind": self.kind}

    def close(self) -> None:
        pass


class MqttBus(Bus):
    """edge/# crosses a real broker; core/, rail/ and agents/ stay in-process."""

    kind = "mqtt"
    REMOTE_PREFIX = "edge/"

    def __init__(self, url: str, client_id: str | None = None, subscribe_remote: bool = True,
                 connect_timeout: float = 10.0):
        super().__init__()
        import paho.mqtt.client as mqtt  # optional dependency

        u = urlparse(url)
        self.url = f"{u.scheme}://{u.hostname}:{u.port or (8883 if u.scheme == 'mqtts' else 1883)}"
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self.connected = False
        self.stats = {"sent": 0, "received": 0, "decode_errors": 0, "reconnects": 0}
        self._subscribe_remote = subscribe_remote
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=client_id or f"gr-{uuid.uuid4().hex[:8]}")
        if u.username:
            self.client.username_pw_set(u.username, u.password)
        if u.scheme == "mqtts":
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.connect_async(u.hostname, u.port or 1883, keepalive=30)
        self.client.loop_start()
        t0 = time.monotonic()
        while not self.connected and time.monotonic() - t0 < connect_timeout:
            time.sleep(0.05)
        if not self.connected:
            log.warning("MQTT broker %s not reachable yet; will keep retrying in the background", self.url)

    # paho network thread ---------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            log.error("MQTT connect refused: %s", reason_code)
            return
        if self.stats["sent"] or self.stats["received"]:
            self.stats["reconnects"] += 1
        self.connected = True
        if self._subscribe_remote:
            client.subscribe(self.REMOTE_PREFIX + "#", qos=1)
        log.info("MQTT connected to %s", self.url)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False
        (log.info if not reason_code.is_failure else log.warning)("MQTT disconnected (%s)", reason_code)

    def _on_message(self, client, userdata, m):
        self._q.put((m.topic, m.payload))

    # simulation thread -----------------------------------------------------
    def publish(self, topic: str, msg: dict, size_bytes: int = 0) -> None:
        if topic.startswith(self.REMOTE_PREFIX):
            payload = json.dumps(msg, separators=(",", ":"), default=str)
            self._account(topic, len(payload))            # real bytes on the wire
            qos = 0 if topic.endswith("/heartbeat") else 1
            self.client.publish(topic, payload, qos=qos)
            self.stats["sent"] += 1
            return                                        # delivered back to us by the broker → pump()
        super().publish(topic, msg, size_bytes)

    def pump(self, max_messages: int = 10_000) -> int:
        n = 0
        while n < max_messages:
            try:
                topic, payload = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                msg = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                self.stats["decode_errors"] += 1
                continue
            self.stats["received"] += 1
            self._deliver(topic, msg)
            n += 1
        return n

    def describe(self) -> dict:
        return {"kind": self.kind, "url": self.url, "connected": self.connected, **self.stats}

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


def make_bus(url: str | None) -> Bus:
    return MqttBus(url, client_id=f"gr-core-{uuid.uuid4().hex[:6]}") if url else Bus()
