"""VerificationAgent — second opinion on borderline detections, OFF the critical path.

Borderline = edge score in [conf_verify_low, conf_high). While a track waits here the
RiskEngine still treats it as a "candidate": ADVISORY normally, CAUTION if it has been on the
track longer than the verify SLA. So a slow or failed verifier can never make the system less
safe — only noisier.

Backends:
  SimulatedVerifier   simulated tracks only — a "heavier model" with ~97% accuracy, 2–5 s latency.
                      It reads simulation ground truth, so it is a stand-in, not a model.
  GroqVisionVerifier  REAL: the keyframe crop the edge uploaded (edge/{tower}/keyframes) goes to a
                      Groq-hosted vision model. Runs in a worker thread; the sim/safety loop only
                      polls the future, it never waits on the network.
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

from ..bus import Bus
from ..core.risk import RiskEngine, TrackStore
from ..llm import GroqClient

PROMPT = ("This image is a crop from a railway-side camera (thermal or colour) in a South Indian forest. "
          "Is the main subject an elephant? Gaur, buffalo, cattle, wild boar, people, vehicles, rocks and "
          "vegetation are NOT elephants. Reply with YES or NO first, then at most one short sentence.")


class SimulatedVerifier:
    accuracy = 0.97
    name = "simulated"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def latency(self) -> float:
        return self.rng.uniform(2.0, 5.0)

    def verify(self, track) -> bool:
        truth = bool(track.gt and track.gt.startswith("H"))  # herd members are the only elephants
        return truth if self.rng.random() < self.accuracy else not truth


class GroqVisionVerifier:
    def __init__(self, client: GroqClient | None = None):
        self.client = client or GroqClient()
        self.name = f"groq:{self.client.vision_model}"

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def verify_image(self, jpeg: bytes) -> tuple[bool, str]:
        text = self.client.classify_image(jpeg, PROMPT)
        head = text.strip().lstrip("*# ").upper()
        return head.startswith("YES"), text[:200]


class VerificationAgent:
    TIMEOUT_S = 90.0   # wall clock incl. rate-limit pacing; the risk engine's 8 s SLA already applies meanwhile
    KEYFRAME_WAIT_S = 4.0

    def __init__(self, store: TrackStore, risk: RiskEngine, bus: Bus, backend: SimulatedVerifier,
                 vision: GroqVisionVerifier | None = None):
        self.store, self.risk, self.bus, self.backend = store, risk, bus, backend
        self.vision = vision if (vision is not None and vision.enabled) else None
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="verify") if self.vision else None
        self.pending: dict[str, dict] = {}
        self._waiting: dict[str, float] = {}
        self.stats = {"requested": 0, "confirmed": 0, "rejected": 0, "reused": 0,
                      "vision_calls": 0, "errors": 0, "timeouts": 0, "unavailable": 0}
        self.recent: list[dict] = []
        # verdict memory: trackers fragment (occlusion, camera hand-off), so a new track id at
        # the spot we just judged inherits that verdict instead of re-triggering the pipeline.
        # Rejections expire after 10 min (a heat artefact can become a real elephant later).
        self.memory: list[tuple[float, float, bool, float]] = []   # (x, y, verdict, expires)
        self.memory_radius = 30.0

    def describe(self) -> dict:
        return {"simulated": self.backend.name, "vision": self.vision.name if self.vision else None}

    def _recall(self, t, now: float):
        best = None
        for x, y, v, exp in self.memory:
            if exp >= now and (x - t.x) ** 2 + (y - t.y) ** 2 <= self.memory_radius ** 2:
                if best is None or v:      # any positive verdict nearby wins (fail-safe)
                    best = v
        return best

    def _near_pending(self, t) -> bool:
        for tid, p in self.pending.items():
            o = self.store.tracks.get(tid)
            if p["kind"] == "vision" and o is not None and \
                    (o.x - t.x) ** 2 + (o.y - t.y) ** 2 <= self.memory_radius ** 2:
                return True
        return False

    def _record(self, clock, tid, score, verdict, via, note=""):
        rec = {"t": clock, "track": tid, "score": round(score, 2), "verdict": verdict, "via": via, "note": note}
        self.recent.append(rec)
        self.recent = self.recent[-20:]
        self.bus.publish("agents/verification", rec, 90)

    def _submit(self, t, now: float, clock: str) -> None:
        kf = self.store.keyframes.get(t.id)
        if t.source == "sim":
            self.pending[t.id] = {"kind": "sim", "due": now + self.backend.latency()}
        elif self.vision and kf:
            self.pending[t.id] = {"kind": "vision", "fut": self.pool.submit(self.vision.verify_image, kf),
                                  "t0": time.monotonic()}
            self.stats["vision_calls"] += 1
        else:
            # real camera but no vision backend / no keyframe: stay a candidate; the risk engine's
            # SLA rule turns it into CAUTION if it sits on the track. Never guess.
            self.stats["unavailable"] += 1
            self._record(clock, t.id, t.score, "unverified", "none",
                         "no vision model configured" if not self.vision else "no keyframe")
            return
        self.stats["requested"] += 1

    def tick(self, now: float, clock: str) -> None:
        self.memory = [m for m in self.memory if m[3] >= now]
        for t in self.store.tracks.values():
            if not t.verify_requested and self.risk.classify(t) == "candidate":
                if t.source != "sim" and self.vision and self._near_pending(t):
                    continue    # a herd: one vision call answers for neighbours (verdict memory), saves rate limit
                if t.source != "sim" and self.vision and t.id not in self.store.keyframes:
                    # the edge uploads the crop just after the track; give it a few seconds to arrive
                    if time.monotonic() - self._waiting.setdefault(t.id, time.monotonic()) < self.KEYFRAME_WAIT_S:
                        continue
                self._waiting.pop(t.id, None)
                t.verify_requested = True
                recalled = self._recall(t, now)
                if recalled is not None:
                    t.verified = recalled
                    self.stats["reused"] += 1
                    continue
                self._submit(t, now, clock)

        for tid, p in list(self.pending.items()):
            t = self.store.tracks.get(tid)
            note, via = "", self.backend.name
            if p["kind"] == "sim":
                if now < p["due"]:
                    continue
                del self.pending[tid]
                if t is None:
                    continue
                verdict = self.backend.verify(t)
            else:
                fut, via = p["fut"], self.vision.name
                if not fut.done():
                    if time.monotonic() - p["t0"] > self.TIMEOUT_S:
                        fut.cancel()
                        del self.pending[tid]
                        self.stats["timeouts"] += 1
                        if t is not None:
                            self._record(clock, tid, t.score, "unverified", via, "timeout")
                    continue
                del self.pending[tid]
                try:
                    verdict, note = fut.result()
                except Exception as e:  # network / API error: stay a candidate (fail-safe via SLA)
                    self.stats["errors"] += 1
                    if t is not None:
                        self._record(clock, tid, t.score, "unverified", via, f"error: {type(e).__name__}")
                    continue
                if t is None:
                    continue
            t.verified = verdict
            self.memory.append((t.x, t.y, verdict, now + (300 if verdict else 600)))
            key = "confirmed" if verdict else "rejected"
            self.stats[key] += 1
            self._record(clock, tid, t.score, key, via, note)
