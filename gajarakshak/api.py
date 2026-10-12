"""Control-room API + live dashboard.

    uvicorn gajarakshak.api:app --host 0.0.0.0 --port 8000

Environment (all optional — without them it runs fully in-process with SQLite):
  GR_MQTT_URL       mqtt://host:1883      edge/# goes through this broker (EMQX in docker-compose)
  GR_DATABASE_URL   postgresql://…        events, alerts, towers, zones in Postgres/PostGIS/Timescale
  GROQ_API_KEY      enables the operator assistant, LLM handover notes and the vision verifier
  GROQ_MODEL / GROQ_VISION_MODEL / GROQ_BASE_URL   see gajarakshak/llm.py
  GR_SEED, GR_DB (SQLite path)

All handlers are `async def` on purpose: they run on the same event loop as the simulation
loop, so there are no cross-thread races on runtime state. Blocking work (vision calls,
database writes) runs in worker threads that never touch runtime state.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents.assistant import OperatorAssistant
from .agents.insights import llm_summary
from .agents.verification import GroqVisionVerifier
from .bus import make_bus
from .config import Config
from .llm import GroqClient
from .runtime import MODES, Runtime
from .sim.scenarios import SCENARIOS
from .store import make_store

logging.basicConfig(level=os.environ.get("GR_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gajarakshak.api")

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
CFG = Config()
LLM = GroqClient()


def build_runtime(seed: int = 7, mode: str = "gajarakshak", auto_spawn: bool = True) -> Runtime:
    """Wire the runtime to the real infrastructure named in the environment."""
    import uuid
    run_id = uuid.uuid4().hex[:8]
    bus = make_bus(os.environ.get("GR_MQTT_URL"))
    store = make_store(os.environ.get("GR_DATABASE_URL"), CFG, run_id, os.environ.get("GR_DB"))
    vision = GroqVisionVerifier(LLM) if LLM.enabled else None
    rt = Runtime(CFG, seed=seed, mode=mode, auto_spawn=auto_spawn, bus=bus, store=store, vision=vision)
    rt.run_id = run_id
    log.info("runtime %s: bus=%s store=%s llm=%s", run_id, bus.describe(), store.describe(), LLM.describe())
    return rt


class State:
    rt: Runtime | None = None


def get_rt() -> Runtime:
    return State.rt


ASSISTANT = OperatorAssistant(get_rt, LLM)


async def sim_loop():
    last = time.perf_counter()
    acc = 0.0
    while True:
        await asyncio.sleep(0.05)
        now = time.perf_counter()
        el, last = now - last, now
        rt = State.rt
        if rt is None:
            continue
        if rt.paused:
            rt.bus.pump()          # keep draining the broker while paused
            continue
        acc += el * rt.speed
        n = 0
        while acc >= rt.dt and n < 600:
            rt.step()
            acc -= rt.dt
            n += 1
        if n >= 600:
            acc = 0.0


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    State.rt = build_runtime(seed=int(os.environ.get("GR_SEED", 7)))
    task = asyncio.create_task(sim_loop())
    yield
    task.cancel()
    State.rt.close()


app = FastAPI(title="GajaRakshak", version="0.2.0", lifespan=lifespan)


class SpeedIn(BaseModel):
    speed: float


class ModeIn(BaseModel):
    mode: str


class ResetIn(BaseModel):
    seed: int = 7
    auto_spawn: bool = True
    mode: str = "gajarakshak"


class AskIn(BaseModel):
    question: str = Field(..., max_length=1000)
    history: list[dict] = Field(default_factory=list)


@app.get("/healthz")
async def healthz():
    rt = State.rt
    return {"ok": rt is not None, "sim_t": rt.world.t if rt else None}


@app.get("/api/config")
async def config():
    return CFG.public()


@app.get("/api/system")
async def system():
    rt = State.rt
    return {"bus": rt.bus.describe(), "store": rt.store.describe(), "llm": LLM.describe(),
            "verifier": rt.verifier.describe(), "run_id": rt.run_id}


@app.get("/api/state")
async def state():
    return State.rt.snapshot()


@app.get("/api/alerts")
async def alerts():
    return State.rt.dispatch.alerts_view(100)


@app.post("/api/alerts/{alert_id}/ack")
async def ack(alert_id: str):
    if not State.rt.dispatch.controller_ack(alert_id):
        raise HTTPException(404, "unknown or already acknowledged")
    return {"ok": True}


@app.get("/api/events")
async def events(limit: int = 100, topic: str | None = None):
    return State.rt.store.recent(min(limit, 1000), topic)


@app.post("/api/scenario/{name}")
async def scenario(name: str):
    if name not in SCENARIOS:
        raise HTTPException(404, f"scenarios: {list(SCENARIOS)}")
    return {"ok": True, "text": State.rt.scenario(name)}


@app.post("/api/speed")
async def speed(body: SpeedIn):
    State.rt.speed = max(0.25, min(body.speed, 120))
    return {"speed": State.rt.speed}


@app.post("/api/pause")
async def pause():
    State.rt.paused = not State.rt.paused
    return {"paused": State.rt.paused}


@app.post("/api/mode")
async def mode(body: ModeIn):
    if body.mode not in MODES:
        raise HTTPException(400, f"modes: {MODES}")
    State.rt.set_mode(body.mode)
    return {"mode": body.mode}


@app.post("/api/reset")
async def reset(body: ResetIn):
    if body.mode not in MODES:
        raise HTTPException(400, f"modes: {MODES}")
    old = State.rt
    sp = old.speed
    State.rt = None                       # sim loop idles while we swap
    await asyncio.to_thread(old.close)    # stop MQTT client / DB writer of the old run
    rt = await asyncio.to_thread(build_runtime, body.seed, body.mode, body.auto_spawn)
    rt.speed = sp
    State.rt = rt
    return {"ok": True, "run_id": rt.run_id}


@app.get("/api/insights")
async def insights(llm: bool = False):
    data = State.rt.insights()
    if llm:
        text = await llm_summary(data["stats"], LLM)   # async HTTP: never blocks the sim loop
        if text:
            data["llm_summary"] = text
    return JSONResponse(data)


@app.post("/api/assistant")
async def assistant(body: AskIn):
    return await ASSISTANT.ask(body.question, body.history)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            rt = State.rt
            if rt is not None:
                await sock.send_json(rt.snapshot())
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        pass


app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="dashboard")
