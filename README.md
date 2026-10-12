# GajaRakshak — elephant–train collision prevention (working demo)

Working system for the Ettimadai–Walayar section (Palakkad Division, Southern Railway), where the A and B lines
cut through elephant corridors between the Walayar forest and the water bodies to the south.

```
 REAL detector (perception/) ──┐
                                ├─ edge PerceptionAgent ──MQTT (EMQX)──► RiskEngine ──► AlertDispatcher ──► pilot / SM / controller
 simulated towers (sim/) ──────┘   tracks + keyframes                    (rules)        (ack, escalation)    (SMS · IVR · VHF · Kavach)
                                                                            ▲   │
         VerificationAgent (Groq vision on keyframes) ── borderline ────────┤   └──► Postgres + PostGIS + Timescale (audit, spatial)
         DeviceHealthAgent ── blind zones ──────────────────────────────────┘
         OperatorAssistant (Groq LLM + read-only tools)  ·  InsightsAgent (handover note)        ← off the safety path
```

## Three ways to run it

**1 · Laptop (no infrastructure)** — in-process bus, SQLite. Everything works; LLM features need a key.
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
export GROQ_API_KEY=gsk_...                       # optional
.venv/bin/uvicorn gajarakshak.api:app --port 8000  # http://localhost:8000
```

**2 · Full stack** — real EMQX broker, PostgreSQL with PostGIS + TimescaleDB, the app in a container.
```bash
cp .env.example .env    # put your GROQ_API_KEY in it
docker compose up --build
# app http://localhost:8000 · EMQX console http://localhost:18083 (admin / public) · Postgres localhost:5432
```
The bar under the KPIs shows what is live: `MQTT … sent/recv`, `Postgres +PostGIS +Timescale`, `Groq <model>`.

**3 · Real perception** — a real detector (YOLOX-S, ONNX Runtime) on real photos.
```bash
pip install -r perception/requirements.txt
python -m perception.download_model                 # 36 MB, checksum-verified
python -m perception.run report --verify             # → perception/out/report.html
python -m perception.run stream --mqtt mqtt://localhost:1883 --seconds 120   # needs mode 2 running
```
`report` runs the detector, projects each detection onto the track with the camera homography and applies the real
rule table (and, with `--verify`, asks the Groq vision model). `stream` acts as a live tower camera: frames go
through the same edge `PerceptionAgent` and are published to the broker; the dashboard shows a **REAL CAMERA**
track and raises STOP/CAUTION from it. Add `--conf-scale 0.6` to mimic a hard frame (range, fog): the track becomes
borderline, the edge uploads a keyframe, and the cloud VerificationAgent sends it to the Groq vision model.
Keep the dashboard speed at 1× or 4× while streaming.

**Your laptop camera as the tower camera:**
```bash
python -m perception.run stream --webcam --seconds 300            # preview window opens; q to quit
python -m perception.run stream --webcam --dry-run                # test camera + detector only, publish nothing
```
Hold a phone (or a printed photo) showing elephants up to the camera: the real detector finds them, the edge confirms
them over 3 frames and publishes to the broker, and the dashboard raises the alert. People in front of the camera are
detected as `person` and dropped at the edge — they never leave the laptop. Detection runs locally (no Groq tokens);
only a borderline track sends one crop to the Groq vision model.

## LLM: Groq

| Feature | Model (env) | Where | On the safety path? |
|---|---|---|---|
| Operator assistant — tool-calling agent | `GROQ_MODEL` = `openai/gpt-oss-120b` (`GROQ_REASONING_EFFORT`=medium) | `agents/assistant.py`, `POST /api/assistant` | No — read-only tools |
| Shift handover note | same | `agents/insights.py`, `GET /api/insights?llm=true` | No |
| Vision second opinion on borderline tracks | `GROQ_VISION_MODEL` = `qwen/qwen3.8-27b` | `agents/verification.py` | No — fails safe via 8 s SLA |

Groq's free tier allows about 3 vision calls a minute (each image counts as ~2.1k input tokens against a 7k/min limit), so vision calls are paced by `GROQ_VISION_RPM` (default 3), 429s are retried after Groq's "try again in" delay, the report verifies only the best detection per image (`--verify-max`), and the cloud verifier makes one call per herd. All calls go through `gajarakshak/llm.py` (OpenAI-compatible REST over httpx; retries on 429/5xx).
Without a key, the assistant answers with a deterministic situation report and the verifier leaves real-camera
tracks unverified (which the risk engine treats as CAUTION if they sit on the track). Check current model IDs at
https://console.groq.com/docs/models — vision models on Groq change more often than chat models.

**Production note:** Groq is hosted outside India. The chat model is open-weights, so for production the same model
moves to a self-hosted vLLM endpoint in an Indian region by changing `GROQ_BASE_URL` only.

## 5-minute demo script

| Step | Do | Say |
|---|---|---|
| 1 | **Reset (quiet)**, speed **10×** | "13 km section, 18 towers, thermal + RGB. Night, 19:00. Bar on top: real MQTT broker, Postgres/PostGIS, Groq." |
| 2 | **🐘 Herd crossing** | "Herd of 5 at H3. Express on B, freight on A, 4–5 km out." |
| 3 | watch ADVISORY → CAUTION → STOP | "Edge confirms in 3 frames, sends 300 bytes over MQTT. A rule table — not an LLM — decides." |
| 4 | open an alert's **audit trail** | "Who was told, which channel, how long the ack took. SMS → IVR → VHF." |
| 5 | **Ask the control room** → "Explain alert AL-…" | "This is the agentic part: gpt-oss-120b on Groq plans tool calls against the live system. Tool trace shown. Read-only by design." |
| 6 | **🔥 False positive** + show ground truth | "Warm rock and a gaur: verifier rejects, no train slowed." |
| 7 | **📵 Camera failure** | "Dead camera at night → BLIND, 40 km/h + ticket. A dead camera never looks like 'no elephants'." |
| 8 | terminal: `perception.run stream` | "Now a real model on a real photo, published from a separate process into the same broker — the alert you see comes from it." |
| 9 | open `perception/out/report.html` | "Detector, projection, rule table, vision second opinion — and the confusers dropped at the edge." |

## What is real vs simulated

| Real | Simulated / stand-in |
|---|---|
| Risk rule table, zones, fail-safes; alert lifecycle, hysteresis, ack ladder, audit | Elephant / gaur / boar / people movement (`sim/world.py`) |
| Edge tracking, 3-of-5 confirmation, filtering, keyframe upload | Detector output for the 18 *simulated* towers (`sim/sensors.py`) |
| **MQTT** transport at the edge→cloud boundary (EMQX / any broker) | Train positions and braking (stand-in for RTIS) |
| **Postgres** store; **PostGIS** spatial queries; Timescale hypertable when available | SMS / IVR / VHF / Kavach delivery (`core/rail.py`) |
| **Groq** tool-calling assistant, LLM handover note, vision verifier for real-camera tracks | Verifier for *simulated* tracks (uses ground truth at 97%) |
| **YOLOX** detector + homography projection on real images (`perception/`) | Camera calibration is a demo homography; RGB model, not thermal-tuned |

Edge and core run in one process for the simulated towers, but every edge message still crosses the broker;
the perception demo is a genuinely separate publisher. Splitting the simulator into its own container is a
deployment change — the MQTT topic contract is already the boundary.

## Tests

```bash
.venv/bin/pytest -q tests                      # unit + mocked-Groq agent tests (no network)
amqtt &                                        # or: docker compose up emqx db
GR_TEST_MQTT_URL=mqtt://localhost:1883 \
GR_TEST_DATABASE_URL=postgresql://postgres:gajarakshak@localhost:5432/gajarakshak \
  .venv/bin/pytest -q tests/test_infra.py      # real broker + Postgres/PostGIS
```

## Benchmark (synthetic, 24 h × 5 seeds, stress world ≈14 herd crossings/day)

| Metric | No system | Blanket 45 km/h at night | GajaRakshak (SMS/IVR/VHF) | + Kavach TSR |
|---|---|---|---|---|
| Collisions / day | 1.4 | 1.2 | **0.0** | **0.0** |
| Train delay, total min / day | 0 | 402.9 | **74.1** | 74.8 |
| Actionable alerts / day (false) | — | — | 41.0 (1.6) | 41.0 (1.6) |
| Warning lead time to train, p50 | — | — | 291 s | 317 s |
| Pilot ack latency, p95 | — | — | 57.5 s | 13.8 s |

Validates the decision logic under the simulator's assumptions; the no-system collision rate is deliberately
exaggerated. Real detector accuracy and safety impact need phase-0 field data.

## Layout

```
gajarakshak/
  llm.py        Groq client (chat, tools, vision)          bus.py   in-process + MQTT bus
  store.py      SQLite + Postgres/PostGIS/Timescale        api.py   FastAPI, WebSocket, env wiring
  sim/          world, sensors, scenarios                  edge/    PerceptionAgent
  core/         risk (rules), dispatch, rail adapter       agents/  assistant, verification, health, insights
  runtime.py    wiring + evaluation                        bench.py A/B benchmark
perception/     YOLOX detector, calibration, report + stream CLI, sample images
config/         section geometry and every threshold
dashboard/      control room UI (vanilla JS + canvas)
```
