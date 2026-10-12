"""OperatorAssistant — an LLM agent (Groq, tool calling) for the control room. OFF the safety path.

The Section Controller asks in plain language ("why is the Sabari Express stopped?", "which
towers need a technician?"); the agent plans which tools to call, calls them against the live
system, and answers from what the tools returned.

Hard boundary: every tool is READ-ONLY. The assistant cannot open, acknowledge, downgrade or
clear an alert, and cannot touch a train. Safety decisions stay with the deterministic risk
engine and dispatcher; the LLM only explains them.

If GROQ_API_KEY is missing or Groq fails, the endpoint degrades to a deterministic situation
report built from the same tools — the operator still gets an answer.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from ..llm import GroqClient, LLMError, strip_think

MAX_STEPS = 6
MAX_TOOL_CHARS = 6000

SYSTEM_PROMPT = """You are the control-room assistant for GajaRakshak, the elephant–train collision
prevention system on the Ettimadai–Walayar section (Palakkad Division, Southern Railway).

Section facts:
- Chainage in km from Ettimadai (km 0) to Walayar (km 13). Zones Z01–Z13 are 1 km each (Z08 = km 7–8).
- Line A runs towards Palakkad (km increasing), line B towards Coimbatore (km decreasing).
- Hotspots: H1 Kanjikode nullah km 2.7, H2 Walayar dam fringe km 5.3, H3 Madukkarai crossing km 7.9, H4 Chavadi slope km 10.5.
- Alert levels: STOP (elephant ≤25 m from a line: train stops short of zone), CAUTION (≤60 m, or ≤150 m and
  approaching: 25 km/h), BLIND (camera coverage lost at a hotspot at night: 40 km/h), ADVISORY (inform only).
- Pilots are told by CUG SMS → IVR call → Station Master on VHF until they acknowledge.

Rules:
- Use the tools to get facts. Never invent alerts, trains, times or numbers. Times are the system clock (HH:MM).
- You are READ-ONLY. If asked to stop a train, clear or acknowledge an alert, or change a rule, say you cannot,
  and say who can (Section Controller on the dashboard / standard operating procedure).
- Answer in at most 8 short lines, plain text, most important first. Cite alert IDs, zones, lines and trains.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_situation",
        "description": "Current overview: clock, active alerts, trains in the section with speed limits, tower problems, blind zones.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "list_alerts",
        "description": "List alerts, newest first.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["active", "cleared", "all"], "description": "default active"},
            "limit": {"type": "integer", "description": "max alerts, default 10"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_alert",
        "description": "Full detail of one alert: audit trail, which trains were told, over which channel, and acknowledgements.",
        "parameters": {"type": "object", "properties": {
            "alert_id": {"type": "string", "description": "e.g. AL-0003"}}, "required": ["alert_id"]}}},
    {"type": "function", "function": {
        "name": "list_trains",
        "description": "Trains in or approaching the section: position, speed, active restriction, last cab messages.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_zone",
        "description": "One zone: km range, hotspots, covering towers, current risk per line, elephant tracks inside.",
        "parameters": {"type": "object", "properties": {
            "zone_id": {"type": "string", "description": "Z01 … Z13"}}, "required": ["zone_id"]}}},
    {"type": "function", "function": {
        "name": "get_tower_health",
        "description": "Camera tower status, telemetry and maintenance tickets. Omit tower_id for all towers with problems.",
        "parameters": {"type": "object", "properties": {
            "tower_id": {"type": "string", "description": "e.g. T12"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "locate",
        "description": "Spatial lookup for a point on the section: zone, nearest hotspot and nearest camera towers.",
        "parameters": {"type": "object", "properties": {
            "km": {"type": "number", "description": "chainage in km from Ettimadai"},
            "lateral_m": {"type": "number", "description": "metres from track centre, + north / forest side; default 0"}},
            "required": ["km"]}}},
    {"type": "function", "function": {
        "name": "get_kpis",
        "description": "Performance numbers since start: alerts by level, false alerts, lead time, pilot ack latency, delays, collisions.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "search_events",
        "description": "Raw event log (audit store), newest first, filtered by topic prefix such as core/alerts, agents/health, agents/verification.",
        "parameters": {"type": "object", "properties": {
            "topic_prefix": {"type": "string"}, "limit": {"type": "integer"}}, "required": []}}},
]


class Tools:
    """Read-only views over the live runtime. Called on the event-loop thread (same as the sim)."""

    def __init__(self, get_rt: Callable):
        self.get_rt = get_rt

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # a broken tool must not break the assistant
            return {"error": f"{type(e).__name__}: {e}"}

    # helpers ---------------------------------------------------------------
    @staticmethod
    def _km(x: float) -> float:
        return round(x / 1000, 2)

    def _train(self, t) -> dict:
        d = t.to_dict()
        return {"id": d["id"], "name": d["name"], "kind": d["kind"], "line": d["line"],
                "direction": "towards Palakkad" if d["dir"] > 0 else "towards Coimbatore",
                "km": self._km(d["front"]), "speed_kmph": d["speed_kmph"], "line_speed_kmph": d["vmax_kmph"],
                "limit_kmph": d["limit_kmph"], "emergency_brake": d["emergency"], "last_cab_messages": d["inbox"][-3:]}

    def _alert(self, a, rt) -> dict:
        age = (rt.world.t - a.opened_at) / 60
        return {"id": a.id, "zone": a.zone_id, "line": a.line, "level": a.level.name, "peak": a.max_level.name,
                "status": a.status, "reason": a.reason, "open_for_min": round(age, 1),
                "controller_acknowledged": a.controller_ack is not None}

    # tools -----------------------------------------------------------------
    def t_get_situation(self) -> dict:
        rt = self.get_rt()
        h = rt.health
        return {
            "clock": rt.world.clock(), "night": rt.world.is_night, "mode": rt.mode,
            "active_alerts": [self._alert(a, rt) for a in
                              sorted(rt.dispatch.active.values(), key=lambda a: -a.level)],
            "trains": [self._train(t) for t in rt.world.trains],
            "towers_not_online": {k: v for k, v in h.status.items() if v != "ONLINE"},
            "blind_zones": sorted(h.blind_zones),
            "open_tickets": sum(1 for t in h.tickets if t["status"] == "OPEN"),
            "collisions_so_far": len(rt.world.collisions),
        }

    def t_list_alerts(self, status: str = "active", limit: int = 10) -> dict:
        rt = self.get_rt()
        d = rt.dispatch
        act = sorted(d.active.values(), key=lambda a: -a.opened_at)
        hist = list(reversed(d.history))
        rows = act if status == "active" else hist if status == "cleared" else act + hist
        return {"clock": rt.world.clock(), "alerts": [self._alert(a, rt) for a in rows[:max(1, min(int(limit), 50))]]}

    def t_get_alert(self, alert_id: str) -> dict:
        rt = self.get_rt()
        a = rt.dispatch.by_id.get(alert_id.strip().upper())
        if a is None:
            return {"error": f"no alert {alert_id}", "known": list(rt.dispatch.by_id)[-10:]}
        deliveries = []
        for (aid, tid), dl in rt.dispatch.deliveries.items():
            if aid != a.id:
                continue
            tr = rt.rail.get(tid)
            deliveries.append({"train": tr.name if tr else tid, "instruction": "RELEASE" if dl.release else dl.level.name,
                               "channels_tried": [at.channel for at in dl.attempts],
                               "acknowledged": dl.acked_at is not None})
        out = self._alert(a, rt)
        out.update({"audit_trail": a.log[-25:], "current_deliveries": deliveries, "tracks": a.track_ids[:6]})
        return out

    def t_list_trains(self) -> dict:
        rt = self.get_rt()
        return {"clock": rt.world.clock(), "trains": [self._train(t) for t in rt.world.trains]}

    def t_get_zone(self, zone_id: str) -> dict:
        rt = self.get_rt()
        z = rt.cfg.zone_by_id.get(zone_id.strip().upper())
        if z is None:
            return {"error": f"no zone {zone_id}", "zones": [q.id for q in rt.cfg.zones]}
        along = rt.cfg.camera["range_along_m"]
        towers = [t["id"] for t in rt.cfg.towers if t["x"] + along > z.start and t["x"] - along < z.end]
        risk = {line: zr.level.name for (zid, line), zr in rt.risk_map.items() if zid == z.id}
        tracks = []
        for t in rt.tracks.tracks.values():
            if z.start - 200 <= t.x <= z.end + 200:
                tracks.append({"id": t.id, "km": self._km(t.x), "lateral_m": round(t.y), "score": round(t.score, 2),
                               "class": rt.risk.classify(t), "source": t.source,
                               "moving_towards_track": (t.y > 0 and t.vy < -0.1) or (t.y < 0 and t.vy > 0.1)})
        return {"zone": z.id, "km_from": z.start / 1000, "km_to": z.end / 1000, "hotspot": z.hotspot,
                "hotspots": [h["id"] + " " + h["name"] for h in rt.cfg.hotspots if z.start - 300 <= h["x"] <= z.end + 300],
                "towers": towers, "tower_status": {t: rt.health.status.get(t) for t in towers},
                "risk_by_line": risk or {"A": "NONE", "B": "NONE"},
                "active_alerts": [a.id for a in rt.dispatch.active.values() if a.zone_id == z.id],
                "elephant_tracks": tracks}

    def t_get_tower_health(self, tower_id: str | None = None) -> dict:
        rt = self.get_rt()
        h = rt.health
        if tower_id:
            tid = tower_id.strip().upper()
            if tid not in h.status:
                return {"error": f"no tower {tower_id}"}
            return {"tower": tid, "status": h.status[tid], "telemetry": h.view()["telemetry"].get(tid),
                    "tickets": [t for t in h.tickets if t["tower"] == tid]}
        return {"towers_total": len(h.status), "not_online": {k: v for k, v in h.status.items() if v != "ONLINE"},
                "blind_zones": sorted(h.blind_zones), "tickets": h.tickets[-10:]}

    def t_locate(self, km: float, lateral_m: float = 0.0) -> dict:
        rt = self.get_rt()
        x, y = float(km) * 1000, float(lateral_m)
        spatial = getattr(rt.store, "nearest_towers", None)
        res = spatial(x, y, 3) if spatial else None
        engine = "postgis" if res is not None else "python"
        if res is None:
            res = sorted(({"id": t["id"], "distance_m": round(((t["x"] - x) ** 2 + (t["y"] - y) ** 2) ** 0.5)}
                          for t in rt.cfg.towers), key=lambda r: r["distance_m"])[:3]
        zone = next((z.id for z in rt.cfg.zones if z.start <= x < z.end), None)
        hs = min(rt.cfg.hotspots, key=lambda h: abs(h["x"] - x))
        return {"km": km, "lateral_m": y, "zone": zone, "nearest_towers": res, "spatial_engine": engine,
                "nearest_hotspot": {"id": hs["id"], "name": hs["name"], "distance_m": round(abs(hs["x"] - x))}}

    def t_get_kpis(self) -> dict:
        s = self.get_rt().stats()
        keep = ("window", "alerts_total", "alerts_by_level", "actionable_alerts", "false_actionable_alerts",
                "detect_to_alert_p50_s", "lead_time_p50_s", "ack_p95_s", "escalations", "trains_restricted",
                "trains_passed", "train_delay_total_min", "collisions", "near_misses", "verification",
                "towers_online", "towers_total", "uplink_kb")
        return {k: s[k] for k in keep}

    def t_search_events(self, topic_prefix: str = "core/alerts", limit: int = 20) -> dict:
        rows = self.get_rt().store.recent(max(1, min(int(limit), 50)), topic_prefix or None)
        return {"events": [{"clock": r["clock"], "topic": r["topic"], "data": r["data"]} for r in rows]}


class OperatorAssistant:
    def __init__(self, get_rt: Callable, client: GroqClient | None = None):
        self.tools = Tools(get_rt)
        self.client = client or GroqClient()

    def describe(self) -> dict:
        return self.client.describe()

    @staticmethod
    def _clean_history(history) -> list[dict]:
        out = []
        for m in (history or [])[-6:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                out.append({"role": m["role"], "content": m["content"][:1500]})
        return out

    def fallback(self, reason: str) -> dict:
        s = self.tools.t_get_situation()
        lines = [f"[{s['clock']}] {len(s['active_alerts'])} active alert(s); {len(s['trains'])} train(s) in the section."]
        for a in s["active_alerts"][:5]:
            lines.append(f"• {a['id']} {a['level']} {a['zone']}/line {a['line']}: {a['reason']}")
        for t in s["trains"]:
            if t["limit_kmph"] is not None:
                lines.append(f"• {t['name']} line {t['line']} km {t['km']}: {t['speed_kmph']} km/h, limit {t['limit_kmph']}")
        if s["towers_not_online"]:
            lines.append("• Towers: " + ", ".join(f"{k} {v}" for k, v in s["towers_not_online"].items()))
        if s["blind_zones"]:
            lines.append("• Blind zones: " + ", ".join(s["blind_zones"]))
        return {"answer": "\n".join(lines), "trace": [{"tool": "get_situation", "args": {}, "ms": 0}],
                "model": None, "degraded": True, "reason": reason}

    async def ask(self, question: str, history=None) -> dict:
        question = (question or "").strip()[:1000]
        if not question:
            return {"answer": "Ask me about alerts, trains, zones or towers.", "trace": [], "model": None}
        if not self.client.enabled:
            return self.fallback("GROQ_API_KEY not set — deterministic situation report instead")

        t0 = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self._clean_history(history),
                    {"role": "user", "content": question}]
        trace: list[dict] = []
        answer = None
        try:
            for _ in range(MAX_STEPS):
                msg = await self.client.chat(messages, tools=TOOLS, max_tokens=2048)
                calls = msg.get("tool_calls") or []
                if not calls:
                    answer = strip_think(msg.get("content")) or \
                        "The model returned no text (token budget used up by reasoning). Try again or set GROQ_REASONING_EFFORT=low."
                    break
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for c in calls:
                    fn = c.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}") or {}
                    except json.JSONDecodeError:
                        args = {}
                    s = time.perf_counter()
                    result = self.tools.call(name, args)
                    trace.append({"tool": name, "args": args, "ms": round((time.perf_counter() - s) * 1000, 1),
                                  "error": result.get("error") if isinstance(result, dict) else None})
                    messages.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                                     "content": json.dumps(result, default=str)[:MAX_TOOL_CHARS]})
            if answer is None:
                answer = "I needed too many steps to answer that. Try a narrower question (an alert ID, a zone or a train)."
        except LLMError as e:
            out = self.fallback(f"Groq unavailable ({e}) — deterministic situation report instead")
            out["trace"] = trace + out["trace"]
            return out
        return {"answer": answer, "trace": trace, "model": self.client.model, "degraded": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000)}
