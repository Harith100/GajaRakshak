"""InsightsAgent — reporting for Railways + Forest Dept. Off the safety path.

Deterministic stats are always computed. If GROQ_API_KEY is set, a Groq-hosted LLM turns
them into a short shift handover note for the control office. It runs on request, never
inside the simulation/safety loop — if the API is down you lose prose, not protection.
"""
from __future__ import annotations

import json

from ..llm import GroqClient, LLMError, strip_think

SYSTEM = ("You write shift handover notes for a Southern Railway section controller and the "
          "Tamil Nadu/Kerala forest range officer. Plain English, at most 6 bullet points, numbers only "
          "from the JSON provided, flag anything needing action. No preamble.")


async def llm_summary(stats: dict, client: GroqClient | None = None) -> str | None:
    client = client or GroqClient()
    if not client.enabled:
        return None
    try:
        msg = await client.chat([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": json.dumps(stats, default=str)}],
                                max_tokens=1500)
        return strip_think(msg.get("content")) or None
    except LLMError as e:  # never let reporting break anything
        return f"(LLM unavailable: {e}; showing template summary)\n\n" + template_summary(stats)


def template_summary(s: dict) -> str:
    z = ", ".join(f"{k} ({v})" for k, v in s["hot_zones"][:3]) or "none"
    lines = [
        f"Shift window {s['window']}: {s['alerts_total']} alerts ({s['alerts_by_level'].get('STOP', 0)} STOP, "
        f"{s['alerts_by_level'].get('CAUTION', 0)} CAUTION, {s['alerts_by_level'].get('BLIND', 0)} BLIND).",
        f"Busiest zones: {z}.",
        f"{s['trains_restricted']} train passes restricted; median warning lead time {s['lead_time_p50_s']} s; "
        f"pilot ack latency p95 {s['ack_p95_s']} s; {s['escalations']} escalations.",
        f"Verifier: {s['verification']['requested']} borderline tracks, {s['verification']['rejected']} rejected as non-elephant.",
        f"Devices: {s['towers_online']}/{s['towers_total']} towers online, {s['open_tickets']} open maintenance tickets.",
        f"Edge→cloud traffic {s['uplink_kb']} KB vs ~{s['video_equiv_gb']} GB if video were streamed 24×7.",
    ]
    if s["collisions"]:
        lines.append(f"⚠ {s['collisions']} collision(s) recorded — review required.")
    return "\n".join("• " + l for l in lines)
