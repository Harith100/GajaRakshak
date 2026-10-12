"""LLM-backed agents against a mocked Groq API (no network, no key needed).

The mock speaks the real OpenAI-compatible wire format Groq uses (tools / tool_calls /
role=tool), so these tests exercise the actual agent loop and request shapes.
"""
import asyncio
import json

import httpx

from gajarakshak.agents.assistant import TOOLS, OperatorAssistant, Tools
from gajarakshak.agents.insights import llm_summary
from gajarakshak.agents.verification import GroqVisionVerifier
from gajarakshak.llm import GroqClient
from gajarakshak.runtime import Runtime


def _reply(message: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})


def crossing_runtime() -> Runtime:
    rt = Runtime(auto_spawn=False)
    rt.scenario("herd_crossing")
    rt.run_for(200)                     # alerts are open and trains restricted by now
    return rt


def test_assistant_runs_tool_loop_against_groq_wire_format():
    rt = crossing_runtime()
    alert_id = next(iter(rt.dispatch.active.values())).id
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        assert req.headers["authorization"] == "Bearer test-key"
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["reasoning_effort"] == "medium" and body["max_completion_tokens"] >= 2048
        assert {t["function"]["name"] for t in body["tools"]} == {t["function"]["name"] for t in TOOLS}
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        if not tool_msgs:   # step 1: the model plans two parallel tool calls
            return _reply({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_situation", "arguments": "{}"}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "get_alert", "arguments": json.dumps({"alert_id": alert_id})}}]})
        # step 2: the model answers from what the tools returned
        alert = json.loads(next(m["content"] for m in tool_msgs if m["tool_call_id"] == "call_2"))
        assert alert["id"] == alert_id and alert["audit_trail"]
        return _reply({"role": "assistant", "content": f"<think>plan</think>{alert_id} is {alert['level']}."})

    client = GroqClient(api_key="test-key", transport=httpx.MockTransport(handler))
    out = asyncio.run(OperatorAssistant(lambda: rt, client).ask("What is happening at H3?"))
    assert out["degraded"] is False
    assert [t["tool"] for t in out["trace"]] == ["get_situation", "get_alert"]
    assert out["answer"].startswith(alert_id) and "<think>" not in out["answer"]
    assert len(seen) == 2


def test_assistant_degrades_without_key_and_on_api_error():
    rt = crossing_runtime()
    out = asyncio.run(OperatorAssistant(lambda: rt, GroqClient(api_key="")).ask("status?"))
    assert out["degraded"] and "active alert" in out["answer"]

    def boom(req):
        return httpx.Response(401, json={"error": {"message": "invalid key"}})
    out = asyncio.run(OperatorAssistant(lambda: rt, GroqClient(api_key="bad", transport=httpx.MockTransport(boom)))
                      .ask("status?"))
    assert out["degraded"] and "Groq unavailable" in out["reason"]


def test_tools_are_read_only_and_robust():
    rt = crossing_runtime()
    tools = Tools(lambda: rt)
    before = (len(rt.dispatch.active), [t.speed for t in rt.world.trains])
    for spec in TOOLS:
        name = spec["function"]["name"]
        args = {"get_alert": {"alert_id": "AL-0001"}, "get_zone": {"zone_id": "Z08"}, "locate": {"km": 7.9}}.get(name, {})
        res = tools.call(name, args)
        assert isinstance(res, dict) and "error" not in res, (name, res)
        json.dumps(res, default=str)
    assert tools.call("get_alert", {"alert_id": "AL-9999"})["error"]
    assert tools.call("nope", {})["error"]
    assert tools.call("get_zone", {"bad": 1})["error"]
    assert (len(rt.dispatch.active), [t.speed for t in rt.world.trains]) == before


def test_insights_and_vision_verifier_use_groq():
    rt = crossing_runtime()

    def handler(req):
        body = json.loads(req.content)
        content = body["messages"][-1]["content"]
        if isinstance(content, list):     # vision request: text + base64 image
            assert body["model"] == "qwen/qwen3.8-27b"
            assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            return _reply({"role": "assistant", "content": "YES. A large grey elephant."})
        return _reply({"role": "assistant", "content": "• All quiet."})

    client = GroqClient(api_key="k", transport=httpx.MockTransport(handler),
                        sync_transport=httpx.MockTransport(handler))
    assert asyncio.run(llm_summary(rt.stats(), client)) == "• All quiet."
    ok, note = GroqVisionVerifier(client).verify_image(b"\xff\xd8fakejpeg")
    assert ok and "elephant" in note


def test_real_camera_track_uses_vision_verifier_not_simulation():
    """A track from a real camera (no ground truth) with a keyframe goes to the vision model."""
    calls = []

    def handler(req):
        calls.append(1)
        return _reply({"role": "assistant", "content": "NO — this is a water buffalo."})

    client = GroqClient(api_key="k", sync_transport=httpx.MockTransport(handler))
    rt = Runtime(auto_spawn=False, vision=GroqVisionVerifier(client))
    import base64
    rt.bus.publish("edge/T12R/keyframes", {"track_id": "T12R-1", "jpeg_b64": base64.b64encode(b"jpg").decode()})
    msg = {"event": "confirmed", "source": "camera",
           "track": {"id": "T12R-1", "tower": "T12R", "x": 7900, "y": -20, "vx": 0, "vy": 0, "score": 0.45,
                     "hits": 3, "first_seen": 0}}
    for _ in range(5):
        rt.bus.publish("edge/T12R/tracks", msg)
        rt.step()
        import time
        time.sleep(0.05)
    assert calls, "vision model was not called"
    assert rt.tracks.tracks["T12R-1"].verified is False
    assert rt.verifier.recent[-1]["via"].startswith("groq:")


def test_groq_404_model_not_found_switches_to_available_model():
    """Groq returns 404 model_not_found when a model ID is wrong, retired or blocked for the org."""
    used = []

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "openai/gpt-oss-120b", "active": True}, {"id": "llama-3.1-8b-instant", "active": True}]})
        model = json.loads(req.content)["model"]
        used.append(model)
        if model == "llama-3.3-70b-versatile":
            return httpx.Response(404, json={"error": {"message": "The model `llama-3.3-70b-versatile` does not exist "
                                                       "or you do not have access to it.", "type": "invalid_request_error",
                                                       "code": "model_not_found"}})
        return _reply({"role": "assistant", "content": "ok"})

    client = GroqClient(api_key="k", model="llama-3.3-70b-versatile", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.chat([{"role": "user", "content": "hi"}]))["content"] == "ok"
    assert used == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    assert client.model == "openai/gpt-oss-120b"


def test_groq_error_message_is_surfaced():
    def handler(req):
        return httpx.Response(401, json={"error": {"message": "Invalid API Key", "code": "invalid_api_key"}})
    rt = crossing_runtime()
    out = asyncio.run(OperatorAssistant(lambda: rt, GroqClient(api_key="bad", transport=httpx.MockTransport(handler)))
                      .ask("status?"))
    assert out["degraded"] and "Invalid API Key" in out["reason"]


def test_rate_limit_429_waits_and_retries():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "Rate limit reached ... Please try again in 120ms.",
                                                       "code": "rate_limit_exceeded"}})
        return _reply({"role": "assistant", "content": "YES"})

    c = GroqClient(api_key="k", sync_transport=httpx.MockTransport(handler))
    c.vision_interval = 0
    assert c.classify_image(b"jpg", "elephant?") == "YES" and len(calls) == 2
